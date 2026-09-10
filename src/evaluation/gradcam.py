"""Grad-CAM interpretability for the baseline CNN and DenseNet121.

Grad-CAM (Selvaraju et al., 2017) weights the final convolutional feature maps
by the gradient of a target score with respect to those maps, giving a coarse
map of which regions most influenced the decision.

SINGLE-LOGIT SUBTLETY
---------------------
Both project models emit ONE pneumonia logit, not a two-class softmax. So the
target score is not "the logit of the predicted class" - for a NORMAL
prediction the relevant score is the *negated* logit, because evidence for
normal is evidence against pneumonia. `target="predicted"` handles this
automatically; getting it wrong silently produces a heatmap that explains the
opposite decision.

DEVICE
------
Everything here runs fine on CPU. Grad-CAM is one forward plus one backward
pass per image, so a handful of report examples costs seconds - a GPU slot is
unnecessary unless you are running it over a whole split.

PRIVACY
-------
Unlike everything else in `src/evaluation/`, these outputs CONTAIN THE X-RAY
IMAGE. The repository data policy (`data/README.md`, `CONTRIBUTING.md`)
forbids committing raw or processed chest X-rays, so the default output
directory is `XRAY_OUTPUT_ROOT`, outside Git. Publishing a small number of
examples in the report PDF is a separate, deliberate team decision - the
dataset is publicly released and de-identified, but that call belongs in
`docs/decisions.md`, not in a default argument.

USAGE
-----
    from src.evaluation.gradcam import GradCAM, resolve_target_layer

    model.eval()
    with GradCAM(model, resolve_target_layer(model)) as cam:
        heatmap = cam(input_tensor, target="predicted")   # (H, W) in [0, 1]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Layer to hook, by model family. These are the last convolutional feature
# maps before global pooling - the standard Grad-CAM target, and the deepest
# point that still carries spatial structure.
KNOWN_TARGET_LAYERS = {
    "DenseNet121Binary": "backbone.features",
    "DenseNet": "features",        # torchvision DenseNet121 -> (B, 1024, 7, 7)
    "BaselineCNN": "features",     # src/models/baseline.py  -> (B, 256, 14, 14)
}


def resolve_target_layer(model: nn.Module, layer_name: Optional[str] = None) -> nn.Module:
    """Pick the convolutional layer to hook.

    Explicit `layer_name` wins. Otherwise a known architecture is matched by
    class name, and failing that the last `Conv2d` in the module tree is used -
    a reasonable default, but one worth stating in the report rather than
    leaving implicit.
    """
    if layer_name is not None:
        modules = dict(model.named_modules())
        if layer_name not in modules:
            raise ValueError(
                f"layer '{layer_name}' not found. Available (first 20): "
                f"{list(modules)[:20]}"
            )
        return modules[layer_name]

    class_name = type(model).__name__
    for known, attribute in KNOWN_TARGET_LAYERS.items():
        if known.lower() in class_name.lower():
            module = dict(model.named_modules()).get(attribute)
            if module is not None:
                return module

    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not conv_layers:
        raise ValueError(f"no Conv2d layer found in {class_name}; pass layer_name")
    return conv_layers[-1]


class GradCAM:
    """Grad-CAM for a single-logit binary classifier.

    Hooks are registered on construction and must be removed afterwards; use
    this as a context manager, or call `close()`, otherwise the hooks keep the
    activation graph alive and leak memory across images.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles: List[Any] = [
            target_layer.register_forward_hook(self._forward_hook)
        ]

    def _forward_hook(self, module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
        self._activations = output
        if output.requires_grad:
            # A tensor hook rather than register_full_backward_hook: it fires
            # for exactly this tensor, so it stays correct for architectures
            # (like DenseNet) whose blocks reuse and concatenate outputs.
            self._handles.append(output.register_hook(self._save_gradients))

    def _save_gradients(self, grad: torch.Tensor) -> None:
        self._gradients = grad

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target: str = "predicted",
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Return a (H, W) heatmap in [0, 1] for a single image.

        `target`:
          "pneumonia" - explain the evidence FOR pneumonia (score = logit)
          "normal"    - explain the evidence FOR normal    (score = -logit)
          "predicted" - whichever class the model actually predicted at
                        `threshold`, which is what a report figure should show
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.shape[0] != 1:
            raise ValueError(
                f"GradCAM handles one image at a time, got batch of "
                f"{input_tensor.shape[0]}"
            )

        self.model.zero_grad(set_to_none=True)
        self._activations = None
        self._gradients = None

        # Explicitly enable grad: callers routinely wrap evaluation in
        # torch.no_grad(), under which Grad-CAM silently cannot work.
        with torch.enable_grad():
            # Input gradients keep a frozen backbone connected to autograd.
            input_tensor = input_tensor.detach().clone().requires_grad_(True)
            logits = self.model(input_tensor)
            logit = logits.reshape(-1)[0]

            if target == "predicted":
                probability = torch.sigmoid(logit).item()
                target = "pneumonia" if probability >= threshold else "normal"

            if target == "pneumonia":
                score = logit
            elif target == "normal":
                score = -logit
            else:
                raise ValueError(
                    f"target must be 'pneumonia', 'normal' or 'predicted', got {target!r}"
                )

            score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "no activations/gradients captured - the target layer did not "
                "participate in the forward pass, or the model was run under "
                "torch.no_grad()"
            )

        activations = self._activations.detach()[0]     # (C, h, w)
        gradients = self._gradients.detach()[0]         # (C, h, w)

        # Channel importance = spatially averaged gradient (Grad-CAM eq. 1).
        weights = gradients.mean(dim=(1, 2))            # (C,)
        cam = torch.einsum("c,chw->hw", weights, activations)

        # ReLU: only regions with POSITIVE influence on the target score are
        # evidence for it; negative regions argue for the other class.
        cam = F.relu(cam)

        cam = F.interpolate(
            cam[None, None], size=input_tensor.shape[-2:],
            mode="bilinear", align_corners=False,
        )[0, 0]

        # Normalise to [0, 1]. An all-zero map (no positive evidence anywhere)
        # is returned as zeros rather than dividing by ~0 and amplifying noise.
        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) < 1e-12:
            return np.zeros(tuple(cam.shape), dtype=np.float32)
        cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.cpu().numpy().astype(np.float32)


def denormalize(tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Undo Normalize() so the heatmap can be drawn over the visible image."""
    image = tensor.detach().cpu().clone()
    if image.dim() == 4:
        image = image[0]
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    image = (image * std_t + mean_t).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def save_gradcam_figure(
    images: Sequence[np.ndarray],
    heatmaps: Sequence[np.ndarray],
    captions: Sequence[str],
    out_path: str | Path,
    *,
    alpha: float = 0.45,
    title: str = "Grad-CAM examples",
    columns: int = 4,
) -> Path:
    """Render an image/heatmap overlay grid.

    Imports matplotlib lazily so that importing this module for the GradCAM
    class alone does not pull in a plotting backend.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    if not (n == len(heatmaps) == len(captions)):
        raise ValueError("images, heatmaps and captions must have equal length")
    if n == 0:
        raise ValueError("nothing to plot")

    columns = min(columns, n)
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.0, rows * 3.4), squeeze=False)

    for index in range(rows * columns):
        ax = axes[index // columns][index % columns]
        ax.axis("off")
        if index >= n:
            continue
        image = images[index]
        if image.ndim == 3 and image.shape[2] == 3:
            image = image.mean(axis=2)  # the 3 channels are a replicated grayscale
        ax.imshow(image, cmap="gray")
        ax.imshow(heatmaps[index], cmap="jet", alpha=alpha)
        ax.set_title(captions[index], fontsize=9)

    fig.suptitle(f"{title}\nGrad-CAM shows network attention, not verified pathology")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def select_examples(
    predictions: "Any", threshold: float, n_per_outcome: int = 2, seed: int = 42
) -> Dict[str, List[int]]:
    """Pick representative TP / TN / FP / FN row indices for a report figure.

    Chooses the most confident example of each outcome first, since a
    confident mistake is far more informative to show than a borderline one.
    Returns positional indices into `predictions`.
    """
    from src.evaluation.error_analysis import classify_outcomes

    scored = classify_outcomes(predictions, threshold).reset_index(drop=True)
    chosen: Dict[str, List[int]] = {}
    for outcome in ("TP", "TN", "FP", "FN"):
        subset = scored[scored["outcome"] == outcome]
        if subset.empty:
            chosen[outcome] = []
            continue
        ordered = subset.sort_values("confidence", ascending=False)
        chosen[outcome] = ordered.head(n_per_outcome).index.tolist()
    return chosen
