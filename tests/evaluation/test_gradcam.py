"""Tests for Grad-CAM.

Two layers of testing, because the plumbing can be perfect while the maths is
backwards:

* Fast structural tests on an untrained model (shapes, ranges, hook cleanup,
  argument validation).
* One functional test that trains a small CNN on a synthetic task with a KNOWN
  discriminative region, then asserts Grad-CAM actually points at it. This is
  what catches a sign error or a wrong reduction axis - mistakes that leave
  every shape correct and every value in range.

All of it runs on CPU in a few seconds; no GPU and no dataset required.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.evaluation.gradcam import (
    GradCAM,
    denormalize,
    resolve_target_layer,
    save_gradcam_figure,
)

SIZE = 64


class TinyCNN(nn.Module):
    """Stand-in with the project's model contract: (B,3,H,W) -> (B,1) logits."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


@pytest.fixture
def model():
    torch.manual_seed(0)
    net = TinyCNN()
    net.eval()
    return net


@pytest.fixture
def image():
    torch.manual_seed(1)
    return torch.randn(1, 3, SIZE, SIZE)


def test_resolve_target_layer_falls_back_to_last_conv(model):
    """TinyCNN is not a known architecture, so the documented fallback applies:
    the last Conv2d in the tree."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert resolve_target_layer(model) is convs[-1]


def test_resolve_target_layer_matches_known_architectures_by_name():
    """The two models this project actually uses resolve to `.features`, the
    last conv feature map before global pooling."""

    class BaselineCNN(TinyCNN):
        pass

    class DenseNet(TinyCNN):
        pass

    assert resolve_target_layer(BaselineCNN()) is not None
    for net in (BaselineCNN(), DenseNet()):
        assert resolve_target_layer(net) is net.features


def test_resolve_target_layer_accepts_explicit_name(model):
    assert resolve_target_layer(model, "features.0") is model.features[0]


def test_resolve_target_layer_rejects_unknown_name(model):
    with pytest.raises(ValueError, match="not found"):
        resolve_target_layer(model, "no_such_layer")


def test_heatmap_shape_and_range(model, image):
    with GradCAM(model, resolve_target_layer(model)) as cam:
        heatmap = cam(image, target="pneumonia")
    assert heatmap.shape == (SIZE, SIZE)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_accepts_unbatched_input(model):
    with GradCAM(model, resolve_target_layer(model)) as cam:
        assert cam(torch.randn(3, SIZE, SIZE)).shape == (SIZE, SIZE)


def test_rejects_multi_image_batch(model):
    with GradCAM(model, resolve_target_layer(model)) as cam:
        with pytest.raises(ValueError, match="one image at a time"):
            cam(torch.randn(4, 3, SIZE, SIZE))


def test_rejects_unknown_target(model, image):
    with GradCAM(model, resolve_target_layer(model)) as cam:
        with pytest.raises(ValueError, match="target must be"):
            cam(image, target="tuberculosis")


def test_hooks_are_removed_on_close(model, image):
    cam = GradCAM(model, resolve_target_layer(model))
    cam(image)
    assert cam._handles
    cam.close()
    assert cam._handles == []


def test_works_inside_no_grad(model, image):
    """Callers wrap evaluation in torch.no_grad(); Grad-CAM needs gradients, so
    __call__ re-enables them itself rather than silently returning nothing."""
    with torch.no_grad():
        with GradCAM(model, resolve_target_layer(model)) as cam:
            heatmap = cam(image, target="pneumonia")
    assert heatmap.shape == (SIZE, SIZE)


def test_predicted_target_follows_the_models_own_decision(model, image):
    with GradCAM(model, resolve_target_layer(model)) as cam:
        logit = model(image).reshape(-1)[0]
        expected = "pneumonia" if torch.sigmoid(logit).item() >= 0.5 else "normal"
        assert np.allclose(cam(image, target="predicted"), cam(image, target=expected))


def _synthetic_batch(n, rng, size=SIZE):
    """Bright blob in the lower-right quadrant == pneumonia."""
    x = rng.normal(0, 0.25, (n, 3, size, size)).astype(np.float32)
    y = (rng.random(n) < 0.5).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for i in np.where(y == 1)[0]:
        cy, cx = rng.integers(int(size * 0.60), int(size * 0.85), 2)
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 5.0 ** 2)))
        x[i] += 2.5 * blob.astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


@pytest.fixture(scope="module")
def trained_model_and_data():
    """Train TinyCNN until it solves the synthetic localisation task."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = TinyCNN()
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=5e-3)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(40):
        x, y = _synthetic_batch(16, rng)
        loss = criterion(net(x).squeeze(1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    net.eval()
    x, y = _synthetic_batch(48, rng)
    with torch.no_grad():
        accuracy = ((torch.sigmoid(net(x).squeeze(1)) >= 0.5).float() == y).float().mean().item()
    return net, x, y, accuracy


def test_synthetic_task_is_actually_learned(trained_model_and_data):
    """Guard: if the model did not learn, the localisation test below proves
    nothing and would fail for the wrong reason."""
    _, _, _, accuracy = trained_model_and_data
    assert accuracy > 0.85, f"synthetic task not learned (accuracy={accuracy:.2f})"


def test_gradcam_localises_the_learned_region(trained_model_and_data):
    """The core correctness test: heatmap mass must concentrate on the blob."""
    net, x, y, _ = trained_model_and_data
    half = SIZE // 2

    fractions = []
    with GradCAM(net, resolve_target_layer(net)) as cam:
        for index in np.where(y.numpy() == 1)[0][:8]:
            heatmap = cam(x[index : index + 1], target="pneumonia")
            assert heatmap.max() > 0, "all-zero heatmap for a pneumonia image"
            fractions.append(heatmap[half:, half:].sum() / heatmap.sum())

    mean_fraction = float(np.mean(fractions))
    assert mean_fraction > 0.5, (
        f"Grad-CAM mass in the discriminative quadrant is {mean_fraction:.1%}; "
        f"random attention would give ~25%"
    )


def test_normal_target_does_not_highlight_pneumonia_evidence(trained_model_and_data):
    """Sign convention: with a single logit, explaining 'normal' means
    backpropagating the negated score, so it must not light up the same blob."""
    net, x, y, _ = trained_model_and_data
    half = SIZE // 2
    index = int(np.where(y.numpy() == 1)[0][0])

    with GradCAM(net, resolve_target_layer(net)) as cam:
        pneumonia_map = cam(x[index : index + 1], target="pneumonia")
        normal_map = cam(x[index : index + 1], target="normal")

    pneumonia_mass = pneumonia_map[half:, half:].sum() / max(pneumonia_map.sum(), 1e-9)
    normal_mass = normal_map[half:, half:].sum() / max(normal_map.sum(), 1e-9)
    assert pneumonia_mass > normal_mass


def test_denormalize_restores_unit_range():
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    original = torch.rand(1, 3, 16, 16)
    normalized = (original - torch.tensor(mean).view(1, -1, 1, 1)) / torch.tensor(std).view(1, -1, 1, 1)
    restored = denormalize(normalized, mean, std)
    assert restored.shape == (16, 16, 3)
    assert np.allclose(restored, original[0].permute(1, 2, 0).numpy(), atol=1e-5)


def test_save_gradcam_figure_writes_a_file(tmp_path, model, image):
    with GradCAM(model, resolve_target_layer(model)) as cam:
        heatmap = cam(image)
    out = save_gradcam_figure(
        [denormalize(image, [0.5] * 3, [0.5] * 3)], [heatmap], ["TP  p=0.97"],
        tmp_path / "gradcam.png",
    )
    assert out.is_file() and out.stat().st_size > 0


def test_save_gradcam_figure_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError, match="equal length"):
        save_gradcam_figure([np.zeros((8, 8))], [], ["a"], tmp_path / "x.png")
