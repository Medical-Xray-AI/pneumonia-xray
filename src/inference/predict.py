"""Checkpoint-based single-image, batch and manifest inference.

The model is rebuilt from the configuration stored inside the checkpoint, so
inference cannot drift from the run that produced it:

* architecture, dropout and freeze strategy come from ``checkpoint["config"]``;
* ImageNet weights are never downloaded (``pretrained`` is forced off because
  the trained weights are loaded immediately afterwards);
* preprocessing is Member 2's ``build_eval_transforms`` with the normalization
  statistics that training resolved from the train split and stored in the
  checkpoint - they are never recomputed here.

The locked test split can only be predicted together with the validation
``frozen_threshold.json`` whose model and run match the checkpoint.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataset import ChestXrayDataset
from src.evaluation.evaluate import load_predictions, validate_frozen_selection, validate_manifest_predictions
from src.inference.schema import PredictionRecord, records_to_frame, validate_threshold
from src.models.factory import create_model
from src.preprocessing.transforms import IMAGENET_MEAN, IMAGENET_STD, build_eval_transforms
from src.training.checkpointing import _torch_load
from src.training.reproducibility import resolve_device


class LockedTestError(RuntimeError):
    """Raised when the locked test split would be touched without a frozen selection."""


@dataclass
class LoadedModel:
    model: torch.nn.Module
    transform: Any
    config: Dict[str, Any]
    run_id: str
    model_name: str
    checkpoint_epoch: int
    checkpoint_path: Path
    device: torch.device

    @property
    def run_dir(self) -> Path:
        """``XRAY_OUTPUT_ROOT/<run_id>`` for a ``checkpoints/best.pt`` layout."""
        return self.checkpoint_path.parent.parent


def normalization_stats(config: Mapping[str, Any]) -> tuple[list, list]:
    """Return the train-derived statistics recorded by the trainer."""
    norm = config.get("normalization") or {}
    mean, std = norm.get("mean"), norm.get("std")
    if mean is None or std is None:
        if norm.get("type", "imagenet") != "imagenet":
            raise ValueError(
                "Checkpoint config has no stored dataset normalization statistics; "
                "inference will not recompute them. Re-export the checkpoint from scripts/train.py."
            )
        return list(IMAGENET_MEAN), list(IMAGENET_STD)
    if len(mean) != 3 or len(std) != 3 or any(float(value) <= 0 for value in std):
        raise ValueError("Stored normalization statistics must have 3 channels and positive std")
    return [float(v) for v in mean], [float(v) for v in std]


def load_model(checkpoint_path: str | Path, device: str | torch.device | None = "auto") -> LoadedModel:
    """Load a training checkpoint for inference in eval mode."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = _torch_load(path, map_location="cpu")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping) or "model" not in config or "data" not in config:
        raise ValueError(f"{path} does not contain a scripts/train.py configuration")
    if "model_state" not in checkpoint:
        raise ValueError(f"{path} does not contain model_state")
    run_id = str(checkpoint.get("run_id") or "").strip()
    if not run_id:
        raise ValueError(f"{path} has no run_id")

    model_config = copy.deepcopy(dict(config))
    model_config["model"] = dict(model_config["model"], pretrained=False)
    model = create_model(model_config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    resolved = resolve_device(str(device) if device is not None else "auto")
    model.to(resolved).eval()

    mean, std = normalization_stats(config)
    transform = build_eval_transforms(config["data"], mean=mean, std=std)
    return LoadedModel(
        model=model,
        transform=transform,
        config=dict(config),
        run_id=run_id,
        model_name=str(config["model"]["name"]),
        checkpoint_epoch=int(checkpoint.get("epoch", 0)),
        checkpoint_path=path,
        device=resolved,
    )


def load_image(path: str | Path) -> Image.Image:
    """Open an X-ray exactly like ``ChestXrayDataset`` does."""
    with Image.open(path) as opened:
        image = opened.convert("L").convert("RGB")
        image.load()
    return image


@torch.no_grad()
def predict_tensor(loaded: LoadedModel, images: torch.Tensor) -> torch.Tensor:
    """Pneumonia probabilities (after sigmoid) for a preprocessed batch."""
    if images.ndim != 4:
        raise ValueError(f"Expected a batch shaped [N, C, H, W], got {tuple(images.shape)}")
    logits = loaded.model(images.to(loaded.device)).float().reshape(images.shape[0], -1)
    if logits.shape[1] != 1:
        raise ValueError("Binary classifiers must emit exactly one logit per image")
    return torch.sigmoid(logits[:, 0]).cpu()


def predict_images(
    loaded: LoadedModel,
    paths: Sequence[str | Path],
    *,
    batch_size: int = 16,
    threshold: Optional[float] = None,
) -> List[PredictionRecord]:
    """Single-image or batch inference on arbitrary image files."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not paths:
        raise ValueError("No images were given")
    if threshold is not None:
        threshold = validate_threshold(threshold)
    records: List[PredictionRecord] = []
    for start in range(0, len(paths), batch_size):
        chunk = [Path(p) for p in paths[start:start + batch_size]]
        missing = [str(p) for p in chunk if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Image(s) not found: {missing}")
        batch = torch.stack([loaded.transform(load_image(p)) for p in chunk])
        for path, probability in zip(chunk, predict_tensor(loaded, batch).tolist()):
            records.append(PredictionRecord.build(
                image_path=path.name, probability=probability, model=loaded.model_name,
                run_id=loaded.run_id, threshold=threshold,
                checkpoint_epoch=loaded.checkpoint_epoch,
            ))
    return records


def read_frozen_selection(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Frozen threshold file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_threshold_for(loaded: LoadedModel, frozen: Mapping[str, Any]) -> float:
    """Return the validation-frozen threshold only if it belongs to this checkpoint."""
    return validate_frozen_selection(frozen, loaded.model_name, loaded.run_id)


def predict_manifest(
    loaded: LoadedModel,
    manifest_path: str | Path,
    split: str,
    *,
    frozen: Optional[Mapping[str, Any]] = None,
    data_root: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> pd.DataFrame:
    """Predict every image of one canonical split manifest.

    The result satisfies the evaluation prediction contract and is checked
    against the manifest before it is returned.
    """
    if split not in {"validation", "test"}:
        raise ValueError("Manifest inference supports the validation and test splits")
    if split == "test" and frozen is None:
        raise LockedTestError(
            "The locked test split requires --threshold-file frozen_threshold.json from "
            "validation; freeze the model and threshold before predicting test."
        )
    threshold = frozen_threshold_for(loaded, frozen) if frozen is not None else None

    manifest_path = Path(manifest_path)
    dataset = ChestXrayDataset(manifest_path, data_root=data_root, transform=loaded.transform,
                               expected_split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=loaded.device.type == "cuda")
    records: List[PredictionRecord] = []
    for batch in loader:
        probabilities = predict_tensor(loaded, batch["image"]).tolist()
        labels = batch["label"].int().tolist()
        for path, label, probability in zip(batch["image_path"], labels, probabilities):
            records.append(PredictionRecord.build(
                image_path=path, probability=probability, model=loaded.model_name,
                run_id=loaded.run_id, split=split, label=label, threshold=threshold,
                checkpoint_epoch=loaded.checkpoint_epoch,
            ))
    frame = records_to_frame(records)
    validate_manifest_predictions(frame, pd.read_csv(manifest_path), split)
    return frame


def default_output_path(loaded: LoadedModel, split: str) -> Path:
    suffix = {"validation": "val", "test": "test"}[split]
    return loaded.run_dir / f"predictions_{suffix}.csv"


def write_predictions(frame: pd.DataFrame, path: str | Path, *, overwrite: bool = False) -> Path:
    """Write a prediction CSV and re-read it through the evaluation loader."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. The locked test is predicted once; pass --overwrite "
            "only if the previous file is known to be invalid."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    load_predictions(path)
    return path


def benchmark(
    loaded: LoadedModel,
    *,
    repeats: int = 20,
    warmup: int = 3,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Model size and forward-pass latency on a preprocessed-size input."""
    size = int(loaded.config["data"]["image"]["size"])
    channels = int(loaded.config["data"]["image"].get("model_channels", 3))
    dummy = torch.zeros(batch_size, channels, size, size)
    for _ in range(max(warmup, 0)):
        predict_tensor(loaded, dummy)
    timings = []
    for _ in range(max(repeats, 1)):
        if loaded.device.type == "cuda":
            torch.cuda.synchronize(loaded.device)
        started = time.perf_counter()
        predict_tensor(loaded, dummy)
        if loaded.device.type == "cuda":
            torch.cuda.synchronize(loaded.device)
        timings.append((time.perf_counter() - started) * 1000.0)
    timings.sort()
    parameters = sum(p.numel() for p in loaded.model.parameters())
    state_bytes = sum(t.numel() * t.element_size() for t in loaded.model.state_dict().values())
    return {
        "model": loaded.model_name,
        "run_id": loaded.run_id,
        "device": str(loaded.device),
        "input_shape": [batch_size, channels, size, size],
        "parameters": int(parameters),
        "weights_mb": round(state_bytes / 1024**2, 3),
        "checkpoint_file_mb": round(loaded.checkpoint_path.stat().st_size / 1024**2, 3),
        "latency_ms_median": round(timings[len(timings) // 2], 3),
        "latency_ms_p90": round(timings[min(len(timings) - 1, int(len(timings) * 0.9))], 3),
        "images_per_second": round(1000.0 * batch_size / timings[len(timings) // 2], 2),
        "repeats": len(timings),
    }
