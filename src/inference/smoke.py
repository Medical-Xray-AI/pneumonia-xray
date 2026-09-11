"""Tiny synthetic workspace for inference smoke checks.

Produces noise images in the Kaggle folder layout, canonical split manifests
and an untrained checkpoint written by the real ``CheckpointManager``. It
exercises the code path only; nothing here is a project result.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.models.factory import create_model
from src.training.checkpointing import CheckpointManager
from src.training.config import PROJECT_ROOT, load_config

FOLDERS = {"train": "train", "validation": "val", "test": "test"}


def build_synthetic_workspace(root: str | Path, n_per_class: int = 2, seed: int = 42) -> Dict[str, Path]:
    """Create ``root/chest_xray`` images and ``root/manifests/<split>.csv``."""
    root = Path(root)
    data_root = root / "chest_xray"
    manifests: Dict[str, Path] = {}
    rng = np.random.default_rng(seed)
    for split, folder in FOLDERS.items():
        rows = []
        for label, class_name in ((0, "NORMAL"), (1, "PNEUMONIA")):
            for index in range(n_per_class):
                relative = f"{folder}/{class_name}/{split}_{class_name.lower()}_{index}.png"
                path = data_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                pixels = rng.integers(0, 60, (40 + index, 56), dtype=np.uint8) + (150 if label else 20)
                Image.fromarray(pixels.astype(np.uint8), mode="L").save(path)
                identity = f"{split}_{label}_{index}"
                rows.append({
                    "image_path": relative, "patient_id": identity, "group_id": identity,
                    "label": label, "pneumonia_subtype": "bacterial" if label else "normal",
                    "split": split, "source_split": f"provided_{folder}",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
        manifests[split] = root / "manifests" / f"{split}.csv"
        manifests[split].parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(manifests[split], index=False)
    return {"data_root": data_root, **manifests}


def build_smoke_checkpoint(workspace: Dict[str, Path], output_root: str | Path, *,
                           model_name: str = "small_cnn", run_id: str = "smoke_run",
                           config_path: str | Path | None = None, seed: int = 42) -> Path:
    """Write ``output_root/<run_id>/checkpoints/best.pt`` like the trainer does."""
    config_path = config_path or PROJECT_ROOT / (
        "configs/baseline.yaml" if model_name == "small_cnn" else "configs/densenet121.yaml")
    cfg = load_config(config_path)
    cfg["data"]["manifests"] = {k: str(workspace[k]) for k in ("train", "validation", "test")}
    cfg["model"].update(name=model_name, pretrained=False)
    if model_name == "densenet121":
        cfg["model"]["freeze_strategy"] = "head_only"
        cfg["normalization"] = {"type": "imagenet", "mean": [0.485, 0.456, 0.406],
                                "std": [0.229, 0.224, 0.225]}
    else:
        # The trainer stores train-derived statistics; use fixed stand-ins here.
        cfg["normalization"].update(mean=[0.4, 0.4, 0.4], std=[0.25, 0.25, 0.25])
    cfg["manifest_sha256"] = {
        split: hashlib.sha256(Path(workspace[split]).read_bytes()).hexdigest()
        for split in ("train", "validation")
    }
    torch.manual_seed(seed)
    model = create_model(cfg)
    optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
    manager = CheckpointManager(Path(output_root) / run_id / "checkpoints")
    manager.save(epoch=1, metric_value=0.5, model=model, optimizer=optimizer, config=cfg,
                 run_id=run_id, history=[], extra={})
    return manager.best_path


def smoke_frozen_selection(model_name: str, run_id: str, threshold: float = 0.5,
                           validation_manifest_sha256: str | None = None) -> Dict[str, object]:
    frozen: Dict[str, object] = {"recommended_model": model_name, "run_id": run_id,
                                 "threshold": threshold, "selected_on": "validation"}
    if validation_manifest_sha256:
        frozen["validation_manifest_sha256"] = validation_manifest_sha256
    return frozen
