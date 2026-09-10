"""Training-side manifest adapter and preprocessing compatibility layer.

Member 1 supplied the leakage-aware ``split_manifest.csv`` and Member 2 owns
final preprocessing. This module keeps Member 3 runnable before all branches
are merged while automatically preferring Member 2's transform API when it is
available.
"""

from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

LABEL_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResizePadSquare:
    """Preserve aspect ratio, resize longest side and zero-pad to square."""

    def __init__(self, size: int, fill: int = 0):
        self.size = int(size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image size {(w, h)}")
        scale = self.size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        image = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        pad_left = (self.size - new_w) // 2
        pad_right = self.size - new_w - pad_left
        pad_top = (self.size - new_h) // 2
        pad_bottom = self.size - new_h - pad_top
        return ImageOps.expand(
            image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill,
        )


def _fallback_transforms(cfg: Mapping[str, Any]) -> Tuple[Any, Any]:
    """Contract-compatible fallback used only until Member 2's branch is merged."""
    size = int(cfg.get("data", {}).get("image_size", 224))
    aug = cfg.get("data", {}).get("augmentation", {}) or {}
    rotation = float(aug.get("rotation_degrees", 5.0))
    translate = float(aug.get("translate_fraction", 0.02))

    train_ops = [ResizePadSquare(size)]
    if bool(aug.get("enabled", True)):
        train_ops.append(
            T.RandomAffine(
                degrees=rotation,
                translate=(translate, translate),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
        )
    train_ops.extend([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    eval_ops = [
        ResizePadSquare(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return T.Compose(train_ops), T.Compose(eval_ops)


def _try_team_transforms(cfg: Mapping[str, Any]) -> Optional[Tuple[Any, Any]]:
    try:
        module = importlib.import_module("src.data.transforms")
    except ImportError:
        return None

    # Preferred explicit API.
    if hasattr(module, "build_train_transform") and hasattr(module, "build_eval_transform"):
        return module.build_train_transform(cfg), module.build_eval_transform(cfg)
    if hasattr(module, "get_train_transform") and hasattr(module, "get_eval_transform"):
        return module.get_train_transform(cfg), module.get_eval_transform(cfg)

    # Common combined APIs.
    for name in ("build_transforms", "get_transforms"):
        fn = getattr(module, name, None)
        if fn is None:
            continue
        built = fn(cfg)
        if isinstance(built, Mapping):
            train_t = built.get("train")
            eval_t = built.get("validation") or built.get("val") or built.get("eval")
            if train_t is not None and eval_t is not None:
                return train_t, eval_t
        if isinstance(built, (tuple, list)) and len(built) == 2:
            return built[0], built[1]
    return None


def build_transforms(cfg: Mapping[str, Any]) -> Tuple[Any, Any, str]:
    team = _try_team_transforms(cfg)
    if team is not None:
        return team[0], team[1], "src.data.transforms"
    fallback = _fallback_transforms(cfg)
    return fallback[0], fallback[1], "member3_fallback"


def resolve_manifest_image_path(row: Mapping[str, Any], data_root: Optional[str | Path]) -> Path:
    """Resolve the teammate manifest on Kaggle or on the shared GPU server.

    The supplied audit manifest currently contains Kaggle absolute paths. The
    project contract wants portable paths; until the data branch is normalized,
    this resolver remaps ``<source split>/<label>/<filename>`` under
    ``XRAY_DATA_ROOT``.
    """
    raw = Path(str(row["path"]))
    if raw.exists():
        return raw

    root = Path(data_root) if data_root else None
    if root is None:
        raise FileNotFoundError(
            f"Image path from manifest does not exist: {raw}. Set XRAY_DATA_ROOT "
            "to the local chest_xray directory."
        )

    # Most robust fallback uses columns included by Member 1's manifest.
    source_split = str(row.get("split", ""))
    label = str(row.get("label", ""))
    filename = str(row.get("filename", raw.name))
    candidate = root / source_split / label / filename
    if candidate.exists():
        return candidate

    # Also handle a root pointing one directory above chest_xray.
    candidate2 = root / "chest_xray" / source_split / label / filename
    if candidate2.exists():
        return candidate2

    raise FileNotFoundError(
        "Could not resolve image referenced by manifest. Tried:\n"
        f"  {raw}\n  {candidate}\n  {candidate2}\n"
        "Check XRAY_DATA_ROOT and ensure the standard train/val/test folders exist."
    )


class ManifestDataset(Dataset):
    """Portable leakage-aware manifest dataset with metadata for predictions."""

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        *,
        split_column: str = "new_split",
        transform: Optional[Any] = None,
        data_root: Optional[str | Path] = None,
    ) -> None:
        df = pd.read_csv(manifest_csv)
        required = {"path", "label", split_column}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        self.df = df[df[split_column] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"Manifest has no rows for {split_column}={split!r}")
        unknown = set(self.df["label"].astype(str).str.upper()) - set(LABEL_TO_IDX)
        if unknown:
            raise ValueError(f"Unknown labels in manifest: {sorted(unknown)}")
        self.transform = transform
        self.data_root = data_root or os.getenv("XRAY_DATA_ROOT")
        self.split = split

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = resolve_manifest_image_path(row, self.data_root)
        with Image.open(image_path) as img:
            # Source is grayscale; convert to RGB to replicate into 3 channels.
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label_name = str(row["label"]).upper()
        label = LABEL_TO_IDX[label_name]
        metadata: Dict[str, Any] = {
            "image_path": str(row.get("filename", image_path.name)),
            "source_split": str(row.get("split", "")),
            "split": self.split,
            "label_name": label_name,
            "group_id": str(row.get("group_id", "")),
            "patient_id": "" if pd.isna(row.get("patient_id", None)) else str(row.get("patient_id")),
        }
        return image, label, metadata

    def class_counts(self) -> Dict[str, int]:
        counts = self.df["label"].astype(str).str.upper().value_counts().to_dict()
        return {k: int(v) for k, v in counts.items()}


def compute_pos_weight_from_train(dataset: ManifestDataset) -> float:
    """BCE ``pos_weight = N_negative/N_positive`` from train split only."""
    counts = dataset.class_counts()
    n_neg = int(counts.get("NORMAL", 0))
    n_pos = int(counts.get("PNEUMONIA", 0))
    if n_neg == 0 or n_pos == 0:
        raise ValueError(f"Both classes are required to compute pos_weight; counts={counts}")
    return float(n_neg / n_pos)
