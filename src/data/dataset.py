"""PyTorch Dataset and DataLoader contract for canonical manifests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as vision_functional

from .common import CANONICAL_COLUMNS, resolve_dataset_root, resolve_image_path


ALLOWED_SUBTYPES = {"normal", "bacterial", "viral", "unknown"}
ALLOWED_SPLITS = {"train", "validation", "test"}


class ChestXrayDataset(Dataset[dict[str, Any]]):
    """Load images plus the metadata needed for leakage-safe evaluation."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
        expected_split: str | None = None,
        validate_paths: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = resolve_dataset_root(data_root)
        self.transform = transform
        self.frame = pd.read_csv(self.manifest_path, keep_default_na=False)

        missing = sorted(set(CANONICAL_COLUMNS) - set(self.frame.columns))
        if missing:
            raise ValueError(f"Manifest is missing columns: {missing}")
        if self.frame.empty:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")
        self.frame["label"] = pd.to_numeric(self.frame["label"], errors="raise").astype(int)
        if not set(self.frame["label"]).issubset({0, 1}):
            raise ValueError("Manifest labels must be 0 or 1")
        if not set(self.frame["split"]).issubset(ALLOWED_SPLITS):
            raise ValueError(f"Unexpected split values: {sorted(set(self.frame['split']))}")
        if not set(self.frame["pneumonia_subtype"]).issubset(ALLOWED_SUBTYPES):
            raise ValueError("Manifest contains an unexpected pneumonia_subtype")
        if expected_split is not None:
            if expected_split not in ALLOWED_SPLITS:
                raise ValueError(f"Unknown expected_split: {expected_split}")
            actual = set(self.frame["split"])
            if actual != {expected_split}:
                raise ValueError(f"Expected only {expected_split}, found {sorted(actual)}")

        if validate_paths:
            missing_paths = [
                value
                for value in self.frame["image_path"]
                if not resolve_image_path(self.root, str(value)).is_file()
            ]
            if missing_paths:
                raise FileNotFoundError(
                    f"{len(missing_paths)} manifest images do not exist below XRAY_DATA_ROOT; "
                    f"examples: {missing_paths[:5]}"
                )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        path = resolve_image_path(self.root, str(row["image_path"]))
        with Image.open(path) as opened:
            image = opened.convert("L").convert("RGB")
            image.load()

        transformed = self.transform(image) if self.transform is not None else image
        if isinstance(transformed, Image.Image):
            transformed = vision_functional.to_tensor(transformed)
        if not torch.is_tensor(transformed):
            raise TypeError("transform must return a torch.Tensor or PIL.Image.Image")

        return {
            "image": transformed,
            # Both configured models use one output logit with BCEWithLogitsLoss.
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "patient_id": str(row["patient_id"]),
            "group_id": str(row["group_id"]),
            "image_path": str(row["image_path"]),
            "source_split": str(row["source_split"]),
            "pneumonia_subtype": str(row["pneumonia_subtype"]),
        }


def build_dataloader(
    dataset: Dataset[Any],
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int | None = None,
    seed: int = 42,
    pin_memory: bool | None = None,
    drop_last: bool = False,
) -> DataLoader[Any]:
    """Build a deterministic DataLoader.

    Member 2 should supply a resize/pad transform before using batch_size > 1,
    because original images have different spatial dimensions.
    """

    workers = num_workers
    if workers is None:
        workers = int(os.getenv("XRAY_NUM_WORKERS", "0"))
    if workers < 0:
        raise ValueError("num_workers cannot be negative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        drop_last=drop_last,
        persistent_workers=workers > 0,
        generator=generator,
    )


def build_split_dataloaders(
    manifest_dir: str | Path,
    data_root: str | Path | None = None,
    transforms_by_split: dict[str, Callable[[Image.Image], Any] | None] | None = None,
    batch_size: int = 1,
    num_workers: int | None = None,
    seed: int = 42,
) -> dict[str, DataLoader[Any]]:
    """Build train/validation/test loaders from the frozen manifest directory."""

    directory = Path(manifest_dir)
    transforms_by_split = transforms_by_split or {}
    loaders: dict[str, DataLoader[Any]] = {}
    for split_name in ("train", "validation", "test"):
        dataset = ChestXrayDataset(
            directory / f"{split_name}.csv",
            data_root=data_root,
            transform=transforms_by_split.get(split_name),
            expected_split=split_name,
        )
        loaders[split_name] = build_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=split_name == "train",
            num_workers=num_workers,
            seed=seed,
        )
    return loaders
