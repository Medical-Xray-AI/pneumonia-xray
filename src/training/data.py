"""Canonical manifest datasets and shared image preprocessing."""
from __future__ import annotations
import hashlib
import math
from src.data.dataset import ChestXrayDataset
from src.preprocessing.transforms import (
    build_stats_transforms, build_train_transforms, build_eval_transforms,
    resolve_normalization_stats,
)


class ManifestDataset(ChestXrayDataset):
    def __init__(self, manifest_csv, split, *, transform=None, data_root=None):
        super().__init__(manifest_csv, data_root=data_root, transform=transform, expected_split=split)
        self.split = split

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["split"] = self.split
        return sample

    def class_counts(self):
        counts = self.frame.label.value_counts()
        return {"NORMAL": int(counts.get(0, 0)), "PNEUMONIA": int(counts.get(1, 0))}


def compute_pos_weight_from_train(dataset):
    if dataset.split != "train":
        raise ValueError("Class weights require the train split")
    counts = dataset.class_counts()
    if not all(counts.values()):
        raise ValueError("Both classes are required in training")
    return counts["NORMAL"] / counts["PNEUMONIA"]


def build_transforms(cfg):
    data = cfg["data"]
    def stats_dataset():
        return ManifestDataset(data["manifests"]["train"], "train", transform=build_stats_transforms(data))
    mean, std = resolve_normalization_stats(cfg["normalization"], train_dataset_builder=stats_dataset)
    if not all(math.isfinite(x) for x in mean + std) or any(x <= 0 for x in std):
        raise ValueError("Training normalization statistics must be finite with positive std")
    cfg["normalization"].update(mean=mean, std=std)
    return (build_train_transforms(data, cfg.get("augmentation", {}), mean, std),
            build_eval_transforms(data, mean, std), "src.preprocessing.transforms")


def validate_development_splits(train, validation):
    for key in ("image_path", "patient_id", "group_id", "sha256"):
        a = set(train.frame[key].astype(str)) - {""}
        b = set(validation.frame[key].astype(str)) - {""}
        if a & b:
            raise ValueError(f"Train/validation overlap in {key}")
