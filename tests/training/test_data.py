from pathlib import Path
import pandas as pd
import pytest
from src.data.audit_data import audit_dataset
from src.data.split_data import create_manifests
from src.training.data import ManifestDataset, compute_pos_weight_from_train, validate_development_splits, build_transforms
from src.training.config import load_config


def test_canonical_manifest_loader(dataset_root, tmp_path, monkeypatch):
    monkeypatch.setenv("XRAY_DATA_ROOT", str(dataset_root))
    audit_dataset(dataset_root, tmp_path / "audit", near_threshold=0)
    create_manifests(tmp_path / "audit/file_manifest.csv", tmp_path / "manifests")
    cfg = load_config(Path(__file__).resolve().parents[2] / "configs/baseline.yaml")
    cfg["data"]["manifests"] = {s: str(tmp_path / "manifests" / f"{s}.csv") for s in ("train", "validation")}
    train_t, eval_t, source = build_transforms(cfg)
    assert source == "src.preprocessing.transforms"
    train = ManifestDataset(cfg["data"]["manifests"]["train"], "train", transform=train_t)
    val = ManifestDataset(cfg["data"]["manifests"]["validation"], "validation", transform=eval_t)
    validate_development_splits(train, val)
    assert train[0]["image"].shape == (3, 224, 224)
    assert val[0]["split"] == "validation"
    assert compute_pos_weight_from_train(train) > 0
    with pytest.raises(ValueError, match="train split"): compute_pos_weight_from_train(val)
    with pytest.raises(ValueError, match="Expected only validation"):
        ManifestDataset(cfg["data"]["manifests"]["train"], "validation")
