from pathlib import Path
import pytest
import yaml
from src.training.config import load_config, save_resolved_config

ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.parametrize("name", ["baseline", "densenet121", "densenet121_frozen", "densenet121_lr2e-4"])
def test_committed_configs_and_resolved_roundtrip(name, tmp_path):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg["seed"] == 42
    assert "train" in cfg["data"]["manifests"]
    saved = tmp_path / "resolved.yaml"
    save_resolved_config(cfg, saved)
    assert load_config(saved)["_meta"]["config_sha256"] == cfg["_meta"]["config_sha256"]

@pytest.mark.parametrize("field,value,match", [("seed", 7, "seed"), ("training_threshold", 0.7, "threshold")])
def test_invalid_config(field, value, match, tmp_path):
    cfg = load_config(ROOT / "configs/baseline.yaml")
    if field == "seed": cfg[field] = value
    else: cfg["training"][field] = value
    p = tmp_path / "bad.yaml"
    save_resolved_config(cfg, p)
    with pytest.raises(ValueError, match=match): load_config(p)
