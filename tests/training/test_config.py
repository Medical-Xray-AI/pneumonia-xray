from pathlib import Path

import pytest

from src.training.config import load_config


BASE = """
seed: 42
split_version: split_v1
data:
  manifest: audit_out/split_manifest.csv
  split_column: new_split
  image_size: 224
model:
  name: densenet121
  pretrained: false
  dropout: 0.2
  freeze_strategy: last_block
training:
  epochs: 2
  batch_size: 2
  amp: false
  monitor: macro_f1
  training_threshold: 0.5
  optimizer: {name: adamw, lr: 0.001}
  scheduler: {name: none}
  loss: {name: bce_with_logits, pos_weight: auto}
  early_stopping: {patience: 2}
"""


def test_load_valid_config(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text(BASE)
    cfg = load_config(p)
    assert cfg["seed"] == 42
    assert cfg["_meta"]["config_sha256"]


def test_rejects_nonproject_seed(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text(BASE.replace("seed: 42", "seed: 7"))
    with pytest.raises(ValueError, match="seed=42"):
        load_config(p)


def test_rejects_threshold_tuning_inside_trainer(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text(BASE.replace("training_threshold: 0.5", "training_threshold: 0.7"))
    with pytest.raises(ValueError, match="threshold"):
        load_config(p)
