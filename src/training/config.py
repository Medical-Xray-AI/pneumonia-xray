"""Resolve the committed experiment and data configs into one training config."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _expand_env(value):
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return os.path.expanduser(os.path.expandvars(value)) if isinstance(value, str) else value


def validate_config(cfg):
    if cfg["seed"] != 42 or cfg["split_version"] != "split_v1":
        raise ValueError("Project contract requires seed=42 and split_version=split_v1")
    model = cfg["model"]
    if model["name"] not in {"small_cnn", "densenet121"}:
        raise ValueError("Unknown model.name")
    if model.get("output_units", 1) != 1:
        raise ValueError("output_units must be 1")
    if not 0 <= float(model.get("dropout", 0.2)) < 1:
        raise ValueError("dropout must be in [0, 1)")
    if model.get("freeze_strategy", "last_block") not in {"head_only", "last_block", "full"}:
        raise ValueError("Invalid freeze_strategy")
    if cfg["data"]["image"]["size"] != 224 or cfg["data"]["image"].get("model_channels", 3) != 3:
        raise ValueError("Expected 224x224 images with 3 channels")
    for name in ("train", "validation"):
        if name not in cfg["data"]["manifests"]:
            raise ValueError(f"Missing {name} manifest")
    training = cfg["training"]
    for name in ("epochs", "batch_size", "grad_accum_steps"):
        value = training.get(name, 1)
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    if training.get("training_threshold", 0.5) != 0.5:
        raise ValueError("Training threshold must remain 0.5")
    if training.get("monitor", "macro_f1") != "macro_f1":
        raise ValueError("Monitor must be validation macro_f1")
    if float(training["optimizer"]["lr"]) <= 0:
        raise ValueError("Learning rate must be positive")
    if cfg["normalization"]["type"] not in {"imagenet", "dataset_statistics"}:
        raise ValueError("Unknown normalization type")
    if model["name"] == "densenet121" and cfg["normalization"]["type"] != "imagenet":
        raise ValueError("DenseNet requires ImageNet normalization")
    return cfg


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    cfg = _expand_env(copy.deepcopy(raw))
    if "data_config" in cfg:
        data_path = (PROJECT_ROOT / cfg.pop("data_config")).resolve()
        data = _expand_env(yaml.safe_load(data_path.read_text(encoding="utf-8")))
        cfg["data"] = data
        cfg["seed"] = cfg["experiment"]["seed"]
        cfg["split_version"] = data["split"]["version"]
        if data["split"]["seed"] != cfg["seed"]:
            raise ValueError("Experiment seed does not match data split seed")
        training = cfg["training"]
        optimizer = training.get("optimizer", "adamw")
        if isinstance(optimizer, str):
            training["optimizer"] = {"name": optimizer,
                "lr": training.pop("learning_rate"),
                "weight_decay": training.pop("weight_decay", 0.0001)}
        loss = training.get("loss", "weighted_bce_with_logits")
        if isinstance(loss, str):
            if loss not in {"weighted_bce_with_logits", "bce_with_logits"}:
                raise ValueError("Only BCEWithLogitsLoss is supported")
            training["loss"] = {"name": "bce_with_logits", "pos_weight": "auto"}
        training["amp"] = training.pop("mixed_precision", False)
        scheduler = cfg.pop("scheduler", {"name": "none"})
        if scheduler.get("monitor", "validation_macro_f1") != "validation_macro_f1" or scheduler.get("mode", "max") != "max":
            raise ValueError("Scheduler must monitor validation_macro_f1 in max mode")
        if scheduler["name"] == "reduce_lr_on_plateau":
            scheduler["name"] = "reduce_on_plateau"
        training["scheduler"] = scheduler
        early = cfg.pop("early_stopping", {"patience": 4})
        if early.get("monitor", "validation_macro_f1") != "validation_macro_f1" or early.get("mode", "max") != "max":
            raise ValueError("Early stopping must monitor validation_macro_f1 in max mode")
        training["early_stopping"] = early
    if "manifests" not in cfg.get("data", {}):
        raise ValueError("Use canonical data manifests from configs/data.yaml")
    cfg["data"]["manifests"] = {
        key: str((PROJECT_ROOT / value).resolve())
        for key, value in cfg["data"]["manifests"].items()
    }
    cfg["model"].setdefault("freeze_strategy", "last_block")
    cfg["training"].setdefault("grad_accum_steps", 1)
    validate_config(cfg)
    cfg["_meta"] = {"source_config": str(path), "config_sha256": config_sha256(cfg)}
    return cfg


def config_sha256(cfg):
    stable = copy.deepcopy(cfg)
    stable.pop("_meta", None)
    return hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def save_resolved_config(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
