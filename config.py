"""Configuration loading and validation for reproducible training runs.

The trainer uses YAML because the project plan standardizes experiments around
committed config files. Environment variables such as ``XRAY_DATA_ROOT`` and
``XRAY_OUTPUT_ROOT`` may be referenced as ``${XRAY_DATA_ROOT}`` so machine-
specific absolute paths never need to be committed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment error, not logic
    raise RuntimeError(
        "PyYAML is required to read project configs. Install it temporarily with "
        "`pip install pyyaml`; the final exact version should be pinned by the "
        "release/environment owner after the GPU server audit."
    ) from exc


REQUIRED_TOP_LEVEL = ("seed", "split_version", "data", "model", "training")


def _expand_env(value: Any) -> Any:
    """Recursively expand environment variables and ``~`` in config strings."""
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def _require_keys(mapping: Dict[str, Any], keys: Iterable[str], where: str) -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ValueError(f"Missing required config key(s) in {where}: {missing}")


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the Member-3 training contract and return ``cfg`` unchanged.

    Validation deliberately catches silent experiment drift before any GPU work
    starts: seed 42, split_v1, DenseNet binary output, validation-only model
    selection, and positive integer training parameters.
    """
    _require_keys(cfg, REQUIRED_TOP_LEVEL, "root")
    _require_keys(cfg["data"], ("manifest", "split_column", "image_size"), "data")
    _require_keys(cfg["model"], ("name", "pretrained", "dropout", "freeze_strategy"), "model")
    _require_keys(
        cfg["training"],
        ("epochs", "batch_size", "optimizer", "scheduler", "loss", "early_stopping", "amp"),
        "training",
    )

    if int(cfg["seed"]) != 42:
        raise ValueError("Project contract requires seed=42 for split_v1 experiments.")
    if str(cfg["split_version"]) != "split_v1":
        raise ValueError("Project contract requires split_version='split_v1'.")
    if str(cfg["model"]["name"]).lower() != "densenet121":
        raise ValueError("Member 3 canonical config must use model.name='densenet121'.")

    image_size = int(cfg["data"]["image_size"])
    if image_size <= 0:
        raise ValueError("data.image_size must be positive.")

    for key in ("epochs", "batch_size"):
        if int(cfg["training"][key]) <= 0:
            raise ValueError(f"training.{key} must be a positive integer.")

    dropout = float(cfg["model"]["dropout"])
    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1).")

    freeze_strategy = str(cfg["model"]["freeze_strategy"]).lower()
    if freeze_strategy not in {"head_only", "last_block", "full"}:
        raise ValueError(
            "model.freeze_strategy must be one of: head_only, last_block, full."
        )

    monitor = str(cfg["training"].get("monitor", "macro_f1")).lower()
    if monitor != "macro_f1":
        raise ValueError(
            "Project contract requires best-model selection by validation Macro F1."
        )

    threshold = float(cfg["training"].get("training_threshold", 0.5))
    if threshold != 0.5:
        raise ValueError(
            "Training-time monitoring threshold must remain 0.5. Final classification "
            "threshold is selected later by Member 4 on validation predictions only."
        )

    return cfg


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load, environment-expand and validate a YAML configuration."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    cfg = _expand_env(copy.deepcopy(raw))
    validate_config(cfg)
    cfg.setdefault("_meta", {})
    cfg["_meta"]["source_config"] = str(path)
    cfg["_meta"]["config_sha256"] = config_sha256(cfg)
    return cfg


def config_sha256(cfg: Dict[str, Any]) -> str:
    """Stable hash for experiment traceability (excluding volatile ``_meta``)."""
    stable = copy.deepcopy(cfg)
    stable.pop("_meta", None)
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_resolved_config(cfg: Dict[str, Any], path: str | Path) -> None:
    """Write the exact resolved config used by a run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
