"""Config-driven model factory shared by baseline and DenseNet branches.

Member 2 owns the first version of this file. This implementation preserves
baseline compatibility where possible and adds the Member-3 ``densenet121``
option. If Member 2's baseline constructor has a different public name, keep
its existing branch in this function during conflict resolution and retain the
DenseNet branch below unchanged.
"""

from __future__ import annotations

from typing import Any, Dict

from .densenet import build_densenet121


def _model_section(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("model", config)


def _create_baseline_compat(model_cfg: Dict[str, Any]):
    """Support common baseline constructor names without hard-coding one API."""
    try:
        from . import baseline as baseline_module
    except ImportError as exc:
        raise ValueError(
            "Requested baseline model, but src/models/baseline.py is not present. "
            "Merge Member 2's feature/baseline-cnn branch first."
        ) from exc

    kwargs = dict(model_cfg.get("kwargs", {}))
    for name in ("build_small_cnn", "create_baseline", "SmallCNN", "BaselineCNN"):
        constructor = getattr(baseline_module, name, None)
        if constructor is not None:
            return constructor(**kwargs)
    raise ValueError(
        "Found src/models/baseline.py but no supported public constructor. During "
        "merge, keep Member 2's existing baseline factory branch and add only the "
        "densenet121 branch from Member 3."
    )


def create_model(config: Dict[str, Any]):
    model_cfg = _model_section(config)
    name = str(model_cfg.get("name", "")).lower()

    if name == "densenet121":
        return build_densenet121(
            pretrained=bool(model_cfg.get("pretrained", True)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            freeze_strategy=str(model_cfg.get("freeze_strategy", "last_block")),
        )
    if name in {"small_cnn", "baseline", "baseline_cnn"}:
        return _create_baseline_compat(model_cfg)
    raise ValueError(
        f"Unknown model.name={name!r}. Supported: densenet121, small_cnn/baseline."
    )
