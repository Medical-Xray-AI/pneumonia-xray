"""Best-checkpoint, last-checkpoint and resume support."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[Mapping[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _torch_load(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    # PyTorch >=2.6 defaults weights_only=True. Our checkpoint intentionally
    # contains optimizer/RNG metadata, so request the classic full checkpoint.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # compatibility with older torch releases
        return torch.load(path, map_location=map_location)


class CheckpointManager:
    """Write ``last.pt`` every epoch and ``best.pt`` only on improvement."""

    def __init__(self, directory: str | Path, monitor: str = "macro_f1", mode: str = "max"):
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_metric: Optional[float] = None

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    @property
    def last_path(self) -> Path:
        return self.directory / "last.pt"

    def _improved(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "max":
            return value > self.best_metric
        return value < self.best_metric

    def save(
        self,
        *,
        epoch: int,
        metric_value: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        early_stopping: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        history: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        metric_value = float(metric_value)
        improved = self._improved(metric_value)
        if improved:
            self.best_metric = metric_value

        payload: Dict[str, Any] = {
            "epoch": int(epoch),
            "monitor": self.monitor,
            "mode": self.mode,
            "metric_value": metric_value,
            "best_metric": self.best_metric,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "early_stopping_state": (
                early_stopping.state_dict() if early_stopping is not None else None
            ),
            "config": config,
            "run_id": run_id,
            "history": history,
            "rng_state": _capture_rng_state(),
            "extra": extra or {},
        }
        _atomic_torch_save(payload, self.last_path)
        if improved:
            _atomic_torch_save(payload, self.best_path)
        return improved

    def restore_best_metric(self, value: Optional[float]) -> None:
        self.best_metric = None if value is None else float(value)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    early_stopping: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
    strict_model: bool = True,
    expected_config_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Restore a full training checkpoint and return its metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    # Load metadata/RNG tensors on CPU; load_state_dict moves parameter states as needed.
    ckpt = _torch_load(path, map_location="cpu")
    if expected_config_hash is not None:
        actual = (ckpt.get("config") or {}).get("_meta", {}).get("config_sha256")
        if actual != expected_config_hash:
            raise ValueError("Resume config or manifest hash does not match checkpoint")
    model.load_state_dict(ckpt["model_state"], strict=strict_model)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    if early_stopping is not None and ckpt.get("early_stopping_state") is not None:
        early_stopping.load_state_dict(ckpt["early_stopping_state"])
    if restore_rng:
        _restore_rng_state(ckpt.get("rng_state"))
    return ckpt
