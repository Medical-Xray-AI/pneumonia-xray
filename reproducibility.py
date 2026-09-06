"""Reproducibility and runtime-device helpers."""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch and configure deterministic behavior."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN settings are relevant only on CUDA; setting them on CPU is harmless.
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

    # ``use_deterministic_algorithms`` can raise for operations that do not have
    # deterministic CUDA kernels. ``warn_only=True`` preserves the run while
    # surfacing such cases in logs.
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except TypeError:  # old PyTorch compatibility
        if deterministic:
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """Deterministically seed DataLoader workers from PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def resolve_device(requested: Optional[str] = None) -> torch.device:
    """Resolve ``auto``/CUDA/CPU with an explicit error for unavailable CUDA."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{requested}', but torch.cuda.is_available() is False."
        )
    return torch.device(requested)


def collect_runtime_metadata(device: torch.device | str | None = None) -> Dict[str, Any]:
    """Collect environment metadata that is safe to persist with a run."""
    device = torch.device(device) if device is not None else resolve_device("auto")
    meta: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        meta.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_bytes": int(props.total_memory),
                "gpu_compute_capability": f"{props.major}.{props.minor}",
            }
        )
    else:
        meta["gpu_name"] = None
    return meta
