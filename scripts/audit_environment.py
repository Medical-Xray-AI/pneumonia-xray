#!/usr/bin/env python3
"""Audit the Python/CUDA/PyTorch environment and write an exact lock file.

Run on the shared GPU server (see docs/GPU_GUIDE.md) after the project tests
pass there:

    python scripts/audit_environment.py
    python scripts/audit_environment.py --write-lock requirements-lock.txt

``requirements.txt`` keeps compatible ranges; the lock file pins the exact
versions that were verified on the server, so a clean clone can reproduce it:

    python -m pip install -r requirements-lock.txt
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def requirement_names(path: Path = PROJECT_ROOT / "requirements.txt") -> List[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        names.append(re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0])
    return names


def installed_versions(names: Sequence[str]) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_summary() -> Dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"torch": None}
    info: Dict[str, object] = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def nvidia_smi() -> Optional[str]:
    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True)
    return result.stdout.strip() or None


def lock_text(versions: Dict[str, Optional[str]], torch_info: Dict[str, object]) -> str:
    missing = sorted(name for name, version in versions.items() if version is None)
    if missing:
        raise SystemExit(f"Cannot write a lock: not installed: {missing}")
    header = [
        "# Exact versions verified by scripts/audit_environment.py.",
        f"# Generated {datetime.now(timezone.utc).date().isoformat()} on Python "
        f"{platform.python_version()} ({platform.system()}).",
        f"# torch CUDA runtime: {torch_info.get('cuda_runtime')}; install the matching torch wheel",
        "# index (e.g. --index-url https://download.pytorch.org/whl/cu121) if pip picks another build.",
    ]
    return "\n".join(header + [f"{name}=={version}" for name, version in versions.items()]) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the environment and optionally write a lock file.")
    parser.add_argument("--write-lock", type=Path, default=None, help="Write exact pins to this file.")
    args = parser.parse_args(argv)

    versions = installed_versions(requirement_names())
    torch_info = torch_summary()
    print(f"python      {platform.python_version()} ({sys.executable})")
    print(f"platform    {platform.platform()}")
    print(f"nvidia-smi  {nvidia_smi() or 'not available'}")
    for key, value in torch_info.items():
        print(f"{key:<11} {value}")
    print("\npackages")
    for name, version in versions.items():
        print(f"  {name:<14} {version or 'NOT INSTALLED'}")
    if args.write_lock:
        args.write_lock.write_text(lock_text(versions, torch_info), encoding="utf-8")
        print(f"\nlock written -> {args.write_lock}")
    return 0 if all(versions.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
