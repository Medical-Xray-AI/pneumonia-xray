"""Stage orchestration behind ``run_all.py``.

Each stage is a plain function that validates its inputs before doing work
(fail-fast on missing configs, environment variables, manifests or
checkpoints) and delegates to the reviewed module that owns the logic:

    check      configs, manifests, environment, optional dataset root
    train      scripts/train.py (Member 3)
    evaluate   scripts/evaluate_model.py (Member 4)
    infer      src/inference/predict.py
    develop    check -> train every config -> validation evaluation/freeze

``develop`` stops at the validation freeze. The locked test is a separate,
explicit ``infer --split test`` + ``evaluate test`` step performed once.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
DEFAULT_CONFIGS = (PROJECT_ROOT / "configs/baseline.yaml", PROJECT_ROOT / "configs/densenet121.yaml")
DEFAULT_REPORT_DIR = PROJECT_ROOT / "report"


class PipelineError(RuntimeError):
    """A stage cannot run with the given inputs."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def load_env_file(path: Path = PROJECT_ROOT / ".env") -> bool:
    """Load the local ``.env`` without overriding variables already set."""
    if not path.is_file():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return bool(load_dotenv(path, override=False))


def require_env(name: str) -> Path:
    value = os.getenv(name)
    if not value:
        raise PipelineError(f"Set {name} (shell variable or .env) before this stage")
    return Path(value).expanduser().resolve()


def require_file(path: str | Path, what: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise PipelineError(f"{what} not found: {path}")
    return path


def default_manifest(split: str) -> Path:
    return MANIFEST_DIR / f"{split}.csv"


def resolve_manifest(split: str, config: Optional[dict] = None, explicit: Optional[Path] = None) -> Path:
    """Explicit path, else the run's configured manifest if present here, else split_v1."""
    if explicit is not None:
        return require_file(explicit, f"{split} manifest")
    configured = ((config or {}).get("data", {}).get("manifests", {}) or {}).get(split)
    if configured and Path(configured).is_file():
        return Path(configured)
    return require_file(default_manifest(split), f"{split} manifest")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- check

def check_environment(configs: Sequence[str | Path] = DEFAULT_CONFIGS, *,
                      require_data: bool = False) -> List[CheckResult]:
    """Validate everything a run needs, without touching images or the GPU."""
    import pandas as pd
    from src.data.common import CANONICAL_COLUMNS

    results: List[CheckResult] = []
    for config in configs:
        path = Path(config)
        try:
            from src.training.config import load_config
            cfg = load_config(require_file(path, "Config"))
            results.append(CheckResult(f"config {path.name}", True,
                                       f"model={cfg['model']['name']} split={cfg['split_version']}"))
        except Exception as exc:  # report every broken config, not only the first
            results.append(CheckResult(f"config {path.name}", False, str(exc)))

    for split in ("train", "validation", "test"):
        path = default_manifest(split)
        if not path.is_file():
            results.append(CheckResult(f"manifest {split}", False, f"missing {path}"))
            continue
        frame = pd.read_csv(path, keep_default_na=False)
        missing = sorted(set(CANONICAL_COLUMNS) - set(frame.columns))
        splits = set(frame.get("split", []))
        ok = not missing and splits == {split} and len(frame) > 0
        detail = f"{len(frame)} images" if ok else f"missing columns {missing} or split values {sorted(splits)}"
        results.append(CheckResult(f"manifest {split}", ok, detail))

    for name in ("XRAY_DATA_ROOT", "XRAY_OUTPUT_ROOT"):
        value = os.getenv(name)
        results.append(CheckResult(name, bool(value), "set" if value else "not set", required=require_data))
    if require_data and os.getenv("XRAY_DATA_ROOT"):
        try:
            from src.data.common import resolve_dataset_root
            results.append(CheckResult("dataset root", True, str(resolve_dataset_root())))
        except Exception as exc:
            results.append(CheckResult("dataset root", False, str(exc)))

    try:
        import torch
        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "CPU only"
        results.append(CheckResult("torch", True, f"{torch.__version__} cuda={torch.version.cuda} ({device})"))
    except Exception as exc:
        results.append(CheckResult("torch", False, f"not importable: {exc}"))
    return results


def checks_passed(results: Sequence[CheckResult]) -> bool:
    return all(result.ok for result in results if result.required)


# --------------------------------------------------------------------------- train

def run_state(run_dir: Path) -> str:
    """``complete``, ``resumable``, ``new`` or ``broken`` for a run directory."""
    if (run_dir / "metrics.json").is_file() and (run_dir / "predictions_val.csv").is_file():
        return "complete"
    if (run_dir / "checkpoints" / "last.pt").is_file():
        return "resumable"
    if not run_dir.exists():
        return "new"
    return "broken"


def train(config: str | Path, *, run_id: Optional[str] = None, device: Optional[str] = None,
          resume: Optional[str | Path] = None, update_registry: bool = False,
          reuse_existing: bool = False) -> Path:
    """Train one config and return its run directory.

    With ``reuse_existing`` and an explicit ``run_id``, a finished run is
    reused and an interrupted one resumes from ``checkpoints/last.pt``.
    """
    from scripts import train as train_cli
    from src.training.config import load_config

    config = require_file(config, "Config")
    output_root = require_env("XRAY_OUTPUT_ROOT")
    require_env("XRAY_DATA_ROOT")
    cfg = load_config(config)

    if reuse_existing and run_id and resume is None:
        run_dir = output_root / run_id
        state = run_state(run_dir)
        if state == "complete":
            print(f"[train] {run_id}: complete run found, reusing {run_dir}")
            return run_dir
        if state == "resumable":
            print(f"[train] {run_id}: resuming from checkpoints/last.pt")
            resume = run_dir / "checkpoints" / "last.pt"
        elif state == "broken":
            raise PipelineError(f"{run_dir} exists without a checkpoint; choose another run id")

    argv = ["--config", str(config)]
    if run_id:
        argv += ["--run-id", run_id]
    if device:
        argv += ["--device", device]
    if resume:
        argv += ["--resume", str(require_file(resume, "Resume checkpoint"))]
    if update_registry:
        argv.append("--update-registry")
    code = train_cli.main(argv)
    if code != 0:
        raise PipelineError(f"Training failed for {config} (exit {code})")
    if resume:
        return Path(resume).resolve().parent.parent
    if run_id:
        return output_root / run_id
    # Auto-generated id: the newest directory for this model.
    candidates = sorted(output_root.glob(f"*_{cfg['model']['name']}_s{cfg['seed']}"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise PipelineError("Training finished but its run directory was not found")
    return candidates[-1]


# --------------------------------------------------------------------------- evaluate

def evaluate(argv: Sequence[str]) -> int:
    """Forward to the evaluation CLI (``validation`` or ``test`` mode)."""
    from scripts import evaluate_model

    return evaluate_model.main(list(argv))


def evaluate_validation(predictions: Dict[str, Path], *, out_dir: Path = DEFAULT_REPORT_DIR,
                        manifest: Optional[Path] = None) -> Path:
    manifest = require_file(manifest or default_manifest("validation"), "Validation manifest")
    argv = ["validation", "--manifest", str(manifest), "--out-dir", str(out_dir)]
    for model, path in predictions.items():
        argv += ["--predictions", f"{model}={require_file(path, 'Validation predictions')}"]
    if evaluate(argv) != 0:
        raise PipelineError("Validation evaluation failed")
    return Path(out_dir) / "tables" / "frozen_threshold.json"


def evaluate_test(model: str, predictions: Path, threshold_file: Path, *,
                  out_dir: Path = DEFAULT_REPORT_DIR, manifest: Optional[Path] = None) -> Path:
    manifest = require_file(manifest or default_manifest("test"), "Test manifest")
    argv = ["test", "--manifest", str(manifest), "--out-dir", str(out_dir),
            "--predictions", f"{model}={require_file(predictions, 'Test predictions')}",
            "--threshold-file", str(require_file(threshold_file, 'Frozen threshold file'))]
    if evaluate(argv) != 0:
        raise PipelineError("Locked-test evaluation failed")
    return Path(out_dir) / "tables" / "test_metrics.csv"


# --------------------------------------------------------------------------- infer

def infer_manifest(checkpoint: str | Path, split: str, *, threshold_file: Optional[Path] = None,
                   manifest: Optional[Path] = None, output: Optional[Path] = None,
                   overwrite: bool = False, device: str = "auto", batch_size: int = 32,
                   num_workers: Optional[int] = None) -> Path:
    """Export ``predictions_<split>.csv`` for a canonical manifest."""
    from src.inference import predict

    if split == "test" and threshold_file is None:
        raise PipelineError("Locked test inference requires --threshold-file frozen_threshold.json")
    require_env("XRAY_DATA_ROOT")
    checkpoint = require_file(checkpoint, "Checkpoint")
    frozen = predict.read_frozen_selection(threshold_file) if threshold_file else None

    loaded = predict.load_model(checkpoint, device=device)
    manifest = resolve_manifest(split, loaded.config, manifest)
    expected = (loaded.config.get("manifest_sha256") or {}).get(split)
    if expected and expected != sha256_file(manifest):
        raise PipelineError(f"{manifest} differs from the {split} manifest used in training")
    if frozen is not None and split == "test":
        recorded = frozen.get("validation_manifest_sha256")
        current = sha256_file(resolve_manifest("validation", loaded.config))
        if recorded and recorded != current:
            raise PipelineError("frozen_threshold.json was selected on a different validation manifest")

    if num_workers is None:
        num_workers = int(os.getenv("XRAY_NUM_WORKERS", "0"))
    frame = predict.predict_manifest(loaded, manifest, split, frozen=frozen,
                                     batch_size=batch_size, num_workers=num_workers)
    path = predict.write_predictions(frame, output or predict.default_output_path(loaded, split),
                                     overwrite=overwrite)
    print(f"[infer] {len(frame)} {split} predictions -> {path}")
    return path


def infer_images(checkpoint: str | Path, images: Sequence[str | Path], *,
                 threshold_file: Optional[Path] = None, device: str = "auto", batch_size: int = 16):
    from src.inference import predict

    loaded = predict.load_model(require_file(checkpoint, "Checkpoint"), device=device)
    threshold = None
    if threshold_file:
        threshold = predict.frozen_threshold_for(loaded, predict.read_frozen_selection(threshold_file))
    return predict.predict_images(loaded, list(images), batch_size=batch_size, threshold=threshold)


# --------------------------------------------------------------------------- develop

def develop(configs: Sequence[str | Path] = DEFAULT_CONFIGS, *, run_ids: Optional[Sequence[str]] = None,
            device: Optional[str] = None, out_dir: Path = DEFAULT_REPORT_DIR,
            update_registry: bool = False) -> Dict[str, object]:
    """Full development pipeline: train every config, then freeze on validation."""
    from src.training.config import load_config

    results = check_environment(configs, require_data=True)
    failed = [f"{r.name}: {r.detail}" for r in results if r.required and not r.ok]
    if failed:
        raise PipelineError("Pre-flight check failed:\n  " + "\n  ".join(failed))
    if run_ids is not None and len(run_ids) != len(configs):
        raise PipelineError("Give one run id per config")

    loaded_configs = [load_config(c) for c in configs]
    names = [cfg["model"]["name"] for cfg in loaded_configs]
    if len(set(names)) != len(names):
        raise PipelineError(f"Compared configs must use distinct model names, got {names}")
    validation_manifests = {cfg["data"]["manifests"]["validation"] for cfg in loaded_configs}
    if len(validation_manifests) != 1:
        raise PipelineError("Compared configs must share one validation manifest")
    validation_manifest = resolve_manifest("validation", loaded_configs[0])

    run_dirs: Dict[str, Path] = {}
    for index, (config, name) in enumerate(zip(configs, names)):
        run_id = run_ids[index] if run_ids else None
        print(f"\n[develop] training {name} from {config}")
        run_dirs[name] = train(config, run_id=run_id, device=device,
                               update_registry=update_registry, reuse_existing=True)

    frozen = evaluate_validation({name: d / "predictions_val.csv" for name, d in run_dirs.items()},
                                 out_dir=out_dir, manifest=validation_manifest)
    print("\n[develop] validation freeze written to", frozen)
    print("[develop] Locked test NOT touched. After team sign-off run once:")
    print("  python run_all.py infer --checkpoint <run>/checkpoints/best.pt --split test "
          f"--threshold-file {frozen}")
    print("  python run_all.py evaluate test --predictions <model>=<run>/predictions_test.csv "
          f"--manifest data/manifests/test.csv --threshold-file {frozen} --out-dir {out_dir}")
    return {"run_dirs": run_dirs, "frozen_threshold": frozen}
