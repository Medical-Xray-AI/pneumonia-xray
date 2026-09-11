#!/usr/bin/env python3
"""Release verification for the pediatric pneumonia repository.

Checks, in order (each prints ok/FAIL/warn; exit code 1 on any FAIL):

  files       required entry points, configs, manifests and docs exist
  configs     every experiment config resolves through src.training.config
  manifests   scripts/verify_manifests.py (schema + zero cross-split leakage);
              with --with-data also every image_path resolves on disk
  repository  tracked files contain no secrets, .env/credentials, checkpoints,
              X-ray images or machine-specific absolute paths
  registry    docs/experiment_registry.csv schema; relative checkpoint paths
              (with --with-data also existing under XRAY_OUTPUT_ROOT)
  freeze      report/tables/frozen_threshold.json, when present, is a valid
              validation selection
  inference   synthetic checkpoint -> load -> manifest/image inference ->
              locked-test guard -> latency benchmark
  clean-clone (--clean-clone) clone HEAD into a temp dir and re-run
              `run_all.py --help`, `run_all.py check` and this script there

USAGE
-----
  python scripts/verify_release.py
  python scripts/verify_release.py --with-data --clean-clone
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_FILES = [
    "run_all.py", "requirements.txt", "README.md", ".gitignore", ".env.example",
    "configs/data.yaml", "configs/baseline.yaml", "configs/densenet121.yaml",
    "data/manifests/train.csv", "data/manifests/validation.csv", "data/manifests/test.csv",
    "docs/experiment_registry.csv", "docs/evaluation_protocol.md", "docs/model_card.md",
    "docs/release_checklist.md", "scripts/train.py", "scripts/evaluate_model.py",
    "src/inference/predict.py", "src/pipeline.py",
]
FORBIDDEN_NAMES = {".env", "kaggle.json", "team5-access.md"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".zip", ".rar", ".7z", ".key", ".pem",
                      ".conf", ".jpeg", ".jpg", ".dcm", ".ipynb"}
# Aggregate figures (plots) may be committed; X-ray images may not.
PNG_ALLOWED_PREFIXES = ("report/figures/", "presentation/", "docs/")
MAX_FILE_BYTES = 5 * 1024 * 1024
SECRET_PATTERNS = {
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "WireGuard private key": re.compile(r"PrivateKey\s*=\s*[A-Za-z0-9+/]{42,}="),
    "Kaggle API key": re.compile(r"(?i)(kaggle_key\s*[=:]\s*['\"]?[0-9a-f]{32}|\"key\"\s*:\s*\"[0-9a-f]{32}\")"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "Jupyter token URL": re.compile(r"[?&]token=[0-9a-f]{32,}"),
}
ABSOLUTE_PATH_PATTERNS = {
    "Windows user path": re.compile(r"\b[A-Za-z]:[\\/]+Users[\\/]+[^\s\\/\"']+"),
    "Linux home path": re.compile(r"(?<![\w.])/home/[a-z_][a-z0-9_-]*/"),
    "macOS user path": re.compile(r"(?<![\w.])/Users/[A-Za-z][^\s/\"']*/"),
}


@dataclass
class Report:
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def ok(self, message: str) -> None:
        print(f"  [ok]   {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"  [FAIL] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"  [warn] {message}")


def tracked_files(root: Path) -> List[str]:
    try:
        output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                      if p.is_file() and ".git" not in p.parts and ".venv" not in p.parts)
    return sorted(name for name in output.decode("utf-8").split("\0") if name)


def scan_repository(root: Path, files: Iterable[str], report: Report) -> None:
    """Flag secrets, credentials, checkpoints, images and absolute paths."""
    problems = 0
    for name in files:
        path = root / name
        pure = PurePosixPath(name)
        if pure.name in FORBIDDEN_NAMES or (pure.name.startswith(".env") and pure.name != ".env.example"):
            report.fail(f"{name}: credential/environment file is tracked")
            problems += 1
            continue
        suffix = pure.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            report.fail(f"{name}: forbidden file type {suffix} is tracked")
            problems += 1
            continue
        if suffix == ".png" and not name.startswith(PNG_ALLOWED_PREFIXES):
            report.fail(f"{name}: image outside report/figures, presentation or docs")
            problems += 1
            continue
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            report.warn(f"{name}: larger than {MAX_FILE_BYTES // 1024**2} MB")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in {**SECRET_PATTERNS, **ABSOLUTE_PATH_PATTERNS}.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                report.fail(f"{name}:{line}: possible {label}")
                problems += 1
    if not problems:
        report.ok("tracked files contain no secrets, credentials, checkpoints, images or user paths")


def check_required_files(root: Path, report: Report) -> None:
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    for name in missing:
        report.fail(f"required file missing: {name}")
    if not missing:
        report.ok(f"all {len(REQUIRED_FILES)} required files present")


def check_configs(root: Path, report: Report) -> None:
    from src.training.config import load_config

    for path in sorted((root / "configs").glob("*.yaml")):
        if path.name == "data.yaml":
            continue
        try:
            cfg = load_config(path)
            report.ok(f"{path.name}: model={cfg['model']['name']} seed={cfg['seed']} split={cfg['split_version']}")
        except Exception as exc:
            report.fail(f"{path.name}: {exc}")


def check_manifests(root: Path, report: Report, with_data: bool) -> None:
    command = [sys.executable, str(root / "scripts" / "verify_manifests.py")]
    if with_data:
        command += ["--check-files", "--check-dataset"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    if result.returncode == 0:
        report.ok("verify_manifests.py: READY" + (" (files and dataset checked)" if with_data else ""))
    else:
        lines = [l for l in result.stdout.splitlines() if "[FAIL]" in l] or result.stderr.splitlines()[-3:]
        report.fail("verify_manifests.py failed: " + " | ".join(l.strip() for l in lines[:5]))


def check_registry(root: Path, report: Report, with_data: bool) -> None:
    from src.training.registry import COLUMNS

    path = root / "docs" / "experiment_registry.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            report.fail("experiment_registry.csv header does not match the shared schema")
            return
        rows = list(reader)
    output_root = os.getenv("XRAY_OUTPUT_ROOT")
    for row in rows:
        reference = row.get("checkpoint_path", "")
        if reference and (Path(reference).is_absolute() or re.match(r"^[A-Za-z]:", reference)):
            report.fail(f"registry {row['run_id']}: checkpoint_path must be relative to XRAY_OUTPUT_ROOT")
        elif reference and with_data and output_root and not (Path(output_root) / reference).is_file():
            report.warn(f"registry {row['run_id']}: checkpoint not found under XRAY_OUTPUT_ROOT")
    tested = [row["run_id"] for row in rows if str(row.get("test_evaluated", "")).lower() == "true"]
    if len(tested) > 1:
        report.warn(f"more than one run marked test_evaluated: {tested}")
    report.ok(f"registry schema valid, {len(rows)} run(s), locked test evaluated for {tested or 'none'}")


def check_freeze(root: Path, report: Report) -> None:
    path = root / "report" / "tables" / "frozen_threshold.json"
    if not path.is_file():
        report.warn("report/tables/frozen_threshold.json not present yet (validation freeze pending)")
        return
    frozen = json.loads(path.read_text(encoding="utf-8"))
    from src.evaluation.evaluate import validate_frozen_selection
    try:
        threshold = validate_frozen_selection(frozen, frozen.get("recommended_model"), frozen.get("run_id"))
        report.ok(f"frozen selection: {frozen['recommended_model']} / {frozen['run_id']} at {threshold:.4f}")
    except ValueError as exc:
        report.fail(f"frozen_threshold.json invalid: {exc}")
    if (root / "report" / "tables" / "test_metrics.csv").is_file():
        report.ok("locked-test metrics recorded (report/tables/test_metrics.csv)")


def check_inference(report: Report) -> None:
    """Exercise the real inference code on a synthetic checkpoint."""
    try:
        from src.inference import predict
        from src.inference.smoke import build_smoke_checkpoint, build_synthetic_workspace, smoke_frozen_selection
    except Exception as exc:
        report.fail(f"inference modules not importable: {exc}")
        return
    with tempfile.TemporaryDirectory(prefix="xray_release_") as tmp:
        tmp = Path(tmp)
        workspace = build_synthetic_workspace(tmp / "data")
        checkpoint = build_smoke_checkpoint(workspace, tmp / "outputs")
        loaded = predict.load_model(checkpoint, device="cpu")
        frame = predict.predict_manifest(loaded, workspace["validation"], "validation",
                                         data_root=workspace["data_root"], batch_size=3)
        if not frame["probability"].between(0, 1).all():
            report.fail("inference produced probabilities outside [0, 1]")
            return
        try:
            predict.predict_manifest(loaded, workspace["test"], "test", data_root=workspace["data_root"])
            report.fail("locked test was predicted without a frozen selection")
            return
        except predict.LockedTestError:
            pass
        frozen = smoke_frozen_selection(loaded.model_name, loaded.run_id)
        test = predict.predict_manifest(loaded, workspace["test"], "test", frozen=frozen,
                                        data_root=workspace["data_root"])
        predict.write_predictions(test, tmp / "predictions_test.csv")
        images = sorted(workspace["data_root"].rglob("*.png"))[:2]
        records = predict.predict_images(loaded, images, threshold=0.5)
        bench = predict.benchmark(loaded, repeats=3, warmup=1)
    report.ok(f"inference smoke: {len(frame)} validation + {len(test)} test rows, "
              f"{len(records)} single images, locked-test guard active, "
              f"{bench['parameters']:,} params, {bench['latency_ms_median']} ms/image on CPU")


def check_clean_clone(root: Path, report: Report) -> None:
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    if status.stdout.strip():
        report.warn("working tree has uncommitted changes; the clean clone tests HEAD only")
    with tempfile.TemporaryDirectory(prefix="xray_clone_") as tmp:
        clone = Path(tmp) / "repo"
        result = subprocess.run(["git", "clone", "--quiet", str(root), str(clone)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            report.fail(f"git clone failed: {result.stderr.strip()}")
            return
        env = {k: v for k, v in os.environ.items() if not k.startswith("XRAY_")}
        steps = [
            [sys.executable, "run_all.py", "--help"],
            [sys.executable, "run_all.py", "check"],
            [sys.executable, "scripts/verify_release.py"],
        ]
        for command in steps:
            result = subprocess.run(command, cwd=clone, env=env, capture_output=True, text=True)
            label = " ".join(Path(part).name if i == 0 else part for i, part in enumerate(command))
            if result.returncode != 0:
                tail = (result.stdout + result.stderr).strip().splitlines()[-5:]
                report.fail(f"clean clone `{label}` exited {result.returncode}: {' | '.join(tail)}")
                return
        report.ok("clean clone: --help, check and release verification pass without local .env/data")


def run(root: Path, *, with_data: bool, clean_clone: bool, skip_inference: bool) -> Report:
    report = Report()
    sections: List[tuple[str, Callable[[], None]]] = [
        ("files", lambda: check_required_files(root, report)),
        ("configs", lambda: check_configs(root, report)),
        ("manifests", lambda: check_manifests(root, report, with_data)),
        ("repository", lambda: scan_repository(root, tracked_files(root), report)),
        ("registry", lambda: check_registry(root, report, with_data)),
        ("freeze", lambda: check_freeze(root, report)),
    ]
    if not skip_inference:
        sections.append(("inference", lambda: check_inference(report)))
    if clean_clone:
        sections.append(("clean clone", lambda: check_clean_clone(root, report)))
    for name, section in sections:
        print(f"\n== {name} ==")
        try:
            section()
        except Exception as exc:  # a crashing check is a failed check
            report.fail(f"{name} check crashed: {type(exc).__name__}: {exc}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("--with-data", action="store_true",
                        help="Also check images under XRAY_DATA_ROOT and registry checkpoints.")
    parser.add_argument("--clean-clone", action="store_true",
                        help="Clone HEAD into a temporary directory and verify there.")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip the synthetic inference smoke test.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run(PROJECT_ROOT, with_data=args.with_data, clean_clone=args.clean_clone,
                 skip_inference=args.skip_inference)
    print(f"\n{len(report.failures)} failure(s), {len(report.warnings)} warning(s)")
    print("RELEASE READY" if not report.failures else "NOT READY")
    return 0 if not report.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
