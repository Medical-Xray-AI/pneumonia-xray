#!/usr/bin/env python3
"""Member 5 release experiments on the frozen validation selection.

Reads ``report/tables/frozen_threshold.json`` and ``validation_metrics.csv``
(written by the validation freeze) plus each run's checkpoint and
``predictions_val.csv`` under ``XRAY_OUTPUT_ROOT``. Never touches the locked
test split. Writes aggregate tables only:

  efficiency.csv              parameters, weights size, latency and throughput
                              per model on GPU (batch 1 and 32) and CPU (batch 1)
  validation_bootstrap_ci.csv 95% bootstrap intervals for macro F1, sensitivity
                              and specificity at the frozen thresholds, plus the
                              paired difference between the two models
  inference_consistency.json  re-running inference from the frozen checkpoint
                              reproduces the trainer's validation probabilities

USAGE
-----
  python scripts/release_experiments.py
  python scripts/release_experiments.py --skip-consistency --bootstrap 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

METRICS = ("macro_f1", "pneumonia_sensitivity", "specificity")


def load_selection(report_dir: Path) -> tuple[Dict[str, str], Dict[str, float], dict]:
    """Model -> run id and model -> frozen threshold from the validation freeze."""
    tables = Path(report_dir) / "tables"
    frozen = json.loads((tables / "frozen_threshold.json").read_text(encoding="utf-8"))
    if frozen.get("selected_on") != "validation":
        raise ValueError("frozen_threshold.json must be a validation selection")
    comparison = pd.read_csv(tables / "validation_metrics.csv")
    runs = dict(zip(comparison["model"], comparison["run_id"].astype(str)))
    thresholds = {model: float(sel["threshold"]) for model, sel in frozen["per_model_selection"].items()}
    missing = set(runs) ^ set(thresholds)
    if missing:
        raise ValueError(f"validation_metrics.csv and frozen_threshold.json disagree on {sorted(missing)}")
    return runs, thresholds, frozen


def efficiency(runs: Dict[str, str], output_root: Path, devices: Sequence[str],
               repeats: int = 30) -> pd.DataFrame:
    from src.inference import predict

    rows = []
    for model, run_id in runs.items():
        for device in devices:
            loaded = predict.load_model(output_root / run_id / "checkpoints" / "best.pt", device=device)
            for batch_size in ((1, 32) if device.startswith("cuda") else (1,)):
                row = predict.benchmark(loaded, repeats=repeats, warmup=5, batch_size=batch_size)
                row["batch_size"] = batch_size
                rows.append(row)
    columns = ["model", "run_id", "device", "batch_size", "parameters", "weights_mb",
               "checkpoint_file_mb", "latency_ms_median", "latency_ms_p90", "images_per_second"]
    return pd.DataFrame(rows)[columns]


def _metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, np.ndarray]:
    """Vectorised macro F1 / sensitivity / specificity over bootstrap rows."""
    tp = ((y == 1) & (pred == 1)).sum(-1)
    tn = ((y == 0) & (pred == 0)).sum(-1)
    fp = ((y == 0) & (pred == 1)).sum(-1)
    fn = ((y == 1) & (pred == 0)).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_pos = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
        f1_neg = np.nan_to_num(2 * tn / (2 * tn + fn + fp))
        return {"macro_f1": (f1_pos + f1_neg) / 2,
                "pneumonia_sensitivity": np.nan_to_num(tp / (tp + fn)),
                "specificity": np.nan_to_num(tn / (tn + fp))}


def bootstrap(runs: Dict[str, str], thresholds: Dict[str, float], output_root: Path,
              n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Percentile intervals, resampling images jointly so model differences are paired."""
    frames = []
    for model, run_id in runs.items():
        frame = pd.read_csv(output_root / run_id / "predictions_val.csv")[["image_path", "label", "probability"]]
        frame["pred"] = (frame["probability"] >= thresholds[model]).astype(int)
        frames.append(frame.rename(columns={"probability": f"p_{model}", "pred": f"y_{model}"})
                      .set_index("image_path"))
    joined = frames[0]
    for frame in frames[1:]:
        if set(frame.index) != set(joined.index):
            raise ValueError("Validation prediction files cover different images")
        if not (frame["label"].reindex(joined.index) == joined["label"]).all():
            raise ValueError("Validation prediction files disagree on labels")
        joined = joined.join(frame.drop(columns="label"), how="inner")
    y = joined["label"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(y), size=(n_boot, len(y)))

    rows: List[dict] = []
    samples: Dict[str, Dict[str, np.ndarray]] = {}
    for model in runs:
        pred = joined[f"y_{model}"].to_numpy()
        point = {k: float(v) for k, v in _metrics(y, pred).items()}
        samples[model] = _metrics(y[idx], pred[idx])
        for metric in METRICS:
            low, high = np.percentile(samples[model][metric], [2.5, 97.5])
            rows.append({"comparison": model, "metric": metric, "value": point[metric],
                         "ci95_low": low, "ci95_high": high})
    models = list(runs)
    if len(models) == 2:
        a, b = sorted(models, key=lambda m: m != "densenet121")  # densenet121 first when present
        for metric in METRICS:
            diff = samples[a][metric] - samples[b][metric]
            point = next(r["value"] for r in rows if r["comparison"] == a and r["metric"] == metric) - \
                next(r["value"] for r in rows if r["comparison"] == b and r["metric"] == metric)
            low, high = np.percentile(diff, [2.5, 97.5])
            rows.append({"comparison": f"{a} - {b}", "metric": metric, "value": point,
                         "ci95_low": low, "ci95_high": high,
                         "share_of_resamples_above_zero": float((diff > 0).mean())})
    table = pd.DataFrame(rows)
    table["n_images"] = len(y)
    table["n_bootstrap"] = n_boot
    return table


def consistency(model: str, run_id: str, threshold: float, output_root: Path,
                device: str = "auto", batch_size: int = 32) -> dict:
    """Re-predict validation from the checkpoint and compare with the trainer export."""
    from src import pipeline
    from src.inference import predict

    loaded = predict.load_model(output_root / run_id / "checkpoints" / "best.pt", device=device)
    manifest = pipeline.resolve_manifest("validation", loaded.config)
    fresh = predict.predict_manifest(loaded, manifest, "validation", batch_size=batch_size,
                                     num_workers=int(os.getenv("XRAY_NUM_WORKERS", "0")))
    trained = pd.read_csv(output_root / run_id / "predictions_val.csv")
    merged = trained.merge(fresh, on="image_path", suffixes=("_train", "_infer"))
    if len(merged) != len(trained) or len(merged) != len(fresh):
        raise ValueError("Inference and training predictions cover different images")
    diff = (merged["probability_train"] - merged["probability_infer"]).abs()
    agree = (merged["probability_train"] >= threshold) == (merged["probability_infer"] >= threshold)
    return {"model": model, "run_id": run_id, "device": str(loaded.device), "n_images": int(len(merged)),
            "threshold": threshold, "max_abs_probability_difference": float(diff.max()),
            "mean_abs_probability_difference": float(diff.mean()),
            "decision_agreement": float(agree.mean()), "decisions_changed": int((~agree).sum())}


def default_devices() -> List[str]:
    import torch
    return (["cuda:0"] if torch.cuda.is_available() else []) + ["cpu"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "report")
    parser.add_argument("--output-root", type=Path, default=None, help="Default: XRAY_OUTPUT_ROOT")
    parser.add_argument("--device", action="append", default=None,
                        help="Benchmark device (repeatable; default: cuda:0 if available, and cpu).")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples.")
    parser.add_argument("--repeats", type=int, default=30, help="Timed forward passes per setting.")
    parser.add_argument("--skip-efficiency", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-consistency", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from src import pipeline
    pipeline.load_env_file()
    output_root = args.output_root or pipeline.require_env("XRAY_OUTPUT_ROOT")
    tables = args.report_dir / "tables"
    runs, thresholds, frozen = load_selection(args.report_dir)
    devices = args.device or default_devices()
    pd.set_option("display.width", 160)

    if not args.skip_efficiency:
        table = efficiency(runs, output_root, devices, repeats=args.repeats)
        table.to_csv(tables / "efficiency.csv", index=False)
        print("\n=== efficiency ===\n" + table.to_string(index=False))
    if not args.skip_bootstrap:
        table = bootstrap(runs, thresholds, output_root, n_boot=args.bootstrap)
        table.round(4).to_csv(tables / "validation_bootstrap_ci.csv", index=False)
        print("\n=== validation 95% bootstrap intervals ===\n" + table.round(4).to_string(index=False))
    if not args.skip_consistency:
        model = frozen["recommended_model"]
        result = consistency(model, runs[model], thresholds[model], output_root, device=devices[0])
        (tables / "inference_consistency.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("\n=== inference consistency ===\n" + json.dumps(result, indent=2))
    print(f"\nTables written to {tables}. The locked test set was not touched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
