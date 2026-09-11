#!/usr/bin/env python3
"""Evaluation CLI for the pediatric pneumonia project.

Two modes, deliberately asymmetric:

  validation  Selects the decision threshold on validation predictions, writes
              the model comparison table, all aggregate figures, and the error
              analysis. Produces `frozen_threshold.json` - the hand-off to
              Uzv 5.

  test        Refuses to select anything. Requires `--threshold-file frozen_threshold.json` and scores
              the locked test set once at that fixed operating point.

The asymmetry is the point: it is not possible to tune a threshold on the
locked test set through this entry point, so the "no test leakage" requirement
in the project brief is enforced by the tool rather than by discipline.

EXAMPLES
--------
  # after Uzv 2 and Uzv 3 hand over validation predictions
  python scripts/evaluate_model.py validation \
      --predictions small_cnn=$XRAY_OUTPUT_ROOT/run_baseline/predictions_val.csv \
      --predictions densenet121=$XRAY_OUTPUT_ROOT/run_densenet/predictions_val.csv \
      --manifest data/manifests/validation.csv \
      --out-dir report

  # once, at the very end, after the team freezes the model and threshold
  python scripts/evaluate_model.py test \
      --predictions densenet121=$XRAY_OUTPUT_ROOT/run_densenet/predictions_test.csv \
      --manifest data/manifests/test.csv \
      --threshold-file report/tables/frozen_threshold.json \
      --out-dir report
"""

from __future__ import annotations

import argparse
import json
import hashlib
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running as `python scripts/evaluate_model.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.evaluation import error_analysis, plots
from src.evaluation.evaluate import (
    build_comparison_table,
    evaluate_predictions,
    evaluate_validation,
    load_predictions,
    prediction_run_id,
    validate_manifest_predictions,
    validate_frozen_selection,
    recommend_model,
    write_comparison_table,
)
from src.evaluation.threshold import DEFAULT_OBJECTIVE


def parse_predictions_argument(value: str) -> Tuple[str, Path]:
    """Parse a `model=path` pair, which keeps model naming explicit."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"expected 'model=path', got '{value}' "
            f"(e.g. densenet121=outputs/run1/predictions_val.csv)"
        )
    model, _, path = value.partition("=")
    model = model.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model):
        raise argparse.ArgumentTypeError(f"empty model name in '{value}'")
    return model, Path(path.strip())


def load_manifest(path: Path) -> pd.DataFrame:
    """Read the mandatory canonical split manifest."""
    if path is None or not path.is_file():
        raise ValueError(f"split manifest not found: {path}")
    return pd.read_csv(path)


def run_validation(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    manifest = load_manifest(args.manifest)
    if len({model for model, _ in args.predictions}) != len(args.predictions):
        raise ValueError("Use distinct model names when comparing predictions")

    results: List[Dict] = []
    curves: Dict[str, Tuple] = {}
    selections: Dict[str, Dict] = {}
    frames: Dict[str, pd.DataFrame] = {}

    for model, path in args.predictions:
        frame = load_predictions(path)
        frames[model] = frame
        metrics, selection = evaluate_validation(
            frame, manifest=manifest, model=model, run_id=args.run_id, objective=args.objective
        )
        results.append(metrics)
        curves[model] = (frame["label"].to_numpy(), frame["probability"].to_numpy())
        selections[model] = selection.to_dict()

        print(f"\n=== {model} (validation, n = {metrics['n_images']}) ===")
        print(f"  selected threshold    {selection.threshold:.4f}  "
              f"({selection.n_tied} tied of {selection.n_candidates} candidates)")
        print(f"  macro F1              {metrics['macro_f1']:.4f}")
        print(f"  pneumonia sensitivity {metrics['pneumonia_sensitivity']:.4f}")
        print(f"  specificity           {metrics['specificity']:.4f}")
        print(f"  ROC-AUC / PR-AUC      {metrics['roc_auc']:.4f} / {metrics['pr_auc']:.4f}")
        print(f"  confusion TN/FP/FN/TP {metrics['true_negatives']}/"
              f"{metrics['false_positives']}/{metrics['false_negatives']}/"
              f"{metrics['true_positives']}")

    table = write_comparison_table(results, tables_dir / "validation_metrics.csv")
    best = recommend_model(table)

    # Figures. Curves are threshold-free, so both models share one pair of axes.
    plots.plot_roc_curves(curves, figures_dir / "roc_validation.png")
    plots.plot_pr_curves(curves, figures_dir / "pr_validation.png")
    for model, frame in frames.items():
        threshold = selections[model]["threshold"]
        plots.plot_confusion_matrix(
            frame["label"], frame["probability"], threshold,
            figures_dir / f"confusion_matrix_{model}_validation.png",
            title=f"Confusion matrix - {model} (validation)",
        )
        plots.plot_threshold_sweep(
            frame["label"], frame["probability"],
            figures_dir / f"threshold_sweep_{model}.png",
            selected_threshold=threshold,
            title=f"Threshold sweep - {model} (validation)",
        )

    # Error analysis for the recommended model only, to keep the report focused.
    best_model = best["model"]
    enriched = error_analysis.attach_manifest_metadata(frames[best_model], manifest)
    threshold = best["threshold"]

    subtype_table = error_analysis.error_rate_by(enriched, threshold, "pneumonia_subtype")
    subtype_table.to_csv(tables_dir / "error_by_subtype.csv", index=False)

    confidence_table = error_analysis.confidence_distribution(enriched, threshold)
    confidence_table.to_csv(tables_dir / "error_by_confidence.csv", index=False)

    summary = error_analysis.worst_errors_summary(enriched, threshold)

    frozen = {
        "recommended_model": best_model,
        "threshold": threshold,
        "objective": args.objective,
        "selected_on": "validation",
        "run_id": best["run_id"],
        "validation_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "validation_macro_f1": best["macro_f1"],
        "validation_pneumonia_sensitivity": best["pneumonia_sensitivity"],
        "validation_specificity": best["specificity"],
        "selection_rule": selections[best_model]["selection_rule"],
        "per_model_selection": selections,
        "error_summary": summary,
    }
    frozen_path = tables_dir / "frozen_threshold.json"
    frozen_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    print("\n=== recommendation ===")
    print(f"  model     {best_model}")
    print(f"  threshold {threshold:.4f}")
    print(f"\n  tables  -> {tables_dir}")
    print(f"  figures -> {figures_dir}")
    print(f"  frozen  -> {frozen_path}")
    print("\nSubgroup error rates:")
    print(subtype_table.to_string(index=False))
    print("\nThe locked test set has NOT been touched. Hand frozen_threshold.json")
    print("to Uzv 5 and run this script in `test` mode exactly once, at the end.")
    return 0


def run_test(args: argparse.Namespace) -> int:
    """Score the locked test set once, at a threshold chosen earlier."""
    if len(args.predictions) != 1:
        raise ValueError("Test mode scores only the single frozen model/run")
    frozen = json.loads(Path(args.threshold_file).read_text(encoding="utf-8"))
    manifest = load_manifest(args.manifest)
    model, path = args.predictions[0]
    frame = load_predictions(path)
    validate_manifest_predictions(frame, manifest, "test")
    run_id = prediction_run_id(frame, model, args.run_id)
    threshold = validate_frozen_selection(frozen, model, run_id)

    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for model, path in args.predictions:
        frame = load_predictions(path)
        metrics = evaluate_predictions(
            frame, threshold=threshold, model=model, split="test",
            run_id=run_id, threshold_source="frozen_from_validation",
        )
        results.append(metrics)

        print(f"\n=== {model} (LOCKED TEST, n = {metrics['n_images']}) ===")
        print(f"  threshold             {threshold:.4f}  (frozen, not tuned here)")
        print(f"  macro F1              {metrics['macro_f1']:.4f}")
        print(f"  pneumonia sensitivity {metrics['pneumonia_sensitivity']:.4f}")
        print(f"  specificity           {metrics['specificity']:.4f}")
        print(f"  ROC-AUC / PR-AUC      {metrics['roc_auc']:.4f} / {metrics['pr_auc']:.4f}")

        plots.plot_confusion_matrix(
            frame["label"], frame["probability"], threshold,
            figures_dir / f"confusion_matrix_{model}_test.png",
            title=f"Confusion matrix - {model} (locked test)",
        )

    write_comparison_table(results, tables_dir / "test_metrics.csv")
    print(f"\n  tables  -> {tables_dir / 'test_metrics.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate prediction CSVs for the pediatric pneumonia project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--predictions", action="append", required=True, metavar="MODEL=PATH",
            type=parse_predictions_argument,
            help="Prediction CSV as 'model=path'; repeat to compare models.",
        )
        subparser.add_argument("--out-dir", type=Path, default=Path("report"),
                               help="Root for figures/ and tables/ (default: report).")
        subparser.add_argument("--run-id", default="",
                               help="Optional assertion of the prediction CSV run identity.")
        subparser.add_argument("--manifest", type=Path, required=True,
                               help="Canonical manifest for the complete evaluated split.")

    validation_parser = subparsers.add_parser(
        "validation", allow_abbrev=False, help="Select the threshold on validation data and report.")
    add_common(validation_parser)
    validation_parser.add_argument(
        "--objective", default=DEFAULT_OBJECTIVE,
        choices=["macro_f1", "pneumonia_sensitivity", "balanced_accuracy"],
        help="Quantity the threshold search maximises (default: macro_f1).")

    test_parser = subparsers.add_parser(
        "test", allow_abbrev=False, help="Score the locked test set once at a frozen threshold.")
    add_common(test_parser)
    test_parser.add_argument("--threshold-file", type=Path, required=True,
                             help="frozen_threshold.json produced by validation mode.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "validation":
            return run_validation(args)
        return run_test(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
