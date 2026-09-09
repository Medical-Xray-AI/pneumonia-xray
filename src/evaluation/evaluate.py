"""Prediction-file schema, evaluation orchestration, and model comparison.

This module owns the hand-off contract between the training owners
(Uzv 2 / Uzv 3) and evaluation (Uzv 4).

PREDICTION FILE CONTRACT
------------------------
Every run must export one CSV per evaluated split, named per the artifact
contract in `docs/TEAM_WORKFLOW.md` (e.g. `predictions_val.csv`), with at
least these three columns:

    image_path    path relative to XRAY_DATA_ROOT; joins back to the split
                  manifest, so no other metadata needs to be carried here
    label         ground-truth label, 0 = normal, 1 = pneumonia
    probability   predicted pneumonia probability in [0, 1], i.e. the model
                  logit AFTER sigmoid - not the raw logit, and not a hard 0/1

Any further columns (run_id, model, split, logit, ...) are preserved and
ignored. Keeping the required set this small is deliberate: subtype, patient
id and source split are recovered by joining to the manifest in
`error_analysis.py`, so the trainers do not have to thread them through.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import LABEL_NORMAL, LABEL_PNEUMONIA, compute_metrics
from src.evaluation.threshold import DEFAULT_OBJECTIVE, ThresholdSelection, select_threshold

REQUIRED_PREDICTION_COLUMNS = ("image_path", "label", "probability")

# Column order for the report's model-comparison table. Fixed here so the
# baseline and DenseNet rows are always directly comparable.
COMPARISON_COLUMNS = (
    "model",
    "split",
    "threshold",
    "threshold_source",
    "macro_f1",
    "pneumonia_sensitivity",
    "specificity",
    "roc_auc",
    "pr_auc",
    "f1_normal",
    "f1_pneumonia",
    "pneumonia_precision",
    "balanced_accuracy",
    "accuracy",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
    "n_images",
    "n_normal",
    "n_pneumonia",
    "run_id",
)


class PredictionSchemaError(ValueError):
    """Raised when a prediction CSV does not satisfy the contract above."""


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Read and validate a prediction CSV.

    Fails loudly and specifically: a silent schema drift here would corrupt
    every downstream number in the report, and the most likely mistakes
    (exporting logits, exporting hard labels, using NORMAL/PNEUMONIA strings)
    all have distinct, recognisable symptoms worth naming in the error.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"prediction file not found: {path}")

    frame = pd.read_csv(path)

    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in frame.columns]
    if missing:
        raise PredictionSchemaError(
            f"{path}: missing required column(s) {missing}. "
            f"Required schema: {list(REQUIRED_PREDICTION_COLUMNS)}; "
            f"found: {list(frame.columns)}"
        )

    if frame.empty:
        raise PredictionSchemaError(f"{path}: contains no rows")

    if frame["image_path"].isna().any():
        raise PredictionSchemaError(f"{path}: image_path contains empty values")

    duplicated = frame["image_path"].duplicated()
    if duplicated.any():
        examples = frame.loc[duplicated, "image_path"].head(3).tolist()
        raise PredictionSchemaError(
            f"{path}: {int(duplicated.sum())} duplicated image_path value(s), "
            f"e.g. {examples}. Each image must be predicted exactly once."
        )

    labels = frame["label"]
    # `is_numeric_dtype` rather than an `object` check: pandas 3 gives string
    # columns a dedicated `str` dtype, so `dtype == object` silently misses them.
    if not pd.api.types.is_numeric_dtype(labels):
        raise PredictionSchemaError(
            f"{path}: label column is text, not numeric. Use the project "
            f"contract 0 = normal, 1 = pneumonia, not 'NORMAL'/'PNEUMONIA'."
        )
    if labels.isna().any():
        raise PredictionSchemaError(f"{path}: label contains missing values")
    bad_labels = sorted(set(labels.unique()) - {LABEL_NORMAL, LABEL_PNEUMONIA})
    if bad_labels:
        raise PredictionSchemaError(
            f"{path}: label must be 0 (normal) or 1 (pneumonia); found {bad_labels}"
        )

    probabilities = frame["probability"]
    if probabilities.isna().any():
        raise PredictionSchemaError(f"{path}: probability contains missing values")
    if not np.all(np.isfinite(probabilities.to_numpy(dtype=float))):
        raise PredictionSchemaError(f"{path}: probability contains NaN or inf")
    low, high = float(probabilities.min()), float(probabilities.max())
    if low < 0.0 or high > 1.0:
        raise PredictionSchemaError(
            f"{path}: probability must lie in [0, 1]; observed [{low:.4f}, {high:.4f}]. "
            f"This usually means raw logits were exported instead of sigmoid outputs."
        )
    if set(np.unique(probabilities)) <= {0.0, 1.0}:
        raise PredictionSchemaError(
            f"{path}: probability contains only 0.0 and 1.0. Threshold selection "
            f"needs continuous probabilities, not hard predictions."
        )

    frame = frame.copy()
    frame["label"] = frame["label"].astype(int)
    frame["probability"] = frame["probability"].astype(float)
    return frame


def evaluate_predictions(
    frame: pd.DataFrame,
    threshold: float,
    *,
    model: str = "unknown",
    split: str = "unknown",
    run_id: str = "",
    threshold_source: str = "provided",
) -> Dict[str, Any]:
    """Compute the full metric bundle for one prediction file at one threshold."""
    metrics = compute_metrics(
        frame["label"].to_numpy(),
        frame["probability"].to_numpy(),
        threshold=threshold,
    )
    metrics.update(
        {
            "model": model,
            "split": split,
            "run_id": run_id,
            "threshold_source": threshold_source,
        }
    )
    return metrics


def evaluate_validation(
    frame: pd.DataFrame,
    *,
    model: str = "unknown",
    run_id: str = "",
    objective: str = DEFAULT_OBJECTIVE,
) -> tuple[Dict[str, Any], ThresholdSelection]:
    """Select a threshold on validation predictions, then score at that point.

    This is the only entry point in the project permitted to choose a
    threshold. Its output threshold is what gets frozen and handed to Uzv 5
    for the single locked-test evaluation.
    """
    selection = select_threshold(
        frame["label"].to_numpy(), frame["probability"].to_numpy(), objective=objective
    )
    metrics = evaluate_predictions(
        frame,
        threshold=selection.threshold,
        model=model,
        split="validation",
        run_id=run_id,
        threshold_source=f"selected_on_validation:{objective}",
    )
    return metrics, selection


def build_comparison_table(results: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Assemble evaluated runs into the report's comparison table.

    Rows are ordered by descending macro F1 so the recommended model is first,
    with the model name as a deterministic tie-break.
    """
    frame = pd.DataFrame(list(results))
    if frame.empty:
        raise ValueError("no results to compare")

    for column in COMPARISON_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    frame = frame[list(COMPARISON_COLUMNS)]
    return frame.sort_values(
        ["macro_f1", "model"], ascending=[False, True]
    ).reset_index(drop=True)


def write_comparison_table(
    results: Iterable[Mapping[str, Any]], path: str | Path, *, decimals: int = 4
) -> pd.DataFrame:
    """Write the comparison table to CSV, rounding float columns for the report."""
    frame = build_comparison_table(results)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rounded = frame.copy()
    for column in rounded.select_dtypes(include=["float"]).columns:
        rounded[column] = rounded[column].round(decimals)
    rounded.to_csv(path, index=False)
    return frame


def recommend_model(frame: pd.DataFrame) -> Dict[str, Any]:
    """Return the top row of a comparison table as the candidate for freezing."""
    if frame.empty:
        raise ValueError("comparison table is empty")
    best = frame.iloc[0]
    return {
        "model": best["model"],
        "threshold": float(best["threshold"]),
        "macro_f1": float(best["macro_f1"]),
        "pneumonia_sensitivity": float(best["pneumonia_sensitivity"]),
        "specificity": float(best["specificity"]),
        "run_id": best["run_id"],
    }
