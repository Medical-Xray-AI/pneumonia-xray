"""Aggregate false-positive / false-negative breakdown.

Purpose: answer "where does the model fail, and is that failure clinically or
statistically interesting?" rather than just reporting a single error count.

Privacy rule: every function here returns *aggregates*. Group sizes below
`MIN_GROUP_SIZE` are collapsed into an `other` bucket so that no row can be
traced back to an individual patient, and no function returns a filename, a
patient id, or an image. That keeps the whole output committable under the
repository data policy.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import LABEL_NORMAL, LABEL_PNEUMONIA, apply_threshold

# Below this many images, a subgroup row is merged into `other` rather than
# reported on its own.
MIN_GROUP_SIZE = 10

# PNEUMONIA filenames encode the subtype: personNNN_bacteria_MMM.jpeg.
SUBTYPE_RE = re.compile(r"person\d+_(bacteria|virus)_\d+", re.IGNORECASE)

# The filename encodes "bacteria"/"virus", but configs/data.yaml
# `audit_metadata.subtype_values` fixes the vocabulary as
# normal/bacterial/viral/unknown. Map to the contract's words so subgroup
# tables in the report use the same terms as the config and the manifest.
FILENAME_SUBTYPE_TO_CONTRACT = {"bacteria": "bacterial", "virus": "viral"}

CONFIDENCE_BINS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def infer_subtype(image_path: str, label: int) -> str:
    """Recover the pneumonia subtype from the filename.

    Used only when subtype metadata is unavailable. The canonical manifest
    already records subtype; it is audit metadata, never a training
    target and never enters model selection.
    """
    if int(label) == LABEL_NORMAL:
        return "normal"
    match = SUBTYPE_RE.search(Path(str(image_path)).name)
    if not match:
        return "unknown"
    return FILENAME_SUBTYPE_TO_CONTRACT[match.group(1).lower()]


def attach_manifest_metadata(
    predictions: pd.DataFrame,
    manifest: Optional[pd.DataFrame] = None,
    *,
    manifest_path_column: str = "image_path",
) -> pd.DataFrame:
    """Join canonical relative image paths without ambiguous basename matches."""
    frame = predictions.copy()
    if manifest is not None:
        if manifest_path_column not in manifest:
            raise ValueError("manifest requires image_path")
        source = manifest.rename(columns={manifest_path_column: "image_path"})
        if source.image_path.duplicated().any() or source.image_path.isna().any():
            raise ValueError("manifest contains duplicate or empty image_path")
        if not set(frame.image_path).issubset(set(source.image_path)):
            raise ValueError("predictions contain image_path absent from manifest")
        carry = [column for column in
                 ("pneumonia_subtype", "source_split", "split", "group_id", "width", "height")
                 if column in source]
        # The audited manifest is authoritative for subgroup metadata.
        frame = frame.drop(columns=[c for c in carry if c in frame]).merge(
            source[["image_path", *carry]], on="image_path", how="left", validate="one_to_one")

    if "pneumonia_subtype" not in frame.columns or frame["pneumonia_subtype"].isna().all():
        frame["pneumonia_subtype"] = [
            infer_subtype(path, label)
            for path, label in zip(frame["image_path"], frame["label"])
        ]
    else:
        frame["pneumonia_subtype"] = frame["pneumonia_subtype"].fillna(
            pd.Series(
                [infer_subtype(p, l) for p, l in zip(frame["image_path"], frame["label"])],
                index=frame.index,
            )
        )

    return frame


def classify_outcomes(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add `predicted`, `outcome` (TP/TN/FP/FN) and `confidence` columns."""
    result = frame.copy()
    result["predicted"] = apply_threshold(result["probability"].to_numpy(), threshold)

    conditions = [
        (result["label"] == LABEL_PNEUMONIA) & (result["predicted"] == LABEL_PNEUMONIA),
        (result["label"] == LABEL_NORMAL) & (result["predicted"] == LABEL_NORMAL),
        (result["label"] == LABEL_NORMAL) & (result["predicted"] == LABEL_PNEUMONIA),
        (result["label"] == LABEL_PNEUMONIA) & (result["predicted"] == LABEL_NORMAL),
    ]
    result["outcome"] = np.select(conditions, ["TP", "TN", "FP", "FN"], default="?")

    # Distance from the operating point: how nearly the model got it right.
    result["confidence"] = np.where(
        result["predicted"] == LABEL_PNEUMONIA,
        result["probability"],
        1.0 - result["probability"],
    )
    return result


def _collapse_small_groups(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """Merge subgroups smaller than MIN_GROUP_SIZE into a single `other` row."""
    sizes = frame.groupby(key).size()
    small = set(sizes[sizes < MIN_GROUP_SIZE].index)
    if not small:
        return frame
    result = frame.copy()
    result[key] = result[key].map(lambda value: "other" if value in small else value)
    return result


def error_rate_by(
    frame: pd.DataFrame, threshold: float, key: str
) -> pd.DataFrame:
    """Error counts and rates for each level of a grouping column.

    Returns one row per subgroup with support, false-negative rate and
    false-positive rate - enough to show whether a subgroup (e.g. viral
    pneumonia) is systematically harder, which is exactly the per-group
    performance the Track 2 brief asks for.
    """
    if key not in frame.columns:
        raise ValueError(f"column '{key}' not present; have {list(frame.columns)}")

    scored = classify_outcomes(frame, threshold)
    scored = scored[scored[key].notna()]
    scored = _collapse_small_groups(scored, key)

    rows: List[Dict[str, Any]] = []
    for value, group in scored.groupby(key, dropna=False):
        n_pneumonia = int((group["label"] == LABEL_PNEUMONIA).sum())
        n_normal = int((group["label"] == LABEL_NORMAL).sum())
        false_negatives = int((group["outcome"] == "FN").sum())
        false_positives = int((group["outcome"] == "FP").sum())

        rows.append(
            {
                key: value,
                "n_images": int(len(group)),
                "n_normal": n_normal,
                "n_pneumonia": n_pneumonia,
                "false_negatives": false_negatives,
                "false_positives": false_positives,
                "false_negative_rate": (
                    false_negatives / n_pneumonia if n_pneumonia else 0.0
                ),
                "false_positive_rate": (
                    false_positives / n_normal if n_normal else 0.0
                ),
                "mean_probability": float(group["probability"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("n_images", ascending=False).reset_index(drop=True)


def confidence_distribution(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Count each outcome type within predicted-probability bins.

    Shows whether the errors are near-misses clustered at the threshold or
    confident mistakes - the latter being far more concerning to report.
    """
    scored = classify_outcomes(frame, threshold)
    scored["probability_bin"] = pd.cut(
        scored["probability"], bins=CONFIDENCE_BINS, include_lowest=True
    )

    table = (
        scored.groupby(["probability_bin", "outcome"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    table["probability_bin"] = table["probability_bin"].astype(str)
    for outcome in ("TN", "FP", "FN", "TP"):
        if outcome not in table.columns:
            table[outcome] = 0
    return table[["probability_bin", "TN", "FP", "FN", "TP"]]


def worst_errors_summary(frame: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    """Headline numbers about the most confident mistakes.

    Deliberately returns statistics, not file paths: the report needs to say
    "the most confident miss was assigned probability 0.03" without publishing
    which radiograph that was.
    """
    scored = classify_outcomes(frame, threshold)
    false_negatives = scored[scored["outcome"] == "FN"]
    false_positives = scored[scored["outcome"] == "FP"]

    def summarize(subset: pd.DataFrame, direction: str) -> Dict[str, Any]:
        if subset.empty:
            return {"count": 0, "most_confident_probability": None, "mean_probability": None}
        probabilities = subset["probability"]
        return {
            "count": int(len(subset)),
            "most_confident_probability": float(
                probabilities.min() if direction == "low" else probabilities.max()
            ),
            "mean_probability": float(probabilities.mean()),
        }

    return {
        "threshold": float(threshold),
        "false_negatives": summarize(false_negatives, "low"),
        "false_positives": summarize(false_positives, "high"),
    }
