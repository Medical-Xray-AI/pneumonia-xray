"""Image-level classification metrics for the pediatric pneumonia task.

Label contract (see `configs/data.yaml`):

    NORMAL    = 0
    PNEUMONIA = 1   <- positive class

Every metric here is a pure function of ground-truth labels and predicted
pneumonia probabilities. Nothing in this module reads the dataset, touches a
checkpoint, or knows which split it is being given, so the same code path
evaluates validation and locked test identically.

Zero-division convention: when a metric's denominator is zero (for example
specificity on a batch containing no NORMAL images) the metric is reported as
0.0 rather than NaN, matching `sklearn`'s `zero_division=0`. The affected
support count is always present in the returned dictionary so a reader can see
that the value is degenerate rather than genuinely bad.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

LABEL_NORMAL = 0
LABEL_PNEUMONIA = 1
POSITIVE_CLASS = "pneumonia"

CLASS_NAMES = {LABEL_NORMAL: "normal", LABEL_PNEUMONIA: "pneumonia"}


def _as_label_array(y_true: Sequence[int]) -> np.ndarray:
    """Validate and normalise ground-truth labels to an int array of 0/1."""
    arr = np.asarray(y_true)
    if arr.ndim != 1:
        raise ValueError(f"y_true must be 1-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("y_true is empty")
    if not np.all(np.isin(arr, [LABEL_NORMAL, LABEL_PNEUMONIA])):
        bad = sorted(set(np.unique(arr)) - {LABEL_NORMAL, LABEL_PNEUMONIA})
        raise ValueError(
            f"y_true must contain only {LABEL_NORMAL} (normal) and "
            f"{LABEL_PNEUMONIA} (pneumonia); found {bad}"
        )
    return arr.astype(int)


def _as_probability_array(y_prob: Sequence[float], expected_length: int) -> np.ndarray:
    """Validate and normalise predicted pneumonia probabilities."""
    arr = np.asarray(y_prob, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"y_prob must be 1-dimensional, got shape {arr.shape}")
    if arr.size != expected_length:
        raise ValueError(
            f"y_true and y_prob length mismatch: {expected_length} vs {arr.size}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("y_prob contains NaN or infinite values")
    if arr.min() < 0.0 or arr.max() > 1.0:
        raise ValueError(
            f"y_prob must lie in [0, 1]; observed range "
            f"[{arr.min():.6f}, {arr.max():.6f}]. Did the trainer export raw "
            f"logits instead of sigmoid probabilities?"
        )
    return arr


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator/denominator, or 0.0 when the denominator is zero."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def apply_threshold(y_prob: Sequence[float], threshold: float) -> np.ndarray:
    """Convert probabilities to hard labels using `probability >= threshold`.

    The `>=` convention is fixed project-wide so that a threshold of 0.0
    predicts pneumonia for every image and a threshold above 1.0 predicts
    normal for every image.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold}")
    arr = np.asarray(y_prob, dtype=float)
    return (arr >= float(threshold)).astype(int)


def confusion_counts(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Tuple[int, int, int, int]:
    """Return `(tn, fp, fn, tp)` with pneumonia as the positive class.

    Computed directly rather than via sklearn so that the ordering is explicit
    and cannot silently change if a split contains only one class.
    """
    truth = _as_label_array(y_true)
    pred = _as_label_array(y_pred)
    if truth.size != pred.size:
        raise ValueError(f"length mismatch: {truth.size} vs {pred.size}")

    tp = int(np.sum((truth == LABEL_PNEUMONIA) & (pred == LABEL_PNEUMONIA)))
    tn = int(np.sum((truth == LABEL_NORMAL) & (pred == LABEL_NORMAL)))
    fp = int(np.sum((truth == LABEL_NORMAL) & (pred == LABEL_PNEUMONIA)))
    fn = int(np.sum((truth == LABEL_PNEUMONIA) & (pred == LABEL_NORMAL)))
    return tn, fp, fn, tp


def pneumonia_sensitivity(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Recall of the pneumonia class: TP / (TP + FN).

    This is the project's clinical metric - it is the fraction of genuinely
    pneumonic radiographs the model does not miss.
    """
    _, _, fn, tp = confusion_counts(y_true, y_pred)
    return _safe_divide(tp, tp + fn)


def specificity(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Recall of the normal class: TN / (TN + FP)."""
    tn, fp, _, _ = confusion_counts(y_true, y_pred)
    return _safe_divide(tn, tn + fp)


def per_class_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """F1 for each class, keyed by class name."""
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)

    pneumonia_precision = _safe_divide(tp, tp + fp)
    pneumonia_recall = _safe_divide(tp, tp + fn)
    normal_precision = _safe_divide(tn, tn + fn)
    normal_recall = _safe_divide(tn, tn + fp)

    def f1(precision: float, recall: float) -> float:
        return _safe_divide(2 * precision * recall, precision + recall)

    return {
        "normal": f1(normal_precision, normal_recall),
        "pneumonia": f1(pneumonia_precision, pneumonia_recall),
    }


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Unweighted mean of the per-class F1 scores - the project's primary metric.

    Macro (rather than weighted) averaging is deliberate: pneumonia is roughly
    three quarters of this dataset, so a weighted average would let a model
    that never predicts NORMAL still look strong.
    """
    scores = per_class_f1(y_true, y_pred)
    return float(np.mean([scores["normal"], scores["pneumonia"]]))


def threshold_free_metrics(
    y_true: Sequence[int], y_prob: Sequence[float]
) -> Dict[str, float]:
    """ROC-AUC and PR-AUC, which do not depend on the decision threshold.

    Both are undefined when only one class is present; in that case they are
    reported as 0.0 and the caller can detect the situation from the support
    counts in `compute_metrics`.
    """
    truth = _as_label_array(y_true)
    prob = _as_probability_array(y_prob, truth.size)

    if len(np.unique(truth)) < 2:
        return {"roc_auc": 0.0, "pr_auc": 0.0}

    return {
        "roc_auc": float(roc_auc_score(truth, prob)),
        "pr_auc": float(average_precision_score(truth, prob)),
    }


def compute_metrics(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float
) -> Dict[str, Any]:
    """Full image-level metric bundle at a given decision threshold.

    Returns every number the report's comparison table needs, plus the raw
    confusion counts and class supports so that any figure can be rebuilt from
    this dictionary alone.
    """
    truth = _as_label_array(y_true)
    prob = _as_probability_array(y_prob, truth.size)
    pred = apply_threshold(prob, threshold)

    tn, fp, fn, tp = confusion_counts(truth, pred)
    class_f1 = per_class_f1(truth, pred)

    metrics: Dict[str, Any] = {
        "threshold": float(threshold),
        "n_images": int(truth.size),
        "n_normal": int(np.sum(truth == LABEL_NORMAL)),
        "n_pneumonia": int(np.sum(truth == LABEL_PNEUMONIA)),
        "macro_f1": float(np.mean([class_f1["normal"], class_f1["pneumonia"]])),
        "f1_normal": class_f1["normal"],
        "f1_pneumonia": class_f1["pneumonia"],
        "pneumonia_sensitivity": _safe_divide(tp, tp + fn),
        "specificity": _safe_divide(tn, tn + fp),
        "pneumonia_precision": _safe_divide(tp, tp + fp),
        "balanced_accuracy": float(
            np.mean([_safe_divide(tp, tp + fn), _safe_divide(tn, tn + fp)])
        ),
        "accuracy": _safe_divide(tp + tn, truth.size),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }
    metrics.update(threshold_free_metrics(truth, prob))
    return metrics
