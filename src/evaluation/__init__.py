"""Image-level evaluation, threshold selection, and interpretation.

Public API kept deliberately small so that training and inference code can
depend on it without importing plotting or torch-only helpers.
"""

from src.evaluation.metrics import (
    LABEL_NORMAL,
    LABEL_PNEUMONIA,
    POSITIVE_CLASS,
    compute_metrics,
    confusion_counts,
    macro_f1,
    pneumonia_sensitivity,
    specificity,
    threshold_free_metrics,
)
from src.evaluation.threshold import ThresholdSelection, select_threshold, sweep_thresholds

__all__ = [
    "LABEL_NORMAL",
    "LABEL_PNEUMONIA",
    "POSITIVE_CLASS",
    "ThresholdSelection",
    "compute_metrics",
    "confusion_counts",
    "macro_f1",
    "pneumonia_sensitivity",
    "select_threshold",
    "specificity",
    "sweep_thresholds",
    "threshold_free_metrics",
]
