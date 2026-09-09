"""Known-answer and edge-case tests for the image-level metrics.

The expected values in `test_toy_confusion_matrix_known_values` are computed
by hand from the fixture below, not from the implementation, so this file
would catch a regression that a self-consistent refactor would hide.

Fixture: 4 pneumonia, 6 normal, threshold 0.5
    predictions -> TP = 3, FN = 1, FP = 1, TN = 5
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    apply_threshold,
    compute_metrics,
    confusion_counts,
    macro_f1,
    per_class_f1,
    pneumonia_sensitivity,
    specificity,
    threshold_free_metrics,
)

Y_TRUE = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
Y_PROB = [0.9, 0.8, 0.7, 0.2, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05]
THRESHOLD = 0.5


def test_toy_confusion_matrix_known_values():
    tn, fp, fn, tp = confusion_counts(Y_TRUE, apply_threshold(Y_PROB, THRESHOLD))
    assert (tn, fp, fn, tp) == (5, 1, 1, 3)


def test_sensitivity_and_specificity_known_values():
    pred = apply_threshold(Y_PROB, THRESHOLD)
    assert pneumonia_sensitivity(Y_TRUE, pred) == pytest.approx(3 / 4)
    assert specificity(Y_TRUE, pred) == pytest.approx(5 / 6)


def test_macro_f1_known_value():
    pred = apply_threshold(Y_PROB, THRESHOLD)
    scores = per_class_f1(Y_TRUE, pred)
    assert scores["pneumonia"] == pytest.approx(0.75)
    assert scores["normal"] == pytest.approx(5 / 6)
    assert macro_f1(Y_TRUE, pred) == pytest.approx((0.75 + 5 / 6) / 2)


def test_compute_metrics_bundle_is_self_consistent():
    metrics = compute_metrics(Y_TRUE, Y_PROB, THRESHOLD)
    assert metrics["n_images"] == 10
    assert metrics["n_normal"] == 6
    assert metrics["n_pneumonia"] == 4
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["balanced_accuracy"] == pytest.approx((3 / 4 + 5 / 6) / 2)
    counts = (
        metrics["true_negatives"] + metrics["false_positives"]
        + metrics["false_negatives"] + metrics["true_positives"]
    )
    assert counts == metrics["n_images"]


def test_threshold_convention_is_greater_or_equal():
    """`probability >= threshold` is fixed project-wide; a value exactly on the
    threshold must be predicted as pneumonia."""
    assert apply_threshold([0.5], 0.5).tolist() == [1]
    assert apply_threshold([0.4999], 0.5).tolist() == [0]


def test_threshold_zero_predicts_all_pneumonia():
    assert apply_threshold(Y_PROB, 0.0).tolist() == [1] * len(Y_PROB)


def test_perfect_and_inverted_predictions():
    perfect = compute_metrics(Y_TRUE, [float(y) for y in Y_TRUE], 0.5)
    assert perfect["macro_f1"] == pytest.approx(1.0)
    assert perfect["roc_auc"] == pytest.approx(1.0)

    inverted = compute_metrics(Y_TRUE, [1.0 - y for y in Y_TRUE], 0.5)
    assert inverted["pneumonia_sensitivity"] == pytest.approx(0.0)
    assert inverted["roc_auc"] == pytest.approx(0.0)


def test_single_class_input_does_not_crash():
    """Threshold-free metrics are undefined with one class present; the
    documented convention is 0.0 rather than an exception."""
    metrics = compute_metrics([1, 1, 1], [0.9, 0.8, 0.7], 0.5)
    assert metrics["roc_auc"] == 0.0
    assert metrics["pr_auc"] == 0.0
    assert metrics["specificity"] == 0.0  # no normal images -> zero-division rule
    assert metrics["pneumonia_sensitivity"] == pytest.approx(1.0)


def test_rejects_probabilities_outside_unit_interval():
    """Catches the most likely trainer bug: exporting logits, not sigmoid."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compute_metrics(Y_TRUE, [2.5] * len(Y_TRUE), 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        compute_metrics(Y_TRUE, [-3.0] * len(Y_TRUE), 0.5)


def test_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="must contain only"):
        compute_metrics([0, 1, 2], [0.1, 0.2, 0.3], 0.5)


def test_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        compute_metrics([0, 1], [0.1, 0.2, 0.3], 0.5)


def test_rejects_nan_probability():
    with pytest.raises(ValueError, match="NaN or infinite"):
        compute_metrics([0, 1], [0.1, np.nan], 0.5)


def test_rejects_out_of_range_threshold():
    with pytest.raises(ValueError, match="threshold must lie"):
        compute_metrics(Y_TRUE, Y_PROB, 1.5)
