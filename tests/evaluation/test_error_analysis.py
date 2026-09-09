"""Tests for subgroup error analysis."""

import pandas as pd
import pytest

from src.evaluation.error_analysis import (
    MIN_GROUP_SIZE,
    classify_outcomes,
    confidence_distribution,
    error_rate_by,
    infer_subtype,
    worst_errors_summary,
)


def test_subtype_vocabulary_matches_configs_data_yaml():
    """configs/data.yaml audit_metadata.subtype_values fixes these four words.
    Filenames say 'bacteria'/'virus'; the contract says 'bacterial'/'viral'."""
    assert infer_subtype("a/person1_bacteria_2.jpeg", 1) == "bacterial"
    assert infer_subtype("a/person63_virus_119.jpeg", 1) == "viral"
    assert infer_subtype("a/NORMAL2-IM-1440-0001.jpeg", 0) == "normal"
    assert infer_subtype("a/unparseable.jpeg", 1) == "unknown"

    allowed = {"normal", "bacterial", "viral", "unknown"}
    for path, label in [("a/person1_bacteria_2.jpeg", 1), ("a/person1_virus_2.jpeg", 1),
                        ("a/IM-0115-0001.jpeg", 0), ("a/x.jpeg", 1)]:
        assert infer_subtype(path, label) in allowed


def _frame():
    return pd.DataFrame({
        "image_path": [f"train/PNEUMONIA/person{i}_bacteria_{i}.jpeg" for i in range(6)],
        "label":       [1, 1, 1, 0, 0, 0],
        "probability": [0.9, 0.8, 0.2, 0.1, 0.3, 0.7],
    })


def test_classify_outcomes_labels_all_four_cases():
    scored = classify_outcomes(_frame(), 0.5)
    assert scored["outcome"].tolist() == ["TP", "TP", "FN", "TN", "TN", "FP"]


def test_confidence_is_distance_from_the_operating_point():
    scored = classify_outcomes(_frame(), 0.5)
    assert scored.loc[0, "confidence"] == pytest.approx(0.9)   # predicted pneumonia
    assert scored.loc[3, "confidence"] == pytest.approx(0.9)   # predicted normal


def test_error_rate_by_reports_supports_and_rates():
    frame = _frame()
    frame["pneumonia_subtype"] = ["bacterial"] * 3 + ["normal"] * 3
    table = error_rate_by(frame, 0.5, "pneumonia_subtype")
    assert set(table.columns) >= {"n_images", "false_negatives", "false_negative_rate"}
    assert table["n_images"].sum() == 6


def test_small_subgroups_are_collapsed_for_privacy():
    """No subgroup row may be small enough to identify an individual."""
    frame = _frame()
    frame["pneumonia_subtype"] = ["bacterial"] * 3 + ["normal"] * 3
    table = error_rate_by(frame, 0.5, "pneumonia_subtype")
    assert MIN_GROUP_SIZE == 10
    assert set(table["pneumonia_subtype"]) == {"other"}  # both groups are < 10


def test_worst_errors_summary_reports_no_identifiers():
    summary = worst_errors_summary(_frame(), 0.5)
    assert summary["false_negatives"]["count"] == 1
    assert summary["false_positives"]["count"] == 1
    flat = str(summary)
    assert "person" not in flat and ".jpeg" not in flat


def test_confidence_distribution_covers_every_row():
    table = confidence_distribution(_frame(), 0.5)
    assert table[["TN", "FP", "FN", "TP"]].to_numpy().sum() == 6
