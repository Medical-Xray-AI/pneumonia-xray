"""Tests for the prediction-file contract and evaluation orchestration.

These are the guard rails for the hand-off from Uzv 2 / Uzv 3: a malformed
prediction export must fail here with a readable message, not silently
produce a plausible-looking number in the report.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluate import (
    PredictionSchemaError,
    REQUIRED_PREDICTION_COLUMNS,
    build_comparison_table,
    evaluate_predictions,
    evaluate_validation,
    load_predictions,
    recommend_model,
    write_comparison_table,
)


def _valid_frame(n: int = 60, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.7).astype(int)
    probabilities = np.clip(rng.normal(np.where(labels == 1, 0.7, 0.3), 0.15), 0.0, 1.0)
    return pd.DataFrame(
        {
            "image_path": [f"train/PNEUMONIA/img_{i:04d}.jpeg" for i in range(n)],
            "label": labels,
            "probability": probabilities,
        }
    )


def _write(frame: pd.DataFrame, tmp_path, name: str = "predictions_val.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_loads_a_valid_prediction_file(tmp_path):
    frame = load_predictions(_write(_valid_frame(), tmp_path))
    assert list(REQUIRED_PREDICTION_COLUMNS) == ["image_path", "label", "probability"]
    assert frame["label"].dtype.kind == "i"
    assert frame["probability"].dtype.kind == "f"


def test_extra_columns_are_preserved(tmp_path):
    frame = _valid_frame()
    frame["run_id"] = "20260905_member3_densenet_s42"
    loaded = load_predictions(_write(frame, tmp_path))
    assert "run_id" in loaded.columns


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_predictions(tmp_path / "does_not_exist.csv")


def test_missing_required_column_is_reported_by_name(tmp_path):
    frame = _valid_frame().drop(columns=["probability"])
    with pytest.raises(PredictionSchemaError, match="probability"):
        load_predictions(_write(frame, tmp_path))


def test_logits_instead_of_probabilities_are_rejected(tmp_path):
    """The single most likely trainer bug - it must not reach the report."""
    frame = _valid_frame()
    frame["probability"] = np.linspace(-4.0, 6.0, len(frame))
    with pytest.raises(PredictionSchemaError, match="logits"):
        load_predictions(_write(frame, tmp_path))


def test_hard_predictions_instead_of_probabilities_are_rejected(tmp_path):
    frame = _valid_frame()
    frame["probability"] = frame["label"].astype(float)
    with pytest.raises(PredictionSchemaError, match="only 0.0 and 1.0"):
        load_predictions(_write(frame, tmp_path))


def test_string_labels_are_rejected_with_a_useful_message(tmp_path):
    frame = _valid_frame()
    frame["label"] = np.where(frame["label"] == 1, "PNEUMONIA", "NORMAL")
    with pytest.raises(PredictionSchemaError, match="0 = normal"):
        load_predictions(_write(frame, tmp_path))


def test_duplicate_image_paths_are_rejected(tmp_path):
    frame = _valid_frame()
    frame.loc[5, "image_path"] = frame.loc[4, "image_path"]
    with pytest.raises(PredictionSchemaError, match="duplicated image_path"):
        load_predictions(_write(frame, tmp_path))


def test_missing_probability_values_are_rejected(tmp_path):
    frame = _valid_frame()
    frame.loc[2, "probability"] = np.nan
    with pytest.raises(PredictionSchemaError, match="missing values"):
        load_predictions(_write(frame, tmp_path))


def test_empty_file_is_rejected(tmp_path):
    empty = pd.DataFrame(columns=list(REQUIRED_PREDICTION_COLUMNS))
    with pytest.raises(PredictionSchemaError, match="no rows"):
        load_predictions(_write(empty, tmp_path))


def test_evaluate_validation_selects_and_records_its_threshold():
    frame = _valid_frame()
    frame["model"], frame["run_id"] = "baseline", "r1"
    manifest = frame[["image_path", "label"]].assign(split="validation")
    metrics, selection = evaluate_validation(frame, manifest=manifest, model="baseline", run_id="r1")

    assert metrics["split"] == "validation"
    assert metrics["threshold"] == pytest.approx(selection.threshold)
    assert metrics["threshold_source"].startswith("selected_on_validation")
    assert metrics["model"] == "baseline"


def test_evaluate_predictions_marks_a_provided_threshold_as_such():
    metrics = evaluate_predictions(_valid_frame(), threshold=0.5, model="densenet121", split="test")
    assert metrics["threshold_source"] == "provided"
    assert metrics["threshold"] == pytest.approx(0.5)


def test_comparison_table_is_sorted_by_macro_f1(tmp_path):
    rows = [
        {"model": "baseline", "macro_f1": 0.81, "threshold": 0.5, "run_id": "r1",
         "pneumonia_sensitivity": 0.9, "specificity": 0.7},
        {"model": "densenet121", "macro_f1": 0.93, "threshold": 0.42, "run_id": "r2",
         "pneumonia_sensitivity": 0.95, "specificity": 0.88},
    ]
    table = build_comparison_table(rows)
    assert table.loc[0, "model"] == "densenet121"

    out = tmp_path / "tables" / "validation_metrics.csv"
    write_comparison_table(rows, out)
    assert out.is_file()
    assert list(pd.read_csv(out)["model"]) == ["densenet121", "baseline"]

    best = recommend_model(table)
    assert best["model"] == "densenet121"
    assert best["threshold"] == pytest.approx(0.42)
