"""Evaluation integration guards use synthetic metadata, never real test data."""
import json

import pandas as pd
import pytest

from scripts.evaluate_model import build_parser, run_validation, run_test
from src.evaluation.evaluate import evaluate_validation, validate_manifest_predictions
from src.evaluation.error_analysis import attach_manifest_metadata


@pytest.fixture
def exports(tmp_path):
    manifest = pd.DataFrame({"image_path": ["train/NORMAL/a.jpeg", "train/PNEUMONIA/b.jpeg"],
                             "label": [0, 1], "split": ["validation"] * 2})
    frame = manifest.assign(probability=[0.2, 0.8], model="small_cnn", run_id="cnn_run")
    return manifest, frame


@pytest.mark.parametrize("failure", ["missing", "extra", "label", "split", "mixed_split", "manifest_split", "duplicate"])
def test_validation_rejects_invalid_split(exports, failure):
    manifest, frame = exports
    if failure == "missing": frame = frame.iloc[:1]
    if failure == "extra": frame = pd.concat([frame, frame.iloc[:1].assign(image_path="extra.jpeg")])
    if failure == "label": frame.loc[0, "label"] = 1
    if failure == "split": frame["split"] = "test"
    if failure == "mixed_split": frame.loc[0, "split"] = "test"
    if failure == "manifest_split": manifest["split"] = "test"
    if failure == "duplicate": frame.loc[1, "image_path"] = frame.loc[0, "image_path"]
    with pytest.raises(ValueError):
        evaluate_validation(frame, manifest=manifest, model="small_cnn")


def test_validation_joins_labels_by_path_not_row_order(exports):
    manifest, frame = exports
    metrics, _ = evaluate_validation(frame.iloc[::-1], manifest=manifest, model="small_cnn")
    assert metrics["macro_f1"] == 1 and metrics["run_id"] == "cnn_run"


def test_error_metadata_uses_full_path():
    frame = pd.DataFrame({"image_path": ["train/a.jpeg", "test/a.jpeg"], "label": [1, 1]})
    manifest = frame.assign(pneumonia_subtype=["bacterial", "viral"])
    assert attach_manifest_metadata(frame, manifest).pneumonia_subtype.tolist() == ["bacterial", "viral"]


def test_comparison_preserves_selected_run_and_rejects_different_set(exports, tmp_path):
    manifest, frame = exports
    manifest_path = tmp_path / "validation.csv"
    manifest.to_csv(manifest_path, index=False)
    baseline, dense = tmp_path / "baseline.csv", tmp_path / "dense.csv"
    frame.assign(probability=[0.7, 0.3]).to_csv(baseline, index=False)
    frame.assign(model="densenet121", run_id="dense_run").to_csv(dense, index=False)
    args = build_parser().parse_args(["validation", "--predictions", f"small_cnn={baseline}",
        "--predictions", f"densenet121={dense}", "--manifest", str(manifest_path),
        "--out-dir", str(tmp_path / "report")])
    assert run_validation(args) == 0
    frozen = json.loads((tmp_path / "report/tables/frozen_threshold.json").read_text())
    assert frozen["recommended_model"] == "densenet121"
    assert frozen["run_id"] == "dense_run"
    frame.iloc[:1].assign(model="densenet121", run_id="dense_run").to_csv(dense, index=False)
    with pytest.raises(ValueError, match="image set mismatch"): run_validation(args)


@pytest.mark.parametrize("failure", [None, "model", "run_id", "selected_on", "threshold", "missing_run"])
def test_locked_test_requires_matching_frozen_identity(exports, tmp_path, failure):
    manifest, frame = exports
    manifest["split"] = frame["split"] = "test"
    manifest_path, prediction_path, frozen_path = [tmp_path / name for name in ("test.csv", "pred.csv", "frozen.json")]
    manifest.to_csv(manifest_path, index=False)
    frame.to_csv(prediction_path, index=False)
    frozen = {"recommended_model": "small_cnn", "run_id": "cnn_run", "selected_on": "validation", "threshold": 0.5}
    if failure == "model": frozen["recommended_model"] = "densenet121"
    if failure == "run_id": frozen["run_id"] = "different_run"
    if failure == "selected_on": frozen["selected_on"] = "test"
    if failure == "threshold": frozen["threshold"] = float("nan")
    if failure == "missing_run": frozen.pop("run_id")
    frozen_path.write_text(json.dumps(frozen))
    args = build_parser().parse_args(["test", "--predictions", f"small_cnn={prediction_path}",
        "--manifest", str(manifest_path), "--threshold-file", str(frozen_path), "--out-dir", str(tmp_path / "report")])
    if failure:
        with pytest.raises(ValueError): run_test(args)
        assert not (tmp_path / "report").exists()
    else:
        assert run_test(args) == 0


@pytest.mark.parametrize("args", [
    ["validation", "--predictions", "small_cnn=x.csv"],
    ["test", "--predictions", "small_cnn=x.csv", "--manifest", "test.csv", "--threshold", "0.5"],
])
def test_cli_has_no_manifest_or_numeric_threshold_bypass(args):
    with pytest.raises(SystemExit): build_parser().parse_args(args)
