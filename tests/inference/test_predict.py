import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.evaluate import load_predictions, validate_manifest_predictions
from src.inference import predict
from src.inference.schema import InferenceSchemaError, PredictionRecord, predict_label, records_to_frame
from src.inference.smoke import build_smoke_checkpoint, build_synthetic_workspace, smoke_frozen_selection
from src.preprocessing.transforms import build_eval_transforms
from src.training.checkpointing import _torch_load


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = build_synthetic_workspace(tmp_path / "data", n_per_class=3)
    monkeypatch.setenv("XRAY_DATA_ROOT", str(ws["data_root"]))
    return ws


@pytest.fixture
def checkpoint(workspace, tmp_path):
    return build_smoke_checkpoint(workspace, tmp_path / "outputs", run_id="unit_run")


def test_load_model_restores_weights_in_eval_mode(checkpoint):
    loaded = predict.load_model(checkpoint, device="cpu")
    saved = _torch_load(checkpoint)["model_state"]
    assert not loaded.model.training
    assert loaded.run_id == "unit_run" and loaded.model_name == "small_cnn"
    assert loaded.checkpoint_epoch == 1
    assert loaded.run_dir == checkpoint.parent.parent
    for key, value in loaded.model.state_dict().items():
        assert torch.equal(value.cpu(), saved[key]), key


def test_preprocessing_reuses_eval_transform_with_stored_statistics(checkpoint, workspace):
    loaded = predict.load_model(checkpoint, device="cpu")
    image = predict.load_image(next(workspace["data_root"].rglob("*.png")))
    expected = build_eval_transforms(loaded.config["data"], mean=[0.4] * 3, std=[0.25] * 3)(image)
    actual = loaded.transform(image)
    assert actual.shape == (3, 224, 224)
    assert torch.equal(actual, expected)


def test_dataset_statistics_are_never_recomputed(checkpoint):
    loaded = predict.load_model(checkpoint, device="cpu")
    config = dict(loaded.config, normalization={"type": "dataset_statistics"})
    with pytest.raises(ValueError, match="normalization"):
        predict.normalization_stats(config)
    assert predict.normalization_stats({"normalization": {"type": "imagenet"}})[0] == [0.485, 0.456, 0.406]


def test_densenet_checkpoint_never_downloads_imagenet_weights(workspace, tmp_path, monkeypatch):
    path = build_smoke_checkpoint(workspace, tmp_path / "outputs", model_name="densenet121", run_id="dn")
    payload = _torch_load(path)
    payload["config"]["model"]["pretrained"] = True  # as in the real training config
    torch.save(payload, path)
    import src.models.densenet as densenet_module
    original = densenet_module.densenet121

    def guarded(*, weights=None, **kwargs):
        assert weights is None, "inference must not fetch ImageNet weights"
        return original(weights=None, **kwargs)

    monkeypatch.setattr(densenet_module, "densenet121", guarded)
    loaded = predict.load_model(path, device="cpu")
    assert loaded.model_name == "densenet121"
    records = predict.predict_images(loaded, sorted(workspace["data_root"].rglob("*.png"))[:2])
    assert all(0 <= r.probability <= 1 for r in records)


def test_probabilities_match_manual_forward_and_batching(checkpoint, workspace):
    loaded = predict.load_model(checkpoint, device="cpu")
    paths = sorted(workspace["data_root"].rglob("*.png"))[:5]
    batched = predict.predict_images(loaded, paths, batch_size=2)
    single = [predict.predict_images(loaded, [p])[0] for p in paths]
    with torch.no_grad():
        tensor = torch.stack([loaded.transform(predict.load_image(p)) for p in paths])
        manual = torch.sigmoid(loaded.model(tensor).reshape(-1)).tolist()
    assert [r.image_path for r in batched] == [p.name for p in paths]
    for b, s, m in zip(batched, single, manual):
        assert 0 <= b.probability <= 1
        assert math.isclose(b.probability, s.probability, rel_tol=1e-5, abs_tol=1e-6)
        assert math.isclose(b.probability, m, rel_tol=1e-5, abs_tol=1e-6)
        assert b.threshold is None and b.predicted_label is None


def test_threshold_uses_project_greater_or_equal_rule(checkpoint, workspace):
    loaded = predict.load_model(checkpoint, device="cpu")
    path = next(workspace["data_root"].rglob("*.png"))
    probability = predict.predict_images(loaded, [path])[0].probability
    at = predict.predict_images(loaded, [path], threshold=probability)[0]
    above = predict.predict_images(loaded, [path], threshold=min(1.0, probability + 1e-3))[0]
    assert at.predicted_label == 1 and at.predicted_class == "PNEUMONIA"
    assert above.predicted_label == 0 and above.predicted_class == "NORMAL"
    assert predict_label(0.5, 0.5) == 1 and predict_label(0.4999, 0.5) == 0
    with pytest.raises(InferenceSchemaError):
        predict.predict_images(loaded, [path], threshold=float("nan"))


def test_manifest_inference_satisfies_evaluation_contract(checkpoint, workspace, tmp_path):
    loaded = predict.load_model(checkpoint, device="cpu")
    frame = predict.predict_manifest(loaded, workspace["validation"], "validation", batch_size=4)
    manifest = pd.read_csv(workspace["validation"])
    assert frame.image_path.tolist() == manifest.image_path.tolist()
    assert frame.label.tolist() == manifest.label.tolist()
    assert set(frame.split) == {"validation"} and set(frame.run_id) == {"unit_run"}
    assert "predicted_label" not in frame  # no threshold before the validation freeze
    path = predict.write_predictions(frame, tmp_path / "predictions_val.csv")
    validate_manifest_predictions(load_predictions(path), manifest, "validation")


def test_locked_test_requires_matching_frozen_selection(checkpoint, workspace, tmp_path):
    loaded = predict.load_model(checkpoint, device="cpu")
    with pytest.raises(predict.LockedTestError):
        predict.predict_manifest(loaded, workspace["test"], "test")
    for wrong in (smoke_frozen_selection("small_cnn", "other_run"),
                  smoke_frozen_selection("densenet121", "unit_run"),
                  dict(smoke_frozen_selection("small_cnn", "unit_run"), selected_on="test"),
                  smoke_frozen_selection("small_cnn", "unit_run", threshold=1.5)):
        with pytest.raises(ValueError):
            predict.predict_manifest(loaded, workspace["test"], "test", frozen=wrong)
    frozen = smoke_frozen_selection("small_cnn", "unit_run", threshold=0.42)
    frame = predict.predict_manifest(loaded, workspace["test"], "test", frozen=frozen)
    assert set(frame.split) == {"test"} and set(frame.threshold) == {0.42}
    assert (frame.predicted_label == (frame.probability >= 0.42).astype(int)).all()


def test_manifest_split_mismatch_is_rejected(checkpoint, workspace):
    loaded = predict.load_model(checkpoint, device="cpu")
    with pytest.raises(ValueError, match="Expected only"):
        predict.predict_manifest(loaded, workspace["train"], "validation")
    with pytest.raises(ValueError, match="validation and test"):
        predict.predict_manifest(loaded, workspace["train"], "train")


def test_prediction_file_is_written_once(checkpoint, workspace, tmp_path):
    loaded = predict.load_model(checkpoint, device="cpu")
    frame = predict.predict_manifest(loaded, workspace["validation"], "validation")
    path = predict.write_predictions(frame, tmp_path / "p.csv")
    with pytest.raises(FileExistsError):
        predict.write_predictions(frame, path)
    predict.write_predictions(frame, path, overwrite=True)
    assert predict.default_output_path(loaded, "test").name == "predictions_test.csv"


def test_missing_inputs_fail_fast(tmp_path, checkpoint):
    with pytest.raises(FileNotFoundError):
        predict.load_model(tmp_path / "missing.pt")
    torch.save({"model_state": {}}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="configuration"):
        predict.load_model(tmp_path / "bad.pt")
    loaded = predict.load_model(checkpoint, device="cpu")
    with pytest.raises(FileNotFoundError):
        predict.predict_images(loaded, [tmp_path / "nope.png"])
    with pytest.raises(ValueError):
        predict.predict_images(loaded, [])


def test_benchmark_reports_size_and_latency(checkpoint):
    loaded = predict.load_model(checkpoint, device="cpu")
    summary = predict.benchmark(loaded, repeats=2, warmup=0)
    assert summary["parameters"] == sum(p.numel() for p in loaded.model.parameters())
    assert summary["weights_mb"] > 0 and summary["checkpoint_file_mb"] > 0
    assert summary["latency_ms_median"] > 0 and summary["input_shape"] == [1, 3, 224, 224]


@pytest.mark.parametrize("kwargs, message", [
    ({"probability": 1.2}, "probability"),
    ({"probability": float("nan")}, "probability"),
    ({"label": 2}, "label"),
    ({"model": "resnet"}, "model"),
    ({"run_id": " "}, "run_id"),
    ({"split": "dev"}, "split"),
    ({"threshold": 0.5, "predicted_label": 0, "predicted_class": "NORMAL", "probability": 0.7}, "disagrees"),
    ({"predicted_label": 1}, "requires a threshold"),
])
def test_schema_rejects_invalid_records(kwargs, message):
    base = dict(image_path="a.png", probability=0.3, model="small_cnn", run_id="r")
    with pytest.raises(InferenceSchemaError, match=message):
        PredictionRecord(**{**base, **kwargs})


def test_records_to_frame_orders_columns():
    records = [PredictionRecord.build(image_path=f"{i}.png", probability=p, model="small_cnn",
                                      run_id="r", split="validation", label=i % 2)
               for i, p in enumerate(np.linspace(0.1, 0.9, 3))]
    frame = records_to_frame(records)
    assert list(frame.columns) == ["run_id", "model", "split", "image_path", "label", "probability"]
    with pytest.raises(InferenceSchemaError):
        records_to_frame([])
