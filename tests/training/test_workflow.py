import csv
import json
import sys

import pandas as pd
import pytest
import torch

from scripts import train
from src.training.checkpointing import _torch_load
from src.training.config import save_resolved_config
from src.training.registry import COLUMNS, append_registry


@pytest.mark.parametrize("name", ["small_cnn", "densenet121"])
def test_training_exports_canonical_validation(training_config, monkeypatch, name):
    cfg, path, output = training_config
    cfg["training"]["epochs"] = 1
    cfg["model"].update(name=name, pretrained=False, freeze_strategy="head_only")
    if name == "densenet121": cfg["normalization"] = {"type": "imagenet"}
    save_resolved_config(cfg, path)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path), "--run-id", name, "--device", "cpu"])
    assert train.main() == 0
    frame = pd.read_csv(output / name / "predictions_val.csv")
    manifest = pd.read_csv(cfg["data"]["manifests"]["validation"])
    assert frame.image_path.tolist() == manifest.image_path.tolist()
    assert frame.label.tolist() == manifest.label.tolist()
    assert frame.probability.between(0, 1).all()
    assert set(frame["split"]) == {"validation"}
    assert set(frame.model) == {name}
    assert set(frame.run_id) == {name}
    assert (output / name / "checkpoints/best.pt").is_file()
    from scripts.evaluate_model import build_parser, run_validation
    report = output / name / "evaluation"
    args = build_parser().parse_args([
        "validation", "--predictions", f"{name}={output / name / 'predictions_val.csv'}",
        "--manifest", cfg["data"]["manifests"]["validation"], "--out-dir", str(report)])
    assert run_validation(args) == 0
    frozen = json.loads((report / "tables/frozen_threshold.json").read_text())
    assert frozen["recommended_model"] == name and frozen["run_id"] == name
    assert frozen["selected_on"] == "validation"
    assert (report / "figures/roc_validation.png").is_file()



def test_interrupted_resume_matches_uninterrupted(training_config, monkeypatch):
    cfg, path, output = training_config
    # Exercise stochastic preprocessing and dropout as well as shuffle state.
    cfg["augmentation"] = {"rotation_degrees": 5, "translation_fraction": 0.02}
    save_resolved_config(cfg, path)
    def invoke(run_id, resume=None):
        args = ["train", "--config", str(path), "--run-id", run_id, "--device", "cpu"]
        if resume: args += ["--resume", str(resume)]
        monkeypatch.setattr(sys, "argv", args)
        return train.main()
    invoke("continuous")
    original = train.train_epoch
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2: raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)
    monkeypatch.setattr(train, "train_epoch", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"): invoke("resumed")
    monkeypatch.setattr(train, "train_epoch", original)
    last = output / "resumed/checkpoints/last.pt"
    invoke("resumed", last)
    expected = _torch_load(output / "continuous/checkpoints/last.pt")
    actual = _torch_load(last)
    assert actual["epoch"] == expected["epoch"] == 2
    for key in expected["model_state"]:
        assert torch.equal(actual["model_state"][key], expected["model_state"][key]), key
    assert actual["early_stopping_state"] == expected["early_stopping_state"]
    cfg["training"]["optimizer"]["lr"] *= 2
    save_resolved_config(cfg, path)
    with pytest.raises(ValueError, match="hash"): invoke("resumed", last)


def test_registry_preserves_header_and_previous_rows(tmp_path):
    path = tmp_path / "registry.csv"
    append_registry(path, {"run_id": "first", "status": "complete"})
    previous = path.read_bytes()
    append_registry(path, {"run_id": "second", "status": "complete", "val_macro_f1": 0.8})
    assert path.read_bytes().startswith(previous)
    with path.open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == COLUMNS
        rows = list(reader)
    assert rows[1]["status"] == "complete"
    assert rows[1]["val_macro_f1"] == "0.8"
    with pytest.raises(ValueError, match="already exists"):
        append_registry(path, {"run_id": "second"})


def test_accumulation_matches_large_batches_with_short_tail():
    from src.training.engine import train_epoch
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(42)
    data = TensorDataset(torch.randn(7, 3), torch.tensor([0., 1., 0., 1., 0., 1., 1.]))
    small = torch.nn.Linear(3, 1)
    large = torch.nn.Linear(3, 1)
    large.load_state_dict(small.state_dict())
    for model, batch, accumulation in [(small, 2, 2), (large, 4, 1)]:
        train_epoch(model, DataLoader(data, batch_size=batch),
                    torch.optim.SGD(model.parameters(), lr=0.1),
                    torch.nn.BCEWithLogitsLoss(), "cpu", amp=False,
                    grad_accum_steps=accumulation)
    for actual, expected in zip(small.parameters(), large.parameters()):
        torch.testing.assert_close(actual, expected)


def test_baseline_can_overfit_small_synthetic_batch():
    from src.models import create_model
    from src.training.engine import train_epoch, validate
    torch.manual_seed(42)
    images = torch.randn(4, 3, 32, 32) * 0.05
    images[2:] += 1.0
    batch = [(images, torch.tensor([0., 0., 1., 1.]))]
    model = create_model({"model": {"name": "small_cnn", "dropout": 0.0}})
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = torch.nn.BCEWithLogitsLoss()
    for _ in range(35):
        train_epoch(model, batch, optimizer, criterion, "cpu", amp=False)
    result = validate(model, batch, criterion, "cpu", amp=False)
    assert result.loss < 0.1
    assert result.metrics["macro_f1"] == 1.0
