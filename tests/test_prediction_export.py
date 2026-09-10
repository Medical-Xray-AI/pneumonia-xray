import pytest
import torch
from torch.utils.data import DataLoader

from scripts.train_baseline import evaluate
from src.models import BaselineCNN, create_model


def test_factory_uses_configuration():
    model = create_model({"model": {"name": "small_cnn", "dropout": 0.15}})
    assert isinstance(model, BaselineCNN)
    assert model.classifier[1].p == 0.15
    with pytest.raises(ValueError, match="Unknown"):
        create_model({"model": {"name": "invalid"}})
    with pytest.raises(ValueError, match="output_units"):
        create_model({"model": {"name": "small_cnn", "output_units": 2}})


def test_export_preserves_sample_order():
    samples = [
        {"image": torch.tensor([float(i)]), "label": i % 2,
         "image_path": f"train/NORMAL/image_{i}.jpeg"}
        for i in [3, 0, 2, 1]
    ]
    metrics, probs, labels, paths = evaluate(
        torch.nn.Linear(1, 1), DataLoader(samples, batch_size=3),
        torch.device("cpu"), torch.nn.BCEWithLogitsLoss(), False, True,
    )
    assert paths == [s["image_path"] for s in samples]
    assert labels.tolist() == [s["label"] for s in samples]
    assert len(probs) == len(paths)
