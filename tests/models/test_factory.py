import pytest

from src.models.factory import create_model
from src.models.densenet import DenseNet121Binary


def test_factory_creates_densenet_without_download():
    model = create_model(
        {
            "model": {
                "name": "densenet121",
                "pretrained": False,
                "dropout": 0.1,
                "freeze_strategy": "head_only",
            }
        }
    )
    assert isinstance(model, DenseNet121Binary)


def test_factory_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown"):
        create_model({"model": {"name": "not_a_model"}})
