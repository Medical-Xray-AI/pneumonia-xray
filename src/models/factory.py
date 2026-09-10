"""Construct binary classifiers from their model configuration."""

from .baseline import build_baseline_from_config


def create_model(config: dict):
    model = config.get("model", config)
    name = str(model.get("name", "")).lower()
    if model.get("output_units", 1) != 1:
        raise ValueError("Binary classifiers require output_units=1")
    if name == "small_cnn":
        return build_baseline_from_config(model)
    raise ValueError(f"Unknown model.name={name!r}")
