"""Construct binary classifiers from their model configuration."""

from .baseline import build_baseline_from_config


def create_model(config: dict):
    model = config.get("model", config)
    name = str(model.get("name", "")).lower()
    if model.get("output_units", 1) != 1:
        raise ValueError("Binary classifiers require output_units=1")
    if name == "small_cnn":
        return build_baseline_from_config(model)
    if name == "densenet121":
        from .densenet import build_densenet121
        return build_densenet121(
            pretrained=bool(model.get("pretrained", True)),
            dropout=float(model.get("dropout", 0.2)),
            freeze_strategy=str(model.get("freeze_strategy", "last_block")),
        )
    raise ValueError(f"Unknown model.name={name!r}")
