import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.engine import EarlyStopping, step_scheduler, train_epoch, validate


def make_easy_loader(n=64, batch_size=16):
    # Linearly separable two-class toy data.
    g = torch.Generator().manual_seed(42)
    x0 = torch.randn(n // 2, 2, generator=g) * 0.2 - 1.0
    x1 = torch.randn(n // 2, 2, generator=g) * 0.2 + 1.0
    x = torch.cat([x0, x1], dim=0)
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)], dim=0)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def test_train_and_validate_reduce_loss():
    torch.manual_seed(42)
    model = nn.Linear(2, 1)
    loader = make_easy_loader()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    before = validate(model, loader, criterion, "cpu", amp=False)
    for _ in range(5):
        train_epoch(model, loader, optimizer, criterion, "cpu", amp=False)
    after = validate(model, loader, criterion, "cpu", amp=False)

    assert after.loss < before.loss
    assert 0.0 <= after.metrics["macro_f1"] <= 1.0
    assert after.metrics["macro_f1"] > 0.95


def test_validate_collects_prediction_rows():
    model = nn.Linear(2, 1)
    loader = make_easy_loader(n=16, batch_size=4)
    criterion = nn.BCEWithLogitsLoss()
    result = validate(model, loader, criterion, "cpu", amp=False, collect_predictions=True)
    assert result.predictions is not None
    assert len(result.predictions) == 16
    assert {"label", "logit", "probability"} <= set(result.predictions[0])
    assert all(0.0 <= r["probability"] <= 1.0 for r in result.predictions)


def test_early_stopping_is_deterministic():
    es = EarlyStopping(patience=2, min_delta=0.01, mode="max")
    assert es.step(0.50) is False
    assert es.step(0.505) is False  # below min_delta improvement
    assert es.step(0.504) is True


def test_scheduler_step_supports_plateau():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=0, factor=0.5
    )
    step_scheduler(scheduler, 0.5)
    step_scheduler(scheduler, 0.4)
    assert optimizer.param_groups[0]["lr"] == 0.05
