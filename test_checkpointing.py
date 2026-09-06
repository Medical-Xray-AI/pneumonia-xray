from pathlib import Path

import torch
from torch import nn

from src.training.checkpointing import CheckpointManager, load_checkpoint
from src.training.engine import EarlyStopping


def test_best_only_rule_and_resume(tmp_path: Path):
    torch.manual_seed(42)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    early = EarlyStopping(patience=3, mode="max")
    manager = CheckpointManager(tmp_path / "checkpoints", monitor="macro_f1", mode="max")

    # First epoch is automatically best.
    assert manager.save(
        epoch=1,
        metric_value=0.60,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early,
        run_id="test",
    )
    best_mtime = manager.best_path.stat().st_mtime_ns

    # Worse epoch updates last.pt but must not replace best.pt.
    assert not manager.save(
        epoch=2,
        metric_value=0.55,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early,
        run_id="test",
    )
    assert manager.best_path.stat().st_mtime_ns == best_mtime

    # Mutate weights then restore last checkpoint.
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(10.0)

    ckpt = load_checkpoint(
        manager.last_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early,
        map_location="cpu",
        restore_rng=False,
    )
    assert ckpt["epoch"] == 2
    for k, v in model.state_dict().items():
        assert torch.allclose(v, original[k])


def test_better_epoch_replaces_best(tmp_path: Path):
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path, mode="max")
    manager.save(epoch=1, metric_value=0.4, model=model, optimizer=optimizer)
    assert manager.save(epoch=2, metric_value=0.7, model=model, optimizer=optimizer)
    ckpt = load_checkpoint(manager.best_path, model=model, map_location="cpu", restore_rng=False)
    assert ckpt["epoch"] == 2
    assert ckpt["best_metric"] == 0.7
