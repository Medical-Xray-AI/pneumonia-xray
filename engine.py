"""Model-agnostic train/validation engine for binary chest-X-ray classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch import nn


@dataclass
class EpochResult:
    loss: float
    metrics: Dict[str, float]
    predictions: Optional[List[Dict[str, Any]]] = None


class EarlyStopping:
    """Validation-monitor early stopping with a deterministic min-delta rule."""

    def __init__(self, patience: int = 4, min_delta: float = 0.0, mode: str = "max"):
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best: Optional[float] = None
        self.bad_epochs = 0

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        """Update state and return ``True`` when training should stop."""
        value = float(value)
        if self._is_improvement(value):
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def state_dict(self) -> Dict[str, Any]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "best": self.best,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.patience = int(state["patience"])
        self.min_delta = float(state["min_delta"])
        self.mode = str(state["mode"])
        self.best = None if state.get("best") is None else float(state["best"])
        self.bad_epochs = int(state.get("bad_epochs", 0))


def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
    """Support team datasets returning tuple/list or dict batches."""
    if isinstance(batch, Mapping):
        if "image" not in batch or "label" not in batch:
            raise KeyError("Dict batch must contain 'image' and 'label'.")
        return batch["image"], batch["label"], batch.get("metadata")
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        metadata = batch[2] if len(batch) >= 3 else None
        return batch[0], batch[1], metadata
    raise TypeError("Expected batch as mapping or tuple/list of (image, label[, metadata]).")


def _flatten_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 1:
        raise ValueError(
            f"Binary model must return one logit per image; got shape {tuple(logits.shape)}"
        )
    return logits


def _binary_metrics(labels: torch.Tensor, probs: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    labels = labels.detach().to(torch.int64).cpu().view(-1)
    probs = probs.detach().to(torch.float32).cpu().view(-1)
    preds = (probs >= float(threshold)).to(torch.int64)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    def safe_div(a: float, b: float) -> float:
        return float(a / b) if b else 0.0

    precision_pos = safe_div(tp, tp + fp)
    recall_pos = safe_div(tp, tp + fn)  # sensitivity
    f1_pos = safe_div(2 * precision_pos * recall_pos, precision_pos + recall_pos)

    precision_neg = safe_div(tn, tn + fn)
    recall_neg = safe_div(tn, tn + fp)  # specificity
    f1_neg = safe_div(2 * precision_neg * recall_neg, precision_neg + recall_neg)

    return {
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "macro_f1": (f1_pos + f1_neg) / 2.0,
        "sensitivity": recall_pos,
        "specificity": recall_neg,
        "precision_pneumonia": precision_pos,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _autocast_context(device: torch.device, enabled: bool):
    # torch.autocast is stable across current CPU/CUDA releases and allows the
    # same engine to be unit-tested on CPU with AMP disabled.
    return torch.autocast(device_type=device.type, enabled=bool(enabled and device.type == "cuda"))


def train_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
    *,
    scaler: Optional[Any] = None,
    amp: bool = True,
    grad_accum_steps: int = 1,
    max_grad_norm: Optional[float] = None,
    threshold: float = 0.5,
) -> EpochResult:
    """Train one epoch and return loss plus image-level binary metrics."""
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    device = torch.device(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_samples = 0
    all_labels: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []

    n_batches = len(loader) if hasattr(loader, "__len__") else None
    for step, batch in enumerate(loader):
        images, labels, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().view(-1)

        with _autocast_context(device, amp):
            logits = _flatten_logits(model(images))
            loss = criterion(logits, labels)
            scaled_loss = loss / grad_accum_steps

        if scaler is not None and scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = ((step + 1) % grad_accum_steps == 0) or (
            n_batches is not None and step + 1 == n_batches
        )
        if should_step:
            if max_grad_norm is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.numel()
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        all_labels.append(labels.detach().cpu())
        all_probs.append(torch.sigmoid(logits.detach()).cpu())

    if total_samples == 0:
        raise ValueError("Training loader produced zero samples.")
    metrics = _binary_metrics(torch.cat(all_labels), torch.cat(all_probs), threshold)
    return EpochResult(loss=total_loss / total_samples, metrics=metrics)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device | str,
    *,
    amp: bool = True,
    threshold: float = 0.5,
    collect_predictions: bool = False,
) -> EpochResult:
    """Evaluate one split without parameter updates.

    ``threshold`` is fixed at 0.5 for training monitoring. Member 4 later uses
    the exported probabilities to choose the final threshold on validation only.
    """
    device = torch.device(device)
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    prediction_rows: List[Dict[str, Any]] = []
    running_index = 0

    for batch in loader:
        images, labels, metadata = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().view(-1)
        with _autocast_context(device, amp):
            logits = _flatten_logits(model(images))
            loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        batch_size = labels.numel()
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

        if collect_predictions:
            for i in range(batch_size):
                row: Dict[str, Any] = {
                    "row_index": running_index + i,
                    "label": int(labels[i].item()),
                    "logit": float(logits[i].item()),
                    "probability": float(probs[i].item()),
                }
                # Optional metadata may be a list of dicts or a collated dict.
                if isinstance(metadata, Mapping):
                    for key, value in metadata.items():
                        try:
                            item = value[i]
                            if torch.is_tensor(item) and item.ndim == 0:
                                item = item.item()
                            row[key] = item
                        except Exception:
                            pass
                elif isinstance(metadata, (list, tuple)) and i < len(metadata):
                    if isinstance(metadata[i], Mapping):
                        row.update(metadata[i])
                prediction_rows.append(row)
            running_index += batch_size

    if total_samples == 0:
        raise ValueError("Validation loader produced zero samples.")
    labels_cat = torch.cat(all_labels)
    probs_cat = torch.cat(all_probs)
    metrics = _binary_metrics(labels_cat, probs_cat, threshold)
    return EpochResult(
        loss=total_loss / total_samples,
        metrics=metrics,
        predictions=prediction_rows if collect_predictions else None,
    )


def step_scheduler(scheduler: Optional[Any], monitored_value: float) -> None:
    """Step either ReduceLROnPlateau or ordinary epoch schedulers."""
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(float(monitored_value))
    else:
        scheduler.step()
