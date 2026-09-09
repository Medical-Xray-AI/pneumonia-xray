"""Aggregate evaluation figures for the report and slides.

Every figure here is an aggregate: curves, counts, and rates. No function in
this module writes an X-ray image, a filename, or any per-patient value, so
its entire output is safe to commit under the repository data policy.

Rendering is headless (`Agg`) so the same code produces identical files on the
GPU server, in a notebook, and in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from src.evaluation.metrics import (
    _as_label_array,
    _as_probability_array,
    apply_threshold,
    confusion_counts,
)
from src.evaluation.threshold import sweep_thresholds

# Colourblind-safe, and legible when the slides are projected.
COLORS = {
    "baseline": "#4C78A8",
    "densenet121": "#2E8B57",
    "reference": "#999999",
    "operating_point": "#D1495B",
}
DEFAULT_DPI = 150


def _color_for(model: str, index: int) -> str:
    palette = ["#4C78A8", "#2E8B57", "#6F4E7C", "#E58606", "#A5AA99"]
    return COLORS.get(model.lower(), palette[index % len(palette)])


def _finalize(fig: plt.Figure, out_path: str | Path, dpi: int = DEFAULT_DPI) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_curves(
    curves: Mapping[str, tuple[Sequence[int], Sequence[float]]],
    out_path: str | Path,
    *,
    title: str = "ROC curves (validation)",
) -> Path:
    """Overlay one ROC curve per model.

    `curves` maps a model name to its `(y_true, y_prob)` pair, so the baseline
    and DenseNet appear on the same axes at the same scale - the comparison
    the research question actually asks about.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))

    for index, (model, (y_true, y_prob)) in enumerate(curves.items()):
        truth = _as_label_array(y_true)
        prob = _as_probability_array(y_prob, truth.size)
        fpr, tpr, _ = roc_curve(truth, prob)
        area = float(auc(fpr, tpr))
        ax.plot(fpr, tpr, label=f"{model} (AUC = {area:.3f})",
                color=_color_for(model, index), linewidth=2)

    ax.plot([0, 1], [0, 1], linestyle="--", color=COLORS["reference"],
            linewidth=1, label="chance")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (pneumonia sensitivity)")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    return _finalize(fig, out_path)


def plot_pr_curves(
    curves: Mapping[str, tuple[Sequence[int], Sequence[float]]],
    out_path: str | Path,
    *,
    title: str = "Precision-recall curves (validation)",
) -> Path:
    """Overlay one precision-recall curve per model.

    The dashed baseline is the pneumonia prevalence, which is what a
    no-skill classifier achieves. On an imbalanced set like this one, PR-AUC
    without that reference line is easy to over-read.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    prevalence: Optional[float] = None

    for index, (model, (y_true, y_prob)) in enumerate(curves.items()):
        truth = _as_label_array(y_true)
        prob = _as_probability_array(y_prob, truth.size)
        precision, recall, _ = precision_recall_curve(truth, prob)
        ap = float(-np.sum(np.diff(recall) * np.array(precision)[:-1]))
        ax.plot(recall, precision, label=f"{model} (AP = {ap:.3f})",
                color=_color_for(model, index), linewidth=2)
        prevalence = float(np.mean(truth))

    if prevalence is not None:
        ax.axhline(prevalence, linestyle="--", color=COLORS["reference"], linewidth=1,
                   label=f"no-skill ({prevalence:.3f})")

    ax.set_xlabel("Recall (pneumonia sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    return _finalize(fig, out_path)


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float,
    out_path: str | Path,
    *,
    title: str = "Confusion matrix",
) -> Path:
    """Render a 2x2 confusion matrix with counts and row percentages.

    Both are shown because the count alone hides how bad a minority-class
    error rate is, and the percentage alone hides how few images it rests on.
    """
    truth = _as_label_array(y_true)
    prob = _as_probability_array(y_prob, truth.size)
    pred = apply_threshold(prob, threshold)
    tn, fp, fn, tp = confusion_counts(truth, pred)

    counts = np.array([[tn, fp], [fn, tp]], dtype=float)
    row_totals = counts.sum(axis=1, keepdims=True)
    percentages = np.divide(
        counts, row_totals, out=np.zeros_like(counts), where=row_totals != 0
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    image = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=1)

    labels = ["normal", "pneumonia"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title}\n(threshold = {threshold:.4f})")

    for row in range(2):
        for col in range(2):
            text_color = "white" if percentages[row, col] > 0.5 else "#222222"
            ax.text(
                col, row,
                f"{int(counts[row, col])}\n{percentages[row, col]:.1%}",
                ha="center", va="center", color=text_color, fontsize=12,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, label="Row-normalised rate")
    return _finalize(fig, out_path)


def plot_threshold_sweep(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    out_path: str | Path,
    *,
    selected_threshold: Optional[float] = None,
    title: str = "Threshold sweep (validation)",
) -> Path:
    """Show macro F1, sensitivity and specificity across all thresholds.

    This is the figure that justifies the operating point: it makes visible
    both where the chosen threshold sits and how sharply the trade-off moves
    around it.
    """
    records = sweep_thresholds(y_true, y_prob)
    thresholds = [r["threshold"] for r in records]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(thresholds, [r["macro_f1"] for r in records],
            label="macro F1", color="#4C78A8", linewidth=2)
    ax.plot(thresholds, [r["pneumonia_sensitivity"] for r in records],
            label="pneumonia sensitivity", color="#2E8B57", linewidth=1.5)
    ax.plot(thresholds, [r["specificity"] for r in records],
            label="specificity", color="#E58606", linewidth=1.5)

    if selected_threshold is not None:
        ax.axvline(selected_threshold, color=COLORS["operating_point"],
                   linestyle="--", linewidth=1.5,
                   label=f"selected = {selected_threshold:.4f}")

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower center", frameon=False, ncol=2)
    ax.grid(alpha=0.25, linewidth=0.5)
    return _finalize(fig, out_path)


def plot_training_history(
    history: Mapping[str, Sequence[float]],
    out_path: str | Path,
    *,
    title: str = "Training history",
    best_epoch: Optional[int] = None,
) -> Path:
    """Plot loss and the monitored validation metric across epochs.

    Expects the `history.csv` columns from the artifact contract in
    `docs/TEAM_WORKFLOW.md`; any missing series is skipped rather than
    raising, so a partially-logged run still produces a usable figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for key, color, label in (
        ("train_loss", "#4C78A8", "train loss"),
        ("val_loss", "#D1495B", "validation loss"),
    ):
        if key in history:
            axes[0].plot(range(1, len(history[key]) + 1), history[key],
                         color=color, label=label, linewidth=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25, linewidth=0.5)

    for key, color, label in (
        ("val_macro_f1", "#2E8B57", "validation macro F1"),
        ("val_pneumonia_sensitivity", "#E58606", "validation sensitivity"),
    ):
        if key in history:
            axes[1].plot(range(1, len(history[key]) + 1), history[key],
                         color=color, label=label, linewidth=1.8)
    if best_epoch is not None:
        axes[1].axvline(best_epoch, color=COLORS["operating_point"],
                        linestyle="--", linewidth=1.5, label=f"best epoch = {best_epoch}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric value")
    axes[1].set_title("Validation metrics")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25, linewidth=0.5)

    fig.suptitle(title)
    return _finalize(fig, out_path)
