"""
Training entry point for the baseline CNN (Member 2 deliverables:
B0 overfit check, B1 full baseline, B2 augmentation/class-weight ablation).

Modes:
    B0:
        python scripts/train_baseline.py --mode overfit

    B1:
        python scripts/train_baseline.py --mode full

    B2:
        python scripts/train_baseline.py --mode full --no_augmentation
        python scripts/train_baseline.py --mode full --no_class_weight
        python scripts/train_baseline.py --mode full --no_augmentation --no_class_weight
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# IMPORTANT ON WINDOWS:
# import torch before numpy/pandas to avoid native DLL conflicts seen on this machine.
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, Subset


# ---------------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline import build_baseline_from_config  # noqa: E402
from src.preprocessing.transforms import (  # noqa: E402
    build_eval_transforms,
    build_no_augmentation_train_transforms,
    build_stats_transforms,
    build_train_transforms,
    resolve_normalization_stats,
)

try:
    from src.data.dataset import ChestXrayDataset as _ExternalDataset
except ImportError:
    _ExternalDataset = None


LABEL_TO_IDX = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}


# ---------------------------------------------------------------------------
# Fallback manifest dataset
# ---------------------------------------------------------------------------

class ManifestDataset(Dataset):
    """
    Fallback manifest dataset.

    Used only if Member 1's ChestXrayDataset cannot be imported with the
    expected signature.
    """

    def __init__(
        self,
        manifest_csv: str,
        data_root: str,
        transform=None,
    ):
        df = pd.read_csv(manifest_csv)

        required = {
            "image_path",
            "label",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{manifest_csv} is missing required columns: {missing}"
            )

        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image

        row = self.df.iloc[idx]

        image = Image.open(
            self.data_root / row["image_path"]
        )

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["label"])

        return image, label

    def class_counts(self) -> dict:
        counts = self.df["label"].value_counts().to_dict()

        out = {
            "NORMAL": 0,
            "PNEUMONIA": 0,
        }

        for key, value in counts.items():
            label_name = (
                "PNEUMONIA"
                if int(key) == LABEL_TO_IDX["PNEUMONIA"]
                else "NORMAL"
            )

            out[label_name] = int(value)

        return out


def load_dataset(
    manifest_path: str,
    data_root: str,
    transform,
):
    """
    Prefer Member 1's dataset implementation.

    Fall back to ManifestDataset if needed.
    """

    if _ExternalDataset is not None:
        try:
            return _ExternalDataset(
                manifest_path,
                transform=transform,
            )
        except TypeError:
            pass

    return ManifestDataset(
        manifest_path,
        data_root,
        transform=transform,
    )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def load_configs(
    baseline_cfg_path: Path,
) -> tuple[dict, dict]:

    exp_cfg = load_yaml(
        baseline_cfg_path
    )

    data_cfg_path = (
        REPO_ROOT
        / exp_cfg["data_config"]
    )

    data_cfg = load_yaml(
        data_cfg_path
    )

    return exp_cfg, data_cfg


def get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=REPO_ROOT,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Reproducibility / normalization cache
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def normalization_cache_metadata(
    data_cfg: dict,
    manifests: dict,
) -> dict:

    train_manifest = (
        REPO_ROOT
        / manifests["train"]
    )

    n_train = len(
        pd.read_csv(
            train_manifest,
            usecols=["label"],
        )
    )

    return {
        "train_manifest_sha256": sha256_file(
            train_manifest
        ),
        "split_version": (
            data_cfg
            .get("split", {})
            .get("version", "unknown")
        ),
        "num_train_images": int(
            n_train
        ),
    }


def resolve_stats(
    exp_cfg: dict,
    data_cfg: dict,
    data_root: str,
    manifests: dict,
    logger: logging.Logger,
):
    normalization_cfg = exp_cfg.get(
        "normalization",
        {
            "type": "imagenet"
        },
    )

    stats_cache_path = (
        REPO_ROOT
        / "data"
        / "manifests"
        / "dataset_stats.json"
    )

    def _build_train_stats_dataset():
        stats_tfms = build_stats_transforms(
            data_cfg
        )

        return load_dataset(
            str(
                REPO_ROOT
                / manifests["train"]
            ),
            data_root,
            stats_tfms,
        )

    cache_meta = normalization_cache_metadata(
        data_cfg,
        manifests,
    )

    mean, std = resolve_normalization_stats(
        normalization_cfg,
        stats_cache_path=stats_cache_path,
        train_dataset_builder=_build_train_stats_dataset,
        expected_cache_metadata=cache_meta,
        cache_metadata=cache_meta,
    )

    logger.info(
        "normalization.type=%s mean=%s std=%s",
        normalization_cfg.get(
            "type",
            "imagenet",
        ),
        [
            round(m, 4)
            for m in mean
        ],
        [
            round(s, 4)
            for s in std
        ],
    )

    return mean, std


# ---------------------------------------------------------------------------
# Experiment mode flags
# ---------------------------------------------------------------------------

def effective_experiment_flags(
    mode: str,
    no_augmentation: bool,
    no_class_weight: bool,
) -> tuple[bool, bool]:

    # B0 is a pure memorisation / pipeline sanity test.
    if mode == "overfit":
        return False, False

    return (
        not no_augmentation,
        not no_class_weight,
    )


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    exp_cfg: dict,
    data_cfg: dict,
    mode: str,
    use_augmentation: bool,
    mean: list,
    std: list,
):

    data_root = os.environ[
        "XRAY_DATA_ROOT"
    ]

    # B0 uses a tiny subset and does not benefit from multiprocessing.
    # This also avoids Windows spawn/pickling issues.
    if mode == "overfit":
        num_workers = 0
    else:
        num_workers = int(
            os.environ.get(
                "XRAY_NUM_WORKERS",
                2,
            )
        )

    train_tfms = (
        build_train_transforms(
            data_cfg,
            exp_cfg.get(
                "augmentation",
                {},
            ),
            mean=mean,
            std=std,
        )
        if use_augmentation
        else build_no_augmentation_train_transforms(
            data_cfg,
            mean=mean,
            std=std,
        )
    )

    eval_tfms = build_eval_transforms(
        data_cfg,
        mean=mean,
        std=std,
    )

    manifests = data_cfg[
        "manifests"
    ]

    train_manifest_path = (
        REPO_ROOT
        / manifests["train"]
    )

    validation_manifest_path = (
        REPO_ROOT
        / manifests["validation"]
    )

    train_ds = load_dataset(
        str(
            train_manifest_path
        ),
        data_root,
        train_tfms,
    )

    val_ds = load_dataset(
        str(
            validation_manifest_path
        ),
        data_root,
        eval_tfms,
    )

    # ------------------------------------------------------------------
    # B0: fixed tiny balanced subset
    # ------------------------------------------------------------------

    if mode == "overfit":

        b0_tfms = (
            build_no_augmentation_train_transforms(
                data_cfg,
                mean=mean,
                std=std,
            )
        )

        train_base = load_dataset(
            str(
                train_manifest_path
            ),
            data_root,
            b0_tfms,
        )

        eval_base = load_dataset(
            str(
                train_manifest_path
            ),
            data_root,
            eval_tfms,
        )

        overfit_cfg = exp_cfg.get(
            "overfit",
            {},
        )

        subset_size = int(
            overfit_cfg.get(
                "subset_size",
                32,
            )
        )

        n = min(
            subset_size,
            len(train_base),
        )

        seed = int(
            exp_cfg["experiment"][
                "seed"
            ]
        )

        # Prefer balanced labels.
        frame = getattr(
            train_base,
            "frame",
            None,
        )

        if frame is None:
            frame = getattr(
                train_base,
                "df",
                None,
            )

        if (
            frame is not None
            and "label" in frame.columns
            and set(
                frame["label"].unique()
            ) >= {0, 1}
        ):

            rng = np.random.default_rng(
                seed
            )

            n0 = n // 2
            n1 = n - n0

            normal_indices = (
                frame.index[
                    frame["label"] == 0
                ].to_numpy()
            )

            pneumonia_indices = (
                frame.index[
                    frame["label"] == 1
                ].to_numpy()
            )

            idx0 = rng.choice(
                normal_indices,
                size=n0,
                replace=False,
            )

            idx1 = rng.choice(
                pneumonia_indices,
                size=n1,
                replace=False,
            )

            idx = np.concatenate(
                [
                    idx0,
                    idx1,
                ]
            ).tolist()

            random.Random(
                seed
            ).shuffle(
                idx
            )

        else:

            generator = (
                torch.Generator()
                .manual_seed(seed)
            )

            idx = (
                torch.randperm(
                    len(train_base),
                    generator=generator,
                )[:n]
                .tolist()
            )

        train_ds = Subset(
            train_base,
            idx,
        )

        val_ds = Subset(
            eval_base,
            idx,
        )

    batch_size = int(
        exp_cfg["training"][
            "batch_size"
        ]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
    )


# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------

def compute_pos_weight(
    dataset,
) -> torch.Tensor:

    if hasattr(
        dataset,
        "class_counts",
    ):

        counts = dataset.class_counts()

        n_neg = counts.get(
            "NORMAL",
            0,
        )

        n_pos = counts.get(
            "PNEUMONIA",
            0,
        )

    elif hasattr(
        dataset,
        "frame",
    ):

        n_neg = int(
            (
                dataset.frame[
                    "label"
                ]
                == 0
            ).sum()
        )

        n_pos = int(
            (
                dataset.frame[
                    "label"
                ]
                == 1
            ).sum()
        )

    elif hasattr(
        dataset,
        "df",
    ):

        n_neg = int(
            (
                dataset.df[
                    "label"
                ]
                == 0
            ).sum()
        )

        n_pos = int(
            (
                dataset.df[
                    "label"
                ]
                == 1
            ).sum()
        )

    else:
        raise TypeError(
            f"{type(dataset).__name__} "
            "does not expose labels in a supported format."
        )

    if n_pos == 0:
        return torch.tensor(
            1.0,
            dtype=torch.float32,
        )

    return torch.tensor(
        n_neg / max(
            n_pos,
            1,
        ),
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def unpack_batch(batch):
    """
    Support both:
        dict batches from ChestXrayDataset
        tuple batches from fallback ManifestDataset
    """

    if isinstance(
        batch,
        dict,
    ):
        return (
            batch["image"],
            batch["label"],
        )

    return batch[0], batch[1]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    criterion,
    use_amp: bool,
    collect_predictions: bool = False,
):

    model.eval()

    total_loss = 0.0

    all_probs = []
    all_labels = []

    for batch in loader:

        images, labels = unpack_batch(
            batch
        )

        images = images.to(
            device
        )

        labels = (
            labels
            .to(device)
            .float()
        )

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):

            logits = (
                model(images)
                .squeeze(1)
            )

            loss = criterion(
                logits,
                labels,
            )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        probabilities = torch.sigmoid(
            logits
        )

        all_probs.append(
            probabilities
            .float()
            .cpu()
        )

        all_labels.append(
            labels.cpu()
        )

    probs = (
        torch.cat(
            all_probs
        )
        .numpy()
    )

    labels_np = (
        torch.cat(
            all_labels
        )
        .numpy()
    )

    preds = (
        probs >= 0.5
    ).astype(int)

    tp = int(
        (
            (preds == 1)
            & (labels_np == 1)
        ).sum()
    )

    tn = int(
        (
            (preds == 0)
            & (labels_np == 0)
        ).sum()
    )

    fp = int(
        (
            (preds == 1)
            & (labels_np == 0)
        ).sum()
    )

    fn = int(
        (
            (preds == 0)
            & (labels_np == 1)
        ).sum()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    metrics = {
        "loss": (
            total_loss
            / len(loader.dataset)
        ),
        "accuracy": accuracy_score(
            labels_np,
            preds,
        ),
        "macro_f1": f1_score(
            labels_np,
            preds,
            average="macro",
        ),
        "sensitivity": recall_score(
            labels_np,
            preds,
            pos_label=1,
            zero_division=0,
        ),
        "specificity": specificity,
    }

    if len(
        set(
            labels_np.tolist()
        )
    ) > 1:

        metrics["roc_auc"] = (
            roc_auc_score(
                labels_np,
                probs,
            )
        )

        metrics["pr_auc"] = (
            average_precision_score(
                labels_np,
                probs,
            )
        )

    else:
        metrics["roc_auc"] = float(
            "nan"
        )

        metrics["pr_auc"] = float(
            "nan"
        )

    if collect_predictions:
        return (
            metrics,
            probs,
            labels_np,
        )

    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    exp_cfg: dict,
    data_cfg: dict,
    mode: str,
    use_augmentation: bool,
    use_class_weight: bool,
    run_dir: Path,
    logger: logging.Logger,
    mean: list,
    std: list,
):

    seed = int(
        exp_cfg["experiment"][
            "seed"
        ]
    )

    set_seed(
        seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        bool(
            exp_cfg["training"].get(
                "mixed_precision",
                False,
            )
        )
        and device.type == "cuda"
    )

    train_loader, val_loader = (
        build_dataloaders(
            exp_cfg,
            data_cfg,
            mode,
            use_augmentation,
            mean=mean,
            std=std,
        )
    )

    logger.info(
        "train batches=%d val batches=%d device=%s amp=%s",
        len(train_loader),
        len(val_loader),
        device,
        use_amp,
    )

    # ------------------------------------------------------------------
    # Model config
    # ------------------------------------------------------------------

    model_cfg = dict(
        exp_cfg["model"]
    )

    overfit_cfg = exp_cfg.get(
        "overfit",
        {},
    )

    if mode == "overfit":
        model_cfg["dropout"] = float(
            overfit_cfg.get(
                "dropout",
                0.0,
            )
        )

    model = (
        build_baseline_from_config(
            model_cfg
        )
        .to(device)
    )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    pos_weight = None

    if use_class_weight:

        base_ds = (
            train_loader.dataset.dataset
            if isinstance(
                train_loader.dataset,
                Subset,
            )
            else train_loader.dataset
        )

        pos_weight = (
            compute_pos_weight(
                base_ds
            )
            .to(device)
        )

        logger.info(
            "pos_weight=%.4f",
            pos_weight.item(),
        )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    train_cfg = exp_cfg[
        "training"
    ]

    if mode == "overfit":

        learning_rate = float(
            overfit_cfg.get(
                "learning_rate",
                0.003,
            )
        )

        weight_decay = float(
            overfit_cfg.get(
                "weight_decay",
                0.0,
            )
        )

    else:

        learning_rate = float(
            train_cfg[
                "learning_rate"
            ]
        )

        weight_decay = float(
            train_cfg.get(
                "weight_decay",
                1e-4,
            )
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    logger.info(
        "optimizer=AdamW lr=%s weight_decay=%s",
        learning_rate,
        weight_decay,
    )

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    scheduler = None

    # IMPORTANT:
    # B0 scheduler is OFF because the purpose is deliberate memorisation.
    if mode == "full":

        sched_cfg = exp_cfg.get(
            "scheduler",
            {},
        )

        if (
            sched_cfg.get(
                "name"
            )
            == "reduce_lr_on_plateau"
        ):

            scheduler = (
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=sched_cfg.get(
                        "mode",
                        "max",
                    ),
                    patience=int(
                        sched_cfg.get(
                            "patience",
                            2,
                        )
                    ),
                )
            )

    logger.info(
        "scheduler=%s",
        (
            scheduler.__class__.__name__
            if scheduler is not None
            else "OFF"
        ),
    )

    # ------------------------------------------------------------------
    # AMP
    # ------------------------------------------------------------------

    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp,
    )

    # ------------------------------------------------------------------
    # Monitoring / stopping
    # ------------------------------------------------------------------

    early_cfg = exp_cfg.get(
        "early_stopping",
        {},
    )

    monitor_key = (
        early_cfg
        .get(
            "monitor",
            "validation_macro_f1",
        )
        .replace(
            "validation_",
            "",
        )
    )

    monitor_mode = early_cfg.get(
        "mode",
        "max",
    )

    patience = int(
        early_cfg.get(
            "patience",
            5,
        )
    )

    if mode == "overfit":
        n_epochs = int(
            overfit_cfg.get(
                "epochs",
                200,
            )
        )
    else:
        n_epochs = int(
            train_cfg[
                "epochs"
            ]
        )

    best_score = (
        -float("inf")
        if monitor_mode == "max"
        else float("inf")
    )

    epochs_no_improve = 0
    best_epoch = 0

    history = []

    start_time = time.time()

    peak_vram_gb = 0.0

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    for epoch in range(
        1,
        n_epochs + 1,
    ):

        model.train()

        running_loss = 0.0

        train_probs = []
        train_labels = []

        for batch in train_loader:

            images, labels = unpack_batch(
                batch
            )

            images = images.to(
                device
            )

            labels = (
                labels
                .to(device)
                .float()
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):

                logits = (
                    model(images)
                    .squeeze(1)
                )

                loss = criterion(
                    logits,
                    labels,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            train_probs.append(
                torch.sigmoid(
                    logits.detach()
                )
                .float()
                .cpu()
            )

            train_labels.append(
                labels.detach().cpu()
            )

        if device.type == "cuda":

            peak_vram_gb = max(
                peak_vram_gb,
                torch.cuda.max_memory_allocated()
                / (1024 ** 3),
            )

        train_loss = (
            running_loss
            / len(
                train_loader.dataset
            )
        )

        train_probs_np = (
            torch.cat(
                train_probs
            )
            .numpy()
        )

        train_labels_np = (
            torch.cat(
                train_labels
            )
            .numpy()
        )

        train_preds_np = (
            train_probs_np >= 0.5
        ).astype(int)

        train_accuracy = accuracy_score(
            train_labels_np,
            train_preds_np,
        )

        train_macro_f1 = f1_score(
            train_labels_np,
            train_preds_np,
            average="macro",
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            use_amp,
        )

        if scheduler is not None:
            scheduler.step(
                val_metrics[
                    monitor_key
                ]
            )

        current_lr = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_macro_f1,
            "learning_rate": current_lr,
            **{
                f"val_{key}": value
                for key, value
                in val_metrics.items()
            },
        }

        history.append(
            row
        )

        logger.info(
            (
                "[epoch %03d] "
                "train_loss=%.4f "
                "train_acc=%.4f "
                "train_macro_f1=%.4f "
                "val_loss=%.4f "
                "val_acc=%.4f "
                "val_macro_f1=%.4f "
                "val_sensitivity=%.4f "
                "val_specificity=%.4f "
                "lr=%.6f"
            ),
            epoch,
            train_loss,
            train_accuracy,
            train_macro_f1,
            val_metrics[
                "loss"
            ],
            val_metrics[
                "accuracy"
            ],
            val_metrics[
                "macro_f1"
            ],
            val_metrics[
                "sensitivity"
            ],
            val_metrics[
                "specificity"
            ],
            current_lr,
        )

        torch.save(
            model.state_dict(),
            run_dir
            / "last_checkpoint.pt",
        )

        current = val_metrics[
            monitor_key
        ]

        improved = (
            current > best_score
            if monitor_mode == "max"
            else current < best_score
        )

        # Full training uses model selection + early stopping.
        if mode == "full":

            if improved:

                best_score = current

                epochs_no_improve = 0

                best_epoch = epoch

                torch.save(
                    model.state_dict(),
                    run_dir
                    / "best_checkpoint.pt",
                )

            else:

                epochs_no_improve += 1

                if (
                    epochs_no_improve
                    >= patience
                ):

                    logger.info(
                        (
                            "early stopping at epoch %d "
                            "(no %s improvement "
                            "for %d epochs)"
                        ),
                        epoch,
                        monitor_key,
                        patience,
                    )

                    break

        # B0 deliberately keeps training and saves latest model.
        else:

            best_epoch = epoch

            torch.save(
                model.state_dict(),
                run_dir
                / "best_checkpoint.pt",
            )

            # Optional early success exit for sanity test.
            if (
                val_metrics[
                    "accuracy"
                ]
                >= 0.999
                and val_metrics[
                    "macro_f1"
                ]
                >= 0.999
            ):

                logger.info(
                    (
                        "B0 PASSED at epoch %d: "
                        "tiny fixed subset fully memorised."
                    ),
                    epoch,
                )

                break

    total_time_min = (
        time.time()
        - start_time
    ) / 60.0

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------

    model.load_state_dict(
        torch.load(
            run_dir
            / "best_checkpoint.pt",
            map_location=device,
        )
    )

    final_metrics, probs, labels_np = evaluate(
        model,
        val_loader,
        device,
        criterion,
        use_amp,
        collect_predictions=True,
    )

    predictions_df = pd.DataFrame(
        {
            "probability": probs,
            "label": labels_np,
            "prediction": (
                probs >= 0.5
            ).astype(int),
        }
    )

    predictions_df.to_csv(
        run_dir
        / "predictions_val.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    if history:

        with open(
            run_dir
            / "history.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    history[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                history
            )

    metrics_out = {
        "mode": mode,
        "use_augmentation": use_augmentation,
        "use_class_weight": use_class_weight,
        "epochs_run": len(
            history
        ),
        "best_epoch": best_epoch,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler": (
            scheduler.__class__.__name__
            if scheduler is not None
            else "OFF"
        ),
        "total_train_time_min": round(
            total_time_min,
            2,
        ),
        "peak_vram_gb": round(
            peak_vram_gb,
            3,
        ),
        "final_validation_metrics": final_metrics,
    }

    with open(
        run_dir
        / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics_out,
            file,
            indent=2,
        )

    logger.info(
        (
            "wrote config.yaml, metrics.json, "
            "history.csv, predictions_val.csv, "
            "checkpoints and train.log to %s"
        ),
        run_dir,
    )

    return (
        metrics_out,
        total_time_min,
        peak_vram_gb,
        best_epoch,
    )


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

def append_to_registry(
    run_id: str,
    exp_cfg_path: Path,
    seed: int,
    device: str,
    start_iso: str,
    end_iso: str,
    duration_min: float,
    peak_vram_gb: float,
    best_epoch: int,
    final_metrics: dict,
):

    registry_path = (
        REPO_ROOT
        / "docs"
        / "experiment_registry.csv"
    )

    header = [
        "run_id",
        "owner",
        "status",
        "commit_sha",
        "config",
        "seed",
        "split_version",
        "device",
        "start_time",
        "end_time",
        "duration_minutes",
        "peak_vram_gb",
        "best_epoch",
        "val_macro_f1",
        "val_pneumonia_sensitivity",
        "val_specificity",
        "val_roc_auc",
        "val_pr_auc",
        "test_evaluated",
        "checkpoint_path",
        "notes",
    ]

    row = {
        "run_id": run_id,
        "owner": "member2",
        "status": "complete",
        "commit_sha": get_git_sha(),
        "config": str(
            exp_cfg_path.relative_to(
                REPO_ROOT
            )
        ),
        "seed": seed,
        "split_version": "split_v1",
        "device": device,
        "start_time": start_iso,
        "end_time": end_iso,
        "duration_minutes": round(
            duration_min,
            2,
        ),
        "peak_vram_gb": round(
            peak_vram_gb,
            3,
        ),
        "best_epoch": best_epoch,
        "val_macro_f1": final_metrics.get(
            "macro_f1"
        ),
        "val_pneumonia_sensitivity": final_metrics.get(
            "sensitivity"
        ),
        "val_specificity": final_metrics.get(
            "specificity"
        ),
        "val_roc_auc": final_metrics.get(
            "roc_auc"
        ),
        "val_pr_auc": final_metrics.get(
            "pr_auc"
        ),
        "test_evaluated": False,
        "checkpoint_path": (
            f"outputs/{run_id}/best_checkpoint.pt"
        ),
        "notes": (
            "baseline CNN run (Member 2)"
        ),
    }

    file_exists = (
        registry_path.exists()
    )

    registry_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        registry_path,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=header,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            row
        )


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def build_logger(
    run_dir: Path,
) -> logging.Logger:

    logger = logging.getLogger(
        "train_baseline"
    )

    logger.setLevel(
        logging.INFO
    )

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )

    file_handler = logging.FileHandler(
        run_dir / "train.log",
        encoding="utf-8",
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        file_handler
    )

    stream_handler = logging.StreamHandler()

    stream_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        stream_handler
    )

    return logger


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to experiment config",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "overfit",
            "full",
        ],
        default="full",
    )

    parser.add_argument(
        "--no_augmentation",
        action="store_true",
        help="B2 ablation: disable train-time augmentation",
    )

    parser.add_argument(
        "--no_class_weight",
        action="store_true",
        help="B2 ablation: disable class weighting",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    load_dotenv(
        REPO_ROOT / ".env"
    )

    for required_var in (
        "XRAY_DATA_ROOT",
        "XRAY_OUTPUT_ROOT",
    ):

        if required_var not in os.environ:

            raise SystemExit(
                (
                    f"{required_var} is not set — "
                    "copy .env.example to .env "
                    "and fill it in first."
                )
            )

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    exp_cfg_path = (
        REPO_ROOT
        / args.config
    )

    exp_cfg, data_cfg = load_configs(
        exp_cfg_path
    )

    use_augmentation, use_class_weight = (
        effective_experiment_flags(
            args.mode,
            args.no_augmentation,
            args.no_class_weight,
        )
    )

    seed = int(
        exp_cfg["experiment"][
            "seed"
        ]
    )

    # ------------------------------------------------------------------
    # Run ID
    # ------------------------------------------------------------------

    time_tag = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    aug_tag = (
        "aug"
        if use_augmentation
        else "noaug"
    )

    cw_tag = (
        "cw"
        if use_class_weight
        else "nocw"
    )

    if args.mode == "overfit":

        experiment_tag = (
            "B0_overfit"
        )

    elif (
        args.no_augmentation
        and not args.no_class_weight
    ):

        experiment_tag = (
            "B2_no_augmentation"
        )

    elif (
        args.no_class_weight
        and not args.no_augmentation
    ):

        experiment_tag = (
            "B2_no_class_weight"
        )

    elif (
        args.no_augmentation
        and args.no_class_weight
    ):

        experiment_tag = (
            "B2_no_aug_no_class_weight"
        )

    else:

        experiment_tag = (
            "B1_baseline"
        )

    run_id = (
        f"{experiment_tag}_"
        f"{time_tag}_"
        f"{aug_tag}_"
        f"{cw_tag}_"
        f"s{seed}"
    )

    run_dir = (
        Path(
            os.environ[
                "XRAY_OUTPUT_ROOT"
            ]
        )
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = build_logger(
        run_dir
    )

    logger.info(
        "run_id=%s mode=%s config=%s",
        run_id,
        args.mode,
        args.config,
    )

    if args.mode == "overfit":

        logger.info(
            (
                "B0 sanity check: "
                "forcing augmentation=OFF "
                "and class_weight=OFF"
            )
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    data_root = os.environ[
        "XRAY_DATA_ROOT"
    ]

    mean, std = resolve_stats(
        exp_cfg,
        data_cfg,
        data_root,
        data_cfg[
            "manifests"
        ],
        logger,
    )

    # ------------------------------------------------------------------
    # Write exact resolved config
    # ------------------------------------------------------------------

    overfit_cfg = exp_cfg.get(
        "overfit",
        {},
    )

    resolved_cfg = {
        **exp_cfg,
        "overfit": {
            "subset_size": int(
                overfit_cfg.get(
                    "subset_size",
                    32,
                )
            ),
            "epochs": int(
                overfit_cfg.get(
                    "epochs",
                    200,
                )
            ),
            "learning_rate": float(
                overfit_cfg.get(
                    "learning_rate",
                    0.003,
                )
            ),
            "weight_decay": float(
                overfit_cfg.get(
                    "weight_decay",
                    0.0,
                )
            ),
            "dropout": float(
                overfit_cfg.get(
                    "dropout",
                    0.0,
                )
            ),
        },
        "mode": args.mode,
        "use_augmentation": use_augmentation,
        "use_class_weight": use_class_weight,
        "resolved_normalization": {
            "type": (
                exp_cfg
                .get(
                    "normalization",
                    {},
                )
                .get(
                    "type",
                    "imagenet",
                )
            ),
            "mean": mean,
            "std": std,
        },
    }

    with open(
        run_dir / "config.yaml",
        "w",
        encoding="utf-8",
    ) as file:

        yaml.safe_dump(
            resolved_cfg,
            file,
            sort_keys=False,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    start_iso = datetime.now().isoformat(
        timespec="seconds"
    )

    (
        metrics_out,
        duration_min,
        peak_vram_gb,
        best_epoch,
    ) = train(
        exp_cfg,
        data_cfg,
        mode=args.mode,
        use_augmentation=use_augmentation,
        use_class_weight=use_class_weight,
        run_dir=run_dir,
        logger=logger,
        mean=mean,
        std=std,
    )

    end_iso = datetime.now().isoformat(
        timespec="seconds"
    )

    device_str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    append_to_registry(
        run_id,
        exp_cfg_path,
        seed,
        device_str,
        start_iso,
        end_iso,
        duration_min,
        peak_vram_gb,
        best_epoch,
        metrics_out[
            "final_validation_metrics"
        ],
    )

    logger.info(
        "appended run to docs/experiment_registry.csv"
    )


if __name__ == "__main__":
    main()