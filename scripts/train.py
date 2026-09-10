#!/usr/bin/env python3
"""Reproducible DenseNet121 training CLI for Member 3.

This command trains on ``train`` and selects the best epoch on ``val`` only.
It never reads the locked ``test`` split. After training, it reloads the best
checkpoint and exports ``predictions_val.csv`` for Member 4's threshold search.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

# Make project root importable when running `python scripts/train.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import create_model
from src.training.checkpointing import CheckpointManager, load_checkpoint
from src.training.config import load_config, save_resolved_config
from src.training.data import ManifestDataset, build_transforms, compute_pos_weight_from_train, validate_development_splits
from src.training.engine import EarlyStopping, step_scheduler, train_epoch, validate
from src.training.reproducibility import (
    collect_runtime_metadata,
    make_generator,
    resolve_device,
    seed_everything,
    seed_worker,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DenseNet121 on leakage-aware split_v1.")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--resume", default=None, help="Path to last.pt to resume")
    p.add_argument("--run-id", default=None, help="Override run id")
    p.add_argument("--device", default=None, help="Override config device: auto/cpu/cuda[:N]")
    p.add_argument(
        "--update-registry",
        action="store_true",
        help="Append the completed run to docs/experiment_registry.csv",
    )
    return p.parse_args()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def make_run_id(config: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        return override
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_member3_densenet121_s{config['seed']}"


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("member3_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]):
    ocfg = cfg["training"]["optimizer"]
    name = str(ocfg.get("name", "adamw")).lower()
    lr = float(ocfg.get("lr", 1e-4))
    wd = float(ocfg.get("weight_decay", 1e-4))
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(ocfg.get("momentum", 0.9)),
            weight_decay=wd,
            nesterov=bool(ocfg.get("nesterov", True)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, cfg: Dict[str, Any]):
    scfg = cfg["training"]["scheduler"] or {}
    name = str(scfg.get("name", "reduce_on_plateau")).lower()
    if name in {"none", "off"}:
        return None
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scfg.get("factor", 0.5)),
            patience=int(scfg.get("patience", 2)),
            min_lr=float(scfg.get("min_lr", 1e-7)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scfg.get("t_max", cfg["training"]["epochs"])),
            eta_min=float(scfg.get("min_lr", 1e-7)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def make_scaler(enabled: bool):
    # New API first; fall back for torch 2.3 used by the teammate requirements.
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_loaders(cfg: Dict[str, Any], logger: logging.Logger):
    train_t, eval_t, transform_source = build_transforms(cfg)
    logger.info("preprocessing_source=%s", transform_source)
    train_ds = ManifestDataset(cfg["data"]["manifests"]["train"], "train", transform=train_t)
    val_ds = ManifestDataset(cfg["data"]["manifests"]["validation"], "validation", transform=eval_t)
    validate_development_splits(train_ds, val_ds)

    loader_cfg = cfg["data"].get("loader", {})
    workers = int(loader_cfg.get("num_workers", 4))
    batch_size = int(cfg["training"]["batch_size"])
    common = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if workers > 0 else None,
        persistent_workers=bool(workers > 0 and loader_cfg.get("persistent_workers", True)),
    )
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=make_generator(int(cfg["seed"])),
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_ds, val_ds, train_loader, val_loader


def build_criterion(cfg: Dict[str, Any], train_ds: ManifestDataset, device: torch.device):
    lcfg = cfg["training"]["loss"]
    if str(lcfg.get("name", "bce_with_logits")).lower() != "bce_with_logits":
        raise ValueError("Member-3 training engine currently supports BCEWithLogitsLoss only.")
    pos_weight_cfg = lcfg.get("pos_weight", "auto")
    if str(pos_weight_cfg).lower() == "auto":
        pos_weight = compute_pos_weight_from_train(train_ds)
    else:
        pos_weight = float(pos_weight_cfg)
    tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=tensor), pos_weight


def current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def append_registry(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "owner",
        "model",
        "git_sha",
        "config",
        "config_sha256",
        "seed",
        "split_version",
        "device",
        "duration_seconds",
        "peak_vram_bytes",
        "best_epoch",
        "val_loss",
        "val_macro_f1_at_0_5",
        "val_sensitivity_at_0_5",
        "val_specificity_at_0_5",
        "status",
    ]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["training"].get("deterministic", True)))
    device = resolve_device(args.device or cfg["training"].get("device", "auto"))

    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        inferred_run_dir = resume_path.parent.parent
        inferred_run_id = inferred_run_dir.name
        run_id = args.run_id or inferred_run_id
        if args.run_id and args.run_id != inferred_run_id:
            raise ValueError(
                f"--run-id={args.run_id!r} does not match resume run directory "
                f"{inferred_run_id!r}. Resume the original run_id."
            )
        run_dir = inferred_run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_id = make_run_id(cfg, args.run_id)
        configured_output = cfg.get("output", {}).get("root")
        # os.path.expandvars intentionally leaves an unset ${VAR} unchanged.
        # Fall back to a local outputs/ directory instead of literally creating
        # a directory named '${XRAY_OUTPUT_ROOT}'.
        if configured_output and "${" not in str(configured_output):
            output_root = Path(configured_output)
        else:
            output_root = Path(os.getenv("XRAY_OUTPUT_ROOT") or "outputs")
        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root
        run_dir = output_root / "member3_densenet" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir)

    sha = git_sha()
    runtime = collect_runtime_metadata(device)
    logger.info("run_id=%s", run_id)
    logger.info("git_sha=%s", sha)
    logger.info("split_version=%s seed=%s device=%s", cfg["split_version"], cfg["seed"], device)
    logger.info("runtime=%s", runtime)
    if not args.resume or not (run_dir / "config.yaml").exists():
        save_resolved_config(cfg, run_dir / "config.yaml")
    write_json(run_dir / "runtime.json", runtime)

    train_ds, val_ds, train_loader, val_loader = build_loaders(cfg, logger)
    logger.info("train_counts=%s val_counts=%s", train_ds.class_counts(), val_ds.class_counts())

    model = create_model(cfg).to(device)
    logger.info(
        "model=densenet121 freeze_strategy=%s trainable_params=%d total_params=%d",
        cfg["model"].get("freeze_strategy", "last_block"),
        getattr(model, "trainable_parameter_count", lambda: -1)(),
        getattr(model, "total_parameter_count", lambda: -1)(),
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion, pos_weight = build_criterion(cfg, train_ds, device)
    logger.info("bce_pos_weight_from_train=%0.8f", pos_weight)

    amp_enabled = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    scaler = make_scaler(amp_enabled)
    ecfg = cfg["training"]["early_stopping"]
    early = EarlyStopping(
        patience=int(ecfg.get("patience", 4)),
        min_delta=float(ecfg.get("min_delta", 0.0)),
        mode="max",
    )
    manager = CheckpointManager(checkpoints_dir, monitor="macro_f1", mode="max")

    start_epoch = 1
    history = []
    if args.resume:
        ckpt = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early,
            map_location=device,
            restore_rng=True,
        )
        previous_cfg = ckpt.get("config") or {}
        previous_hash = previous_cfg.get("_meta", {}).get("config_sha256")
        current_hash = cfg.get("_meta", {}).get("config_sha256")
        if previous_hash and current_hash and previous_hash != current_hash:
            raise ValueError(
                "Resume config does not match the checkpoint config hash. Use the exact "
                "same committed/resolved config for an interrupted run rather than "
                "silently changing hyperparameters."
            )
        start_epoch = int(ckpt["epoch"]) + 1
        history = list(ckpt.get("history") or [])
        manager.restore_best_metric(ckpt.get("best_metric"))
        logger.info("resumed_from=%s next_epoch=%d", resume_path, start_epoch)

    max_epochs = int(cfg["training"]["epochs"])
    threshold = float(cfg["training"].get("training_threshold", 0.5))
    grad_accum = int(cfg["training"].get("grad_accum_steps", 1))
    max_grad_norm = cfg["training"].get("max_grad_norm")
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(start_epoch, max_epochs + 1):
        epoch_started = time.perf_counter()
        train_res = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            amp=amp_enabled,
            grad_accum_steps=grad_accum,
            max_grad_norm=None if max_grad_norm is None else float(max_grad_norm),
            threshold=threshold,
        )
        val_res = validate(
            model,
            val_loader,
            criterion,
            device,
            amp=amp_enabled,
            threshold=threshold,
            collect_predictions=False,
        )
        step_scheduler(scheduler, val_res.metrics["macro_f1"])
        should_stop = early.step(val_res.metrics["macro_f1"])
        epoch_seconds = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch,
            "lr": current_lr(optimizer),
            "train_loss": train_res.loss,
            "train_macro_f1": train_res.metrics["macro_f1"],
            "val_loss": val_res.loss,
            "val_macro_f1": val_res.metrics["macro_f1"],
            "val_sensitivity": val_res.metrics["sensitivity"],
            "val_specificity": val_res.metrics["specificity"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        improved = manager.save(
            epoch=epoch,
            metric_value=val_res.metrics["macro_f1"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early,
            config=cfg,
            run_id=run_id,
            history=history,
            extra={"git_sha": sha, "pos_weight": pos_weight, "runtime": runtime},
        )
        logger.info(
            "epoch=%d/%d train_loss=%.5f train_f1=%.4f val_loss=%.5f val_f1=%.4f "
            "sens=%.4f spec=%.4f lr=%.3g best=%s time=%.1fs",
            epoch,
            max_epochs,
            train_res.loss,
            train_res.metrics["macro_f1"],
            val_res.loss,
            val_res.metrics["macro_f1"],
            val_res.metrics["sensitivity"],
            val_res.metrics["specificity"],
            current_lr(optimizer),
            improved,
            epoch_seconds,
        )

        if should_stop:
            logger.info("early_stopping_triggered epoch=%d", epoch)
            break

    duration = time.perf_counter() - started
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0

    # Reload best epoch before exporting validation probabilities for Member 4.
    best_ckpt = load_checkpoint(
        manager.best_path,
        model=model,
        map_location=device,
        restore_rng=False,
    )
    best_epoch = int(best_ckpt["epoch"])
    final_val = validate(
        model,
        val_loader,
        criterion,
        device,
        amp=amp_enabled,
        threshold=threshold,
        collect_predictions=True,
    )
    predictions = pd.DataFrame(final_val.predictions or [])
    predictions.insert(0, "run_id", run_id)
    predictions.insert(1, "model", cfg["model"]["name"])
    predictions.insert(2, "checkpoint_epoch", best_epoch)
    predictions.to_csv(run_dir / "predictions_val.csv", index=False)

    metrics = {
        "run_id": run_id,
        "git_sha": sha,
        "config_sha256": cfg.get("_meta", {}).get("config_sha256"),
        "seed": int(cfg["seed"]),
        "split_version": cfg["split_version"],
        "model": cfg["model"]["name"],
        "freeze_strategy": cfg["model"].get("freeze_strategy", "last_block"),
        "best_epoch": best_epoch,
        "selection_metric": "validation_macro_f1_at_threshold_0.5",
        "validation_threshold_for_training_monitor_only": threshold,
        "val_loss": final_val.loss,
        "val_metrics_at_0_5": final_val.metrics,
        "pos_weight_from_train": pos_weight,
        "duration_seconds": duration,
        "peak_vram_bytes": peak_vram,
        "best_checkpoint": str(manager.best_path),
        "test_evaluated": False,
    }
    write_json(run_dir / "metrics.json", metrics)

    logger.info("best_epoch=%d best_val_macro_f1=%.4f", best_epoch, final_val.metrics["macro_f1"])
    logger.info("duration_seconds=%.1f peak_vram_bytes=%d", duration, peak_vram)
    logger.info("validation_predictions=%s", run_dir / "predictions_val.csv")
    logger.info("LOCKED TEST WAS NOT EVALUATED.")

    if args.update_registry:
        registry_path = PROJECT_ROOT / "docs" / "experiment_registry.csv"
        append_registry(
            registry_path,
            {
                "run_id": run_id,
                "owner": "member3",
                "model": cfg["model"]["name"],
                "git_sha": sha,
                "config": cfg.get("_meta", {}).get("source_config"),
                "config_sha256": cfg.get("_meta", {}).get("config_sha256"),
                "seed": cfg["seed"],
                "split_version": cfg["split_version"],
                "device": str(device),
                "duration_seconds": f"{duration:.3f}",
                "peak_vram_bytes": peak_vram,
                "best_epoch": best_epoch,
                "val_loss": f"{final_val.loss:.8f}",
                "val_macro_f1_at_0_5": f"{final_val.metrics['macro_f1']:.8f}",
                "val_sensitivity_at_0_5": f"{final_val.metrics['sensitivity']:.8f}",
                "val_specificity_at_0_5": f"{final_val.metrics['specificity']:.8f}",
                "status": "complete",
            },
        )
        logger.info("registry_appended=%s", registry_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
