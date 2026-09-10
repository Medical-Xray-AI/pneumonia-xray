#!/usr/bin/env python3
"""Reproducible binary-classifier training CLI.

This command trains on ``train`` and selects the best epoch on ``val`` only.
It never reads the locked ``test`` split. After training, it reloads the best
checkpoint and exports ``predictions_val.csv`` for Member 4's threshold search.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import re
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
from src.training.config import load_config, save_resolved_config, config_sha256
from src.training.registry import append_registry
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
    p = argparse.ArgumentParser(description="Train a configured binary classifier on split_v1.")
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
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", override):
            raise ValueError("run_id must be a filename-safe identifier")
        return override
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{config['model']['name']}_s{config['seed']}"


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
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
        persistent_workers=False,
    )
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=make_generator(int(cfg["seed"])),
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, generator=make_generator(int(cfg["seed"]) + 1), **common)
    return train_ds, val_ds, train_loader, val_loader


def build_criterion(cfg: Dict[str, Any], train_ds: ManifestDataset, device: torch.device):
    lcfg = cfg["training"]["loss"]
    if str(lcfg.get("name", "bce_with_logits")).lower() != "bce_with_logits":
        raise ValueError("Training supports BCEWithLogitsLoss only.")
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


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"], deterministic=cfg["training"].get("deterministic", True))
    device = resolve_device(args.device or cfg["training"].get("device", "auto"))
    output_value = os.getenv("XRAY_OUTPUT_ROOT")
    if not output_value:
        raise ValueError("Set XRAY_OUTPUT_ROOT before training")
    output_root = Path(output_value).expanduser().resolve()
    resume_path = Path(args.resume).resolve() if args.resume else None
    if resume_path:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        run_dir = resume_path.parent.parent
        if run_dir.parent != output_root or resume_path.name != "last.pt":
            raise ValueError("Resume must use XRAY_OUTPUT_ROOT/<run_id>/checkpoints/last.pt")
        run_id = make_run_id(cfg, args.run_id or run_dir.name)
        if run_id != run_dir.name:
            raise ValueError("Resume run_id does not match the checkpoint directory")
    else:
        run_id = make_run_id(cfg, args.run_id)
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_logger(run_dir)
    try:
        return run_training(cfg, args, device, run_id, run_dir, output_root, resume_path, logger)
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def run_training(cfg, args, device, run_id, run_dir, output_root, resume_path, logger):
    train_ds, val_ds, train_loader, val_loader = build_loaders(cfg, logger)
    # Hash the actual manifests, not just their paths, so resume detects split drift.
    cfg["manifest_sha256"] = {name: hashlib.sha256(Path(cfg["data"]["manifests"][name]).read_bytes()).hexdigest()
                              for name in ("train", "validation")}
    cfg["_meta"]["config_sha256"] = config_sha256(cfg)
    model_cfg = copy.deepcopy(cfg)
    if resume_path:
        model_cfg["model"]["pretrained"] = False
    model = create_model(model_cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion, pos_weight = build_criterion(cfg, train_ds, device)
    amp = bool(cfg["training"].get("amp", False) and device.type == "cuda")
    scaler = make_scaler(amp)
    ecfg = cfg["training"]["early_stopping"]
    early = EarlyStopping(patience=int(ecfg.get("patience", 4)), min_delta=float(ecfg.get("min_delta", 0)))
    manager = CheckpointManager(run_dir / "checkpoints")
    history, start_epoch = [], 1
    sha = git_sha()
    started_at = datetime.now(timezone.utc).isoformat()
    if resume_path:
        ckpt = load_checkpoint(resume_path, model=model, optimizer=optimizer, scheduler=scheduler,
                               scaler=scaler, early_stopping=early, map_location=device,
                               expected_config_hash=cfg["_meta"]["config_sha256"])
        if ckpt.get("run_id") != run_id:
            raise ValueError("Checkpoint run_id mismatch")
        extra = ckpt["extra"]
        train_loader.generator.set_state(extra["train_generator_state"].cpu())
        val_loader.generator.set_state(extra["validation_generator_state"].cpu())
        history = list(ckpt["history"])
        start_epoch = int(ckpt["epoch"]) + 1
        manager.restore_best_metric(ckpt["best_metric"])
        sha = extra["git_sha"]
        started_at = extra["started_at"]
        if not manager.best_path.is_file():
            raise FileNotFoundError("Resume requires the original best.pt alongside last.pt")
    save_resolved_config(cfg, run_dir / "config.yaml")
    runtime = collect_runtime_metadata(device)
    write_json(run_dir / "runtime.json", runtime)
    logger.info("run_id=%s model=%s split=%s device=%s", run_id, cfg["model"]["name"], cfg["split_version"], device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, int(cfg["training"]["epochs"]) + 1):
        if early.bad_epochs >= early.patience:
            break
        started = time.perf_counter()
        train_res = train_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler,
                                amp=amp, grad_accum_steps=cfg["training"]["grad_accum_steps"],
                                max_grad_norm=cfg["training"].get("max_grad_norm"))
        val_res = validate(model, val_loader, criterion, device, amp=amp)
        step_scheduler(scheduler, val_res.metrics["macro_f1"])
        should_stop = early.step(val_res.metrics["macro_f1"])
        history.append({"epoch": epoch, "lr": current_lr(optimizer), "train_loss": train_res.loss,
                        "train_macro_f1": train_res.metrics["macro_f1"], "val_loss": val_res.loss,
                        "val_macro_f1": val_res.metrics["macro_f1"],
                        "val_pneumonia_sensitivity": val_res.metrics["sensitivity"],
                        "val_specificity": val_res.metrics["specificity"],
                        "epoch_seconds": time.perf_counter() - started})
        manager.save(epoch=epoch, metric_value=val_res.metrics["macro_f1"], model=model,
                     optimizer=optimizer, scheduler=scheduler, scaler=scaler, early_stopping=early,
                     config=cfg, run_id=run_id, history=history,
                     extra={"git_sha": sha, "started_at": started_at, "pos_weight": pos_weight,
                            "train_generator_state": train_loader.generator.get_state(),
                            "validation_generator_state": val_loader.generator.get_state()})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        logger.info("epoch=%d train_loss=%.5f val_loss=%.5f val_macro_f1=%.4f", epoch, train_res.loss, val_res.loss, val_res.metrics["macro_f1"])
        if should_stop:
            break
    best = load_checkpoint(manager.best_path, model=model, map_location=device, restore_rng=False,
                           expected_config_hash=cfg["_meta"]["config_sha256"])
    final = validate(model, val_loader, criterion, device, amp=amp, collect_predictions=True)
    predictions = pd.DataFrame(final.predictions)
    predictions.insert(0, "run_id", run_id)
    predictions.insert(1, "model", cfg["model"]["name"])
    predictions.insert(2, "checkpoint_epoch", best["epoch"])
    predictions.to_csv(run_dir / "predictions_val.csv", index=False)
    duration = sum(row["epoch_seconds"] for row in history)
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    reference = manager.best_path.relative_to(output_root).as_posix()
    metrics = {"run_id": run_id, "model": cfg["model"]["name"], "git_sha": sha,
               "config_sha256": cfg["_meta"]["config_sha256"], "seed": cfg["seed"],
               "split_version": cfg["split_version"], "best_epoch": best["epoch"],
               "selection_metric": "validation_macro_f1_at_threshold_0.5", "threshold": 0.5,
               "val_loss": final.loss, "val_metrics_at_0_5": final.metrics,
               "pos_weight_from_train": pos_weight, "duration_seconds": duration,
               "peak_vram_bytes": peak_vram, "best_checkpoint": reference, "test_evaluated": False}
    write_json(run_dir / "metrics.json", metrics)
    if args.update_registry:
        source_config = Path(cfg["_meta"]["source_config"])
        try:
            config_reference = source_config.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            config_reference = (run_dir / "config.yaml").relative_to(output_root).as_posix()
        append_registry(PROJECT_ROOT / "docs/experiment_registry.csv", {
            "run_id": run_id, "owner": "", "status": "complete", "commit_sha": sha,
            "config": config_reference, "seed": cfg["seed"], "split_version": cfg["split_version"],
            "device": str(device), "start_time": started_at, "end_time": datetime.now(timezone.utc).isoformat(),
            "duration_minutes": duration / 60, "peak_vram_gb": peak_vram / 1024**3,
            "best_epoch": best["epoch"], "val_macro_f1": final.metrics["macro_f1"],
            "val_pneumonia_sensitivity": final.metrics["sensitivity"], "val_specificity": final.metrics["specificity"],
            "val_roc_auc": final.metrics["roc_auc"], "val_pr_auc": final.metrics["pr_auc"],
            "test_evaluated": False, "checkpoint_path": reference,
            "notes": f"model={cfg['model']['name']}; config_sha256={cfg['_meta']['config_sha256']}"})
    logger.info("Validation predictions saved. Locked test was not evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
