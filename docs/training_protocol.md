# Training and resume

Run commands from the repository root with the project dependencies installed.
`src/models.create_model(config)` supports `small_cnn` and `densenet121`.
Training modules live under `src/training`; the shared CLI is `scripts/train.py`.

## Paths and commands

Set environment variables in the current shell. For PowerShell:

```powershell
$env:XRAY_DATA_ROOT = 'C:/datasets/chest_xray'
$env:XRAY_OUTPUT_ROOT = 'C:/xray_outputs'
$env:XRAY_NUM_WORKERS = '0'
python scripts/train.py --config configs/baseline.yaml --run-id baseline_run
python scripts/train.py --config configs/densenet121.yaml --run-id densenet_run
```

For Bash use `export XRAY_DATA_ROOT=/path/to/chest_xray` and
`export XRAY_OUTPUT_ROOT=/path/to/xray_outputs`. Training requires those
variables; creating `.env` alone does not set shell variables for this CLI.
Add `--device cpu` for a CPU run or `--device cuda:0` for a selected GPU.
DenseNet with `pretrained: true` needs cached torchvision weights or access to
download them. Tests use `pretrained: false` and synthetic images.

Committed config files include `baseline.yaml`, `densenet121.yaml`,
`densenet121_frozen.yaml` and `densenet121_lr2e-4.yaml`. The loader resolves
`data_config`, environment variables and optimizer/scheduler settings into one
format, saved as `config_resolved.yaml` for each run.

## Shared data contract

Both models use `ChestXrayDataset` and canonical numeric labels (0 normal,
1 pneumonia), relative `image_path` and `train`/`validation` manifests.
The committed `split_v1` is reused unchanged. Training never loads test images.
Class weights and dataset normalization statistics are computed from train only.
DenseNet uses ImageNet normalization; baseline retains its configured choice.
Both use the shared aspect-preserving resize/padding and augmentation functions.
Validation has no random augmentation.

## Artifacts and registry

Each new run gets a unique directory below `XRAY_OUTPUT_ROOT/<run_id>/`:

- `checkpoints/last.pt` and `checkpoints/best.pt`
- `config_resolved.yaml`, `runtime.json`, `train.log`, `history.csv`
- `predictions_val.csv` and `metrics.json`

The best epoch is selected by validation macro F1 at threshold 0.5. Predictions
are regenerated from that checkpoint and retain image paths, labels, split,
model, run ID, checkpoint epoch and dataset metadata in matching row order.
Threshold tuning happens later in the evaluation CLI.

`--update-registry` appends a completed run to the existing 21-column
`docs/experiment_registry.csv`. Existing rows are preserved and duplicate run
IDs are rejected. Checkpoint references are relative to `XRAY_OUTPUT_ROOT`.
Checkpoint files, logs and image outputs stay outside Git.

## Interrupted resume

```powershell
python scripts/train.py --config configs/baseline.yaml --resume "$env:XRAY_OUTPUT_ROOT/baseline_run/checkpoints/last.pt"
```

Resume uses the original run directory and requires its `best.pt` alongside
`last.pt`. Keep the original config, manifests and output root. Config and
manifest hashes are checked before restoring the model. Checkpoints include
optimizer, scheduler, AMP scaler, early stopping, history, random number state
and both DataLoader generators. An unfinished epoch restarts from the last
completed epoch. Already exhausted early stopping is respected.

CPU tests compare an interrupted and resumed run with a continuous run using
shuffle, augmentation and dropout; their final model tensors are identical.
CUDA/AMP behavior still needs verification on the team's GPU environment.

## Verification and remaining release work

```bash
python -m pytest tests -q
```

Tests cover both actual model wrappers, synthetic training to validation export
to evaluation, small baseline overfit, checkpoint resume, split/label guards
and Grad-CAM for frozen backbones. These do not replace full dataset training.
`python run_all.py develop` runs both configs and the validation freeze in
one command, reusing finished runs and resuming interrupted ones. Real
experiment tables, the single locked-test run and the final report remain
subsequent work.
