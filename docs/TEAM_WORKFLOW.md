# Team workflow

## Shared objective

Build and compare a custom CNN baseline and a pretrained DenseNet121 for image-level normal-versus-pneumonia classification using the Kermany pediatric chest X-ray dataset distributed on Kaggle.

The shared data contract is `configs/data.yaml`. Changes to labels, split policy, image handling, or primary metrics require a documented team decision and pull request.

## Responsibilities

### Member 1: data and audit

- Verify the leader-provided Kaggle download and implement the reviewed audit/preparation pipeline.
- Inventory the provided train, validation, and test folders.
- Audit class balance, corrupt files, patient identifiers, exact hashes, and near-duplicates.
- Create deterministic patient/group-level manifests.
- Verify that the locked test policy is safe or trigger a complete group-level re-split.
- Implement the dataset and DataLoader contract.

Deliverables: manifests, audit summary, leakage checks, split statistics, and data-loading code.

### Member 2: preprocessing and baseline

- Implement aspect-ratio-preserving resize and padding.
- Implement conservative training-only augmentation.
- Implement the custom CNN baseline.
- Run a small-subset overfit test and baseline experiments.
- Document class-imbalance handling fitted from training data only.

Deliverables: preprocessing, baseline model, training evidence, config, and validation results.

### Member 3: pretrained model and training

- Implement pretrained DenseNet121 with a binary logit head.
- Implement the training loop, AMP, checkpointing, and resume support.
- Record training time and peak GPU memory.

Deliverables: reusable trainer, best validation checkpoint reference, and run logs.

### Member 4: evaluation and interpretation

- Implement image-level metrics and prediction export.
- Select the decision threshold using validation data only.
- Produce a confusion matrix, ROC/PR results, ablations, error analysis, and Grad-CAM examples.
- Report pneumonia sensitivity and specificity explicitly.

Deliverables: evaluation tables, safe aggregate figures, and analysis notes.

### Member 5: integration and reproducibility

- Maintain repository contracts and team instructions.
- Organize shared GPU access and dataset location without exposing credentials.
- Implement inference and the final `run_all.py` entry point.
- Finalize pinned requirements after the GPU audit.
- Verify a clean-clone run, latency, model size, demo, report integration, and release package.

Deliverables: reproducible end-to-end command, inference output, final release, and submission QA.

## Review chain

1. Member 2 reviews Member 1.
2. Member 3 reviews Member 2.
3. Member 4 reviews Member 3.
4. Member 5 reviews Member 4.
5. Member 1 reviews Member 5.

At least one approval is required before merge.

## Integration gates

### Gate 1: repository and data contract

- Repository reflects the pediatric pneumonia task.
- Credentials and raw data are ignored.
- Dataset location is environment-based.
- Every member has run `python scripts/download_data.py` on their own computer.
- Roles and PR rules are understood.

### Gate 2: split freeze

- Dataset inventory and corrupt-file audit are complete.
- Patient/group and exact-hash overlap between splits is zero.
- Near-duplicate handling is documented.
- Manifest schema is valid.
- Seed 42 and `split_v1` are recorded.

No member may create a personal alternative split after this gate.

Member 1 contributes the audit/split implementation and creates the audited team split once. Other members download the pinned raw dataset locally and use the reviewed `split_v1` manifests; they do not generate personal splits.

### Gate 3: smoke tests

- One batch loads correctly.
- Image and label shapes match the contract.
- Baseline and DenseNet forward passes work.
- A small subset can be intentionally overfit.

### Gate 4: full experiments

- Runs use registered configs and commit SHAs.
- Checkpoints and logs are stored outside Git.
- Only validation results are used for model and threshold selection.

### Gate 5: final evaluation and release

- Model and threshold are frozen.
- Locked test is evaluated once for the final comparison.
- A clean clone can reproduce inference.
- Report, presentation, and repository contain no secrets or patient data.

## Daily status format

Each member reports:

1. Completed result and PR/commit link
2. Evidence produced
3. Next measurable result
4. GPU time needed
5. Blocker, if any

Avoid percentage-only updates without evidence.

## Artifact contract

Each experiment directory should contain:

```text
<run_id>/
|-- config.yaml
|-- metrics.json
|-- history.csv
|-- train.log
`-- predictions_val.csv
```

Large run directories remain outside Git. Register every meaningful run in `docs/experiment_registry.csv`.
