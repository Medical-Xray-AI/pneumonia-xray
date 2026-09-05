# Data

This repository does not contain raw or processed medical images.

## Dataset

The project uses the pediatric chest X-ray subset distributed as **Chest X-Ray Images (Pneumonia)**:

- Original dataset: Kermany pediatric chest X-rays
- Distribution: Kaggle, `paultimothymooney/chest-xray-pneumonia`
- Population: children approximately 1-5 years old
- Task: binary classification
- Labels: `NORMAL` and `PNEUMONIA`
- Prediction level: image

Sources:

- https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- https://data.mendeley.com/datasets/rscbjbr9sj/3
- https://doi.org/10.1016/j.cell.2018.02.010

## Expected dataset structure

The directory referenced by `XRAY_DATA_ROOT` should contain:

```text
chest_xray/
|-- train/
|   |-- NORMAL/
|   `-- PNEUMONIA/
|-- val/
|   |-- NORMAL/
|   `-- PNEUMONIA/
`-- test/
    |-- NORMAL/
    `-- PNEUMONIA/
```

Some archive versions contain duplicate nested `chest_xray` directories and macOS metadata. The download script selects the shallowest complete non-`__MACOSX` copy. `XRAY_DATA_ROOT` is written to the directory that directly contains `train`, `val`, and `test`.

## Required local download

Every team member downloads the exact same public Kaggle Version 2 into their own working copy:

```bash
python scripts/download_data.py
```

Authentication is normally unnecessary. Use `--login` only if Kaggle explicitly reports an authorization error. The default destination is `data/raw/kaggle/`, which is ignored by Git. The command validates the official class counts and automatically sets `XRAY_DATA_ROOT` in the local `.env` file. To register an existing download instead:

```bash
python scripts/download_data.py --data-root "/absolute/path/to/chest_xray"
```

Each working copy therefore has its own ignored `.env`:

```env
XRAY_DATA_ROOT=/absolute/path/to/chest_xray
XRAY_OUTPUT_ROOT=outputs
XRAY_NUM_WORKERS=4
```

The `.env`, Kaggle API token, `kaggle.json`, downloaded archive, and raw images must never be committed.

## Ownership boundary

The leader-owned download command acquires the pinned Kaggle release, validates its expected 5,856-image folder inventory, and configures the local path. Member 1 separately owns integrity auditing, duplicate detection, patient/group-aware splitting, manifest generation, data loading, and their tests. Those implementations and results must be submitted from Member 1's branch for review. Team members must not create personal alternative splits after the reviewed split is frozen.

## Split policy

The provided Kaggle `val` folder contains only 16 images and is not suitable as the project's validation set.

The data owner must:

1. Inventory all images and original folder labels.
2. Parse patient identifiers from filenames when reliable.
3. Compute exact file hashes and inspect near-duplicate groups.
4. Combine provided `train` and `val` for development.
5. Create a stratified internal validation split at patient/group level using seed 42.
6. Audit the provided `test` folder against development data.
7. Keep the provided test as locked final test only if no patient or duplicate overlap exists.
8. Rebuild all three splits at patient/group level if test contamination is found.

All images belonging to one patient or duplicate group must remain in exactly one split. The resulting contract is recorded as `split_v1`.

## Class imbalance

Pneumonia is the majority class. Class counts must be reported for every generated split, and imbalance handling must be fitted from training data only.

## Repository policy

Allowed in Git:

- Relative-path manifests
- Aggregate dataset statistics
- Data audit reports
- Small aggregate plots without identifying information

Not allowed in Git:

- X-ray images or dataset archives
- Processed image copies
- Absolute machine-specific paths
- Patient-identifiable information
- Kaggle credentials

Use the dataset only under its source terms and do not redistribute it through this repository.
