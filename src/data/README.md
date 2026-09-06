# Data module contract

## Modules

- `audit_data.py`: inventory, integrity, identity parsing, exact hashes, perceptual hashes, duplicate groups, and leakage reports.
- `split_data.py`: locked-test decision and deterministic group-level manifests.
- `dataset.py`: safe relative-path resolution plus PyTorch Dataset/DataLoader helpers.
- `visualize_data.py`: class counts, deterministic sample grid, and original image dimensions.
- `common.py`: schema, path, hashing, identity, and deterministic helper functions.

## Canonical sample

Each `ChestXrayDataset` item contains:

```text
image
label
patient_id
group_id
image_path
source_split
pneumonia_subtype
```

`image` is a three-channel tensor unless a supplied transform returns another tensor representation. `label` is a scalar `torch.float32` value for the configured one-logit `BCEWithLogitsLoss`: 0 for normal and 1 for pneumonia.

## Leakage grouping

A group joins images sharing at least one of the following:

1. a conservatively parsed patient ID;
2. an exact SHA-256 hash;
3. both pHash and dHash distance at or below the selected threshold.

All members of a connected group receive one deterministic `group_id`. A split is rejected if group, known patient, or exact hash values cross its boundaries.

Unparseable patient filenames keep an empty `patient_id`; this limitation must remain visible and must not be described as fully patient-level coverage.
