# Member 1 corrected data package

This package is a corrected, standalone replacement for the two submitted Member 1 archives. It does not contain the raw X-ray dataset and it does not modify the main repository.

## What this package delivers

- deterministic inventory of the Kaggle `train`, `val`, and `test` folders;
- corrupt-image, SHA-256 exact-duplicate, pHash/dHash near-duplicate, and patient/group leakage audits;
- subtype-aware pneumonia patient keys and namespace-aware normal patient keys;
- portable manifests using paths relative to `XRAY_DATA_ROOT`;
- locked official test when its audit is clean, otherwise a deterministic full group-level rebuild;
- PyTorch Dataset/DataLoader output with all metadata required by the repository contract;
- aggregate figures and automated sanity tests.

## Important correction

`person1_bacteria_...` and `person1_virus_...` must not both become patient ID `1`. Their stable keys are:

```text
pneumonia:bacterial:person1
pneumonia:viral:person1
```

The rejected audit omitted the subtype namespace and consequently reported false patient overlap. Its generated CSV and figures are deliberately not carried into this corrected package.

## Run on the real dataset

The value of `XRAY_DATA_ROOT` may point either to the extracted `chest_xray` directory or to its immediate parent.

PowerShell:

```powershell
$env:XRAY_DATA_ROOT = "D:\datasets\chest_xray"
python -m src.data.audit_data --output-dir audit_out --near-threshold 4
python -m src.data.split_data --audit-manifest audit_out/file_manifest.csv --output-dir data/manifests
python -m src.data.visualize_data --manifest data/manifests/split_manifest.csv --output-dir audit_out/figures
pytest -q
```

Linux/server:

```bash
export XRAY_DATA_ROOT=/data/chest_xray
python -m src.data.audit_data --output-dir audit_out --near-threshold 4
python -m src.data.split_data --audit-manifest audit_out/file_manifest.csv --output-dir data/manifests
python -m src.data.visualize_data --manifest data/manifests/split_manifest.csv --output-dir audit_out/figures
pytest -q
```

Run all commands from this package's top-level directory.

## Required review before split freeze

Inspect these generated files:

```text
audit_out/audit_summary.md
audit_out/audit_summary.json
audit_out/exact_duplicate_report.csv
audit_out/near_duplicate_report.csv
audit_out/patient_overlap_report.csv
audit_out/test_overlap_report.csv
data/manifests/split_summary.json
data/manifests/split_statistics.csv
```

When `provided_test_safe` is `true`, the 624 provided test images remain locked and only the provided train+val pool is divided into train and validation. When it is `false`, the configured 70/15/15 group-level rebuild is used. Seed `42` and version `split_v1` are recorded.

## Boundary with Member 2

This package converts grayscale images to three channels and loads samples. Aspect-ratio-preserving resize, padding, normalization, and training-only augmentation remain Member 2's responsibility and should be passed as the Dataset transform.
