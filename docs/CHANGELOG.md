# Member 5: inference, integration and release

- Added checkpoint-based single-image, batch and manifest inference (`src/inference/`) with a validated output schema.
- Inference rebuilds the model from the checkpoint config, never downloads ImageNet weights and reuses the stored train-derived normalization.
- Locked-test inference requires the matching validation `frozen_threshold.json` and never overwrites an existing test prediction file.
- `run_all.py` now provides `check`, `train`, `develop`, `evaluate`, `infer`, `benchmark` and `verify-release`; `develop` reuses finished runs and resumes interrupted ones.
- Added `scripts/verify_release.py` (secret, path and file scan, registry, freeze, inference smoke, clean clone) and `scripts/audit_environment.py` (GPU audit and lock file).
- `requirements.txt` now uses tested compatible ranges; the exact lock is generated on the GPU server.
- Added the model card and release checklist.

# Corrections made to Member 1 submission

- Replaced subtype-blind numeric pneumonia IDs with subtype-aware patient keys.
- Added conservative NORMAL filename parsing while preserving `IM` and `NORMAL2-IM` namespaces.
- Replaced absolute Kaggle paths with paths relative to `XRAY_DATA_ROOT`.
- Implemented the repository's canonical manifest columns and numeric labels.
- Added pHash plus dHash near-duplicate detection and leakage grouping.
- Preserved the provided test when clean; full rebuilding now occurs only on audited overlap or explicit `--force-rebuild`.
- Sorted paths and groups before deterministic seed/hash assignment.
- Made failed leakage checks raise errors instead of printing and continuing.
- Added metadata-rich Dataset/DataLoader output.
- Expanded tests to cover patient namespaces, path portability, locked-test policy, rebuild policy, reproducibility, leakage, and sample shape.
- Updated source documentation to Mendeley Data Version 3.
- Removed environment-specific dependency pins; GPU-compatible torch versions must be selected after the shared server audit.
