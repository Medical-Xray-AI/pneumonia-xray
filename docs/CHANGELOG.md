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
