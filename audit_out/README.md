# Generated audit evidence

This directory intentionally contains no inherited CSV or figure from the rejected audit. The previous report used a subtype-blind pneumonia patient key and therefore reported false train/test overlap.

Generate fresh evidence from the real extracted dataset:

```bash
python -m src.data.audit_data --output-dir audit_out --near-threshold 4
python -m src.data.split_data --audit-manifest audit_out/file_manifest.csv --output-dir data/manifests
python -m src.data.visualize_data --manifest data/manifests/split_manifest.csv --output-dir audit_out/figures
​```

Do not call this audit complete until it has run against all 5,856 real images and the resulting audit_summary.json, leakage reports, manifests, and figures have been reviewed.
