# Generated manifests

Run `python -m src.data.split_data` after the full audit. It creates:

```text
train.csv
validation.csv
test.csv
split_manifest.csv
split_statistics.csv
split_summary.json
```

Every CSV row uses this canonical schema:

```text
image_path,patient_id,group_id,label,pneumonia_subtype,split,source_split,sha256
```

Generated image paths are relative to `XRAY_DATA_ROOT`.
