"""Append completed runs without changing the shared CSV schema."""
import csv
from pathlib import Path

COLUMNS = "run_id,owner,status,commit_sha,config,seed,split_version,device,start_time,end_time,duration_minutes,peak_vram_gb,best_epoch,val_macro_f1,val_pneumonia_sensitivity,val_specificity,val_roc_auc,val_pr_auc,test_evaluated,checkpoint_path,notes".split(",")


def append_registry(path, row):
    path = Path(path)
    if not row.get("run_id"):
        raise ValueError("Registry requires run_id")
    exists = path.is_file() and path.stat().st_size > 0
    if exists:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != COLUMNS:
                raise ValueError("Registry header does not match the shared schema")
            if any(existing["run_id"] == row["run_id"] for existing in reader):
                raise ValueError("run_id already exists in registry")
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise ValueError(f"Unknown registry columns: {sorted(unknown)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in COLUMNS})
