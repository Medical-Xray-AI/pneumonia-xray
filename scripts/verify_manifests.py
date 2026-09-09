"""
verify_manifests.py

Run this INSIDE the real repo (pneumonia-xray/) to check whether Member 1's
manifests + dataset.py are actually ready for Member 2's training code to
consume -- no synthetic data, no guessing, just concrete pass/fail checks
against data/manifests/README.md's documented contract.

USAGE
-----
    cd pneumonia-xray/
    python scripts/verify_manifests.py

    # if you already have XRAY_DATA_ROOT set (e.g. via .env / download_data.py),
    # this also validates that every image_path actually resolves to a real file:
    python scripts/verify_manifests.py --check-files

    # and to actually instantiate the real ChestXrayDataset and load one
    # sample per split (the strongest possible check -- exercises the exact
    # code path train_baseline.py will use):
    python scripts/verify_manifests.py --check-files --check-dataset

Prints a clear READY / NOT READY summary at the end with concrete reasons.
Exits with code 0 if ready, 1 if not -- safe to use in a pre-flight check
before kicking off a real training run.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_COLUMNS = [
    "image_path", "patient_id", "group_id", "label",
    "pneumonia_subtype", "split", "source_split", "sha256",
]
ALLOWED_SUBTYPES = {"normal", "bacterial", "viral", "unknown"}
SPLIT_FILES = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}

problems = []
warnings = []


def fail(msg: str):
    problems.append(msg)
    print(f"  [FAIL] {msg}")


def warn(msg: str):
    warnings.append(msg)
    print(f"  [WARN] {msg}")


def ok(msg: str):
    print(f"  [ok]   {msg}")


def load_csv(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_files_exist(manifests_dir: Path) -> dict:
    print("\n== 1. Manifest files exist ==")
    found = {}
    for split, fname in SPLIT_FILES.items():
        p = manifests_dir / fname
        if p.is_file():
            ok(f"{fname} found ({p})")
            found[split] = p
        else:
            fail(f"{fname} MISSING at {p}")
    return found


def check_schema(split: str, rows: list):
    print(f"\n== 2. Schema check: {split}.csv ==")
    if not rows:
        fail(f"{split}.csv has 0 rows")
        return
    actual_cols = set(rows[0].keys())
    missing = set(CANONICAL_COLUMNS) - actual_cols
    extra = actual_cols - set(CANONICAL_COLUMNS)
    if missing:
        fail(f"{split}.csv missing columns: {sorted(missing)}")
    else:
        ok(f"{split}.csv has all required columns")
    if extra:
        warn(f"{split}.csv has extra columns (not necessarily a problem): {sorted(extra)}")

    bad_labels = {r["label"] for r in rows} - {"0", "1"}
    if bad_labels:
        fail(f"{split}.csv has non-0/1 label values: {bad_labels}")
    else:
        ok(f"{split}.csv labels are all 0/1")

    bad_split_vals = {r["split"] for r in rows} - {split}
    if bad_split_vals:
        fail(f"{split}.csv contains rows whose 'split' column isn't '{split}': {bad_split_vals}")
    else:
        ok(f"{split}.csv 'split' column is consistent")

    bad_subtypes = {r.get("pneumonia_subtype", "") for r in rows} - ALLOWED_SUBTYPES
    if bad_subtypes:
        fail(f"{split}.csv has unexpected pneumonia_subtype values: {bad_subtypes}")
    else:
        ok(f"{split}.csv pneumonia_subtype values are all valid")

    print(f"  -> {len(rows)} rows total, "
          f"{sum(1 for r in rows if r['label'] == '0')} normal / "
          f"{sum(1 for r in rows if r['label'] == '1')} pneumonia")


def check_leakage(all_rows: dict):
    print("\n== 3. Cross-split leakage checks ==")

    def collect(field):
        by_split = {}
        for split, rows in all_rows.items():
            vals = {r[field] for r in rows if r.get(field)}
            by_split[split] = vals
        return by_split

    for field in ("group_id", "sha256"):
        by_split = collect(field)
        splits = list(by_split.keys())
        leaked = False
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                overlap = by_split[splits[i]] & by_split[splits[j]]
                if overlap:
                    fail(f"{field} overlaps between {splits[i]} and {splits[j]}: "
                         f"{len(overlap)} shared values, e.g. {list(overlap)[:3]}")
                    leaked = True
        if not leaked:
            ok(f"no {field} overlap across splits")

    by_patient = collect("patient_id")
    splits = list(by_patient.keys())
    leaked = False
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = by_patient[splits[i]] & by_patient[splits[j]]
            if overlap:
                fail(f"patient_id overlaps between {splits[i]} and {splits[j]}: "
                     f"{len(overlap)} shared patients, e.g. {list(overlap)[:3]}")
                leaked = True
    if not leaked:
        ok("no non-empty patient_id overlap across splits")

    empty_patient_counts = {s: sum(1 for r in rows if not r.get("patient_id")) for s, rows in all_rows.items()}
    if any(empty_patient_counts.values()):
        warn(f"rows with empty patient_id (unparseable filename) per split: {empty_patient_counts} "
             "-- expected per src/data/README.md, not itself a failure")


def check_files_on_disk(all_rows: dict, sample_size: int = None):
    print("\n== 4. image_path resolves under XRAY_DATA_ROOT ==")
    data_root_env = os.environ.get("XRAY_DATA_ROOT")
    if not data_root_env:
        warn("XRAY_DATA_ROOT is not set -- skipping file-existence check. "
             "Run `python scripts/download_data.py` first, or export XRAY_DATA_ROOT manually.")
        return

    sys.path.insert(0, str(REPO_ROOT))
    try:
        from src.data.common import resolve_dataset_root, resolve_image_path
    except Exception as exc:
        fail(f"could not import src.data.common to resolve paths: {exc}")
        return

    try:
        root = resolve_dataset_root(data_root_env)
    except Exception as exc:
        fail(f"resolve_dataset_root() failed: {exc}")
        return
    ok(f"XRAY_DATA_ROOT resolves to {root}")

    for split, rows in all_rows.items():
        check_rows = rows if sample_size is None else rows[:sample_size]
        missing = 0
        for r in check_rows:
            try:
                p = resolve_image_path(root, r["image_path"])
            except Exception as exc:
                fail(f"{split}: bad image_path {r['image_path']!r}: {exc}")
                missing += 1
                continue
            if not p.is_file():
                missing += 1
        if missing:
            fail(f"{split}: {missing}/{len(check_rows)} checked image_path values do not exist on disk")
        else:
            ok(f"{split}: all {len(check_rows)} checked image_path values exist on disk")


def check_real_dataset_class(manifests_dir: Path):
    print("\n== 5. Instantiate the real ChestXrayDataset ==")
    if not os.environ.get("XRAY_DATA_ROOT"):
        warn("XRAY_DATA_ROOT not set -- skipping (needs a real data root to load an actual image)")
        return

    sys.path.insert(0, str(REPO_ROOT))
    try:
        from src.data.dataset import ChestXrayDataset
    except Exception as exc:
        fail(f"could not import ChestXrayDataset: {exc}")
        return

    for split, fname in SPLIT_FILES.items():
        path = manifests_dir / fname
        if not path.is_file():
            continue
        try:
            ds = ChestXrayDataset(path, expected_split=split)
        except Exception as exc:
            fail(f"ChestXrayDataset({split}) failed to construct: {exc}")
            continue
        try:
            sample = ds[0]
        except Exception as exc:
            fail(f"ChestXrayDataset({split})[0] failed: {exc}")
            continue
        expected_keys = {"image", "label", "patient_id", "group_id", "image_path", "source_split", "pneumonia_subtype"}
        if set(sample.keys()) != expected_keys:
            fail(f"{split}: sample keys {set(sample.keys())} != expected {expected_keys}")
        else:
            ok(f"{split}: ChestXrayDataset loads real samples, keys match documented contract "
               f"(image shape={tuple(sample['image'].shape)}, label={sample['label'].item()})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests-dir", default=str(REPO_ROOT / "data" / "manifests"))
    ap.add_argument("--check-files", action="store_true", help="Verify every image_path resolves to a real file under XRAY_DATA_ROOT")
    ap.add_argument("--check-dataset", action="store_true", help="Instantiate the real ChestXrayDataset and load one sample per split")
    ap.add_argument("--sample-size", type=int, default=None, help="Only check this many rows per split for file existence (default: all)")
    args = ap.parse_args()

    manifests_dir = Path(args.manifests_dir)
    print(f"Checking manifests in: {manifests_dir}")

    found = check_files_exist(manifests_dir)
    all_rows = {}
    for split, path in found.items():
        rows = load_csv(path)
        all_rows[split] = rows
        check_schema(split, rows)

    if len(all_rows) == 3:
        check_leakage(all_rows)
    else:
        warn("skipping leakage check -- not all 3 manifest files were found")

    if args.check_files and all_rows:
        check_files_on_disk(all_rows, sample_size=args.sample_size)

    if args.check_dataset:
        check_real_dataset_class(manifests_dir)

    print("\n" + "=" * 60)
    if problems:
        print(f"NOT READY -- {len(problems)} problem(s) found:")
        for p in problems:
            print(f"  - {p}")
        if warnings:
            print(f"\n({len(warnings)} warning(s), see above)")
        sys.exit(1)
    else:
        print("READY -- no blocking problems found.")
        if warnings:
            print(f"({len(warnings)} non-blocking warning(s), see above)")
        sys.exit(0)


if __name__ == "__main__":
    main()
