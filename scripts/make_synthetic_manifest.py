"""
make_synthetic_manifest.py

Generates a tiny FAKE dataset (random-noise JPEGs) plus manifest CSVs that
match the real schema exactly (data/manifests/README.md), so Member 2 can
smoke-test the full pipeline -- dataset.py -> transforms.py -> baseline.py
-> train_baseline.py -- WITHOUT waiting for Member 1's real audited
manifests and WITHOUT downloading the real Kaggle dataset.

THIS IS A TESTING SHORTCUT, NOT A DELIVERABLE. Never train the real
baseline/report numbers on this data -- it's random noise, not X-rays.
Delete the synthetic root once real manifests land.

USAGE
-----
    python scripts/make_synthetic_manifest.py --out /tmp/xray_synthetic

Then, for a smoke-test run:

    export XRAY_DATA_ROOT=/tmp/xray_synthetic
    export XRAY_OUTPUT_ROOT=outputs
    export XRAY_NUM_WORKERS=0

    # point a scratch copy of configs/data.yaml at the synthetic manifests
    python scripts/make_synthetic_manifest.py --out /tmp/xray_synthetic \
        --write-data-config configs/data.synthetic.yaml

    python scripts/train_baseline.py \
        --config configs/baseline.synthetic.yaml --mode overfit

(configs/baseline.synthetic.yaml is just configs/baseline.yaml with
data_config: configs/data.synthetic.yaml -- copy-edit one line.)
"""

import argparse
import hashlib
import random
from pathlib import Path

import yaml
from PIL import Image

LABELS = {"NORMAL": 0, "PNEUMONIA": 1}
SOURCE_SPLIT_MAP = {"train": "provided_train", "val": "provided_val", "test": "provided_test"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_fake_image(path: Path, seed: int, size: int = 256) -> str:
    """Writes a random-noise grayscale JPEG and returns its sha256."""
    rng = random.Random(seed)
    img = Image.new("L", (size, size))
    pixels = [rng.randint(0, 255) for _ in range(size * size)]
    img.putdata(pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=90)
    return sha256_bytes(path.read_bytes())


def build_synthetic_dataset(out_root: Path, n_per_split_per_class: int = 6, seed: int = 42):
    """
    Creates out_root/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg
    (matching the REAL Kaggle folder layout, per data/README.md) and a
    canonical-schema manifest CSV for train/validation/test splits.

    Patients are fully separated across splits by construction (each
    synthetic "patient" only ever appears in one split) so this also
    smoke-tests that your training code doesn't accidentally assume a
    particular leakage pattern.
    """
    rows = []
    global_seed = seed
    patient_counter = 0

    # kaggle_split -> our_split : keep it simple, 1:1, synthetic data
    # doesn't need the real train+val-combine-then-resplit logic -- that's
    # Member 1's concern, not something this smoke test needs to reproduce.
    split_map = {"train": "train", "val": "validation", "test": "test"}

    for kaggle_split, our_split in split_map.items():
        for class_name, label in LABELS.items():
            for i in range(n_per_split_per_class):
                patient_counter += 1
                patient_id = f"synthP{patient_counter:04d}"
                group_id = patient_id
                fname = f"{patient_id}-{i:03d}.jpeg"
                # image_path is relative to the RESOLVED data root, which
                # resolve_dataset_root() sets to <XRAY_DATA_ROOT>/chest_xray
                # -- so it must NOT include the "chest_xray/" prefix itself
                # (data/manifests/README.md's example row confirms this:
                # "train/NORMAL/IM-0115-0001.jpeg", no chest_xray/ prefix).
                rel_path = f"{kaggle_split}/{class_name}/{fname}"
                abs_path = out_root / "chest_xray" / rel_path
                global_seed += 1
                sha = make_fake_image(abs_path, seed=global_seed)

                subtype = "normal" if label == 0 else random.choice(["bacterial", "viral"])
                rows.append(
                    {
                        "image_path": rel_path,
                        "patient_id": patient_id,
                        "group_id": group_id,
                        "label": label,
                        "pneumonia_subtype": subtype,
                        "split": our_split,
                        "source_split": SOURCE_SPLIT_MAP[kaggle_split],
                        "sha256": sha,
                    }
                )
    return rows


def write_manifests(rows, manifests_dir: Path):
    import csv

    manifests_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "patient_id", "group_id", "label", "pneumonia_subtype", "split", "source_split", "sha256"]

    for split_name in ("train", "validation", "test"):
        split_rows = [r for r in rows if r["split"] == split_name]
        out_path = manifests_dir / f"{split_name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"[ok] wrote {len(split_rows)} rows to {out_path}")


def write_synthetic_data_config(template_path: Path, manifests_dir: Path, out_path: Path):
    """
    Copies configs/data.yaml but repoints `manifests:` at the synthetic
    CSVs, so you don't have to touch the real config file at all.
    """
    with open(template_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["manifests"] = {
        "train": str(manifests_dir / "train.csv"),
        "validation": str(manifests_dir / "validation.csv"),
        "test": str(manifests_dir / "test.csv"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[ok] wrote synthetic data config to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Root dir for synthetic images (this becomes XRAY_DATA_ROOT)")
    ap.add_argument("--manifests-dir", default=None, help="Where to write train/validation/test.csv (default: <out>/manifests)")
    ap.add_argument("--n-per-class", type=int, default=6, help="Images per class per split (default 6 -> 12 per split)")
    ap.add_argument("--write-data-config", default=None, help="If set, writes a scratch configs/data.*.yaml pointing at the synthetic manifests")
    ap.add_argument("--data-config-template", default="configs/data.yaml", help="Real config to copy structure from")
    args = ap.parse_args()

    out_root = Path(args.out)
    manifests_dir = Path(args.manifests_dir) if args.manifests_dir else out_root / "manifests"

    rows = build_synthetic_dataset(out_root, n_per_split_per_class=args.n_per_class)
    write_manifests(rows, manifests_dir)

    print(f"\n[done] synthetic dataset ready.")
    print(f"  export XRAY_DATA_ROOT={out_root.resolve()}")
    print(f"  manifests written to: {manifests_dir.resolve()}")

    if args.write_data_config:
        write_synthetic_data_config(Path(args.data_config_template), manifests_dir, Path(args.write_data_config))


if __name__ == "__main__":
    main()
