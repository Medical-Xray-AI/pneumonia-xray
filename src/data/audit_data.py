"""Audit the Kermany pediatric chest X-ray dataset without changing images."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError

from .common import (
    AUDIT_COLUMNS,
    IMAGE_EXTENSIONS,
    SOURCE_SPLITS,
    UnionFind,
    difference_hash,
    hamming_distance,
    parse_identity,
    perceptual_hash,
    relative_posix,
    resolve_dataset_root,
    sha256_file,
    stable_group_id,
)


def discover_images(root: Path) -> list[tuple[Path, str, int]]:
    """Return sorted (path, source_split, label) records."""

    records: list[tuple[Path, str, int]] = []
    for folder_split, source_split in SOURCE_SPLITS.items():
        split_dir = root / folder_split
        for class_name, label in (("NORMAL", 0), ("PNEUMONIA", 1)):
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing required directory: {class_dir}")
            for path in sorted(class_dir.rglob("*"), key=lambda item: item.as_posix().lower()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append((path, source_split, label))
    if not records:
        raise RuntimeError(f"No supported image files found below {root}")
    return sorted(records, key=lambda item: relative_posix(item[0], root).lower())


def _join_equivalent_rows(
    frame: pd.DataFrame,
    union_find: UnionFind,
    column: str,
) -> None:
    for value, indices in frame.groupby(column, sort=True).groups.items():
        if not str(value).strip():
            continue
        ordered = sorted(int(index) for index in indices)
        for index in ordered[1:]:
            union_find.union(ordered[0], index)


def _near_duplicate_pairs(
    frame: pd.DataFrame,
    union_find: UnionFind,
    threshold: int,
) -> list[dict[str, Any]]:
    """Find conservative near duplicates using both pHash and dHash."""

    pairs: list[dict[str, Any]] = []
    valid = frame.loc[~frame["is_corrupted"]].copy()
    for label, label_rows in valid.groupby("label", sort=True):
        items = [
            (
                int(index),
                row["phash"],
                row["dhash"],
                row["sha256"],
                row["image_path"],
                row["source_split"],
            )
            for index, row in label_rows.sort_values("image_path").iterrows()
        ]
        for position, left in enumerate(items):
            left_index, left_phash, left_dhash, left_sha, left_path, left_source = left
            for right in items[position + 1 :]:
                right_index, right_phash, right_dhash, right_sha, right_path, right_source = right
                if left_sha and left_sha == right_sha:
                    continue
                p_distance = hamming_distance(left_phash, right_phash)
                if p_distance > threshold:
                    continue
                d_distance = hamming_distance(left_dhash, right_dhash)
                if d_distance > threshold:
                    continue
                union_find.union(left_index, right_index)
                pairs.append(
                    {
                        "left_image_path": left_path,
                        "right_image_path": right_path,
                        "label": int(label),
                        "phash_distance": p_distance,
                        "dhash_distance": d_distance,
                        "left_source_split": left_source,
                        "right_source_split": right_source,
                        "cross_source_split": left_source != right_source,
                    }
                )
    return pairs


def _build_groups(frame: pd.DataFrame, near_threshold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    union_find = UnionFind(len(frame))
    healthy = frame.loc[~frame["is_corrupted"]]
    _join_equivalent_rows(healthy, union_find, "patient_id")
    _join_equivalent_rows(healthy, union_find, "sha256")
    near_pairs = _near_duplicate_pairs(frame, union_find, near_threshold)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(frame)):
        members[union_find.find(index)].append(index)
    for indices in members.values():
        group_id = stable_group_id(frame.loc[indices, "image_path"].tolist())
        frame.loc[indices, "group_id"] = group_id

    pair_columns = [
        "left_image_path",
        "right_image_path",
        "label",
        "phash_distance",
        "dhash_distance",
        "left_source_split",
        "right_source_split",
        "cross_source_split",
    ]
    return frame, pd.DataFrame(near_pairs, columns=pair_columns)


def _test_overlap_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_id, group in frame.loc[~frame["is_corrupted"]].groupby("group_id", sort=True):
        sources = sorted(group["source_split"].unique().tolist())
        contains_test = "provided_test" in sources
        contains_development = any(value in {"provided_train", "provided_val"} for value in sources)
        if contains_test and contains_development:
            rows.append(
                {
                    "group_id": group_id,
                    "source_splits": "|".join(sources),
                    "image_count": len(group),
                    "patient_ids": "|".join(sorted(value for value in group["patient_id"].unique() if value)),
                    "image_paths": "|".join(sorted(group["image_path"].tolist())),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["group_id", "source_splits", "image_count", "patient_ids", "image_paths"],
    )


def _write_markdown_summary(summary: dict[str, Any], class_counts: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Data audit summary",
        "",
        f"- Dataset root used at runtime: `{summary['dataset_root']}` (not stored in manifests)",
        f"- Images scanned: **{summary['image_count']}**",
        f"- Corrupt or unreadable images: **{summary['corrupt_count']}**",
        f"- Images with a parsed patient identifier: **{summary['patient_id_coverage']}**",
        f"- Exact duplicate groups: **{summary['exact_duplicate_groups']}**",
        f"- Conservative near-duplicate pairs: **{summary['near_duplicate_pairs']}**",
        f"- Leakage groups touching both development data and provided test: **{summary['test_overlap_groups']}**",
        f"- Provided test is safe to lock: **{str(summary['provided_test_safe']).lower()}**",
        "",
        "## Class distribution",
        "",
        "| Source split | Label | Count |",
        "|---|---:|---:|",
    ]
    for row in class_counts.itertuples(index=False):
        lines.append(f"| {row.source_split} | {row.label} | {row.count} |")
    lines.extend(
        [
            "",
            "## Identity and duplicate policy",
            "",
            "Pneumonia identifiers include the bacterial/viral namespace, for example "
            "`pneumonia:bacterial:person1` and `pneumonia:viral:person1`. Normal identifiers "
            "retain the `IM` versus `NORMAL2-IM` namespace. Unparseable names keep an empty "
            "patient ID and are controlled by exact/perceptual duplicate groups.",
            "",
            "Near duplicates are conservative candidates that pass both pHash and dHash "
            f"Hamming-distance thresholds (<= {summary['near_duplicate_threshold']}).",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_dataset(
    data_root: str | Path | None,
    output_dir: str | Path,
    near_threshold: int = 4,
) -> dict[str, Any]:
    """Run the audit and write portable evidence files."""

    if near_threshold < 0 or near_threshold > 16:
        raise ValueError("near_threshold must be between 0 and 16")
    root = resolve_dataset_root(data_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for path, source_split, label in discover_images(root):
        patient_id, subtype = parse_identity(path.name, label)
        row: dict[str, Any] = {
            "image_path": relative_posix(path, root),
            "patient_id": patient_id,
            "group_id": "",
            "label": label,
            "pneumonia_subtype": subtype,
            "split": "",
            "source_split": source_split,
            "sha256": "",
            "phash": "",
            "dhash": "",
            "width": "",
            "height": "",
            "is_corrupted": False,
        }
        try:
            row["sha256"] = sha256_file(path)
            with Image.open(path) as image:
                image.load()
                row["width"], row["height"] = image.size
                row["phash"] = perceptual_hash(image)
                row["dhash"] = difference_hash(image)
        except (OSError, ValueError, UnidentifiedImageError):
            row["is_corrupted"] = True
        rows.append(row)

    frame = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    frame = frame.sort_values("image_path", kind="stable").reset_index(drop=True)
    frame, near_pairs = _build_groups(frame, near_threshold)

    exact_counts = frame.loc[frame["sha256"] != "", "sha256"].value_counts()
    duplicate_hashes = set(exact_counts[exact_counts > 1].index)
    exact_duplicates = frame.loc[frame["sha256"].isin(duplicate_hashes)].copy()
    exact_duplicates.insert(0, "exact_duplicate_id", exact_duplicates["sha256"].str[:20])

    test_overlap = _test_overlap_report(frame)
    class_counts = (
        frame.groupby(["source_split", "label"], as_index=False, sort=True)
        .size()
        .rename(columns={"size": "count"})
    )
    patient_overlap_rows = []
    for patient_id, group in frame.loc[frame["patient_id"] != ""].groupby("patient_id", sort=True):
        sources = sorted(group["source_split"].unique().tolist())
        if len(sources) > 1:
            patient_overlap_rows.append(
                {
                    "patient_id": patient_id,
                    "source_splits": "|".join(sources),
                    "image_count": len(group),
                }
            )
    patient_overlap = pd.DataFrame(
        patient_overlap_rows,
        columns=["patient_id", "source_splits", "image_count"],
    )

    frame.to_csv(output / "file_manifest.csv", index=False)
    exact_duplicates.to_csv(output / "exact_duplicate_report.csv", index=False)
    near_pairs.to_csv(output / "near_duplicate_report.csv", index=False)
    patient_overlap.to_csv(output / "patient_overlap_report.csv", index=False)
    test_overlap.to_csv(output / "test_overlap_report.csv", index=False)
    class_counts.to_csv(output / "class_distribution.csv", index=False)

    summary: dict[str, Any] = {
        "dataset_root": str(root),
        "image_count": int(len(frame)),
        "corrupt_count": int(frame["is_corrupted"].sum()),
        "patient_id_coverage": int((frame["patient_id"] != "").sum()),
        "unique_patient_ids": int(frame.loc[frame["patient_id"] != "", "patient_id"].nunique()),
        "exact_duplicate_groups": int(len(duplicate_hashes)),
        "near_duplicate_pairs": int(len(near_pairs)),
        "cross_source_near_duplicate_pairs": int(
            near_pairs["cross_source_split"].sum() if not near_pairs.empty else 0
        ),
        "patient_overlap_ids": int(len(patient_overlap)),
        "test_overlap_groups": int(len(test_overlap)),
        "provided_test_safe": bool(test_overlap.empty),
        "near_duplicate_threshold": int(near_threshold),
        "identity_policy": "subtype-aware pneumonia IDs; namespace-aware normal IDs",
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown_summary(summary, class_counts, output / "audit_summary.md")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None, help="Defaults to XRAY_DATA_ROOT")
    parser.add_argument("--output-dir", default="audit_out")
    parser.add_argument("--near-threshold", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = audit_dataset(args.data_root, args.output_dir, args.near_threshold)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
