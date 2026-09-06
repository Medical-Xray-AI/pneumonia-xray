"""Create deterministic leakage-aware train/validation/test manifests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import pandas as pd

from .common import CANONICAL_COLUMNS, stable_rank


REQUIRED_AUDIT_COLUMNS = set(CANONICAL_COLUMNS) | {"is_corrupted"}
ALLOWED_SOURCES = {"provided_train", "provided_val", "provided_test"}
ALLOWED_SPLITS = {"train", "validation", "test"}


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def load_audit_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = sorted(REQUIRED_AUDIT_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Audit manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Audit manifest is empty")

    frame = frame.copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not set(frame["label"]).issubset({0, 1}):
        raise ValueError("Labels must be numeric 0 (normal) or 1 (pneumonia)")
    if not set(frame["source_split"]).issubset(ALLOWED_SOURCES):
        raise ValueError(f"Unexpected source_split values: {sorted(set(frame['source_split']) - ALLOWED_SOURCES)}")
    if frame["image_path"].duplicated().any():
        raise ValueError("Every image_path must appear exactly once")
    for value in frame["image_path"]:
        path_value = PurePosixPath(str(value))
        if path_value.is_absolute() or ".." in path_value.parts:
            raise ValueError(f"image_path must be a safe relative POSIX path: {value}")
    if (frame["group_id"].astype(str).str.strip() == "").any():
        raise ValueError("Every non-corrupt row must have a group_id")
    corrupt = _truthy(frame["is_corrupted"])
    if corrupt.any():
        examples = frame.loc[corrupt, "image_path"].head(5).tolist()
        raise ValueError(f"Resolve or exclude corrupt images before splitting; examples: {examples}")

    mixed_groups = frame.groupby("group_id")["label"].nunique()
    if (mixed_groups > 1).any():
        examples = mixed_groups[mixed_groups > 1].index[:5].tolist()
        raise ValueError(f"Leakage groups contain mixed binary labels: {examples}")
    return frame.sort_values("image_path", kind="stable").reset_index(drop=True)


def _group_table(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("group_id", as_index=False, sort=True)
        .agg(label=("label", "first"), image_count=("image_path", "size"))
        .sort_values("group_id", kind="stable")
        .reset_index(drop=True)
    )


def _select_fraction_groups(
    frame: pd.DataFrame,
    fraction: float,
    seed: int,
    salt: str,
) -> set[str]:
    """Select whole groups per class, targeting an image fraction."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1")
    selected: set[str] = set()
    groups = _group_table(frame)
    for label, label_groups in groups.groupby("label", sort=True):
        ordered = label_groups.copy()
        ordered["rank"] = ordered["group_id"].map(
            lambda value: stable_rank(seed, f"{salt}|label={label}", str(value))
        )
        ordered = ordered.sort_values(["rank", "group_id"], kind="stable").reset_index(drop=True)
        if len(ordered) < 2:
            raise ValueError(f"Need at least two leakage groups for label {label}")

        target = int(round(int(ordered["image_count"].sum()) * fraction))
        cumulative = ordered["image_count"].cumsum().tolist()
        candidate_counts = range(1, len(ordered))
        best_count = min(candidate_counts, key=lambda count: (abs(cumulative[count - 1] - target), count))
        selected.update(ordered.iloc[:best_count]["group_id"].astype(str).tolist())
    return selected


def _groups_touching_test_and_development(frame: pd.DataFrame) -> list[str]:
    unsafe: list[str] = []
    for group_id, group in frame.groupby("group_id", sort=True):
        sources = set(group["source_split"])
        if "provided_test" in sources and sources.intersection({"provided_train", "provided_val"}):
            unsafe.append(str(group_id))
    return unsafe


def _assign_locked_test(
    frame: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> pd.Series:
    development = frame[frame["source_split"].isin({"provided_train", "provided_val"})]
    validation_groups = _select_fraction_groups(
        development,
        fraction=validation_fraction,
        seed=seed,
        salt="locked-test-validation",
    )
    assigned = pd.Series("train", index=frame.index, dtype="object")
    assigned.loc[frame["group_id"].isin(validation_groups)] = "validation"
    assigned.loc[frame["source_split"] == "provided_test"] = "test"
    return assigned


def _assign_rebuilt_all(
    frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.Series:
    total = train_fraction + validation_fraction + test_fraction
    if any(value <= 0 for value in (train_fraction, validation_fraction, test_fraction)):
        raise ValueError("All rebuild fractions must be positive")
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Rebuild fractions must sum to 1")

    test_groups = _select_fraction_groups(frame, test_fraction, seed, "rebuild-test")
    remaining = frame.loc[~frame["group_id"].isin(test_groups)]
    remaining_validation_fraction = validation_fraction / (train_fraction + validation_fraction)
    validation_groups = _select_fraction_groups(
        remaining,
        remaining_validation_fraction,
        seed,
        "rebuild-validation",
    )
    assigned = pd.Series("train", index=frame.index, dtype="object")
    assigned.loc[frame["group_id"].isin(validation_groups)] = "validation"
    assigned.loc[frame["group_id"].isin(test_groups)] = "test"
    return assigned


def _cross_split_values(frame: pd.DataFrame, column: str, ignore_blank: bool = True) -> list[str]:
    values: defaultdict[str, set[str]] = defaultdict(set)
    for row in frame[[column, "split"]].itertuples(index=False):
        value = str(row[0]).strip()
        if ignore_blank and not value:
            continue
        values[value].add(str(row[1]))
    return sorted(value for value, splits in values.items() if len(splits) > 1)


def validate_split_manifest(frame: pd.DataFrame, strategy: str) -> dict[str, Any]:
    if set(frame["split"]) != ALLOWED_SPLITS:
        raise AssertionError(f"Expected all three splits, got {sorted(set(frame['split']))}")
    failures: dict[str, list[str]] = {
        "group_overlap": _cross_split_values(frame, "group_id"),
        "patient_overlap": _cross_split_values(frame, "patient_id"),
        "sha256_overlap": _cross_split_values(frame, "sha256"),
    }
    nonempty_failures = {name: values for name, values in failures.items() if values}
    if nonempty_failures:
        preview = {name: values[:5] for name, values in nonempty_failures.items()}
        raise AssertionError(f"Leakage validation failed: {preview}")

    if strategy == "locked_provided_test":
        provided_test_splits = set(frame.loc[frame["source_split"] == "provided_test", "split"])
        if provided_test_splits != {"test"}:
            raise AssertionError("Every provided_test image must remain in the locked test split")
        if (frame.loc[frame["split"] == "test", "source_split"] != "provided_test").any():
            raise AssertionError("Locked test contains images outside provided_test")

    for split_name, split_rows in frame.groupby("split", sort=True):
        if set(split_rows["label"]) != {0, 1}:
            raise AssertionError(f"Split {split_name} does not contain both labels")
    return {name: len(values) for name, values in failures.items()}


def create_manifests(
    audit_manifest: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    validation_fraction: float = 0.15,
    rebuild_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    force_rebuild: bool = False,
    version: str = "split_v1",
) -> dict[str, Any]:
    frame = load_audit_manifest(audit_manifest)
    unsafe_test_groups = _groups_touching_test_and_development(frame)

    if unsafe_test_groups or force_rebuild:
        strategy = "rebuild_all_groups"
        frame["split"] = _assign_rebuilt_all(frame, *rebuild_ratios, seed)
    else:
        strategy = "locked_provided_test"
        frame["split"] = _assign_locked_test(frame, validation_fraction, seed)

    checks = validate_split_manifest(frame, strategy)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    canonical = frame[CANONICAL_COLUMNS].sort_values(["split", "image_path"], kind="stable")
    canonical.to_csv(output / "split_manifest.csv", index=False)
    for split_name in ("train", "validation", "test"):
        canonical.loc[canonical["split"] == split_name].to_csv(
            output / f"{split_name}.csv",
            index=False,
        )

    split_stats = (
        canonical.groupby(["split", "label"], as_index=False, sort=True)
        .size()
        .rename(columns={"size": "count"})
    )
    split_stats.to_csv(output / "split_statistics.csv", index=False)
    summary: dict[str, Any] = {
        "version": version,
        "seed": int(seed),
        "strategy": strategy,
        "validation_fraction_of_development": float(validation_fraction),
        "rebuild_ratios": {
            "train": rebuild_ratios[0],
            "validation": rebuild_ratios[1],
            "test": rebuild_ratios[2],
        },
        "unsafe_provided_test_groups": len(unsafe_test_groups),
        "row_count": int(len(canonical)),
        "checks": checks,
        "split_counts": {
            name: int(count) for name, count in canonical["split"].value_counts().sort_index().items()
        },
    }
    (output / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", default="audit_out/file_manifest.csv")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--version", default="split_v1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = create_manifests(
        audit_manifest=args.audit_manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        force_rebuild=args.force_rebuild,
        version=args.version,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
