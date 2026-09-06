from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.audit_data import audit_dataset
from src.data.common import (
    CANONICAL_COLUMNS,
    difference_hash,
    hamming_distance,
    parse_identity,
    perceptual_hash,
    resolve_image_path,
)
from src.data.dataset import ChestXrayDataset, build_dataloader
from src.data.split_data import create_manifests


def test_identity_parser_keeps_subtype_namespaces_separate() -> None:
    bacterial, bacterial_subtype = parse_identity("person1_bacteria_1.jpeg", 1)
    viral, viral_subtype = parse_identity("person1_virus_6.jpeg", 1)
    assert bacterial == "pneumonia:bacterial:person1"
    assert viral == "pneumonia:viral:person1"
    assert bacterial != viral
    assert bacterial_subtype == "bacterial"
    assert viral_subtype == "viral"


def test_identity_parser_keeps_normal_namespaces_separate() -> None:
    im_patient, _ = parse_identity("IM-0001-0001.jpeg", 0)
    normal2_patient, _ = parse_identity("NORMAL2-IM-0001-0001.jpeg", 0)
    assert im_patient == "normal:im-0001"
    assert normal2_patient == "normal:normal2-im-0001"
    assert im_patient != normal2_patient


def test_audit_and_locked_test_split_are_portable_and_reproducible(
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    audit_dir = tmp_path / "audit"
    summary = audit_dataset(dataset_root, audit_dir, near_threshold=0)
    assert summary["image_count"] == 16
    assert summary["corrupt_count"] == 0
    assert summary["provided_test_safe"] is True

    audit = pd.read_csv(audit_dir / "file_manifest.csv", keep_default_na=False)
    assert not audit["image_path"].str.match(r"^(?:[A-Za-z]:|/|\\\\)").any()
    assert audit["group_id"].ne("").all()
    assert set(audit["pneumonia_subtype"]) == {"normal", "bacterial", "viral"}

    first_dir = tmp_path / "manifests_a"
    second_dir = tmp_path / "manifests_b"
    first = create_manifests(audit_dir / "file_manifest.csv", first_dir, seed=42)
    second = create_manifests(audit_dir / "file_manifest.csv", second_dir, seed=42)
    assert first["strategy"] == "locked_provided_test"
    assert first == second
    assert (first_dir / "split_manifest.csv").read_bytes() == (
        second_dir / "split_manifest.csv"
    ).read_bytes()

    combined = pd.read_csv(first_dir / "split_manifest.csv", keep_default_na=False)
    assert list(combined.columns) == CANONICAL_COLUMNS
    assert set(combined["split"]) == {"train", "validation", "test"}
    assert set(combined.loc[combined["source_split"] == "provided_test", "split"]) == {"test"}
    assert set(combined.loc[combined["split"] == "test", "source_split"]) == {"provided_test"}
    for column in ("group_id", "patient_id", "sha256"):
        known = combined.loc[combined[column] != ""]
        assert known.groupby(column)["split"].nunique().max() == 1
    for relative_path in combined["image_path"]:
        assert resolve_image_path(dataset_root, relative_path).is_file()


def test_cross_test_exact_duplicate_triggers_full_rebuild(dataset_root: Path, tmp_path: Path) -> None:
    source = dataset_root / "train" / "NORMAL" / "IM-0001-0001.jpeg"
    target = dataset_root / "test" / "NORMAL" / "IM-0200-0001.jpeg"
    shutil.copyfile(source, target)
    audit_dir = tmp_path / "unsafe_audit"
    summary = audit_dataset(dataset_root, audit_dir, near_threshold=0)
    assert summary["provided_test_safe"] is False
    split_summary = create_manifests(audit_dir / "file_manifest.csv", tmp_path / "rebuilt", seed=42)
    assert split_summary["strategy"] == "rebuild_all_groups"
    assert split_summary["checks"] == {
        "group_overlap": 0,
        "patient_overlap": 0,
        "sha256_overlap": 0,
    }


def test_near_duplicate_hashes_survive_jpeg_reencoding(tmp_path: Path) -> None:
    pixels = np.random.default_rng(42).integers(0, 256, size=(128, 128), dtype=np.uint8)
    original = Image.fromarray(pixels, mode="L")
    first = tmp_path / "first.jpeg"
    second = tmp_path / "second.jpeg"
    original.save(first, quality=98)
    original.save(second, quality=75)
    with Image.open(first) as left, Image.open(second) as right:
        assert hamming_distance(perceptual_hash(left), perceptual_hash(right)) <= 4
        assert hamming_distance(difference_hash(left), difference_hash(right)) <= 4


def test_dataset_returns_required_metadata_and_three_channels(
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    audit_dir = tmp_path / "audit"
    manifest_dir = tmp_path / "manifests"
    audit_dataset(dataset_root, audit_dir, near_threshold=0)
    create_manifests(audit_dir / "file_manifest.csv", manifest_dir, seed=42)

    dataset = ChestXrayDataset(
        manifest_dir / "train.csv",
        data_root=dataset_root,
        expected_split="train",
    )
    sample = dataset[0]
    assert set(sample) == {
        "image",
        "label",
        "patient_id",
        "group_id",
        "image_path",
        "source_split",
        "pneumonia_subtype",
    }
    assert sample["image"].shape[0] == 3
    assert sample["label"].dtype == torch.float32
    loader = build_dataloader(dataset, batch_size=2, num_workers=0, seed=42)
    batch = next(iter(loader))
    assert batch["image"].shape[0] == 2


def test_machine_readable_summaries_are_valid_json(dataset_root: Path, tmp_path: Path) -> None:
    audit_dir = tmp_path / "audit"
    manifest_dir = tmp_path / "manifests"
    audit_dataset(dataset_root, audit_dir, near_threshold=0)
    create_manifests(audit_dir / "file_manifest.csv", manifest_dir)
    assert json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    assert json.loads((manifest_dir / "split_summary.json").read_text(encoding="utf-8"))
