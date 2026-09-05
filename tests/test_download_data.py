from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.data.download import (
    DATASET_HANDLE,
    ensure_local_dataset,
    locate_dataset_root,
    update_env_data_root,
    validate_dataset_inventory,
)


FIXTURE_COUNTS = {
    ("train", "NORMAL"): 4,
    ("train", "PNEUMONIA"): 4,
    ("val", "NORMAL"): 2,
    ("val", "PNEUMONIA"): 2,
    ("test", "NORMAL"): 2,
    ("test", "PNEUMONIA"): 2,
}


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "chest_xray"
    seed = 0
    for (split, label), count in FIXTURE_COUNTS.items():
        directory = root / split / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            seed += 1
            (directory / f"image_{seed}_{index}.jpeg").write_bytes(b"test-image")
    return root


def test_existing_dataset_is_reused(dataset_root: Path, monkeypatch) -> None:
    monkeypatch.delenv("XRAY_DATA_ROOT", raising=False)

    def fail_downloader(handle: str, output_dir: str) -> str:
        raise AssertionError("Downloader must not run for a valid local dataset")

    root, downloaded_now = ensure_local_dataset(
        data_root=dataset_root,
        expected_counts=FIXTURE_COUNTS,
        downloader=fail_downloader,
    )
    assert root == dataset_root.resolve()
    assert downloaded_now is False
    assert (root / ".dataset_source.json").is_file()


def test_download_uses_pinned_handle(dataset_root: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("XRAY_DATA_ROOT", raising=False)
    destination = tmp_path / "download"
    calls: list[tuple[str, str]] = []

    def fake_downloader(handle: str, output_dir: str) -> str:
        calls.append((handle, output_dir))
        shutil.copytree(dataset_root, Path(output_dir) / "chest_xray")
        return output_dir

    root, downloaded_now = ensure_local_dataset(
        download_dir=destination,
        expected_counts=FIXTURE_COUNTS,
        downloader=fake_downloader,
    )
    assert downloaded_now is True
    assert calls == [(DATASET_HANDLE, str(destination.resolve()))]
    assert root == (destination / "chest_xray").resolve()
    assert sum(validate_dataset_inventory(root, FIXTURE_COUNTS).values()) == 16


def test_locator_ignores_nested_and_macos_copies(dataset_root: Path, tmp_path: Path) -> None:
    destination = tmp_path / "download" / "chest_xray"
    shutil.copytree(dataset_root, destination)
    shutil.copytree(dataset_root, destination / "chest_xray")
    shutil.copytree(dataset_root, destination / "__MACOSX" / "chest_xray")

    assert locate_dataset_root(destination.parent, FIXTURE_COUNTS) == destination.resolve()


def test_env_update_preserves_other_settings(dataset_root: Path, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "XRAY_OUTPUT_ROOT=outputs\nXRAY_DATA_ROOT=C:/old/path\nXRAY_NUM_WORKERS=4\n",
        encoding="utf-8",
    )
    update_env_data_root(dataset_root, env_file)
    content = env_file.read_text(encoding="utf-8")
    assert content.count("XRAY_DATA_ROOT=") == 1
    assert dataset_root.resolve().as_posix() in content
    assert "XRAY_OUTPUT_ROOT=outputs" in content
    assert "XRAY_NUM_WORKERS=4" in content
