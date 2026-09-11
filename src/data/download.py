"""Safe, version-pinned local acquisition for the approved Kaggle dataset."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Mapping

try:
    from dotenv import load_dotenv
except ImportError:  # Explicit --data-root remains available in minimal setups.
    load_dotenv = None


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
if load_dotenv is not None and os.getenv("XRAY_SKIP_DOTENV") != "1":
    load_dotenv(REPOSITORY_ROOT / ".env", override=False)


DATASET_HANDLE = "paultimothymooney/chest-xray-pneumonia/versions/2"
EXPECTED_COUNTS: dict[tuple[str, str], int] = {
    ("train", "NORMAL"): 1341,
    ("train", "PNEUMONIA"): 3875,
    ("val", "NORMAL"): 8,
    ("val", "PNEUMONIA"): 8,
    ("test", "NORMAL"): 234,
    ("test", "PNEUMONIA"): 390,
}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())


def default_download_dir() -> Path:
    return REPOSITORY_ROOT / "data" / "raw" / "kaggle"


def _has_layout(path: Path) -> bool:
    return all(
        (path / split / label).is_dir()
        for split in ("train", "val", "test")
        for label in ("NORMAL", "PNEUMONIA")
    )


def locate_dataset_root(
    start: str | Path,
    expected_counts: Mapping[tuple[str, str], int] = EXPECTED_COUNTS,
) -> Path:
    """Find the canonical directory that directly contains train/val/test."""

    base = Path(start).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Dataset location does not exist: {base}")

    candidates = [base, base / "chest_xray", base / "chest_xray" / "chest_xray"]
    if base.is_dir():
        for train_dir in sorted(base.rglob("train"), key=lambda value: value.as_posix().lower()):
            try:
                relative_depth = len(train_dir.relative_to(base).parts)
            except ValueError:
                continue
            if relative_depth <= 5:
                candidates.append(train_dir.parent)

    valid: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and _has_layout(resolved):
            seen.add(resolved)
            valid.append(resolved)
    if not valid:
        raise FileNotFoundError(
            f"Could not find the required train/val/test and NORMAL/PNEUMONIA layout below {base}"
        )
    if len(valid) > 1:
        exact = []
        for candidate in valid:
            try:
                validate_dataset_inventory(candidate, expected_counts)
                exact.append(candidate)
            except ValueError:
                continue

        # Kaggle Version 2 contains a complete nested copy and macOS resource
        # fork metadata beside the canonical top-level data. Prefer the
        # shallowest exact root, while never selecting anything in __MACOSX.
        ordinary = [
            candidate
            for candidate in exact
            if not any(part.casefold() == "__macosx" for part in candidate.parts)
        ]
        preferred = ordinary or exact
        if preferred:
            depths = {
                candidate: len(candidate.relative_to(base).parts)
                for candidate in preferred
            }
            minimum_depth = min(depths.values())
            shallowest = [
                candidate for candidate in preferred if depths[candidate] == minimum_depth
            ]
            if len(shallowest) == 1:
                return shallowest[0]

        raise RuntimeError(f"Multiple candidate dataset roots found: {[str(value) for value in valid]}")
    return valid[0]


def list_dataset_images(root: str | Path) -> list[Path]:
    resolved = Path(root).resolve()
    images: list[Path] = []
    for split, label in EXPECTED_COUNTS:
        directory = resolved / split / label
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing dataset directory: {directory}")
        images.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return sorted(images, key=lambda value: value.relative_to(resolved).as_posix().lower())


def dataset_inventory(root: str | Path) -> dict[tuple[str, str], int]:
    resolved = Path(root).resolve()
    return {
        (split, label): sum(
            1
            for path in (resolved / split / label).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for split, label in EXPECTED_COUNTS
    }


def validate_dataset_inventory(
    root: str | Path,
    expected_counts: Mapping[tuple[str, str], int] = EXPECTED_COUNTS,
) -> dict[tuple[str, str], int]:
    """Reject incomplete, wrong-version, or unexpectedly modified downloads."""

    resolved = Path(root).resolve()
    actual = dataset_inventory(resolved)
    expected = dict(expected_counts)
    mismatches = {
        f"{split}/{label}": {"expected": expected[(split, label)], "actual": actual[(split, label)]}
        for split, label in expected
        if actual.get((split, label)) != expected[(split, label)]
    }
    if mismatches:
        raise ValueError(
            "Dataset inventory does not match the approved Kaggle Version 2: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return actual


def _default_downloader(handle: str, output_dir: str) -> str:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install project requirements before downloading: pip install -r requirements.txt") from exc
    try:
        return kagglehub.dataset_download(handle, output_dir=output_dir)
    except Exception as exc:
        raise RuntimeError(
            "Kaggle download failed. Run `python scripts/download_data.py --login` once or "
            "provide a Kaggle API token through the environment, "
            "then retry. Never commit the token or kaggle.json."
        ) from exc


def authenticate_kaggle() -> None:
    """Start KaggleHub's interactive authentication flow.

    Credential validity is established by the real download request. Skipping
    KaggleHub's separate diagnostics call also avoids a known failure mode in
    which that endpoint returns an empty or non-JSON response.
    """

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install project requirements first: pip install -r requirements.txt") from exc
    kagglehub.login(validate_credentials=False)


def _write_source_marker(root: Path, inventory: Mapping[tuple[str, str], int]) -> None:
    marker = {
        "dataset_handle": DATASET_HANDLE,
        "expected_total": EXPECTED_TOTAL,
        "inventory": {
            f"{split}/{label}": count for (split, label), count in sorted(inventory.items())
        },
    }
    (root / ".dataset_source.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def ensure_local_dataset(
    download_dir: str | Path | None = None,
    data_root: str | Path | None = None,
    expected_counts: Mapping[tuple[str, str], int] = EXPECTED_COUNTS,
    downloader: Callable[[str, str], str] | None = None,
) -> tuple[Path, bool]:
    """Return a validated root; download only when no valid local root exists.

    Returns `(root, downloaded_now)`. Existing data is never deleted or
    overwritten by this function.
    """

    configured = data_root or os.getenv("XRAY_DATA_ROOT")
    if configured and Path(configured).expanduser().exists():
        root = locate_dataset_root(configured, expected_counts)
        inventory = validate_dataset_inventory(root, expected_counts)
        _write_source_marker(root, inventory)
        return root, False

    target = Path(download_dir) if download_dir else default_download_dir()
    target = target.expanduser().resolve()
    if target.exists():
        try:
            root = locate_dataset_root(target, expected_counts)
        except FileNotFoundError:
            root = None
        if root is not None:
            inventory = validate_dataset_inventory(root, expected_counts)
            _write_source_marker(root, inventory)
            return root, False
    target.mkdir(parents=True, exist_ok=True)

    download = downloader or _default_downloader
    downloaded_path = Path(download(DATASET_HANDLE, str(target))).expanduser().resolve()
    search_root = downloaded_path if downloaded_path.exists() else target
    root = locate_dataset_root(search_root, expected_counts)
    inventory = validate_dataset_inventory(root, expected_counts)
    _write_source_marker(root, inventory)
    return root, True


def update_env_data_root(root: str | Path, env_file: str | Path | None = None) -> Path:
    """Atomically set XRAY_DATA_ROOT in the ignored local .env file."""

    path = Path(env_file) if env_file else REPOSITORY_ROOT / ".env"
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    assignment = f"XRAY_DATA_ROOT={Path(root).resolve().as_posix()}"
    pattern = re.compile(r"^\s*XRAY_DATA_ROOT\s*=")
    output: list[str] = []
    replaced = False
    for line in existing:
        if pattern.match(line):
            if not replaced:
                output.append(assignment)
                replaced = True
            continue
        output.append(line)
    if not replaced:
        if output and output[-1] != "":
            output.append("")
        output.append(assignment)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write("\n".join(output) + "\n")
    os.replace(temporary, path)
    os.environ["XRAY_DATA_ROOT"] = str(Path(root).resolve())
    return path
