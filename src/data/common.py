"""Shared constants and deterministic helpers for the data package."""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
SOURCE_SPLITS = {
    "train": "provided_train",
    "val": "provided_val",
    "test": "provided_test",
}
CANONICAL_COLUMNS = [
    "image_path",
    "patient_id",
    "group_id",
    "label",
    "pneumonia_subtype",
    "split",
    "source_split",
    "sha256",
]
AUDIT_COLUMNS = CANONICAL_COLUMNS + [
    "phash",
    "dhash",
    "width",
    "height",
    "is_corrupted",
]

_PNEUMONIA_RE = re.compile(
    r"^person(?P<patient>\d+)_(?P<kind>bacteria|virus)_\d+(?:_\d+)?$",
    flags=re.IGNORECASE,
)
_NORMAL_RE = re.compile(
    r"^(?P<patient>(?:NORMAL2-)?IM-\d+)-\d+(?:-\d+)?$",
    flags=re.IGNORECASE,
)


def resolve_dataset_root(value: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the directory whose direct children are train/val/test."""

    raw = value or os.getenv("XRAY_DATA_ROOT")
    if not raw:
        raise ValueError("Set XRAY_DATA_ROOT or pass --data-root.")

    start = Path(raw).expanduser().resolve()
    candidates = [start, start / "chest_xray", start / "chest_xray" / "chest_xray"]
    for candidate in candidates:
        if all((candidate / split).is_dir() for split in SOURCE_SPLITS):
            return candidate
    raise FileNotFoundError(
        f"Could not find train/val/test below {start}. "
        "XRAY_DATA_ROOT must point to the extracted chest_xray directory or its parent."
    )


def relative_posix(path: Path, root: Path) -> str:
    """Return a portable path below the dataset root."""

    return path.resolve().relative_to(root.resolve()).as_posix()


def resolve_image_path(root: Path, relative_path: str) -> Path:
    """Resolve a manifest path while preventing absolute paths and traversal."""

    rel = Path(str(relative_path))
    if rel.is_absolute():
        raise ValueError(f"Manifest path must be relative: {relative_path}")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes XRAY_DATA_ROOT: {relative_path}") from exc
    return candidate


def parse_identity(filename: str, label: int) -> tuple[str, str]:
    """Return a conservative patient key and audit subtype.

    Bacterial and viral filename counters are separate namespaces in this
    dataset, so the subtype is deliberately part of a pneumonia patient key.
    """

    stem = Path(filename).stem
    if int(label) == 1:
        match = _PNEUMONIA_RE.fullmatch(stem)
        if not match:
            return "", "unknown"
        kind = match.group("kind").lower()
        subtype = "bacterial" if kind == "bacteria" else "viral"
        number = int(match.group("patient"))
        return f"pneumonia:{subtype}:person{number}", subtype

    match = _NORMAL_RE.fullmatch(stem)
    if not match:
        return "", "normal"
    patient = match.group("patient").lower()
    return f"normal:{patient}", "normal"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _dct_matrix(size: int) -> np.ndarray:
    x = np.arange(size, dtype=np.float64)
    k = np.arange(size, dtype=np.float64)[:, None]
    matrix = np.cos((np.pi / size) * (x + 0.5) * k)
    matrix[0, :] *= np.sqrt(1.0 / size)
    matrix[1:, :] *= np.sqrt(2.0 / size)
    return matrix


def perceptual_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute a 64-bit pHash without an additional imagehash dependency."""

    size = hash_size * 4
    pixels = np.asarray(
        image.convert("L").resize((size, size), Image.Resampling.LANCZOS),
        dtype=np.float64,
    )
    matrix = _dct_matrix(size)
    transformed = matrix @ pixels @ matrix.T
    low = transformed[:hash_size, :hash_size]
    threshold = np.median(low.ravel()[1:])
    bits = low > threshold
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute a 64-bit horizontal difference hash."""

    pixels = np.asarray(
        image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS),
        dtype=np.int16,
    )
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def stable_rank(seed: int, salt: str, value: str) -> str:
    payload = f"{seed}|{salt}|{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_group_id(paths: Iterable[str]) -> str:
    payload = "\n".join(sorted(paths)).encode("utf-8")
    return f"group:{hashlib.sha256(payload).hexdigest()[:20]}"


class UnionFind:
    """Small deterministic disjoint-set implementation."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1
