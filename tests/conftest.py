from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def write_image(path: Path, seed: int, size: tuple[int, int] = (32, 24)) -> None:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(size[1], size[0]), dtype=np.uint8)
    pixels[:, seed % size[0]] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="L").save(path, quality=94)


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "chest_xray"
    records = [
        ("train", "NORMAL", "IM-0001-0001.jpeg", 1),
        ("train", "NORMAL", "NORMAL2-IM-0001-0001.jpeg", 2),
        ("train", "NORMAL", "IM-0002-0001.jpeg", 3),
        ("train", "NORMAL", "IM-0003-0001.jpeg", 4),
        ("train", "PNEUMONIA", "person1_bacteria_1.jpeg", 5),
        ("train", "PNEUMONIA", "person1_bacteria_2.jpeg", 6),
        ("train", "PNEUMONIA", "person1_virus_6.jpeg", 7),
        ("train", "PNEUMONIA", "person2_virus_7.jpeg", 8),
        ("val", "NORMAL", "IM-0100-0001.jpeg", 9),
        ("val", "NORMAL", "NORMAL2-IM-0100-0001.jpeg", 10),
        ("val", "PNEUMONIA", "person100_bacteria_100.jpeg", 11),
        ("val", "PNEUMONIA", "person100_virus_101.jpeg", 12),
        ("test", "NORMAL", "IM-0200-0001.jpeg", 13),
        ("test", "NORMAL", "NORMAL2-IM-0200-0001.jpeg", 14),
        ("test", "PNEUMONIA", "person200_bacteria_200.jpeg", 15),
        ("test", "PNEUMONIA", "person200_virus_201.jpeg", 16),
    ]
    for split, label, filename, seed in records:
        write_image(root / split / label / filename, seed)
    return root
