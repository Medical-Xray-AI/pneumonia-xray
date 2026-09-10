from pathlib import Path

import pandas as pd
from PIL import Image

from src.training.data import ManifestDataset, compute_pos_weight_from_train


def test_manifest_path_remaps_under_data_root(tmp_path: Path):
    root = tmp_path / "chest_xray"
    for label in ["NORMAL", "PNEUMONIA"]:
        d = root / "train" / label
        d.mkdir(parents=True, exist_ok=True)
        Image.new("L", (40, 20), color=128).save(d / f"{label}.jpeg")

    manifest = tmp_path / "split_manifest.csv"
    pd.DataFrame(
        [
            {
                "path": "/kaggle/input/old/train/NORMAL/NORMAL.jpeg",
                "filename": "NORMAL.jpeg",
                "split": "train",
                "label": "NORMAL",
                "group_id": "a",
                "new_split": "train",
            },
            {
                "path": "/kaggle/input/old/train/PNEUMONIA/PNEUMONIA.jpeg",
                "filename": "PNEUMONIA.jpeg",
                "split": "train",
                "label": "PNEUMONIA",
                "group_id": "b",
                "new_split": "train",
            },
        ]
    ).to_csv(manifest, index=False)

    ds = ManifestDataset(manifest, "train", data_root=root)
    image, label, metadata = ds[0]
    assert image.size == (40, 20)
    assert label == 0
    assert metadata["image_path"] == "NORMAL.jpeg"
    assert compute_pos_weight_from_train(ds) == 1.0
