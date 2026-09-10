import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.training.config import load_config, save_resolved_config


@pytest.fixture
def training_config(tmp_path, monkeypatch):
    root = tmp_path / "images"
    for folder in ("train", "val", "test"):
        (root / folder).mkdir(parents=True)
    manifests = {}
    for split, folder in (("train", "train"), ("validation", "val")):
        rows = []
        for i in range(4):
            label = i % 2
            name = f"{folder}/{'PNEUMONIA' if label else 'NORMAL'}/{split}_{i}.png"
            path = root / name
            path.parent.mkdir(exist_ok=True)
            pixels = np.random.default_rng(i + (100 if split == "validation" else 0)).integers(0, 20, (48, 64), dtype=np.uint8)
            pixels += 180 if label else 20
            Image.fromarray(pixels).save(path)
            rows.append({"image_path": name, "label": label, "patient_id": f"{split}_{i}",
                         "group_id": f"{split}_{i}", "pneumonia_subtype": "bacterial" if label else "normal",
                         "split": split, "source_split": f"provided_{folder}",
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        manifest = tmp_path / f"{split}.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        manifests[split] = str(manifest)
    monkeypatch.setenv("XRAY_DATA_ROOT", str(root))
    monkeypatch.setenv("XRAY_OUTPUT_ROOT", str(tmp_path / "outputs"))
    cfg = load_config(Path(__file__).resolve().parents[2] / "configs/baseline.yaml")
    cfg["data"]["manifests"] = manifests
    cfg["data"]["loader"] = {"num_workers": 0}
    cfg["training"].update(epochs=2, batch_size=2, amp=False)
    cfg["augmentation"] = {}
    cfg["model"]["dropout"] = 0.1
    path = tmp_path / "config.yaml"
    save_resolved_config(cfg, path)
    return cfg, path, tmp_path / "outputs"
