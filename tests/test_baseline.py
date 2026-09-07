"""
tests/test_baseline.py

Automated checks for Member 2's deliverables. Run with:

    pytest tests/test_baseline.py -v

These are exactly the "Gate 3: smoke tests" items from
docs/TEAM_WORKFLOW.md that fall under Member 2's ownership:
    - "Baseline ... forward pass works"
    - image/label shapes match the model contract

No dataset or GPU is required — everything here runs on dummy tensors
or synthetic in-memory images, so this file can (and should) pass in CI.
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline import BaselineCNN, build_baseline_from_config  # noqa: E402
from src.preprocessing.transforms import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    AspectPreservingResize,
    build_eval_transforms,
    build_no_augmentation_train_transforms,
    build_train_transforms,
    compute_dataset_statistics,
    resolve_normalization_stats,
)


def test_baseline_forward_pass_shape():
    model = BaselineCNN(output_units=1)
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (4, 1)


def test_baseline_from_config_respects_output_units():
    model = build_baseline_from_config({"output_units": 1, "dropout": 0.3})
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1)


def test_baseline_param_count_is_small():
    model = BaselineCNN()
    # sanity ceiling from the project brief: baseline should be a small,
    # cheap model, not a second pretrained backbone in disguise.
    assert model.num_parameters() < 5_000_000


def test_aspect_preserving_resize_produces_square_output():
    img = Image.new("RGB", (640, 320), color=(10, 20, 30))
    resize = AspectPreservingResize(size=224)
    out = resize(img)
    assert out.size == (224, 224)


def test_eval_transforms_output_tensor_shape():
    data_cfg = {"image": {"size": 224, "model_channels": 3}}
    tfms = build_eval_transforms(data_cfg)
    img = Image.new("L", (500, 300), color=128)
    tensor = tfms(img)
    assert tensor.shape == (3, 224, 224)


def test_train_transforms_are_deterministic_when_augmentation_disabled():
    data_cfg = {"image": {"size": 224, "model_channels": 3}}
    aug_cfg = {
        "horizontal_flip_probability": 0.0,
        "rotation_degrees": 0,
        "translation_fraction": 0.0,
        "brightness_contrast_probability": 0.0,
    }
    tfms = build_train_transforms(data_cfg, aug_cfg)
    img = Image.new("L", (400, 400), color=200)
    out1, out2 = tfms(img), tfms(img)
    assert torch.equal(out1, out2)


# --------------------------------------------------------------------------
# Regression tests for the normalization-hardcode fix
# (configs/baseline.yaml sets `normalization.type: dataset_statistics`,
# which was previously silently ignored in favour of ImageNet stats).
# --------------------------------------------------------------------------
def test_resolve_normalization_stats_imagenet_type():
    mean, std = resolve_normalization_stats({"type": "imagenet"})
    assert mean == IMAGENET_MEAN
    assert std == IMAGENET_STD


def test_resolve_normalization_stats_dataset_statistics_requires_source():
    """
    If normalization.type is 'dataset_statistics' and there's neither a
    cache file nor a way to compute one, this must raise -- NOT silently
    fall back to ImageNet stats (that silent fallback was the original bug).
    """
    import pytest

    with pytest.raises(ValueError):
        resolve_normalization_stats(
            {"type": "dataset_statistics"},
            stats_cache_path=Path("/tmp/definitely_missing_stats_file.json"),
            train_dataset_builder=None,
        )


def test_resolve_normalization_stats_dataset_statistics_uses_cache(tmp_path):
    cache_path = tmp_path / "dataset_stats.json"
    cache_path.write_text(json.dumps({"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}))

    mean, std = resolve_normalization_stats(
        {"type": "dataset_statistics"},
        stats_cache_path=cache_path,
        train_dataset_builder=None,  # must not be needed when cache exists
    )
    assert mean == [0.1, 0.2, 0.3]
    assert std == [0.4, 0.5, 0.6]


class _ConstantColorDataset(Dataset):
    """Every sample is a solid-color 4x4 tensor -> known, hand-checkable mean/std."""

    def __init__(self, value: float, n: int = 8):
        self.value = value
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.full((3, 4, 4), self.value), 0


def test_compute_dataset_statistics_matches_known_constant_input():
    ds = _ConstantColorDataset(value=0.5, n=8)
    mean, std = compute_dataset_statistics(ds, batch_size=4, num_workers=0)
    for m in mean:
        assert abs(m - 0.5) < 1e-6
    for s in std:
        assert abs(s - 0.0) < 1e-6  # constant input -> zero variance


class _DictBatchDataset(Dataset):
    """
    Mimics the REAL src/data/dataset.py ChestXrayDataset: __getitem__
    returns a dict, not a (image, label) tuple. Regression test for the
    bug where compute_dataset_statistics() only worked against a
    tuple-yielding dataset and crashed on the real one with
    "ValueError: too many values to unpack".
    """

    def __init__(self, value: float, n: int = 8):
        self.value = value
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"image": torch.full((3, 4, 4), self.value), "label": torch.tensor(0.0)}


def test_compute_dataset_statistics_works_with_dict_batches():
    ds = _DictBatchDataset(value=0.5, n=8)
    mean, std = compute_dataset_statistics(ds, batch_size=4, num_workers=0)
    for m in mean:
        assert abs(m - 0.5) < 1e-6
    for s in std:
        assert abs(s - 0.0) < 1e-6


class _FrameOnlyDataset:
    """
    Mimics the REAL ChestXrayDataset's public surface for label counting:
    a `.frame` DataFrame with an int 0/1 `label` column, and deliberately
    NO class_counts() method. Regression test for the bug where
    compute_pos_weight() assumed every dataset has class_counts()
    (only true of the old ManifestDataset fallback) and crashed with
    "AttributeError: 'ChestXrayDataset' object has no attribute
    'class_counts'" against the real dataset.
    """

    def __init__(self, n_normal: int, n_pneumonia: int):
        import pandas as pd

        self.frame = pd.DataFrame(
            {"label": [0] * n_normal + [1] * n_pneumonia}
        )


def test_compute_pos_weight_works_without_class_counts():
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_baseline import compute_pos_weight  # noqa: E402

    ds = _FrameOnlyDataset(n_normal=30, n_pneumonia=10)
    pos_weight = compute_pos_weight(ds)
    assert abs(pos_weight.item() - 3.0) < 1e-6  # 30 normal / 10 pneumonia


def test_color_jitter_strengths_come_from_config():
    from torchvision import transforms

    data_cfg = {"image": {"size": 224, "model_channels": 3}}
    aug_cfg = {
        "horizontal_flip_probability": 0.0,
        "rotation_degrees": 0,
        "translation_fraction": 0.0,
        "brightness_contrast_probability": 1.0,
        "brightness": 0.07,
        "contrast": 0.11,
    }
    tfms = build_train_transforms(data_cfg, aug_cfg)
    random_apply = next(t for t in tfms.transforms if isinstance(t, transforms.RandomApply))
    jitter = random_apply.transforms[0]
    import pytest
    assert jitter.brightness == pytest.approx((0.93, 1.07))
    assert jitter.contrast == pytest.approx((0.89, 1.11))


def test_no_augmentation_transform_is_deterministic():
    data_cfg = {"image": {"size": 224, "model_channels": 3}}
    tfms = build_no_augmentation_train_transforms(data_cfg)
    img = Image.new("L", (411, 287), color=173)
    assert torch.equal(tfms(img), tfms(img))


def test_normalization_cache_metadata_mismatch_recomputes(tmp_path):
    cache_path = tmp_path / "dataset_stats.json"
    cache_path.write_text(json.dumps({
        "mean": [0.1, 0.1, 0.1],
        "std": [0.2, 0.2, 0.2],
        "metadata": {"train_manifest_sha256": "old", "split_version": "split_v0", "num_train_images": 10},
    }))

    expected = {"train_manifest_sha256": "new", "split_version": "split_v1", "num_train_images": 8}
    mean, std = resolve_normalization_stats(
        {"type": "dataset_statistics"},
        stats_cache_path=cache_path,
        train_dataset_builder=lambda: _ConstantColorDataset(value=0.5, n=8),
        expected_cache_metadata=expected,
        cache_metadata=expected,
    )
    assert all(abs(m - 0.5) < 1e-6 for m in mean)
    payload = json.loads(cache_path.read_text())
    assert payload["metadata"] == expected


def test_b0_forces_no_augmentation_and_no_class_weight():
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_baseline import effective_experiment_flags

    assert effective_experiment_flags("overfit", False, False) == (False, False)
    assert effective_experiment_flags("overfit", True, True) == (False, False)
    assert effective_experiment_flags("full", False, False) == (True, True)
    assert effective_experiment_flags("full", True, False) == (False, True)
