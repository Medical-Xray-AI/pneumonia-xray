"""
Preprocessing and augmentation pipelines for the pediatric chest X-ray
baseline, driven by configs/data.yaml + configs/baseline.yaml.

Contract:
    - Stochastic augmentation is applied to TRAINING data only.
    - Resize preserves aspect ratio and pads to a square.
    - Grayscale images are converted to the configured number of channels.
    - Validation/test preprocessing is deterministic.
    - Dataset normalization statistics are computed from TRAIN split only.
    - All transform objects are Windows DataLoader multiprocessing-safe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Normalization constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def compute_dataset_statistics(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
) -> tuple[list[float], list[float]]:
    """
    Compute per-channel mean/std from the TRAINING dataset.

    The dataset must already:
      - resize the image
      - convert to the correct number of channels
      - convert to Tensor

    It must NOT already apply Normalize().
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sumsq = torch.zeros(3, dtype=torch.float64)

    n_pixels = 0
    seen = 0

    for batch in loader:

        # ChestXrayDataset returns a dictionary.
        # Support tuple-style datasets too for unit tests.
        if isinstance(batch, dict):
            images = batch["image"]
        else:
            images = batch[0]

        images = images.double()

        batch_size_actual, _, height, width = images.shape

        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sumsq += (images ** 2).sum(dim=(0, 2, 3))

        n_pixels += batch_size_actual * height * width
        seen += batch_size_actual

        if max_samples is not None and seen >= max_samples:
            break

    if n_pixels == 0:
        raise ValueError(
            "compute_dataset_statistics() received an empty dataset."
        )

    mean = channel_sum / n_pixels

    variance = (
        channel_sumsq / n_pixels
    ) - mean ** 2

    variance = variance.clamp(min=0)

    std = torch.sqrt(variance)

    return mean.tolist(), std.tolist()


def resolve_normalization_stats(
    normalization_cfg: dict,
    stats_cache_path: Optional[Path] = None,
    train_dataset_builder: Optional[Callable[[], Dataset]] = None,
    expected_cache_metadata: Optional[dict[str, Any]] = None,
    cache_metadata: Optional[dict[str, Any]] = None,
) -> tuple[list[float], list[float]]:
    """
    Resolve normalization statistics according to configuration.

    Supported normalization types:

        imagenet
        dataset_statistics

    For dataset_statistics, cached values are reused only if their metadata
    matches the expected training-manifest metadata.
    """

    norm_type = normalization_cfg.get(
        "type",
        "imagenet",
    )

    # ------------------------------------------------------------------
    # ImageNet normalization
    # ------------------------------------------------------------------

    if norm_type == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD

    # ------------------------------------------------------------------
    # Dataset statistics
    # ------------------------------------------------------------------

    if norm_type == "dataset_statistics":

        if stats_cache_path is not None:

            stats_cache_path = Path(stats_cache_path)

            if stats_cache_path.is_file():

                with open(
                    stats_cache_path,
                    "r",
                    encoding="utf-8",
                ) as file:
                    cached = json.load(file)

                metadata_ok = True

                if expected_cache_metadata is not None:

                    cached_metadata = cached.get(
                        "metadata",
                        {},
                    )

                    metadata_ok = all(
                        cached_metadata.get(key) == value
                        for key, value
                        in expected_cache_metadata.items()
                    )

                if metadata_ok:
                    return (
                        cached["mean"],
                        cached["std"],
                    )

        # Cache doesn't exist or is stale.
        if train_dataset_builder is None:

            reason = "no valid cached normalization statistics were found"

            if expected_cache_metadata is not None:
                reason += " because the cache is missing or metadata differs"

            raise ValueError(
                "normalization.type='dataset_statistics', but "
                f"{reason}. "
                "Provide train_dataset_builder or use "
                "normalization.type='imagenet'."
            )

        train_dataset = train_dataset_builder()

        mean, std = compute_dataset_statistics(
            train_dataset,
            num_workers=0,
        )

        if stats_cache_path is not None:

            stats_cache_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            payload = {
                "mean": mean,
                "std": std,
                "source": "computed_from_training_split",
            }

            metadata = dict(
                cache_metadata
                or expected_cache_metadata
                or {}
            )

            if metadata:
                payload["metadata"] = metadata

            with open(
                stats_cache_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    payload,
                    file,
                    indent=2,
                )

        return mean, std

    raise ValueError(
        f"Unknown normalization.type: {norm_type!r}"
    )


# ---------------------------------------------------------------------------
# Windows-safe transform objects
# ---------------------------------------------------------------------------

class AspectPreservingResize:
    """
    Resize an image while preserving aspect ratio.

    The longest side becomes `size`; the remaining space is padded with
    black pixels until the result is size x size.
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(
        self,
        img: Image.Image,
    ) -> Image.Image:

        width, height = img.size

        scale = self.size / max(
            width,
            height,
        )

        new_width = max(
            1,
            round(width * scale),
        )

        new_height = max(
            1,
            round(height * scale),
        )

        img = img.resize(
            (new_width, new_height),
            Image.BILINEAR,
        )

        pad_width = self.size - new_width
        pad_height = self.size - new_height

        left = pad_width // 2
        top = pad_height // 2

        canvas = Image.new(
            img.mode,
            (self.size, self.size),
            color=0,
        )

        canvas.paste(
            img,
            (left, top),
        )

        return canvas


class ConvertChannels:
    """
    Convert a PIL image to the configured model channel count.

    IMPORTANT:
    This is a top-level callable class instead of transforms.Lambda(lambda...).

    Windows DataLoader workers use multiprocessing spawn and therefore need
    transform objects to be pickleable. Local lambdas are not pickleable.
    """

    def __init__(
        self,
        model_channels: int,
    ):
        if model_channels not in (1, 3):
            raise ValueError(
                "model_channels must be either 1 or 3."
            )

        self.model_channels = model_channels

    def __call__(
        self,
        img: Image.Image,
    ) -> Image.Image:

        if self.model_channels == 3:
            return img.convert("RGB")

        return img.convert("L")


def _channels_transform(
    model_channels: int,
) -> ConvertChannels:
    """
    Return a Windows multiprocessing-safe channel conversion transform.

    Do NOT replace this with:
        transforms.Lambda(lambda ...)

    because local lambda functions cannot be pickled by Windows DataLoader
    worker processes.
    """

    return ConvertChannels(
        model_channels=model_channels,
    )


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def build_stats_transforms(
    data_cfg: dict,
) -> transforms.Compose:
    """
    Transform used ONLY while computing TRAIN-set mean/std.

    No augmentation.
    No normalization.

    Pipeline:
        resize
        channel conversion
        ToTensor
    """

    size = data_cfg["image"]["size"]

    model_channels = data_cfg["image"].get(
        "model_channels",
        3,
    )

    return transforms.Compose(
        [
            AspectPreservingResize(size),
            _channels_transform(model_channels),
            transforms.ToTensor(),
        ]
    )


def build_train_transforms(
    data_cfg: dict,
    aug_cfg: dict,
    mean: Optional[list] = None,
    std: Optional[list] = None,
) -> transforms.Compose:
    """
    Training preprocessing pipeline.

    Pipeline:
        aspect-preserving resize
        optional TRAIN-ONLY augmentation
        channel conversion
        ToTensor
        Normalize

    Augmentation parameters are config-driven.
    """

    size = data_cfg["image"]["size"]

    model_channels = data_cfg["image"].get(
        "model_channels",
        3,
    )

    norm_mean = (
        mean
        if mean is not None
        else IMAGENET_MEAN
    )

    norm_std = (
        std
        if std is not None
        else IMAGENET_STD
    )

    rotation_degrees = aug_cfg.get(
        "rotation_degrees",
        0,
    )

    translation_fraction = aug_cfg.get(
        "translation_fraction",
        0.0,
    )

    horizontal_flip_probability = aug_cfg.get(
        "horizontal_flip_probability",
        0.0,
    )

    brightness_contrast_probability = aug_cfg.get(
        "brightness_contrast_probability",
        0.0,
    )

    brightness = aug_cfg.get(
        "brightness",
        0.0,
    )

    contrast = aug_cfg.get(
        "contrast",
        0.0,
    )

    pipeline = [
        AspectPreservingResize(size)
    ]

    # ---------------------------------------------------------------
    # Horizontal flip
    # ---------------------------------------------------------------

    if horizontal_flip_probability > 0:

        pipeline.append(
            transforms.RandomHorizontalFlip(
                p=horizontal_flip_probability
            )
        )

    # ---------------------------------------------------------------
    # Conservative geometric augmentation
    # ---------------------------------------------------------------

    if (
        rotation_degrees > 0
        or translation_fraction > 0
    ):

        translate = None

        if translation_fraction > 0:
            translate = (
                translation_fraction,
                translation_fraction,
            )

        pipeline.append(
            transforms.RandomAffine(
                degrees=rotation_degrees,
                translate=translate,
            )
        )

    # ---------------------------------------------------------------
    # Brightness / contrast augmentation
    # ---------------------------------------------------------------

    if brightness_contrast_probability > 0:

        jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
        )

        pipeline.append(
            transforms.RandomApply(
                [jitter],
                p=brightness_contrast_probability,
            )
        )

    # ---------------------------------------------------------------
    # Final deterministic preprocessing
    # ---------------------------------------------------------------

    pipeline.extend(
        [
            _channels_transform(
                model_channels
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=norm_mean,
                std=norm_std,
            ),
        ]
    )

    return transforms.Compose(
        pipeline
    )


def build_eval_transforms(
    data_cfg: dict,
    mean: Optional[list] = None,
    std: Optional[list] = None,
) -> transforms.Compose:
    """
    Deterministic transform for VALIDATION and TEST.

    No stochastic augmentation is allowed here.
    """

    size = data_cfg["image"]["size"]

    model_channels = data_cfg["image"].get(
        "model_channels",
        3,
    )

    norm_mean = (
        mean
        if mean is not None
        else IMAGENET_MEAN
    )

    norm_std = (
        std
        if std is not None
        else IMAGENET_STD
    )

    return transforms.Compose(
        [
            AspectPreservingResize(size),
            _channels_transform(
                model_channels
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=norm_mean,
                std=norm_std,
            ),
        ]
    )


def build_no_augmentation_train_transforms(
    data_cfg: dict,
    mean: Optional[list] = None,
    std: Optional[list] = None,
) -> transforms.Compose:
    """
    Training transform without stochastic augmentation.

    Used for:
        B0 — small-subset overfit sanity test
        B2 — augmentation ON vs OFF comparison

    It intentionally uses the same deterministic preprocessing as evaluation.
    """

    return build_eval_transforms(
        data_cfg,
        mean=mean,
        std=std,
    )