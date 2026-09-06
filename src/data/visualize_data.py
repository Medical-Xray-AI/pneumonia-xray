"""Create report-safe data figures from a canonical manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from .common import CANONICAL_COLUMNS, resolve_dataset_root, resolve_image_path


SPLIT_ORDER = ["train", "validation", "test"]
LABEL_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


def _load_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = sorted(set(CANONICAL_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    return frame


def plot_class_distribution(frame: pd.DataFrame, output_path: Path) -> None:
    counts = (
        frame.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(SPLIT_ORDER, fill_value=0)
    )
    counts = counts.rename(columns=LABEL_NAMES)
    axis = counts.plot(kind="bar", color=["#2878B5", "#F28E2B"], figsize=(9, 5))
    axis.set_title("Class distribution per frozen split")
    axis.set_xlabel("Split")
    axis.set_ylabel("Number of images")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Class")
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=160)
    plt.close(axis.figure)


def plot_sample_grid(
    frame: pd.DataFrame,
    root: Path,
    output_path: Path,
    samples_per_class: int = 4,
    seed: int = 42,
) -> None:
    figure, axes = plt.subplots(2, samples_per_class, figsize=(3 * samples_per_class, 6))
    for row_index, label in enumerate((0, 1)):
        candidates = frame.loc[(frame["split"] == "train") & (frame["label"] == label)]
        if len(candidates) < samples_per_class:
            raise ValueError(f"Not enough training examples for label {label}")
        sampled = candidates.sample(samples_per_class, random_state=seed + label)
        for column_index, record in enumerate(sampled.itertuples(index=False)):
            path = resolve_image_path(root, record.image_path)
            with Image.open(path) as opened:
                image = opened.convert("L")
                image.load()
            axis = axes[row_index, column_index]
            axis.imshow(image, cmap="gray")
            axis.axis("off")
            if column_index == 0:
                axis.set_title(LABEL_NAMES[label], loc="left", fontsize=11, fontweight="bold")
    figure.suptitle("Deterministic training samples by class")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_image_sizes(frame: pd.DataFrame, root: Path, output_path: Path) -> None:
    widths: list[int] = []
    heights: list[int] = []
    for relative_path in frame["image_path"]:
        with Image.open(resolve_image_path(root, str(relative_path))) as image:
            width, height = image.size
        widths.append(width)
        heights.append(height)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(widths, heights, s=8, alpha=0.25, color="#2878B5")
    axis.set_title("Original image dimensions")
    axis.set_xlabel("Width (pixels)")
    axis.set_ylabel("Height (pixels)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def create_figures(
    manifest_path: str | Path,
    output_dir: str | Path,
    data_root: str | Path | None = None,
    seed: int = 42,
) -> None:
    frame = _load_manifest(manifest_path)
    root = resolve_dataset_root(data_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(frame, output / "class_distribution.png")
    plot_sample_grid(frame, root, output / "sample_grid.png", seed=seed)
    plot_image_sizes(frame, root, output / "image_size_distribution.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/split_manifest.csv")
    parser.add_argument("--output-dir", default="audit_out/figures")
    parser.add_argument("--data-root", default=None, help="Defaults to XRAY_DATA_ROOT")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    create_figures(args.manifest, args.output_dir, args.data_root, args.seed)


if __name__ == "__main__":
    main()
