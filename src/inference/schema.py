"""Typed output contract for checkpoint-based inference.

One ``PredictionRecord`` describes one image. Manifest exports keep the
evaluation prediction-file columns (``image_path``, ``label``, ``probability``,
``model``, ``run_id``, ``split``) so ``scripts/evaluate_model.py`` consumes them
unchanged. Thresholded fields are filled only when a frozen threshold is given.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

LABEL_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}
MODEL_NAMES = {"small_cnn", "densenet121"}
SPLITS = {"train", "validation", "test", "external"}

# Column order of every exported prediction CSV.
OUTPUT_COLUMNS = [
    "run_id",
    "model",
    "checkpoint_epoch",
    "split",
    "image_path",
    "label",
    "probability",
    "threshold",
    "predicted_label",
    "predicted_class",
]


class InferenceSchemaError(ValueError):
    """Raised when an inference output violates the contract."""


def validate_threshold(value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise InferenceSchemaError(f"threshold must be a number, got {value!r}") from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise InferenceSchemaError(f"threshold must be finite and in [0, 1], got {value!r}")
    return threshold


def predict_label(probability: float, threshold: float) -> int:
    """Project-wide decision rule: pneumonia when ``probability >= threshold``."""
    return int(probability >= threshold)


@dataclass(frozen=True)
class PredictionRecord:
    image_path: str
    probability: float
    model: str
    run_id: str
    split: str = "external"
    label: Optional[int] = None
    threshold: Optional[float] = None
    predicted_label: Optional[int] = None
    predicted_class: Optional[str] = None
    checkpoint_epoch: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.image_path, str) or not self.image_path.strip():
            raise InferenceSchemaError("image_path must be a nonempty string")
        probability = float(self.probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise InferenceSchemaError(
                f"probability must be in [0, 1] (sigmoid output), got {self.probability!r}"
            )
        if self.model not in MODEL_NAMES:
            raise InferenceSchemaError(f"unknown model {self.model!r}")
        if not str(self.run_id).strip():
            raise InferenceSchemaError("run_id must be nonempty")
        if self.split not in SPLITS:
            raise InferenceSchemaError(f"unknown split {self.split!r}")
        if self.label is not None and self.label not in (0, 1):
            raise InferenceSchemaError(f"label must be 0 or 1, got {self.label!r}")
        if self.threshold is None:
            if self.predicted_label is not None or self.predicted_class is not None:
                raise InferenceSchemaError("predicted_label requires a threshold")
            return
        threshold = validate_threshold(self.threshold)
        expected = predict_label(probability, threshold)
        if self.predicted_label != expected:
            raise InferenceSchemaError(
                f"predicted_label {self.predicted_label!r} disagrees with "
                f"probability {probability:.6f} >= threshold {threshold:.6f}"
            )
        if self.predicted_class != LABEL_NAMES[expected]:
            raise InferenceSchemaError("predicted_class does not match predicted_label")

    @classmethod
    def build(
        cls,
        *,
        image_path: str,
        probability: float,
        model: str,
        run_id: str,
        split: str = "external",
        label: Optional[int] = None,
        threshold: Optional[float] = None,
        checkpoint_epoch: Optional[int] = None,
    ) -> "PredictionRecord":
        """Derive the thresholded fields instead of trusting the caller."""
        predicted = None if threshold is None else predict_label(float(probability), float(threshold))
        return cls(
            image_path=image_path,
            probability=float(probability),
            model=model,
            run_id=run_id,
            split=split,
            label=None if label is None else int(label),
            threshold=None if threshold is None else float(threshold),
            predicted_label=predicted,
            predicted_class=None if predicted is None else LABEL_NAMES[predicted],
            checkpoint_epoch=checkpoint_epoch,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def records_to_frame(records: Iterable[PredictionRecord]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [record.to_dict() for record in records]
    if not rows:
        raise InferenceSchemaError("no predictions were produced")
    frame = pd.DataFrame(rows)
    frame = frame[OUTPUT_COLUMNS]
    # Columns that are entirely unset carry no information in the CSV.
    return frame.dropna(axis=1, how="all")
