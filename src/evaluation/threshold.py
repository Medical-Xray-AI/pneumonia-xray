"""Decision-threshold selection, restricted to validation data by construction.

Project rule (`docs/decisions.md`): the operating threshold is chosen on
validation data only. Selecting it on the locked test set would leak final
evaluation information and inflate the reported result.

This module cannot enforce that rule by itself - it only sees arrays - so the
enforcement lives one level up, in `scripts/evaluate_model.py`, which refuses
to search for a threshold when it is run in test mode and instead demands a
frozen value. The docstring convention here is that every function taking
`y_true`/`y_prob` in this module is a *validation* function.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from src.evaluation.metrics import (
    _as_label_array,
    _as_probability_array,
    apply_threshold,
    macro_f1,
    pneumonia_sensitivity,
    specificity,
)

# Objective the threshold search maximises by default. Macro F1 is the
# project's primary metric, so the operating point is chosen for the same
# quantity the models are compared on.
DEFAULT_OBJECTIVE = "macro_f1"

OBJECTIVES = {
    "macro_f1": macro_f1,
    "pneumonia_sensitivity": pneumonia_sensitivity,
    "balanced_accuracy": lambda t, p: float(
        np.mean([pneumonia_sensitivity(t, p), specificity(t, p)])
    ),
}


@dataclass(frozen=True)
class ThresholdSelection:
    """The chosen operating point plus the evidence behind it."""

    threshold: float
    objective: str
    objective_value: float
    macro_f1: float
    pneumonia_sensitivity: float
    specificity: float
    n_candidates: int
    n_tied: int
    selection_rule: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def candidate_thresholds(y_prob: Sequence[float]) -> np.ndarray:
    """Deterministic candidate set: every distinct achievable operating point.

    Sweeping a fixed grid (0.01, 0.02, ...) can step straight over the optimum
    when probabilities cluster, so instead we use the sorted unique predicted
    probabilities. Under the project's `probability >= threshold` rule, each
    such value is the exact point at which one more image flips to pneumonia,
    which makes this set both minimal and exhaustive.

    0.0 and 1.0 are appended so the degenerate all-pneumonia and all-normal
    operating points are always considered.
    """
    arr = np.asarray(y_prob, dtype=float)
    values = np.unique(np.concatenate([arr, [0.0, 1.0]]))
    return np.sort(values)


def sweep_thresholds(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    objective: str = DEFAULT_OBJECTIVE,
) -> List[Dict[str, float]]:
    """Evaluate every candidate threshold on VALIDATION data.

    Returns one record per candidate, in ascending threshold order, suitable
    for both `select_threshold` and the threshold-sweep plot.
    """
    if objective not in OBJECTIVES:
        raise ValueError(
            f"unknown objective '{objective}'; expected one of {sorted(OBJECTIVES)}"
        )

    truth = _as_label_array(y_true)
    prob = _as_probability_array(y_prob, truth.size)
    score_fn = OBJECTIVES[objective]

    records: List[Dict[str, float]] = []
    for threshold in candidate_thresholds(prob):
        pred = apply_threshold(prob, threshold)
        records.append(
            {
                "threshold": float(threshold),
                "objective_value": float(score_fn(truth, pred)),
                "macro_f1": macro_f1(truth, pred),
                "pneumonia_sensitivity": pneumonia_sensitivity(truth, pred),
                "specificity": specificity(truth, pred),
            }
        )
    return records


def select_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    objective: str = DEFAULT_OBJECTIVE,
) -> ThresholdSelection:
    """Choose the operating threshold from VALIDATION labels and probabilities.

    Selection rule, applied in order so the outcome is fully deterministic and
    independent of input row order:

    1. Maximise the objective (default: macro F1).
    2. Break ties by higher pneumonia sensitivity - a missed pneumonia is the
       costlier error in this domain, so among equally good operating points
       we prefer the one that misses fewer cases.
    3. Break remaining ties by the lower threshold, which is the more
       conservative choice for the same reason.

    Re-running this function on identical inputs always returns an identical
    threshold; there is no randomness and no dependence on floating-point
    accumulation order.
    """
    records = sweep_thresholds(y_true, y_prob, objective=objective)

    best_value = max(record["objective_value"] for record in records)
    tied = [
        record
        for record in records
        if np.isclose(record["objective_value"], best_value, rtol=0.0, atol=1e-12)
    ]

    # Rule 2 then rule 3. `min` over a negated sensitivity keeps a single pass
    # and makes the ordering explicit rather than relying on sort stability.
    best = min(tied, key=lambda record: (-record["pneumonia_sensitivity"], record["threshold"]))

    return ThresholdSelection(
        threshold=float(best["threshold"]),
        objective=objective,
        objective_value=float(best["objective_value"]),
        macro_f1=float(best["macro_f1"]),
        pneumonia_sensitivity=float(best["pneumonia_sensitivity"]),
        specificity=float(best["specificity"]),
        n_candidates=len(records),
        n_tied=len(tied),
        selection_rule=(
            f"argmax {objective}; ties -> max pneumonia_sensitivity; "
            f"ties -> min threshold"
        ),
    )
