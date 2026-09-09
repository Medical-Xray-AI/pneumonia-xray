"""Tests for validation-only, deterministic threshold selection."""

import numpy as np
import pytest

from src.evaluation.metrics import macro_f1, apply_threshold
from src.evaluation.threshold import (
    candidate_thresholds,
    select_threshold,
    sweep_thresholds,
)


def _synthetic(seed: int = 42, n: int = 300):
    """Separable-but-noisy validation-like predictions."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.72).astype(int)
    p = np.clip(rng.normal(np.where(y == 1, 0.70, 0.30), 0.18), 0.0, 1.0)
    return y, p


def test_selection_is_deterministic_across_repeated_calls():
    y, p = _synthetic()
    first = select_threshold(y, p)
    for _ in range(5):
        assert select_threshold(y, p).threshold == first.threshold


def test_selection_is_invariant_to_row_order():
    """Shuffling the input rows must not move the operating point - otherwise
    the frozen threshold would depend on DataLoader ordering."""
    y, p = _synthetic()
    baseline = select_threshold(y, p).threshold

    rng = np.random.default_rng(7)
    for _ in range(3):
        order = rng.permutation(len(y))
        assert select_threshold(y[order], p[order]).threshold == pytest.approx(baseline)


def test_selected_threshold_maximises_macro_f1_over_the_candidate_grid():
    y, p = _synthetic()
    selection = select_threshold(y, p)
    best_possible = max(
        macro_f1(y, apply_threshold(p, t)) for t in candidate_thresholds(p)
    )
    assert selection.macro_f1 == pytest.approx(best_possible)


def test_tie_break_prefers_higher_sensitivity_then_lower_threshold():
    """Two thresholds give identical macro F1 here; the clinical tie-break must
    pick the one that misses fewer pneumonia cases."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.10, 0.20, 0.80, 0.90])
    selection = select_threshold(y, p)

    assert selection.macro_f1 == pytest.approx(1.0)
    # Every threshold in (0.20, 0.80] is perfect; the lowest such candidate is
    # 0.80 in the candidate set, and no perfect candidate has higher sensitivity.
    assert selection.pneumonia_sensitivity == pytest.approx(1.0)
    assert selection.threshold == pytest.approx(0.80)
    assert selection.n_tied >= 1


def test_perfectly_separable_data_reaches_macro_f1_of_one():
    y = np.array([0] * 20 + [1] * 20)
    p = np.concatenate([np.linspace(0.01, 0.30, 20), np.linspace(0.70, 0.99, 20)])
    assert select_threshold(y, p).macro_f1 == pytest.approx(1.0)


def test_sweep_covers_degenerate_endpoints():
    y, p = _synthetic()
    records = sweep_thresholds(y, p)
    thresholds = [r["threshold"] for r in records]

    assert thresholds == sorted(thresholds)
    assert thresholds[0] == pytest.approx(0.0)
    assert thresholds[-1] == pytest.approx(1.0)
    # Threshold 0.0 predicts pneumonia for everything -> perfect sensitivity.
    assert records[0]["pneumonia_sensitivity"] == pytest.approx(1.0)


def test_alternative_objective_changes_the_operating_point():
    y, p = _synthetic()
    by_f1 = select_threshold(y, p, objective="macro_f1")
    by_sensitivity = select_threshold(y, p, objective="pneumonia_sensitivity")
    assert by_sensitivity.pneumonia_sensitivity >= by_f1.pneumonia_sensitivity


def test_unknown_objective_is_rejected():
    y, p = _synthetic()
    with pytest.raises(ValueError, match="unknown objective"):
        select_threshold(y, p, objective="accuracy_at_all_costs")


def test_selection_record_documents_its_own_rule():
    """The frozen threshold has to be explainable in the report without
    re-reading the source."""
    y, p = _synthetic()
    selection = select_threshold(y, p)
    assert "argmax macro_f1" in selection.selection_rule
    assert selection.objective == "macro_f1"
    assert selection.n_candidates > 1
    assert set(selection.to_dict()) >= {"threshold", "macro_f1", "selection_rule"}
