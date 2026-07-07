"""Unit tests for evalkit.bootstrap.paired_bootstrap -- pure statistics, no
model calls."""
from __future__ import annotations

from deep_research_toolkit.evalkit.bootstrap import paired_bootstrap


def test_all_positive_deltas_are_significant():
    deltas = [0.1] * 20
    out = paired_bootstrap(deltas, b=500, seed=7)
    assert out["mean"] == 0.1
    lo, hi = out["ci95"]
    assert lo > 0
    assert out["significant"] is True


def test_zero_mean_mixed_deltas_are_not_significant():
    deltas = [0.2, -0.2] * 10
    out = paired_bootstrap(deltas, b=500, seed=7)
    assert out["mean"] == 0.0
    lo, hi = out["ci95"]
    assert lo <= 0 <= hi
    assert out["significant"] is False


def test_ci_bounds_are_ordered_and_contain_mean_region():
    deltas = [0.05, 0.06, 0.04, 0.05, 0.07, 0.03, 0.05, 0.06]
    out = paired_bootstrap(deltas, b=1000, seed=7)
    lo, hi = out["ci95"]
    assert lo <= hi


def test_empty_deltas_returns_zero_mean_and_not_significant():
    out = paired_bootstrap([], b=200, seed=7)
    assert out["mean"] == 0.0
    assert out["significant"] is False


def test_same_seed_is_deterministic():
    deltas = [0.1, -0.05, 0.2, 0.0, 0.15]
    out1 = paired_bootstrap(deltas, b=300, seed=7)
    out2 = paired_bootstrap(deltas, b=300, seed=7)
    assert out1 == out2


def test_different_seed_can_change_ci_but_not_mean():
    deltas = [0.1, -0.05, 0.2, 0.0, 0.15]
    out1 = paired_bootstrap(deltas, b=300, seed=7)
    out2 = paired_bootstrap(deltas, b=300, seed=99)
    assert out1["mean"] == out2["mean"]
