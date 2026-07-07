"""Unit tests for evalkit.flake: Wilson-interval sanity + run_flaky counting.

Pure statistics helpers for the irreducibly stochastic live-model tier -- no
backend, no network, so these run in the fast suite."""
from __future__ import annotations

from deep_research_toolkit.evalkit.flake import run_flaky, wilson_interval


def test_wilson_interval_perfect_score_is_narrow_and_bounded():
    lo, hi = wilson_interval(5, 5)
    assert 0.0 < lo <= 1.0
    assert hi == 1.0
    assert lo < hi


def test_wilson_interval_zero_score_is_narrow_and_bounded():
    lo, hi = wilson_interval(0, 5)
    assert lo == 0.0
    assert 0.0 <= hi < 1.0


def test_wilson_interval_contains_the_observed_rate():
    successes, n = 3, 5
    lo, hi = wilson_interval(successes, n)
    p = successes / n
    assert lo <= p <= hi


def test_wilson_interval_widens_with_fewer_samples_at_the_same_rate():
    lo_big, hi_big = wilson_interval(30, 50)
    lo_small, hi_small = wilson_interval(3, 5)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_wilson_interval_zero_n_is_the_full_unit_range():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_run_flaky_all_pass():
    result = run_flaky(lambda: True, runs=5)
    assert result["runs"] == 5
    assert result["passes"] == 5
    assert result["rate"] == 1.0
    assert result["errors"] == []
    lo, hi = result["ci95"]
    assert 0.0 < lo <= hi <= 1.0


def test_run_flaky_counts_exceptions_as_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("boom")
        return True

    result = run_flaky(flaky, runs=4)
    assert result["passes"] == 2
    assert result["rate"] == 0.5
    assert len(result["errors"]) == 2
    assert result["errors"][0] == "RuntimeError: boom"


def test_run_flaky_never_raises_even_when_fn_always_raises():
    def always_raises():
        raise ValueError("nope")

    result = run_flaky(always_raises, runs=3)
    assert result["passes"] == 0
    assert result["rate"] == 0.0
    assert len(result["errors"]) == 3
    assert all("ValueError: nope" in e for e in result["errors"])


def test_run_flaky_errors_are_capped_at_three():
    def always_raises():
        raise ValueError("nope")

    result = run_flaky(always_raises, runs=8)
    assert result["passes"] == 0
    assert len(result["errors"]) == 3


def test_run_flaky_falsy_return_counts_as_failure_not_error():
    result = run_flaky(lambda: False, runs=3)
    assert result["passes"] == 0
    assert result["errors"] == []
