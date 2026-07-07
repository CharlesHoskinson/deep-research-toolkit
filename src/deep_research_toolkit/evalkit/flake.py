"""N-run pass-rate helpers for irreducibly stochastic live-model tests."""
from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def run_flaky(fn, runs: int = 5) -> dict:
    """Run fn() `runs` times; fn returns truthy on pass. Never raises --
    exceptions count as failures and are recorded."""
    if runs <= 0:
        return {"runs": 0, "passes": 0, "rate": 0.0, "ci95": (0.0, 1.0), "errors": []}
    passes, errors = 0, []
    for _ in range(runs):
        try:
            if fn():
                passes += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")
    lo, hi = wilson_interval(passes, runs)
    return {"runs": runs, "passes": passes, "rate": passes / runs,
            "ci95": (round(lo, 3), round(hi, 3)), "errors": errors[:3]}
