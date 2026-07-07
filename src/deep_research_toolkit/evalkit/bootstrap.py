"""Paired bootstrap over per-chunk metric deltas -- the model/version A-vs-B
comparison the design calls for (design doc §3.4): "improved" requires a 95%
CI that excludes zero, not just a higher point estimate on a handful of
stochastic runs."""
from __future__ import annotations

import random


def paired_bootstrap(deltas: list[float], b: int = 2000, seed: int = 7) -> dict:
    """Resample `deltas` with replacement `b` times, compute the mean of each
    resample, and report the observed mean plus the 95% percentile interval
    over the resampled means. `significant` is True when that interval
    excludes zero entirely (all-positive or all-negative), matching the
    "CI excluding zero" criterion in the design doc.

    Deterministic for a fixed (deltas, b, seed) triple via `random.Random(seed)`.
    An empty `deltas` list is not an error -- it reports a zero mean, a
    degenerate (0.0, 0.0) interval, and not significant."""
    n = len(deltas)
    if n == 0:
        return {"mean": 0.0, "ci95": (0.0, 0.0), "significant": False}

    mean = sum(deltas) / n
    rng = random.Random(seed)
    resampled_means = []
    for _ in range(b):
        resample_sum = 0.0
        for _ in range(n):
            resample_sum += deltas[rng.randrange(n)]
        resampled_means.append(resample_sum / n)
    resampled_means.sort()

    lo_idx = int(0.025 * b)
    hi_idx = min(int(0.975 * b), b - 1)
    lo = resampled_means[lo_idx]
    hi = resampled_means[hi_idx]
    significant = lo > 0 or hi < 0

    return {"mean": mean, "ci95": (lo, hi), "significant": significant}
