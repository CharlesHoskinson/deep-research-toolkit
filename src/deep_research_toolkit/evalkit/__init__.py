"""Eval-suite helpers: flake-rate statistics for the live-model tier (Task 5),
plus the extraction/prose/adjudication metrics, paired-bootstrap comparison,
and the raw-completion RecordingBackend wrapper consumed by
scripts/eval-pipeline.py (Task 7). Pure Python (RecordingBackend excepted,
which wraps whatever backend the runner gives it) -- no model calls of its
own; the live tier and the eval runner call into this package.

Also exports pool_gold (Task 12): the pure union+dedup that
scripts/build-pooled-gold.py wraps to pool gate-passing claims from multiple
models into one gold set (the fixed eval denominator / Phase-2 SFT target)."""
from __future__ import annotations

from ..llm.selfconsistency import union_claims


def pool_gold(claim_lists: list[list[dict]]) -> list[dict]:
    """Union + dedup gate-passing claims from multiple models into one gold set
    (min_support=1: any model's gate-passed claim is gold). Dedup by claim_key."""
    return union_claims(claim_lists, min_support=1)
