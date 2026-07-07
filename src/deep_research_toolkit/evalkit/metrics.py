"""Deterministic, no-LLM-judge metrics for the eval runner (Task 7).

Every function here is pure: it takes plain claim/dict lists and returns a
dict or scalar, so it is unit-testable with synthetic claim lists and never
touches a backend. `scripts/eval-pipeline.py` is the only caller that feeds
these live-model output.
"""
from __future__ import annotations

import re

from ..llm.response import CLAIM_MARKER_RE

#: Bare `[<id>]` markers -- the same character class `normalize_claim_markers`
#: builds dynamically from an allowed-id set, but scoped generically here (no
#: allowed-id list is available at metrics time). A prefixed `[claim:<id>]`
#: marker never matches this pattern: the literal colon in "claim:" is not in
#: the character class, so the two regexes partition a reply's brackets
#: without double-counting.
_BARE_MARKER_RE_SOURCE = r"\[([A-Za-z0-9_\-\.]+)\]"


def _quotes(claim: dict) -> list[str]:
    return [q for q in (ev.get("quote") for ev in claim.get("supporting_evidence") or []) if q]


def quote_overlap_match(produced: list[dict], reference: list[dict]) -> dict:
    """A reference claim counts as recalled when any produced evidence quote
    overlaps one of its quotes -- substring either direction, the same rule
    `scripts/validate-local-llm.py`'s `_recovered` uses. Returns:
      {"recalled": [reference claim, ...], "missed": [reference claim, ...],
       "matched_produced": {produced claim_id, ...}}
    `matched_produced` is every produced claim_id that overlapped at least one
    reference quote (used for precision_proxy), independent of which
    reference claim it matched."""
    prod_quotes = [(c.get("claim_id"), q) for c in produced for q in _quotes(c)]

    matched_produced: set = set()
    recalled_ids: set = set()
    for ref in reference:
        ref_quotes = _quotes(ref)
        if not ref_quotes:
            continue
        hit = False
        for pid, pq in prod_quotes:
            if any(pq in rq or rq in pq for rq in ref_quotes):
                matched_produced.add(pid)
                hit = True
        if hit:
            recalled_ids.add(ref.get("claim_id"))

    recalled = [r for r in reference if r.get("claim_id") in recalled_ids]
    missed = [r for r in reference if r.get("claim_id") not in recalled_ids]
    return {"recalled": recalled, "missed": missed, "matched_produced": matched_produced}


def extract_metrics(produced: list[dict], reference: list[dict],
                    dropped: list, parse_failures: int) -> dict:
    """Per-doc/per-run extraction metrics.

    - gate_pass_rate: len(produced) / (len(produced) + len(dropped)); None
      when nothing was attempted at all (nothing produced and nothing dropped).
    - recall: share of reference claims recalled (quote_overlap_match); None
      when reference is empty (nothing to recall).
    - precision_proxy: share of produced claims that matched some reference
      claim; None when nothing was produced.
    - atomicity: len(produced) / len(reference), flags over/under-splitting;
      None when reference is empty.
    - parse_failures: passed through unchanged (batches that never yielded
      parseable JSON).
    """
    match = quote_overlap_match(produced, reference)

    denom = len(produced) + len(dropped)
    gate_pass_rate = (len(produced) / denom) if denom else None
    recall = (len(match["recalled"]) / len(reference)) if reference else None
    precision_proxy = (len(match["matched_produced"]) / len(produced)) if produced else None
    atomicity = (len(produced) / len(reference)) if reference else None

    return {
        "gate_pass_rate": gate_pass_rate,
        "recall": recall,
        "precision_proxy": precision_proxy,
        "atomicity": atomicity,
        "parse_failures": parse_failures,
        "recalled": len(match["recalled"]),
        "missed_claim_ids": [r.get("claim_id") for r in match["missed"]],
    }


def bait_rejection(produced_bait_chunk_claims: list[dict], bait_source_text: str) -> float | None:
    """Share of a bait chunk's produced (already chunk-gate-passed) claims
    whose quotes do NOT also appear verbatim in the bait chunk's *source*
    text. Every claim here already passed the verbatim gate against its own
    (bait) chunk; this measures whether the gate held chunk-scope -- i.e. it
    did not accept a claim whose quote is really the near-copied source
    sentence bleeding across chunks. 1.0 is perfect discipline. None when
    there are no bait-chunk claims to score (nothing extracted from the bait
    chunk at all)."""
    if not produced_bait_chunk_claims:
        return None
    held = 0
    for claim in produced_bait_chunk_claims:
        quotes = _quotes(claim)
        if quotes and all(q not in bait_source_text for q in quotes):
            held += 1
    return held / len(produced_bait_chunk_claims)


def marker_rates(raw_texts: list[str]) -> dict:
    """Bare vs `[claim:...]`-prefixed marker counts across a batch of raw
    (pre-normalization) prose-role replies -- the marker-fidelity signal the
    live canary (`test_marker_fidelity_smoke`) and the eval runner both
    track. `bare_rate` is bare / (bare + prefixed), None when no markers of
    either form appear (nothing to rate)."""
    prefixed = sum(len(CLAIM_MARKER_RE.findall(t)) for t in raw_texts)
    bare_re = re.compile(_BARE_MARKER_RE_SOURCE)
    bare = sum(len(bare_re.findall(t)) for t in raw_texts)
    total = bare + prefixed
    return {"bare": bare, "prefixed": prefixed, "bare_rate": (bare / total) if total else None}
