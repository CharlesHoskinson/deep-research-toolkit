"""Unit tests for evalkit.metrics: pure, synthetic-claim-list driven, no
model calls. Mirrors the quote-overlap rule already used by
scripts/validate-local-llm.py (`_recovered`): a reference claim is recalled
when any produced evidence quote overlaps one of its quotes, substring
either direction."""
from __future__ import annotations

from deep_research_toolkit.evalkit.metrics import (
    bait_rejection,
    extract_metrics,
    marker_rates,
    quote_overlap_match,
)


def _claim(claim_id: str, quote: str | None, locator: str = "doc#c001") -> dict:
    evidence = [{"locator": locator, "quote": quote}] if quote else []
    return {"claim_id": claim_id, "claim": claim_id, "supporting_evidence": evidence}


# --------------------------------------------------------------------------
# quote_overlap_match
# --------------------------------------------------------------------------

def test_quote_overlap_exact_match_recalls_and_marks_produced():
    produced = [_claim("p1", "settles instantly")]
    reference = [_claim("r1", "settles instantly")]
    out = quote_overlap_match(produced, reference)
    assert out["recalled"] == reference
    assert out["missed"] == []
    assert out["matched_produced"] == {"p1"}


def test_quote_overlap_matches_when_produced_quote_is_a_superstring():
    produced = [_claim("p1", "the head settles instantly among participants")]
    reference = [_claim("r1", "settles instantly")]
    out = quote_overlap_match(produced, reference)
    assert out["recalled"] == reference
    assert out["matched_produced"] == {"p1"}


def test_quote_overlap_matches_when_reference_quote_is_a_superstring():
    produced = [_claim("p1", "settles instantly")]
    reference = [_claim("r1", "the head settles instantly among participants")]
    out = quote_overlap_match(produced, reference)
    assert out["recalled"] == reference
    assert out["matched_produced"] == {"p1"}


def test_quote_overlap_no_match_is_missed():
    produced = [_claim("p1", "completely unrelated text")]
    reference = [_claim("r1", "settles instantly")]
    out = quote_overlap_match(produced, reference)
    assert out["recalled"] == []
    assert out["missed"] == reference
    assert out["matched_produced"] == set()


def test_quote_overlap_empty_reference_returns_empty():
    out = quote_overlap_match([_claim("p1", "x")], [])
    assert out == {"recalled": [], "missed": [], "matched_produced": set()}


def test_quote_overlap_claim_without_evidence_is_ignored_not_crashed():
    produced = [_claim("p1", None)]  # no supporting_evidence at all
    reference = [_claim("r1", "settles instantly")]
    out = quote_overlap_match(produced, reference)
    assert out["missed"] == reference
    assert out["matched_produced"] == set()


def test_quote_overlap_one_produced_claim_can_recall_multiple_references():
    produced = [_claim("p1", "settles instantly among all participants")]
    reference = [_claim("r1", "settles instantly"), _claim("r2", "among all participants")]
    out = quote_overlap_match(produced, reference)
    assert {r["claim_id"] for r in out["recalled"]} == {"r1", "r2"}
    assert out["matched_produced"] == {"p1"}


def test_quote_overlap_short_substring_does_not_match():
    # "42" is a substring of the reference quote, but the shorter quote is
    # under the 12-char floor -- coincidental containment must not count.
    produced = [_claim("p1", "42")]
    reference = [_claim("r1", "the answer to everything is 42 exactly")]
    out = quote_overlap_match(produced, reference)
    assert out["recalled"] == []
    assert out["matched_produced"] == set()


def test_quote_overlap_short_exact_equality_still_matches():
    # Equality always counts, even below the substring-length floor.
    produced = [_claim("p1", "42")]
    reference = [_claim("r1", "42")]
    out = quote_overlap_match(produced, reference)
    assert out["recalled"] == reference
    assert out["matched_produced"] == {"p1"}


def test_quote_overlap_twelve_char_substring_matches():
    # Exactly at the floor: a 12-char shorter quote in either direction counts.
    produced = [_claim("p1", "abcdefghijkl")]  # 12 chars
    reference = [_claim("r1", "xx abcdefghijkl yy")]
    out = quote_overlap_match(produced, reference)
    assert out["recalled"] == reference


# --------------------------------------------------------------------------
# extract_metrics
# --------------------------------------------------------------------------

def test_extract_metrics_basic_rates():
    produced = [_claim("p1", "settles instantly"), _claim("p2", "unrelated")]
    reference = [_claim("r1", "settles instantly"), _claim("r2", "b"), _claim("r3", "c")]
    dropped = ["d1"]
    out = extract_metrics(produced, reference, dropped, parse_failures=2)
    assert out["gate_pass_rate"] == 2 / 3
    assert out["recall"] == 1 / 3
    assert out["precision_proxy"] == 1 / 2  # only p1 matched something
    assert out["atomicity"] == 2 / 3
    assert out["parse_failures"] == 2


def test_extract_metrics_guards_zero_written_and_dropped():
    out = extract_metrics([], [_claim("r1", "x")], [], parse_failures=0)
    assert out["gate_pass_rate"] is None


def test_extract_metrics_guards_empty_reference():
    out = extract_metrics([_claim("p1", "x")], [], [], parse_failures=0)
    assert out["recall"] is None
    assert out["atomicity"] is None


def test_extract_metrics_guards_empty_produced():
    out = extract_metrics([], [_claim("r1", "x")], ["d1"], parse_failures=0)
    assert out["precision_proxy"] is None
    assert out["gate_pass_rate"] == 0.0


def test_extract_metrics_perfect_recall_and_precision():
    produced = [_claim("p1", "x"), _claim("p2", "y")]
    reference = [_claim("r1", "x"), _claim("r2", "y")]
    out = extract_metrics(produced, reference, [], parse_failures=0)
    assert out["gate_pass_rate"] == 1.0
    assert out["recall"] == 1.0
    assert out["precision_proxy"] == 1.0
    assert out["atomicity"] == 1.0


# --------------------------------------------------------------------------
# bait_rejection
# --------------------------------------------------------------------------

def test_bait_rejection_all_quotes_absent_from_source_is_perfect():
    bait_claims = [_claim("b1", "the near-copy sentence text")]
    source_text = "a totally different sentence lives in the source chunk"
    assert bait_rejection(bait_claims, source_text) == 1.0


def test_bait_rejection_penalizes_quotes_that_leak_into_source():
    bait_claims = [
        _claim("b1", "unique to the bait chunk"),
        _claim("b2", "shared sentence text"),
    ]
    source_text = "this chunk contains shared sentence text verbatim"
    assert bait_rejection(bait_claims, source_text) == 0.5


def test_bait_rejection_empty_claims_returns_none():
    assert bait_rejection([], "some source text") is None


# --------------------------------------------------------------------------
# marker_rates
# --------------------------------------------------------------------------

def test_marker_rates_counts_prefixed_and_bare():
    raw = [
        "Fact one [claim:c1]. Fact two [c2].",
        "Fact three [claim:c3].",
    ]
    out = marker_rates(raw)
    assert out["prefixed"] == 2
    assert out["bare"] == 1
    assert out["bare_rate"] == 1 / 3


def test_marker_rates_no_markers_returns_none_rate():
    out = marker_rates(["no markers at all here"])
    assert out["prefixed"] == 0
    assert out["bare"] == 0
    assert out["bare_rate"] is None
