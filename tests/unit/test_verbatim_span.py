# tests/unit/test_verbatim_span.py
from deep_research_toolkit.common.verbatim import slice_span, span_ok

SRC = "The mempool orders transactions by fee density before inclusion."

def test_slice_span_in_bounds():
    assert slice_span(SRC, 4, 11) == "mempool"

def test_slice_span_out_of_bounds_or_empty():
    assert slice_span(SRC, -1, 5) is None
    assert slice_span(SRC, 5, 5) is None
    assert slice_span(SRC, 5, 999) is None
    assert slice_span(SRC, 11, 4) is None

def test_span_ok_bounds_only():
    assert span_ok(4, 11, SRC) is True
    assert span_ok(5, 5, SRC) is False
    assert span_ok(5, 999, SRC) is False

def test_span_ok_matches_claimed_quote():
    assert span_ok(4, 11, SRC, "mempool") is True
    # near-quote bait: claimed text is NOT what the span actually covers
    assert span_ok(4, 11, SRC, "mempols") is False
