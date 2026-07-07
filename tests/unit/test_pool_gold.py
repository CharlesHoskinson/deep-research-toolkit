"""pool_gold: union + dedup of gate-passing claims from multiple models into
one gold set (Task 12). Pure -- wraps selfconsistency.union_claims with
min_support=1, so any model's gate-passed claim is gold and duplicates
collapse by normalized claim key."""
from deep_research_toolkit.evalkit import pool_gold


def _c(text, s, e):
    return {"claim": text, "supporting_evidence": [{"locator": "d#c1", "start_char": s, "end_char": e}]}


def test_pool_dedups_across_models():
    e4b = [_c("Leaders rotate each epoch", 0, 10)]
    qwen = [_c("leaders  rotate each epoch", 0, 10), _c("Followers verify", 20, 30)]
    pooled = pool_gold([e4b, qwen])
    texts = sorted(c["claim"].lower().strip() for c in pooled)
    assert len(pooled) == 2  # the rotate claim collapses; verify is added
    assert texts == ["followers verify", "leaders rotate each epoch"]  # first-seen row (e4b's) wins
