# tests/unit/test_selfconsistency.py
from deep_research_toolkit.llm.selfconsistency import union_claims, claim_key

def _c(cid, text, loc="d#c1", s=0, e=10):
    return {"claim_id": cid, "claim": text,
            "supporting_evidence": [{"locator": loc, "start_char": s, "end_char": e}]}

def test_claim_key_ignores_id_and_casing_and_spacing():
    assert claim_key(_c("a", "Fee  density Orders it.")) == claim_key(_c("b", "fee density orders it."))

def test_union_min_support_filters_singletons():
    p1 = [_c("a", "claim one"), _c("b", "claim two")]
    p2 = [_c("c", "claim one")]  # only "claim one" repeats
    p3 = [_c("d", "claim one")]
    kept = union_claims([p1, p2, p3], min_support=2)
    texts = {c["claim"].lower() for c in kept}
    assert "claim one" in texts and "claim two" not in texts

def test_union_min_support_1_is_plain_union_deduped():
    p1 = [_c("a", "same claim")]
    p2 = [_c("b", "same claim")]
    kept = union_claims([p1, p2], min_support=1)
    assert len(kept) == 1

def test_claim_key_ignores_jittered_offsets_same_locator():
    # Same claim, same source chunk, but different samples picked slightly
    # different char offsets -- must dedup to ONE claim (else the union explodes
    # ~N-fold under samples=N). This is the self-consistency dedup bug fix.
    assert claim_key(_c("a", "leaders rotate", s=0, e=14)) == \
           claim_key(_c("b", "leaders rotate", s=2, e=16))

def test_claim_key_distinguishes_different_locators():
    # Same text citing a genuinely different chunk stays distinct.
    assert claim_key(_c("a", "leaders rotate", loc="d#c1")) != \
           claim_key(_c("b", "leaders rotate", loc="d#c9"))

def test_union_collapses_jittered_offset_duplicates_across_samples():
    # 5 samples each emit the "same" claim with jittered offsets -> ONE claim,
    # not five (atomicity stays ~1, not ~5).
    passes = [[_c(str(i), "the mempool orders by fee", s=i, e=25 + i)] for i in range(5)]
    kept = union_claims(passes, min_support=1)
    assert len(kept) == 1
