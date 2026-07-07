from deep_research_toolkit.llm.response import validate_citations


def test_absolute_rule_when_few_claims_all_cited():
    r = validate_citations("prose [claim:c1] and [claim:c2].", ["c1", "c2"])
    assert r["rule"] == "absolute" and r["coverage_ok"] is True


def test_absolute_rule_fails_if_a_citable_claim_uncited():
    r = validate_citations("prose [claim:c1] only.", ["c1", "c2"])
    assert r["rule"] == "absolute" and r["coverage_ok"] is False


def test_ratio_rule_applies_with_enough_claims():
    ids = [f"c{i}" for i in range(6)]
    text = " ".join(f"[claim:{i}]" for i in ids[:2])  # 2/6 = 0.33
    r = validate_citations(text, ids, min_citable_for_ratio=4)
    assert r["rule"] == "ratio"
