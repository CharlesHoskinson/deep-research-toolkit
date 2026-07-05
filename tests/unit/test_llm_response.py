from deep_research_toolkit.llm.response import extract_claim_ids, parse_json_block, unfence, validate_citations


def test_extract_claim_ids_in_order_with_dupes_removed():
    text = "Praos is adaptive [claim:c1]. It tolerates delay [claim:c2] [claim:c1]."
    assert extract_claim_ids(text) == ["c1", "c2"]


def test_validate_citations_flags_unknown_and_reports_coverage():
    text = "A [claim:c1]. B [claim:zz]."
    rep = validate_citations(text, allowed_ids=["c1", "c2", "c3"])
    assert rep["cited"] == ["c1"]
    assert rep["unknown"] == ["zz"]
    assert rep["coverage"] == 1 / 3


def test_parse_json_block_prefers_output_tags():
    text = 'noise {"a": 1} noise <output>[{"b": 2}]</output>'
    assert parse_json_block(text) == [{"b": 2}]


def test_parse_json_block_falls_back_to_bracket_slice():
    assert parse_json_block('prose... [{"x": 1}, {"x": 2}] trailing') == [{"x": 1}, {"x": 2}]


def test_parse_json_block_returns_none_on_garbage():
    assert parse_json_block("no json here") is None


def test_parse_json_block_object_wrapping_array_returns_object():
    text = 'Here is the result: {"summary": "ok", "citations": [1, 2, 3]} done.'
    assert parse_json_block(text) == {"summary": "ok", "citations": [1, 2, 3]}


def test_parse_json_block_fenced_output_block():
    text = 'Plan: notes. <output>```json\n{"marker": "REAL", "items": [1, 2]}\n```</output>'
    assert parse_json_block(text) == {"marker": "REAL", "items": [1, 2]}


def test_parse_json_block_object_only_fallback():
    assert parse_json_block('x {"a": 1} y') == {"a": 1}


def test_parse_json_block_output_block_is_authoritative():
    # Garbage inside <output> must NOT fall back to JSON outside it.
    assert parse_json_block('<output>garbage</output> {"a": 1}') is None


def test_unfence_unwraps_whole_reply_fence():
    assert unfence("```markdown\nbody [claim:c1]\n```") == "body [claim:c1]"


def test_unfence_passthrough_when_not_fenced():
    text = "plain body\n"
    assert unfence(text) == text


def test_unfence_leaves_mid_body_fences_alone():
    text = "intro\n```python\ncode\n```\noutro"
    assert unfence(text) == text
