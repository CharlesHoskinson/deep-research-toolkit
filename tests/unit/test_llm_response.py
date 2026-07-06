from deep_research_toolkit.llm.response import (
    extract_claim_ids,
    has_repetition_loop,
    normalize_claim_markers,
    parse_json_block,
    unfence,
    validate_citations,
)


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


def test_normalize_claim_markers_rewrites_bare_known_ids():
    # Gemma 4 tic measured in the e2e run: [b00_c_0001] instead of [claim:b00_c_0001]
    text = "Hydra scales eUTxO [b00_c_0001]. Instant settlement [b00_c_0002]."
    out = normalize_claim_markers(text, ["b00_c_0001", "b00_c_0002"])
    assert out == ("Hydra scales eUTxO [claim:b00_c_0001]. "
                   "Instant settlement [claim:b00_c_0002].")


def test_normalize_claim_markers_leaves_unknown_and_prefixed_alone():
    text = "Known [c1]. Unknown [zz]. Already tagged [claim:c1]. Cite [1]."
    out = normalize_claim_markers(text, ["c1", "c2"])
    assert out == "Known [claim:c1]. Unknown [zz]. Already tagged [claim:c1]. Cite [1]."


def test_normalize_claim_markers_empty_allowed_is_identity():
    text = "Anything [c1] at all."
    assert normalize_claim_markers(text, []) == text


def test_repetition_loop_detected_on_repeated_phrase():
    text = "The ledger records " + ("the same value " * 30)
    assert has_repetition_loop(text)


def test_repetition_loop_ignores_normal_prose():
    text = ("Hydra is a family of Layer-2 protocols. Transactions settle "
            "instantly among participants. The main chain reconciles state "
            "when the head closes. Four phases structure the lifecycle.")
    assert not has_repetition_loop(text)


def test_repetition_loop_ignores_short_texts():
    assert not has_repetition_loop("yes yes yes")


def test_repetition_loop_found_inside_json_string():
    import json as _json
    reply = "<output>" + _json.dumps([{"rationale": "loop " * 50}]) + "</output>"
    assert has_repetition_loop(reply)


_VARIED_PROSE = (
    "The settlement layer batches transactions before anchoring them on the "
    "main chain, while validators rotate through committee assignments each "
    "epoch. Fee markets respond to congestion by repricing inclusion, and "
    "light clients verify headers without replaying full state transitions. "
    "Checkpoint intervals bound how far any rollback can reach."
)


def test_repetition_loop_ignores_table_separator_rows():
    text = ("| col | col | col |\n|---|---|---|\n"
            "| a | 1 | x |\n| b | 2 | y |\n" + _VARIED_PROSE)
    assert not has_repetition_loop(text)


def test_repetition_loop_ignores_na_cells():
    text = "| n/a | n/a | n/a | n/a | n/a |\n" + _VARIED_PROSE
    assert not has_repetition_loop(text)


def test_repetition_loop_catches_punctuation_jitter():
    reps = "".join(["Error occurred, retrying." if i % 2 else "Error occurred retrying!" for i in range(30)])
    assert has_repetition_loop(reps.replace(".", ". ").replace("!", "! "))
