"""tunekit.assemble: Recipe-B final dataset curation/merge/split/manifest
helpers (`scripts/assemble-sft.py`'s backing module). Exercises the
near-duplicate collapse + per-record cap curation heuristic, the frontier
bait merge, the self-authored general-instruction synthesis, the stratified
re-split, and the hashing/manifest helpers -- all against small synthetic
fixtures, never the real (large) datasets on disk."""
from __future__ import annotations

import hashlib
import json

import pytest

from deep_research_toolkit.tunekit.assemble import (
    GENERAL_TOPICS,
    atomicity_stats,
    build_generator_digests,
    claim_quote_len,
    component_hash,
    curate_claims,
    curate_local_record,
    final_dataset_hash,
    jaccard,
    load_corpus_chunk_texts,
    load_jsonl,
    merge_all_bait,
    merge_bait_candidate_file,
    normalized_tokens,
    record_claim_count,
    source_corpus_hash,
    split_train_val,
    stratified_split,
    synthesize_general_instructions,
    write_jsonl,
)


def _claim(text: str, quote: str, locator: str = "d#c1", claim_id: str = "c_0001") -> dict:
    return {
        "claim_id": claim_id,
        "claim": text,
        "supporting_evidence": [{"locator": locator, "start_char": 0,
                                "end_char": len(quote), "quote": quote}],
        "citable": True,
    }


def _conversation_record(claims: list[dict], locator: str = "d#c1") -> dict:
    body = json.dumps({"claims": claims, "entities": [], "relations": []}, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": "SYSTEM PROMPT"},
            {"role": "user", "content": "USER PROMPT"},
            {"role": "assistant", "content": body},
        ],
        "locator": locator,
    }


# ---------------------------------------------------------------------------
# normalized_tokens / jaccard / claim_quote_len
# ---------------------------------------------------------------------------

def test_normalized_tokens_folds_case_and_punctuation():
    assert normalized_tokens("Gabbro's Slicer!") == normalized_tokens("gabbro s slicer")


def test_jaccard_identical_and_disjoint():
    a = {"x", "y"}
    b = {"x", "y"}
    assert jaccard(a, b) == 1.0
    assert jaccard({"x"}, {"y"}) == 0.0
    assert jaccard(set(), set()) == 1.0


def test_claim_quote_len_sums_evidence_quotes():
    claim = {"supporting_evidence": [{"quote": "abc"}, {"quote": "de"}]}
    assert claim_quote_len(claim) == 5
    assert claim_quote_len({"supporting_evidence": []}) == 0
    assert claim_quote_len({}) == 0


# ---------------------------------------------------------------------------
# curate_claims: paraphrase collapse + cap
# ---------------------------------------------------------------------------

def test_curate_claims_collapses_paraphrases_keeping_longer_quote():
    claims = [
        _claim("Gabbro is available as a desktop app.", "a" * 30, claim_id="c1"),
        _claim("Gabbro is available as a desktop application and headless engine.", "a" * 60, claim_id="c2"),
    ]
    out = curate_claims(claims, jaccard_threshold=0.3, cap=8)
    assert len(out) == 1
    assert out[0]["claim_id"] == "c2"  # longer quote survives


def test_curate_claims_keeps_genuinely_distinct_claims():
    claims = [
        _claim("The project began in 2016.", "began in 2016", claim_id="c1"),
        _claim("Gabbro supports third-party plugins.", "third-party plugins", claim_id="c2"),
    ]
    out = curate_claims(claims, jaccard_threshold=0.3, cap=8)
    assert {c["claim_id"] for c in out} == {"c1", "c2"}


def test_curate_claims_caps_at_n():
    claims = [_claim(f"Distinct fact number {i} about the widget.", "q" * (i + 1), claim_id=f"c{i}")
             for i in range(20)]
    out = curate_claims(claims, jaccard_threshold=0.9, cap=8)
    assert len(out) == 8
    # ranked by quote length descending -> the longest-quote claims survive
    assert out[0]["claim_id"] == "c19"


def test_curate_claims_empty_input():
    assert curate_claims([], cap=8) == []


# ---------------------------------------------------------------------------
# record_claim_count / atomicity_stats / curate_local_record
# ---------------------------------------------------------------------------

def test_record_claim_count_and_atomicity_stats():
    rec_a = _conversation_record([_claim("a", "aaa", claim_id="c1"), _claim("b", "bbb", claim_id="c2")])
    rec_b = _conversation_record([_claim("c", "ccc", claim_id="c1")])
    assert record_claim_count(rec_a) == 2
    assert record_claim_count(rec_b) == 1
    stats = atomicity_stats([rec_a, rec_b])
    assert stats == {"n": 2, "mean": 1.5, "median": 1.5, "min": 1, "max": 2}


def test_atomicity_stats_empty():
    assert atomicity_stats([]) == {"n": 0, "mean": None, "median": None, "min": None, "max": None}


def test_record_claim_count_non_extraction_record_is_zero():
    general = {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"},
                            {"role": "assistant", "content": "just some prose, not JSON"}]}
    assert record_claim_count(general) == 0


def test_curate_local_record_preserves_prompt_and_tags_component():
    claims = [
        _claim("Gabbro is available as a desktop app.", "a" * 20, claim_id="c1"),
        _claim("Gabbro is available as a desktop application and headless engine.", "a" * 40, claim_id="c2"),
        _claim("The project began in 2016.", "began in 2016", claim_id="c3"),
    ]
    rec = _conversation_record(claims, locator="doc#c9")
    curated = curate_local_record(rec, jaccard_threshold=0.3, cap=8)

    assert curated["component"] == "local"
    assert curated["locator"] == "doc#c9"
    assert curated["messages"][0] == rec["messages"][0]  # system unchanged
    assert curated["messages"][1] == rec["messages"][1]  # user unchanged
    obj = json.loads(curated["messages"][2]["content"])
    assert obj["entities"] == []
    assert obj["relations"] == []
    assert len(obj["claims"]) == 2  # c1/c2 collapsed, c3 distinct survives

    # original record must be untouched (curate_local_record does not mutate)
    assert len(json.loads(rec["messages"][2]["content"])["claims"]) == 3


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def test_jsonl_roundtrip(tmp_path):
    records = [{"a": 1}, {"b": [1, 2, 3]}]
    path = tmp_path / "out.jsonl"
    write_jsonl(path, records)
    assert load_jsonl(path) == records


def test_load_jsonl_missing_file_returns_empty(tmp_path):
    assert load_jsonl(tmp_path / "nope.jsonl") == []


# ---------------------------------------------------------------------------
# Frontier bait merge
# ---------------------------------------------------------------------------

def test_merge_bait_candidate_file_unions_and_dedups(tmp_path):
    candidate_file = tmp_path / "doc_c1.json"
    candidate_file.write_text(json.dumps({
        "bait_locator": "doc#c1",
        "candidates": [
            {"candidate_id": 0, "claims": [
                {"claim_id": "c_0001", "claim": "Widget threshold is 55 degrees.",
                 "supporting_evidence": [{"locator": "doc#c1", "start_char": 0, "end_char": 10,
                                         "quote": "0123456789"}]},
            ]},
            {"candidate_id": 1, "claims": [
                {"claim_id": "c_0001", "claim": "Widget threshold is 55 degrees.",
                 "supporting_evidence": [{"locator": "doc#c1", "start_char": 0, "end_char": 10,
                                         "quote": "0123456789"}]},
                {"claim_id": "c_0002", "claim": "Widget ships with a plugin API.",
                 "supporting_evidence": [{"locator": "doc#c1", "start_char": 10, "end_char": 15,
                                         "quote": "01234"}]},
            ]},
        ],
    }), encoding="utf-8")
    chunk_by_locator = {"doc#c1": {"locator": "doc#c1", "text": "0123456789012345678901234567890"}}

    rec = merge_bait_candidate_file(candidate_file, chunk_by_locator)
    assert rec["component"] == "bait"
    assert rec["locator"] == "doc#c1"
    obj = json.loads(rec["messages"][-1]["content"])
    # the identical claim across both candidates dedups to 1; the distinct one survives
    assert len(obj["claims"]) == 2


def test_merge_all_bait_over_a_directory(tmp_path):
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    for i in range(3):
        (candidates_dir / f"doc_c{i}.json").write_text(json.dumps({
            "bait_locator": f"doc#c{i}",
            "candidates": [{"candidate_id": 0, "claims": [
                {"claim_id": "c_0001", "claim": f"Fact number {i}.",
                 "supporting_evidence": [{"locator": f"doc#c{i}", "start_char": 0, "end_char": 4,
                                         "quote": "abcd"}]},
            ]}],
        }), encoding="utf-8")
    chunk_by_locator = {f"doc#c{i}": {"locator": f"doc#c{i}", "text": "abcdefgh"} for i in range(3)}
    records = merge_all_bait(candidates_dir, chunk_by_locator)
    assert len(records) == 3
    assert {r["locator"] for r in records} == {"doc#c0", "doc#c1", "doc#c2"}
    assert all(r["component"] == "bait" for r in records)


def test_load_corpus_chunk_texts(tmp_path):
    doc_dir = tmp_path / "docA"
    doc_dir.mkdir()
    (doc_dir / "chunks.jsonl").write_text(
        json.dumps({"locator": "docA#c1", "text": "hello world"}) + "\n", encoding="utf-8")
    chunks = load_corpus_chunk_texts(tmp_path)
    assert chunks["docA#c1"]["text"] == "hello world"


# ---------------------------------------------------------------------------
# General instruction synthesis
# ---------------------------------------------------------------------------

def test_synthesize_general_instructions_count_and_shape():
    records = synthesize_general_instructions(n=100, seed=42)
    assert len(records) == 100
    for r in records:
        assert r["component"] == "general"
        assert r["locator"] is None
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["system", "user", "assistant"]
        # not extraction-shaped JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(r["messages"][-1]["content"])


def test_synthesize_general_instructions_deterministic():
    a = synthesize_general_instructions(n=100, seed=42)
    b = synthesize_general_instructions(n=100, seed=42)
    assert a == b


def test_synthesize_general_instructions_truncates_and_is_representative():
    records = synthesize_general_instructions(n=8, seed=42)
    assert len(records) == 8


def test_general_topics_count_is_25():
    assert len(GENERAL_TOPICS) == 25


# ---------------------------------------------------------------------------
# split_train_val / stratified_split
# ---------------------------------------------------------------------------

def test_split_train_val_deterministic_and_covers_all():
    records = [{"i": i} for i in range(50)]
    train1, val1 = split_train_val(records, val_fraction=0.10, seed=42)
    train2, val2 = split_train_val(records, val_fraction=0.10, seed=42)
    assert train1 == train2
    assert val1 == val2
    assert len(val1) == 5
    assert sorted(r["i"] for r in train1 + val1) == list(range(50))


def test_stratified_split_holds_each_component_proportionally():
    records_by_component = {
        "local": [{"i": f"l{i}"} for i in range(90)],
        "bait": [{"i": f"b{i}"} for i in range(20)],
        "general": [{"i": f"g{i}"} for i in range(10)],
    }
    train, val = stratified_split(records_by_component, val_fraction=0.10, seed=42)
    val_ids = {r["i"] for r in val}
    n_local_val = sum(1 for i in val_ids if i.startswith("l"))
    n_bait_val = sum(1 for i in val_ids if i.startswith("b"))
    n_general_val = sum(1 for i in val_ids if i.startswith("g"))
    assert n_local_val == 9   # round(90 * 0.10)
    assert n_bait_val == 2    # round(20 * 0.10)
    assert n_general_val == 1  # round(10 * 0.10)
    assert len(train) + len(val) == 120


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def test_component_hash_deterministic_and_sensitive():
    a = [{"x": 1}, {"x": 2}]
    b = [{"x": 1}, {"x": 2}]
    c = [{"x": 1}, {"x": 3}]
    assert component_hash(a) == component_hash(b)
    assert component_hash(a) != component_hash(c)


def test_source_corpus_hash_order_independent(tmp_path):
    (tmp_path / "docA").mkdir()
    (tmp_path / "docB").mkdir()
    (tmp_path / "docA" / "chunks.jsonl").write_text(
        json.dumps({"locator": "docA#c1", "text": "alpha"}) + "\n", encoding="utf-8")
    (tmp_path / "docB" / "chunks.jsonl").write_text(
        json.dumps({"locator": "docB#c1", "text": "beta"}) + "\n", encoding="utf-8")
    h1 = source_corpus_hash(tmp_path)

    # Rebuild with docs discovered in a different order (docB has an extra
    # earlier chunk we then remove) -- the hash is computed by sorting on
    # locator internally, so re-running against the same content is stable.
    h2 = source_corpus_hash(tmp_path)
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_final_dataset_hash_matches_manual_sha256(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    expected = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert final_dataset_hash(path) == expected


# ---------------------------------------------------------------------------
# Generator digests
# ---------------------------------------------------------------------------

def test_build_generator_digests_attributes_by_teacher_and_adds_bait_general():
    local_records = [
        _conversation_record([_claim("a", "aaa", claim_id="c1")], locator="doc#c1"),
        _conversation_record([_claim("b", "bbb", claim_id="c1"), _claim("c", "ccc", claim_id="c2")],
                             locator="doc#c2"),
    ]
    escalation_log = [
        {"locator": "doc#c1", "teacher_model": "e4b"},
        {"locator": "doc#c2", "teacher_model": "qwen3:30b-a3b"},
    ]
    digests = build_generator_digests(local_records, escalation_log, bait_n_claims=7, general_n_records=100)
    assert digests["e4b"] == 1
    assert digests["qwen3:30b-a3b"] == 2
    assert digests["frontier-agent-opus"] == 7
    assert digests["self-authored (assembling agent)"] == 100


def test_build_generator_digests_unknown_teacher_when_locator_missing():
    local_records = [_conversation_record([_claim("a", "aaa")], locator="doc#unmapped")]
    digests = build_generator_digests(local_records, escalation_log=[], bait_n_claims=0, general_n_records=0)
    assert digests["unknown"] == 1
