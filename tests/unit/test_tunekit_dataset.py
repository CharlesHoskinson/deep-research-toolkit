"""tunekit.dataset: the SFT dataset harness skeleton (design doc §6.1).
Exercises the DART k-escalation trigger, near-dup dedup, the hard
contamination guard, the provenance manifest, teacher-call budgets, and
output-format fidelity to the extract contract -- all against a SCRIPTED
fake teacher, never a live model."""
from __future__ import annotations

import json
import logging

import pytest

from deep_research_toolkit.llm.extract import parse_extraction_response
from deep_research_toolkit.tunekit.dataset import (
    DEFAULT_ROUTER_TABLE,
    DEFAULT_TEACHER_ROUTE,
    BannedTeacherError,
    ContaminationError,
    TeacherRoute,
    assert_not_contaminated,
    build_manifest,
    build_sft_dataset,
    dataset_hash,
    dedup_claims,
    escalating_k_sample,
    gate_claim,
    is_a_priori_difficult,
    load_contamination_index,
    route_for_chunk,
    split_train_val,
    to_conversation_record,
)


def _chunk(locator: str, text: str, slices: list[str] | None = None) -> dict:
    return {"locator": locator, "text": text, "slices": slices or ["prose"]}


def _claim_json(text: str, locator: str, start: int, end: int, claim_id: str = "c_0001") -> dict:
    return {
        "claim_id": claim_id,
        "claim": text,
        "supporting_evidence": [{"locator": locator, "start_char": start, "end_char": end}],
    }


def _completion(*claims: dict) -> str:
    return json.dumps({"claims": list(claims), "entities": [], "relations": []})


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def test_route_for_chunk_bait_slice_gets_frontier():
    chunk = _chunk("d#c1", "text", slices=["bait"])
    route = route_for_chunk(chunk)
    assert route.model == "frontier"
    assert route.role == "bait"


def test_route_for_chunk_dense_facts_gets_recall_teacher():
    chunk = _chunk("d#c1", "text", slices=["dense-facts"])
    route = route_for_chunk(chunk)
    assert route.model == "qwen3:30b-a3b"


def test_route_for_chunk_default_is_bulk_e4b():
    chunk = _chunk("d#c1", "text", slices=["prose"])
    route = route_for_chunk(chunk)
    assert route == DEFAULT_TEACHER_ROUTE
    assert route.model == "e4b"


def test_route_for_chunk_banned_model_raises():
    with pytest.raises(BannedTeacherError):
        route_for_chunk(_chunk("d#c1", "text", slices=["prose"]),
                        router_table={}, fallback=TeacherRoute("bulk", "gemma4:31b"))


def test_route_table_never_bans_the_defaults():
    for route in (*DEFAULT_ROUTER_TABLE.values(), DEFAULT_TEACHER_ROUTE):
        route_for_chunk(_chunk("d#c1", "x", slices=[route.role]), router_table={route.role: route})


# ---------------------------------------------------------------------------
# is_a_priori_difficult
# ---------------------------------------------------------------------------

def test_a_priori_difficult_flags_bait_and_dense_facts():
    assert is_a_priori_difficult(_chunk("d#c1", "x", slices=["bait"])) is True
    assert is_a_priori_difficult(_chunk("d#c1", "x", slices=["dense-facts"])) is True
    assert is_a_priori_difficult(_chunk("d#c1", "x", slices=["prose"])) is False


# ---------------------------------------------------------------------------
# gate_claim -- mirrors the production span gate
# ---------------------------------------------------------------------------

def test_gate_claim_accepts_valid_span():
    chunk = _chunk("d#c1", "Leaders rotate each epoch.")
    claim = _claim_json("Leaders rotate each epoch", "d#c1", 0, 25)
    gated = gate_claim(claim, chunk)
    assert gated is not None
    assert gated["citable"] is True
    assert gated["supporting_evidence"][0]["quote"] == "Leaders rotate each epoch"


def test_gate_claim_rejects_out_of_bounds_span():
    chunk = _chunk("d#c1", "short")
    claim = _claim_json("bogus", "d#c1", 0, 999)
    assert gate_claim(claim, chunk) is None


def test_gate_claim_rejects_wrong_locator():
    chunk = _chunk("d#c1", "Leaders rotate each epoch.")
    claim = _claim_json("Leaders rotate each epoch", "OTHER#c9", 0, 26)
    assert gate_claim(claim, chunk) is None


def test_gate_claim_rejects_no_evidence():
    chunk = _chunk("d#c1", "text")
    assert gate_claim({"claim": "x", "supporting_evidence": []}, chunk) is None


def test_gate_claim_does_not_mutate_input():
    chunk = _chunk("d#c1", "Leaders rotate each epoch.")
    claim = _claim_json("Leaders rotate each epoch", "d#c1", 0, 26)
    original_evidence = claim["supporting_evidence"][0]
    gate_claim(claim, chunk)
    assert "quote" not in original_evidence  # the input claim dict is untouched


# ---------------------------------------------------------------------------
# escalating_k_sample -- DART escalation trigger
# ---------------------------------------------------------------------------

def test_escalation_triggers_on_low_yield_and_stops_once_floor_cleared():
    chunk = _chunk("d#c1", "Leaders rotate each epoch.", slices=["prose"])
    calls = []

    def teacher(chunk_batch, k, temperature):
        calls.append(k)
        if k == 4:
            return [_completion()]  # nothing gate-passable at k=4 -> low yield
        # k=16: one gate-passing claim -> clears yield_floor=1, should stop here
        return [_completion(_claim_json("Leaders rotate each epoch", "d#c1", 0, 26))]

    result = escalating_k_sample(chunk, teacher, k_ladder=(4, 16, 64), yield_floor=1)
    assert calls == [4, 16]  # escalated once, then stopped -- 64 never tried
    assert result["escalated"] is True
    assert len(result["claims"]) == 1


def test_no_escalation_when_first_rung_clears_floor():
    chunk = _chunk("d#c1", "Leaders rotate each epoch.", slices=["prose"])
    calls = []

    def teacher(chunk_batch, k, temperature):
        calls.append(k)
        return [_completion(_claim_json("Leaders rotate each epoch", "d#c1", 0, 26))]

    result = escalating_k_sample(chunk, teacher, k_ladder=(4, 16, 64), yield_floor=1)
    assert calls == [4]
    assert result["escalated"] is False


def test_a_priori_difficult_chunk_skips_cheapest_rung():
    chunk = _chunk("d#c1", "Leaders rotate each epoch.", slices=["bait"])
    calls = []

    def teacher(chunk_batch, k, temperature):
        calls.append(k)
        return [_completion(_claim_json("Leaders rotate each epoch", "d#c1", 0, 26))]

    result = escalating_k_sample(chunk, teacher, k_ladder=(4, 16, 64), yield_floor=1)
    assert calls == [16]  # k=4 skipped because the chunk is a priori hard (bait)
    assert result["escalated"] is True


def test_escalation_exhausts_ladder_when_never_clearing_floor():
    chunk = _chunk("d#c1", "text", slices=["prose"])
    calls = []

    def teacher(chunk_batch, k, temperature):
        calls.append(k)
        return [_completion()]  # never gate-passable

    result = escalating_k_sample(chunk, teacher, k_ladder=(4, 16, 64), yield_floor=1)
    assert calls == [4, 16, 64]
    assert result["claims"] == []


def test_escalating_k_sample_parses_multiple_claims_per_completion():
    chunk = _chunk("d#c1", "Leaders rotate. Followers verify.")
    claim_a = _claim_json("Leaders rotate", "d#c1", 0, 15, claim_id="c1")
    claim_b = _claim_json("Followers verify", "d#c1", 16, 33, claim_id="c2")

    def teacher(chunk_batch, k, temperature):
        return [_completion(claim_a, claim_b)]

    result = escalating_k_sample(chunk, teacher, k_ladder=(4,), yield_floor=1)
    assert len(result["claims"]) == 2


def test_rounds_record_parsed_claim_counts_not_completion_counts():
    """I5 probe: ONE completion carrying 5 claims, 4 of which the gate
    rejects. The round must report parsed=5 / accepted=1 (claim units), so
    parsed - accepted == 4 gate rejections -- a completion-count round would
    report raw=1 and make the rejection arithmetic nonsense."""
    text = "Leaders rotate each epoch."
    chunk = _chunk("d#c1", text)
    good = _claim_json("Leaders rotate each epoch", "d#c1", 0, 25, claim_id="ok")
    bad = [_claim_json(f"bogus {i}", "d#c1", 0, 999, claim_id=f"bad{i}") for i in range(4)]

    def teacher(chunk_batch, k, temperature):
        return [_completion(good, *bad)]  # ONE completion, five claims

    result = escalating_k_sample(chunk, teacher, k_ladder=(4,), yield_floor=1)
    assert result["rounds"][0]["completions"] == 1
    assert result["rounds"][0]["parsed"] == 5
    assert result["rounds"][0]["accepted"] == 1
    assert result["rounds"][0]["parsed"] - result["rounds"][0]["accepted"] == 4


def test_per_chunk_completion_budget_stops_escalation():
    """M2: a rung that would exceed max_completions_per_chunk is not
    attempted -- with cap 10 and ladder (4, 16, 64), only k=4 runs."""
    chunk = _chunk("d#c1", "text", slices=["prose"])
    calls = []

    def teacher(chunk_batch, k, temperature):
        calls.append(k)
        return [_completion()] * k  # never gate-passable -> wants to escalate

    result = escalating_k_sample(chunk, teacher, k_ladder=(4, 16, 64), yield_floor=1,
                                 max_completions_per_chunk=10)
    assert calls == [4]                       # 4+16 > 10 -> k=16 never attempted
    assert result["budget_exhausted"] is True
    assert result["completions_used"] == 4


# ---------------------------------------------------------------------------
# dedup_claims
# ---------------------------------------------------------------------------

def test_dedup_claims_collapses_near_duplicates():
    a = _claim_json("Leaders rotate each epoch", "d#c1", 0, 26, claim_id="c1")
    b = _claim_json("leaders  rotate each epoch", "d#c1", 0, 27, claim_id="c2")  # near-dup: whitespace/case/offset jitter
    c = _claim_json("Followers verify blocks", "d#c1", 30, 54, claim_id="c3")
    deduped = dedup_claims([a, b, c])
    assert len(deduped) == 2
    assert deduped[0]["claim_id"] == "c1"  # first occurrence wins


def test_dedup_claims_keeps_distinct_locators_distinct():
    a = _claim_json("Same text", "d#c1", 0, 9, claim_id="c1")
    b = _claim_json("Same text", "d#c2", 0, 9, claim_id="c2")
    assert len(dedup_claims([a, b])) == 2


# ---------------------------------------------------------------------------
# Contamination guard (HARD)
# ---------------------------------------------------------------------------

def _write_eval_corpus(tmp_path):
    corpus_dir = tmp_path / "eval-corpus"
    doc_dir = corpus_dir / "distributed-consensus"
    doc_dir.mkdir(parents=True)
    (doc_dir / "chunks.jsonl").write_text(
        json.dumps({"locator": "distributed-consensus#c001", "text": "Leaders rotate each epoch."}) + "\n",
        encoding="utf-8")
    (corpus_dir / "corpus-index.json").write_text(
        json.dumps({"chunks": {"distributed-consensus#c001": {"slices": ["prose"], "doc": "distributed-consensus"}}}),
        encoding="utf-8")
    return corpus_dir


def test_contamination_guard_raises_on_locator_match(tmp_path):
    corpus_dir = _write_eval_corpus(tmp_path)
    index = load_contamination_index(corpus_dir)
    contaminated = _chunk("distributed-consensus#c001", "some other text entirely")
    with pytest.raises(ContaminationError):
        assert_not_contaminated(contaminated, index)


def test_contamination_guard_raises_on_text_hash_match_even_with_different_locator(tmp_path):
    corpus_dir = _write_eval_corpus(tmp_path)
    index = load_contamination_index(corpus_dir)
    # Different locator, but the EXACT SAME text as the eval chunk -- must
    # still be caught by the content-hash half of the guard.
    duplicate = _chunk("training-corpus#c999", "Leaders rotate each epoch.")
    with pytest.raises(ContaminationError):
        assert_not_contaminated(duplicate, index)


def test_contamination_guard_passes_clean_chunk(tmp_path):
    corpus_dir = _write_eval_corpus(tmp_path)
    index = load_contamination_index(corpus_dir)
    clean = _chunk("training-corpus#c001", "Completely different training text.")
    assert_not_contaminated(clean, index)  # does not raise


def test_load_contamination_index_missing_dir_degrades_to_empty(tmp_path):
    index = load_contamination_index(tmp_path / "does-not-exist")
    assert index == {"locators": set(), "text_hashes": set()}


def test_contamination_text_hash_ignores_case_and_whitespace(tmp_path):
    """M1: a copy of an eval chunk that differs only by case or reflowed
    whitespace must still be caught by the content-hash half of the guard."""
    corpus_dir = _write_eval_corpus(tmp_path)  # eval text: "Leaders rotate each epoch."
    index = load_contamination_index(corpus_dir)
    sneaky = _chunk("training#c777", "  LEADERS   rotate\teach epoch. ")
    with pytest.raises(ContaminationError):
        assert_not_contaminated(sneaky, index)


def test_load_contamination_index_warns_on_missing_chunks_jsonl(tmp_path, caplog):
    """M1: a doc dir without chunks.jsonl silently weakens the text-hash
    half of the guard -- it must WARN, not degrade in silence."""
    corpus_dir = tmp_path / "eval-corpus"
    bare_doc = corpus_dir / "doc-without-chunks"
    bare_doc.mkdir(parents=True)
    (corpus_dir / "corpus-index.json").write_text(
        json.dumps({"chunks": {"doc-without-chunks#c001": {"slices": ["prose"]}}}),
        encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="deep_research_toolkit.tunekit.dataset"):
        index = load_contamination_index(corpus_dir)

    assert index["locators"] == {"doc-without-chunks#c001"}
    assert index["text_hashes"] == set()
    assert any("no chunks.jsonl" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# to_conversation_record: output-format fidelity to the extract contract
# ---------------------------------------------------------------------------

def test_conversation_record_round_trips_through_the_extract_parser():
    chunk = _chunk("d#c1", "Leaders rotate each epoch.")
    claims = [gate_claim(_claim_json("Leaders rotate each epoch", "d#c1", 0, 26), chunk)]
    record = to_conversation_record(chunk, claims, producer="web", thinking=False)

    roles = [m["role"] for m in record["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert "d#c1" in record["messages"][1]["content"]  # locator rendered into the user turn

    parsed = parse_extraction_response(record["messages"][2]["content"])
    assert len(parsed["claims"]) == 1
    assert parsed["claims"][0]["claim"] == "Leaders rotate each epoch"


def test_conversation_record_thinking_mode_wraps_output_tags():
    chunk = _chunk("d#c1", "Leaders rotate each epoch.")
    claims = [gate_claim(_claim_json("Leaders rotate each epoch", "d#c1", 0, 26), chunk)]
    record = to_conversation_record(chunk, claims, producer="web", thinking=True)
    assistant = record["messages"][2]["content"]
    assert assistant.startswith("<output>") and assistant.endswith("</output>")
    parsed = parse_extraction_response(assistant)
    assert len(parsed["claims"]) == 1


# ---------------------------------------------------------------------------
# split_train_val
# ---------------------------------------------------------------------------

def test_split_train_val_is_deterministic_and_covers_all_records():
    records = [{"i": i} for i in range(20)]
    train1, val1 = split_train_val(records, val_fraction=0.10, seed=42)
    train2, val2 = split_train_val(records, val_fraction=0.10, seed=42)
    assert train1 == train2 and val1 == val2
    assert len(val1) == 2  # round(20 * 0.10)
    assert len(train1) == 18
    assert sorted(r["i"] for r in train1 + val1) == list(range(20))


def test_split_train_val_empty_input():
    assert split_train_val([]) == ([], [])


def test_split_train_val_at_least_one_val_record_when_fraction_positive():
    records = [{"i": i} for i in range(3)]
    train, val = split_train_val(records, val_fraction=0.10, seed=1)
    assert len(val) == 1


# ---------------------------------------------------------------------------
# dataset_hash / build_manifest
# ---------------------------------------------------------------------------

def test_dataset_hash_deterministic():
    records = [{"a": 1}, {"b": 2}]
    assert dataset_hash(records) == dataset_hash(records)
    assert dataset_hash(records).startswith("sha256:")


def test_dataset_hash_order_sensitive():
    assert dataset_hash([{"a": 1}, {"b": 2}]) != dataset_hash([{"b": 2}, {"a": 1}])


def test_build_manifest_stats():
    train = [{"i": 1}, {"i": 2}]
    val = [{"i": 3}]
    manifest = build_manifest(train, val, generator_model_digests={"e4b": 3},
                              source_corpus_hash="sha256:corpus", n_accepted_claims=3,
                              n_gate_rejected=5, n_dedup_dropped=2)
    assert manifest["n_accepted"] == 3
    assert manifest["n_rejected"] == 7
    assert manifest["n_train"] == 2
    assert manifest["n_val"] == 1
    assert manifest["acceptance_rate"] == pytest.approx(3 / 10)
    assert manifest["dataset_hash"] == dataset_hash(train + val)


def test_build_manifest_acceptance_rate_is_in_claim_units():
    """I5: acceptance_rate must be accepted_claims / parsed_claims -- record
    counts (n_train/n_val, one per chunk) never enter the rate."""
    train = [{"one_record_covering": "many claims"}]
    manifest = build_manifest(train, [], generator_model_digests={"e4b": 6},
                              source_corpus_hash=None, n_accepted_claims=6,
                              n_gate_rejected=4, n_dedup_dropped=0)
    assert manifest["n_accepted"] == 6            # claims, not the 1 record
    assert manifest["acceptance_rate"] == pytest.approx(0.6)
    assert manifest["n_train"] == 1


def test_build_manifest_zero_total_has_none_acceptance_rate():
    manifest = build_manifest([], [], generator_model_digests={}, source_corpus_hash=None,
                              n_accepted_claims=0, n_gate_rejected=0, n_dedup_dropped=0)
    assert manifest["acceptance_rate"] is None


# ---------------------------------------------------------------------------
# build_sft_dataset: end-to-end with a scripted fake teacher
# ---------------------------------------------------------------------------

def _fake_teacher_factory(claims_by_locator: dict[str, list[dict]]):
    """Returns a TeacherFn that always returns the scripted claims for the
    chunk it's called on, regardless of k -- enough to drive
    build_sft_dataset end-to-end without ever hitting yield-floor escalation
    logic (that's exercised separately above)."""
    def teacher(chunk_batch, k, temperature):
        chunk = chunk_batch[0]
        loc = chunk.get("locator") or chunk.get("node_id")
        claims = claims_by_locator.get(loc, [])
        return [_completion(*claims)] if claims else [_completion()]
    return teacher


def test_build_sft_dataset_end_to_end(tmp_path):
    corpus_dir = _write_eval_corpus(tmp_path)  # for the contamination index only
    contamination_index = load_contamination_index(corpus_dir)

    # Deliberately DIFFERENT text from _write_eval_corpus's eval chunk (else
    # the (working-as-intended) text-hash contamination guard would trip).
    chunk1 = _chunk("training#c001", "Validators exchange heartbeats every cycle.", slices=["prose"])
    chunk2 = _chunk("training#c002", "Auditors record checkpoints hourly.", slices=["prose"])
    claims_by_locator = {
        "training#c001": [_claim_json("Validators exchange heartbeats every cycle", "training#c001", 0, 42)],
        "training#c002": [_claim_json("Auditors record checkpoints hourly", "training#c002", 0, 34)],
    }
    teachers = {"e4b": _fake_teacher_factory(claims_by_locator)}

    result = build_sft_dataset([chunk1, chunk2], teachers, contamination_index,
                               val_fraction=0.5, seed=1, source_corpus_hash="sha256:training-corpus")

    all_records = result["train"] + result["val"]
    assert len(all_records) == 2
    assert result["manifest"]["n_accepted"] == 2
    assert result["manifest"]["source_corpus_hash"] == "sha256:training-corpus"
    assert result["manifest"]["generator_model_digests"] == {"e4b": 2}
    assert len(result["escalation_log"]) == 2

    # Round-trips through the production parser for every written example.
    for record in all_records:
        parsed = parse_extraction_response(record["messages"][2]["content"])
        assert len(parsed["claims"]) == 1


def test_build_sft_dataset_raises_on_contaminated_chunk(tmp_path):
    corpus_dir = _write_eval_corpus(tmp_path)
    contamination_index = load_contamination_index(corpus_dir)
    contaminated_chunk = _chunk("distributed-consensus#c001", "Leaders rotate each epoch.")
    teachers = {"e4b": _fake_teacher_factory({})}

    with pytest.raises(ContaminationError):
        build_sft_dataset([contaminated_chunk], teachers, contamination_index)


def test_build_sft_dataset_dedups_across_chunks(tmp_path):
    corpus_dir = _write_eval_corpus(tmp_path)
    contamination_index = load_contamination_index(corpus_dir)

    # Two different chunks whose teacher, oddly, both assert the exact same
    # fact citing the SAME locator (a plausible near-dup from re-sampling
    # escalation) -- claim_key collapses them. Text is deliberately DIFFERENT
    # from _write_eval_corpus's eval chunk (else the text-hash contamination
    # guard would (correctly) trip).
    chunk1 = _chunk("training#c001", "Validators exchange heartbeats every cycle.", slices=["prose"])
    dup_claim_1 = _claim_json("Validators exchange heartbeats every cycle", "training#c001", 0, 42, claim_id="a")
    dup_claim_2 = _claim_json("validators  exchange heartbeats every cycle", "training#c001", 0, 42, claim_id="b")
    teachers = {"e4b": _fake_teacher_factory({"training#c001": [dup_claim_1, dup_claim_2]})}

    result = build_sft_dataset([chunk1], teachers, contamination_index)
    all_records = result["train"] + result["val"]
    assert len(all_records) == 1
    parsed = parse_extraction_response(all_records[0]["messages"][2]["content"])
    assert len(parsed["claims"]) == 1  # the dup collapsed
    assert result["manifest"]["n_dedup_dropped"] == 1


def test_gate_rejected_counts_claims_not_completions(tmp_path):
    """I5 probe, end to end: ONE completion with 5 claims, 4 gate-rejected
    -> manifest n_gate_rejected == 4 and acceptance_rate == 1/5. The old
    completion-unit arithmetic would have reported 0 rejections (1
    completion - 1 'accepted')."""
    corpus_dir = _write_eval_corpus(tmp_path)
    contamination_index = load_contamination_index(corpus_dir)

    text = "Validators exchange heartbeats every cycle."
    chunk = _chunk("training#c001", text, slices=["prose"])
    good = _claim_json("Validators exchange heartbeats every cycle", "training#c001", 0, 42, claim_id="ok")
    bad = [_claim_json(f"bogus {i}", "training#c001", 0, 999, claim_id=f"bad{i}") for i in range(4)]

    def teacher(chunk_batch, k, temperature):
        return [_completion(good, *bad)]  # ONE completion, five claims

    result = build_sft_dataset([chunk], {"e4b": teacher}, contamination_index)
    assert result["manifest"]["n_gate_rejected"] == 4
    assert result["manifest"]["n_accepted"] == 1
    assert result["manifest"]["acceptance_rate"] == pytest.approx(1 / 5)


def test_total_completions_budget_skips_later_chunks(tmp_path, caplog):
    """M2: with max_total_completions=4 and a 4-completion first chunk, the
    second chunk is skipped (logged + recorded), never half-sampled."""
    corpus_dir = _write_eval_corpus(tmp_path)
    contamination_index = load_contamination_index(corpus_dir)

    chunk1 = _chunk("training#c001", "Validators exchange heartbeats every cycle.", slices=["prose"])
    chunk2 = _chunk("training#c002", "Auditors record checkpoints hourly.", slices=["prose"])
    good1 = _claim_json("Validators exchange heartbeats every cycle", "training#c001", 0, 42)
    teacher_calls = []

    def teacher(chunk_batch, k, temperature):
        teacher_calls.append(chunk_batch[0]["locator"])
        return [_completion(good1)] * k  # k completions per call

    with caplog.at_level(logging.WARNING, logger="deep_research_toolkit.tunekit.dataset"):
        result = build_sft_dataset([chunk1, chunk2], {"e4b": teacher}, contamination_index,
                                   k_ladder=(4,), max_total_completions=4)

    assert teacher_calls == ["training#c001"]  # chunk2's teacher never called
    skipped = [e for e in result["escalation_log"] if e.get("skipped")]
    assert len(skipped) == 1
    assert skipped[0]["locator"] == "training#c002"
    assert skipped[0]["skipped"] == "total-completions-budget"
    assert any("max_total_completions" in rec.message for rec in caplog.records)


def test_unlimited_total_budget_logs_usage(tmp_path, caplog):
    """M2: max_total_completions=None is allowed but the total used must be
    logged, so an unbounded run is at least visible."""
    corpus_dir = _write_eval_corpus(tmp_path)
    contamination_index = load_contamination_index(corpus_dir)
    chunk = _chunk("training#c001", "Validators exchange heartbeats every cycle.", slices=["prose"])
    good = _claim_json("Validators exchange heartbeats every cycle", "training#c001", 0, 42)

    def teacher(chunk_batch, k, temperature):
        return [_completion(good)]

    with caplog.at_level(logging.INFO, logger="deep_research_toolkit.tunekit.dataset"):
        build_sft_dataset([chunk], {"e4b": teacher}, contamination_index)

    assert any("without a max_total_completions cap" in rec.message for rec in caplog.records)
