"""Unit tests for scripts/eval-pipeline.py's importable pure logic: corpus
loading/limiting, stratified sampling, the adjudicate synthetic-candidate
protocol ("pair-claims-v2"), report assembly, history lines, and baseline
comparison. Extraction/prose-role wiring and role-level aggregation are
exercised end-to-end with fake in-process backends (never a live model) so
the whole pipeline is drivable in the fast suite; only the CLI's actual
`get_backend(config, ...)` construction against a live endpoint is left for
the live tier / Task 8 runbook."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "eval_pipeline", REPO_ROOT / "scripts" / "eval-pipeline.py")
eval_pipeline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_pipeline)

DEFAULT_CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "eval-corpus"


# ---------------------------------------------------------------------------
# corpus helpers (pure filesystem reads over a synthetic tmp corpus)
# ---------------------------------------------------------------------------

def _write_doc(root: Path, doc_id: str, chunks: list[dict], claims: list[dict]) -> None:
    doc_dir = root / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "manifest.json").write_text(
        json.dumps({"document_id": doc_id, "producer": "web"}), encoding="utf-8")
    with open(doc_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(doc_dir / "reference-claims.jsonl", "w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def _claim(claim_id: str, locator: str, quote: str) -> dict:
    return {"claim_id": claim_id, "claim": f"claim about {quote[:20]}",
            "supporting_evidence": [{"locator": locator, "quote": quote, "url": None}]}


def _build_tiny_corpus(root: Path) -> None:
    _write_doc(
        root, "doc-a",
        chunks=[
            {"locator": "doc-a#c001", "text": "Alpha fact one. Alpha fact two."},
            {"locator": "doc-a#c002", "text": "Beta fact one. Beta fact two."},
        ],
        claims=[
            _claim("a_c1", "doc-a#c001", "Alpha fact one"),
            _claim("a_c2", "doc-a#c002", "Beta fact one"),
        ],
    )
    _write_doc(
        root, "doc-b",
        chunks=[{"locator": "doc-b#c001", "text": "Gamma fact one. Gamma fact two."}],
        claims=[_claim("b_c1", "doc-b#c001", "Gamma fact one")],
    )
    index = {
        "chunks": {
            "doc-a#c001": {"slices": ["prose"], "doc": "doc-a"},
            "doc-a#c002": {"slices": ["bait"], "doc": "doc-a"},
            "doc-b#c001": {"slices": ["dense-facts"], "doc": "doc-b"},
        },
        "bait_sources": {"doc-a#c002": "doc-a#c001"},
        "contradiction_pairs": [
            {"a": "doc-a#c001", "b": "doc-b#c001",
             "claim_a": "a_c1", "claim_b": "b_c1",
             "verdict": "contradiction", "note": "widget count: 10 vs 20"},
        ],
        "corpus_version": "sha256:test",
    }
    (root / "corpus-index.json").write_text(json.dumps(index), encoding="utf-8")


def test_load_claims_by_chunk_maps_every_cited_locator(tmp_path):
    _build_tiny_corpus(tmp_path)
    by_chunk = eval_pipeline.load_claims_by_chunk(tmp_path)
    assert {c["claim_id"] for c in by_chunk["doc-a#c001"]} == {"a_c1"}
    assert {c["claim_id"] for c in by_chunk["doc-b#c001"]} == {"b_c1"}


def test_select_docs_for_limit_none_returns_all_docs_uncapped(tmp_path):
    _build_tiny_corpus(tmp_path)
    selection = eval_pipeline.select_docs_for_limit(tmp_path, None)
    assert selection == [("doc-a", None), ("doc-b", None)]


def test_select_docs_for_limit_caps_mid_doc_and_omits_the_rest(tmp_path):
    _build_tiny_corpus(tmp_path)
    # doc-a has 2 chunks, doc-b has 1 -- a limit of 1 should take only doc-a's
    # first chunk and never reach doc-b.
    selection = eval_pipeline.select_docs_for_limit(tmp_path, 1)
    assert selection == [("doc-a", 1)]


def test_select_docs_for_limit_spanning_two_docs(tmp_path):
    _build_tiny_corpus(tmp_path)
    selection = eval_pipeline.select_docs_for_limit(tmp_path, 3)
    assert selection == [("doc-a", None), ("doc-b", None)]  # exactly covers both, uncapped


# ---------------------------------------------------------------------------
# stratified_sample_chunks
# ---------------------------------------------------------------------------

def test_stratified_sample_respects_k_and_is_deterministic():
    index = {"chunks": {
        f"doc#c{i:03d}": {"slices": ["prose" if i % 2 == 0 else "dense-facts"]}
        for i in range(20)
    }}
    a = eval_pipeline.stratified_sample_chunks(index, k=6, seed=7)
    b = eval_pipeline.stratified_sample_chunks(index, k=6, seed=7)
    assert a == b
    assert len(a) == 6
    assert len(set(a)) == 6


def test_stratified_sample_draws_from_multiple_slices_when_available():
    index = {"chunks": {
        "p1": {"slices": ["prose"]}, "p2": {"slices": ["prose"]}, "p3": {"slices": ["prose"]},
        "d1": {"slices": ["dense-facts"]}, "d2": {"slices": ["dense-facts"]},
    }}
    picked = eval_pipeline.stratified_sample_chunks(index, k=4, seed=7)
    slices_hit = set()
    for loc in picked:
        slices_hit.update(index["chunks"][loc]["slices"])
    assert slices_hit == {"prose", "dense-facts"}


def test_stratified_sample_caps_at_available_chunks():
    index = {"chunks": {"only-one": {"slices": ["prose"]}}}
    picked = eval_pipeline.stratified_sample_chunks(index, k=10, seed=7)
    assert picked == ["only-one"]


def test_stratified_sample_excludes_chunks_with_no_gold_claims():
    # A prose role can't be exercised on a chunk with nothing to cite, so a
    # claim-less chunk must never enter the sample when the gold map is given.
    index = {"chunks": {
        "with-claims": {"slices": ["prose"]},
        "no-claims": {"slices": ["prose"]},
    }}
    claims_by_chunk = {"with-claims": [{"claim_id": "c1"}]}
    picked = eval_pipeline.stratified_sample_chunks(
        index, k=5, seed=7, claims_by_chunk=claims_by_chunk)
    assert picked == ["with-claims"]


# ---------------------------------------------------------------------------
# adjudicate synthetic-candidate protocol ("pair-claims-v2")
# ---------------------------------------------------------------------------

def test_slugify_basic():
    assert eval_pipeline.slugify("MCB v2 release year") == "mcb-v2-release-year"


def test_load_claims_by_id_maps_every_claim(tmp_path):
    _build_tiny_corpus(tmp_path)
    by_id = eval_pipeline.load_claims_by_id(tmp_path)
    assert set(by_id) == {"a_c1", "a_c2", "b_c1"}
    assert by_id["a_c1"]["supporting_evidence"][0]["locator"] == "doc-a#c001"


def test_build_adjudicate_candidates_shape_from_pinned_gold_claims():
    claims_by_id = {
        "a1": _claim("a1", "doc-a#c001", "widget count was 10"),
        "b1": _claim("b1", "doc-b#c001", "widget count was 20"),
    }
    pairs = [
        {"a": "doc-a#c001", "b": "doc-b#c001", "claim_a": "a1", "claim_b": "b1",
         "verdict": "contradiction", "note": "widget count: 10 vs 20"},
    ]
    candidates, meta, warnings = eval_pipeline.build_adjudicate_candidates(pairs, claims_by_id)
    assert warnings == []
    assert len(candidates) == 1 and len(meta) == 1
    cand = candidates[0]
    assert cand["kind"] == "relation"
    assert cand["predicate"] == "asserts"
    assert cand["subject"] == "widget-count-10-vs-20"  # full note, slugified
    assert cand["relation_ids"] == ["a1", "b1"]
    assert cand["objects"] == ["claim about widget count was 10", "claim about widget count was 20"]
    assert cand["source_ids"] == ["doc-a#c001", "doc-b#c001"]
    assert meta[0]["gold_verdict"] == "contradiction"
    assert meta[0]["subject"] == "widget-count-10-vs-20"
    assert meta[0]["predicate"] == "asserts"


def test_build_adjudicate_candidates_dedupes_colliding_subjects():
    claims_by_id = {
        "a1": _claim("a1", "doc-a#c001", "x"), "b1": _claim("b1", "doc-b#c001", "y"),
        "c1": _claim("c1", "doc-c#c001", "z"), "d1": _claim("d1", "doc-d#c001", "w"),
    }
    pairs = [
        {"a": "doc-a#c001", "b": "doc-b#c001", "claim_a": "a1", "claim_b": "b1",
         "verdict": "contradiction", "note": "same topic"},
        {"a": "doc-c#c001", "b": "doc-d#c001", "claim_a": "c1", "claim_b": "d1",
         "verdict": "not_contradiction", "note": "same topic"},
    ]
    candidates, meta, warnings = eval_pipeline.build_adjudicate_candidates(pairs, claims_by_id)
    assert warnings == []
    subjects = [c["subject"] for c in candidates]
    assert len(subjects) == len(set(subjects))  # never collide


def test_build_adjudicate_candidates_skips_missing_gold_claim_with_warning():
    claims_by_id = {"a1": _claim("a1", "doc-a#c001", "x")}
    pairs = [
        {"a": "doc-a#c001", "b": "doc-b#c001", "claim_a": "a1", "claim_b": "ghost",
         "verdict": "contradiction", "note": "n"},
    ]
    candidates, meta, warnings = eval_pipeline.build_adjudicate_candidates(pairs, claims_by_id)
    assert candidates == [] and meta == []
    assert len(warnings) == 1
    assert "ghost" in warnings[0] and "skipped" in warnings[0]


def test_build_adjudicate_candidates_truncates_long_note_subjects():
    claims_by_id = {
        "a1": _claim("a1", "doc-a#c001", "x"), "b1": _claim("b1", "doc-b#c001", "y"),
    }
    long_note = "an extremely long note " * 10 + ": 1 vs 2"
    pairs = [{"a": "doc-a#c001", "b": "doc-b#c001", "claim_a": "a1", "claim_b": "b1",
              "verdict": "contradiction", "note": long_note}]
    candidates, _, _ = eval_pipeline.build_adjudicate_candidates(pairs, claims_by_id)
    assert len(candidates[0]["subject"]) <= 60
    assert not candidates[0]["subject"].endswith("-")


def test_score_adjudicate_exact_match_is_full_credit():
    meta = [{"subject": "s1", "predicate": "asserts", "gold_verdict": "contradiction", "note": "n"}]
    result = {"verdicts": [{"subject": "s1", "predicate": "asserts", "verdict": "contradiction",
                            "relation_ids": ["a1", "b1"], "rationale": "x"}],
              "invalid": [], "parse_failures": 0}
    out = eval_pipeline.score_adjudicate(result, meta)
    assert out["accuracy"] == 1.0
    assert out["adjudicate_protocol"] == "pair-claims-v2"


def test_score_adjudicate_insufficient_evidence_is_half_credit():
    meta = [{"subject": "s1", "predicate": "asserts", "gold_verdict": "contradiction", "note": "n"}]
    result = {"verdicts": [{"subject": "s1", "predicate": "asserts", "verdict": "insufficient_evidence",
                            "relation_ids": ["a1", "b1"], "rationale": "x"}],
              "invalid": [], "parse_failures": 0}
    out = eval_pipeline.score_adjudicate(result, meta)
    assert out["accuracy"] == 0.5


def test_score_adjudicate_wrong_verdict_is_zero_credit():
    meta = [{"subject": "s1", "predicate": "asserts", "gold_verdict": "contradiction", "note": "n"}]
    result = {"verdicts": [{"subject": "s1", "predicate": "asserts", "verdict": "not_contradiction",
                            "relation_ids": ["a1", "b1"], "rationale": "x"}],
              "invalid": [], "parse_failures": 0}
    out = eval_pipeline.score_adjudicate(result, meta)
    assert out["accuracy"] == 0.0


def test_score_adjudicate_missing_verdict_is_zero_credit():
    meta = [{"subject": "s1", "predicate": "asserts", "gold_verdict": "contradiction", "note": "n"}]
    result = {"verdicts": [], "invalid": [{"row": {}, "reason": "bad"}], "parse_failures": 0}
    out = eval_pipeline.score_adjudicate(result, meta)
    assert out["accuracy"] == 0.0
    assert out["schema_invalid"] == 1


def test_score_adjudicate_averages_over_multiple_pairs():
    meta = [
        {"subject": "s1", "predicate": "asserts", "gold_verdict": "contradiction", "note": "n"},
        {"subject": "s2", "predicate": "asserts", "gold_verdict": "not_contradiction", "note": "n"},
    ]
    result = {"verdicts": [
        {"subject": "s1", "predicate": "asserts", "verdict": "contradiction", "relation_ids": [], "rationale": "x"},
        {"subject": "s2", "predicate": "asserts", "verdict": "insufficient_evidence", "relation_ids": [], "rationale": "x"},
    ], "invalid": [], "parse_failures": 0}
    out = eval_pipeline.score_adjudicate(result, meta)
    assert out["accuracy"] == 0.75  # (1.0 + 0.5) / 2


# ---------------------------------------------------------------------------
# extraction wiring with a fake backend (no live model)
# ---------------------------------------------------------------------------

class _FakeExtractBackend:
    def __init__(self, by_locator: dict):
        self.by_locator = by_locator

    def complete(self, system, user, **kw):
        for locator, payload in self.by_locator.items():
            if locator in user:
                return payload
        return "{}"


def test_run_extract_for_doc_scores_against_untouched_reference(tmp_path):
    _build_tiny_corpus(tmp_path)
    from types import SimpleNamespace
    config = SimpleNamespace(research_runs_path=tmp_path / "unused", pdf_runs_path=tmp_path / "unused2")
    payload = json.dumps([
        {"claim_id": "p1", "claim": "x", "supporting_evidence":  # chunk[0:14] == "Alpha fact one"
         [{"locator": "doc-a#c001", "start_char": 0, "end_char": 14, "url": None}]},
    ])
    backend = _FakeExtractBackend({"doc-a#c001": payload})
    out = eval_pipeline.run_extract_for_doc(tmp_path / "doc-a", config, backend)
    assert out["written"] == 1
    assert len(out["reference"]) == 2  # untouched reference-claims.jsonl, both claims
    # the original doc dir must be untouched (no claims.jsonl written there)
    assert not (tmp_path / "doc-a" / "claims.jsonl").exists()


def test_run_extract_for_doc_respects_chunk_limit(tmp_path):
    _build_tiny_corpus(tmp_path)
    from types import SimpleNamespace
    config = SimpleNamespace(research_runs_path=tmp_path / "unused", pdf_runs_path=tmp_path / "unused2")
    backend = _FakeExtractBackend({})  # returns "{}" always -> nothing extracted
    out = eval_pipeline.run_extract_for_doc(tmp_path / "doc-a", config, backend, chunk_limit=1)
    # only c001's claims should remain as "reference" once c002 is excluded by the cap
    assert {c["claim_id"] for c in out["reference"]} == {"a_c1"}


# ---------------------------------------------------------------------------
# aggregation end-to-end (fake backends, no live calls)
# ---------------------------------------------------------------------------

def test_run_extract_for_model_aggregates_recall_across_docs(tmp_path):
    # End-to-end over the whole extract aggregation path: gold comes from the
    # UNTOUCHED reference-claims.jsonl files, pooled across docs. The fake
    # backend recovers exactly one of the corpus's three reference claims, so
    # overall recall must be exactly 1/3 -- a wrong-gold-file regression (e.g.
    # scoring against the model's own output) would move this number.
    _build_tiny_corpus(tmp_path)
    from types import SimpleNamespace
    config = SimpleNamespace(research_runs_path=tmp_path / "unused", pdf_runs_path=tmp_path / "unused2")
    payload = json.dumps([
        {"claim_id": "p1", "claim": "x", "supporting_evidence":  # chunk[0:14] == "Alpha fact one"
         [{"locator": "doc-a#c001", "start_char": 0, "end_char": 14, "url": None}]},
    ])
    backend = _FakeExtractBackend({"doc-a#c001": payload})  # doc-b gets "{}"
    index = json.loads((tmp_path / "corpus-index.json").read_text(encoding="utf-8"))
    doc_selection = eval_pipeline.select_docs_for_limit(tmp_path, None)

    out = eval_pipeline.run_extract_for_model(tmp_path, index, config, backend, doc_selection)

    assert out["recall"] == pytest.approx(1 / 3)     # 1 of 3 pooled reference claims
    assert out["gate_pass_rate"] == 1.0              # nothing dropped
    assert out["gold_match_rate"] == 1.0             # the one produced claim matched
    assert out["docs"] == 2
    assert out["per_doc"]["doc-a"]["recall"] == pytest.approx(0.5)   # 1 of doc-a's 2
    assert out["per_doc"]["doc-b"]["recall"] == 0.0                  # doc-b's 1 missed


class _KeyedProseBackend:
    """Returns a canned reply based on which claim id appears in the prompt."""

    def __init__(self, replies_by_key: dict):
        self.replies_by_key = replies_by_key

    def complete(self, system, user, **sampling):
        for key, reply in self.replies_by_key.items():
            if key in user:
                return reply
        raise AssertionError(f"no canned reply matches prompt: {user[:120]}")


def test_run_prose_role_aggregates_pass_rate_and_bare_marker_rate(tmp_path):
    # End-to-end over the prose aggregation path with controlled raw replies:
    # chunk 1's reply cites with the claim: prefix, chunk 2's reply cites
    # bare -- both pass the gate (bare markers are normalized), so
    # mean_pass_rate is 1.0 while bare_marker_rate (measured on the RAW
    # replies via RecordingBackend) is exactly 1/2.
    _build_tiny_corpus(tmp_path)
    claims_by_chunk = eval_pipeline.load_claims_by_chunk(tmp_path)
    backend = _KeyedProseBackend({
        "a_c1": "Alpha holds [claim:a_c1].",
        "a_c2": "Beta holds [a_c2].",  # bare marker in the raw reply
    })
    out = eval_pipeline.run_prose_role_with_backend(
        "wiki_write", ["doc-a#c001", "doc-a#c002"], claims_by_chunk, backend, runs=1)
    assert out["mean_pass_rate"] == 1.0
    assert out["mean_coverage"] == 1.0
    assert out["marker_counts"] == {"bare": 1, "prefixed": 1}
    assert out["bare_marker_rate"] == pytest.approx(0.5)
    assert set(out["per_chunk"]) == {"doc-a#c001", "doc-a#c002"}


def test_real_corpus_adjudicate_candidates_carry_the_distinguishing_values():
    # The regression test for the v1 critical: every candidate built from the
    # committed corpus must put the note's distinguishing values in front of
    # the model. Every digit run in a pair's note (the conflicting years/
    # counts/sizes) must appear in the union of the two pinned claim texts.
    index = json.loads((DEFAULT_CORPUS_DIR / "corpus-index.json").read_text(encoding="utf-8"))
    pairs = index["contradiction_pairs"]
    claims_by_id = eval_pipeline.load_claims_by_id(DEFAULT_CORPUS_DIR)
    candidates, meta, warnings = eval_pipeline.build_adjudicate_candidates(pairs, claims_by_id)
    assert warnings == []
    assert len(candidates) == len(pairs) == 12
    for pair, cand in zip(pairs, candidates):
        assert cand["relation_ids"] == [pair["claim_a"], pair["claim_b"]]
        union = " ".join(cand["objects"])
        for digits in re.findall(r"\d+", pair["note"]):
            assert digits in union, (
                f"pair note {pair['note']!r}: value {digits!r} missing from the "
                f"candidate objects: {union!r}")


# ---------------------------------------------------------------------------
# Task 13 wiring: samples/min-support/parallel/pooled-gold flags,
# truncated_calls in the report, entailment recall via an injected embedder
# ---------------------------------------------------------------------------

class _FakeMetaExtractBackend:
    """Like _FakeExtractBackend but answers on the per-call meta channel
    (complete_with_meta), reporting every call's finish_reason -- so the
    truncation counter can be exercised without a live model."""

    def __init__(self, by_locator: dict, finish_reason: str | None = "length"):
        self.by_locator = by_locator
        self.finish_reason = finish_reason

    def complete_with_meta(self, system, user, **kw):
        for locator, payload in self.by_locator.items():
            if locator in user:
                return payload, self.finish_reason
        return "{}", self.finish_reason


def _make_one_hot_embedder(dim: int = 16):
    """Deterministic fake embedder callable (texts -> unit vectors): every
    distinct text gets its own one-hot axis, so cosine is exactly 1.0 for
    identical texts and 0.0 otherwise -- no model, no network."""
    index: dict[str, int] = {}

    def embed(texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            i = index.setdefault(t, len(index))
            assert i < dim, "one-hot fake embedder ran out of axes"
            v = [0.0] * dim
            v[i] = 1.0
            vecs.append(v)
        return vecs

    return embed


def _tiny_corpus_config(tmp_path):
    from types import SimpleNamespace
    return SimpleNamespace(research_runs_path=tmp_path / "unused",
                           pdf_runs_path=tmp_path / "unused2")


_ALPHA_PAYLOAD = json.dumps([
    {"claim_id": "p1", "claim": "claim about Alpha fact one", "supporting_evidence":
     [{"locator": "doc-a#c001", "start_char": 0, "end_char": 14, "url": None}]},
])
_GAMMA_PAYLOAD = json.dumps([
    {"claim_id": "p2", "claim": "some unrelated paraphrase", "supporting_evidence":
     [{"locator": "doc-b#c001", "start_char": 0, "end_char": 14, "url": None}]},
])


def test_run_extract_for_model_reports_truncated_calls_and_batches(tmp_path):
    # Every call reports finish_reason == "length"; both docs parse on their
    # first (only) batch, so truncated_calls must be exactly 2 (one per doc)
    # and it must survive into build_report's per-model extract metrics.
    _build_tiny_corpus(tmp_path)
    config = _tiny_corpus_config(tmp_path)
    backend = _FakeMetaExtractBackend(
        {"doc-a#c001": _ALPHA_PAYLOAD, "doc-b#c001": _GAMMA_PAYLOAD})
    index = json.loads((tmp_path / "corpus-index.json").read_text(encoding="utf-8"))
    doc_selection = eval_pipeline.select_docs_for_limit(tmp_path, None)

    out = eval_pipeline.run_extract_for_model(tmp_path, index, config, backend, doc_selection)
    assert out["truncated_calls"] == 2
    assert out["batches"] == 2

    report = eval_pipeline.build_report(
        corpus_dir=str(tmp_path), join_keys={}, role_results={"extract": {"models": {"m1": out}}})
    assert report["roles"]["extract"]["models"]["m1"]["truncated_calls"] == 2


def test_run_extract_for_model_truncated_calls_zero_without_meta_backend(tmp_path):
    # A plain complete()-only backend (no meta channel) must still yield the
    # key, at zero -- never a crash, never a missing field.
    _build_tiny_corpus(tmp_path)
    config = _tiny_corpus_config(tmp_path)
    backend = _FakeExtractBackend({"doc-a#c001": _ALPHA_PAYLOAD})
    index = json.loads((tmp_path / "corpus-index.json").read_text(encoding="utf-8"))
    doc_selection = eval_pipeline.select_docs_for_limit(tmp_path, None)
    out = eval_pipeline.run_extract_for_model(tmp_path, index, config, backend, doc_selection)
    assert out["truncated_calls"] == 0


def test_run_extract_for_model_recall_entailment_with_fake_embedder(tmp_path):
    # The produced claim TEXT equals gold a_c1's text exactly, so the one-hot
    # embedder credits exactly 1 of the 3 pooled reference claims.
    _build_tiny_corpus(tmp_path)
    config = _tiny_corpus_config(tmp_path)
    backend = _FakeExtractBackend({"doc-a#c001": _ALPHA_PAYLOAD})
    index = json.loads((tmp_path / "corpus-index.json").read_text(encoding="utf-8"))
    doc_selection = eval_pipeline.select_docs_for_limit(tmp_path, None)

    out = eval_pipeline.run_extract_for_model(
        tmp_path, index, config, backend, doc_selection,
        embedder=_make_one_hot_embedder())

    assert out["recall_entailment"] == pytest.approx(1 / 3)
    assert out["f_fact"] is None  # no self_faithfulness signal here
    assert out["per_doc"]["doc-a"]["recall_entailment"] == pytest.approx(0.5)
    assert out["per_doc"]["doc-b"]["recall_entailment"] == 0.0

    report = eval_pipeline.build_report(
        corpus_dir=str(tmp_path), join_keys={}, role_results={"extract": {"models": {"m1": out}}})
    assert report["roles"]["extract"]["models"]["m1"]["recall_entailment"] == pytest.approx(1 / 3)


def test_run_extract_for_model_without_embedder_omits_entailment_fields(tmp_path):
    # Default flags must reproduce today's report shape exactly.
    _build_tiny_corpus(tmp_path)
    config = _tiny_corpus_config(tmp_path)
    backend = _FakeExtractBackend({"doc-a#c001": _ALPHA_PAYLOAD})
    index = json.loads((tmp_path / "corpus-index.json").read_text(encoding="utf-8"))
    doc_selection = eval_pipeline.select_docs_for_limit(tmp_path, None)
    out = eval_pipeline.run_extract_for_model(tmp_path, index, config, backend, doc_selection)
    assert "recall_entailment" not in out
    assert "f_fact" not in out


def test_run_extract_for_model_degrades_when_embedder_raises(tmp_path):
    # An embedder whose calls fail (endpoint down) must degrade to the
    # no-embedder metric shape, never crash the run.
    _build_tiny_corpus(tmp_path)
    config = _tiny_corpus_config(tmp_path)
    backend = _FakeExtractBackend({"doc-a#c001": _ALPHA_PAYLOAD})
    index = json.loads((tmp_path / "corpus-index.json").read_text(encoding="utf-8"))
    doc_selection = eval_pipeline.select_docs_for_limit(tmp_path, None)

    def broken(texts):
        raise RuntimeError("no embedding endpoint")

    out = eval_pipeline.run_extract_for_model(
        tmp_path, index, config, backend, doc_selection, embedder=broken)
    assert "recall_entailment" not in out
    assert out["recall"] == pytest.approx(1 / 3)  # base metrics intact


def test_run_extract_for_model_embedder_failure_isolated_to_one_aggregate(tmp_path):
    # Anomaly 2: a batch that dies (e.g. the embedding runner crashing on an
    # oversized request) must fail entailment for ONLY the aggregate it was
    # scoring -- never permanently disable the embedder for the rest of the
    # run. doc-a is scored first (the first _metrics() call, which calls the
    # embedder for its non-empty reference texts) and is made to fail; doc-b
    # and the final corpus-level aggregate come after and must still get
    # recall_entailment, proving the failure didn't leak forward.
    _build_tiny_corpus(tmp_path)
    config = _tiny_corpus_config(tmp_path)
    backend = _FakeExtractBackend({"doc-a#c001": _ALPHA_PAYLOAD})
    index = json.loads((tmp_path / "corpus-index.json").read_text(encoding="utf-8"))
    doc_selection = eval_pipeline.select_docs_for_limit(tmp_path, None)

    good = _make_one_hot_embedder()
    calls = {"n": 0}

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("embedding runner died on this batch")
        return good(texts)

    out = eval_pipeline.run_extract_for_model(
        tmp_path, index, config, backend, doc_selection, embedder=flaky)

    assert "recall_entailment" not in out["per_doc"]["doc-a"]  # the failed aggregate
    assert "recall_entailment" in out["per_doc"]["doc-b"]      # unaffected by doc-a's failure
    assert out["recall_entailment"] is not None                # corpus-level aggregate unaffected too


def test_run_extract_for_doc_prefers_pooled_gold_reference(tmp_path):
    _build_tiny_corpus(tmp_path)
    pooled = [_claim("pg1", "doc-a#c001", "Alpha fact one"),
              _claim("pg2", "doc-a#c001", "Alpha fact two"),
              _claim("pg3", "doc-a#c002", "Beta fact one")]
    with open(tmp_path / "doc-a" / "pooled-gold.jsonl", "w", encoding="utf-8") as f:
        for c in pooled:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    config = _tiny_corpus_config(tmp_path)
    backend = _FakeExtractBackend({})

    out = eval_pipeline.run_extract_for_doc(
        tmp_path / "doc-a", config, backend, pooled_gold=True)
    assert {c["claim_id"] for c in out["reference"]} == {"pg1", "pg2", "pg3"}
    assert out["reference_source"] == "pooled-gold.jsonl"

    # doc-b ships no pooled-gold.jsonl -> falls back to reference-claims.jsonl
    out_b = eval_pipeline.run_extract_for_doc(
        tmp_path / "doc-b", config, backend, pooled_gold=True)
    assert {c["claim_id"] for c in out_b["reference"]} == {"b_c1"}
    assert out_b["reference_source"] == "reference-claims.jsonl"

    # without the flag the pooled file is ignored even when present
    out_default = eval_pipeline.run_extract_for_doc(tmp_path / "doc-a", config, backend)
    assert {c["claim_id"] for c in out_default["reference"]} == {"a_c1", "a_c2"}
    assert out_default["reference_source"] == "reference-claims.jsonl"


class _CountingExtractBackend:
    def __init__(self, by_locator: dict):
        self.by_locator = by_locator
        self.calls = 0

    def complete(self, system, user, **kw):
        self.calls += 1
        for locator, payload in self.by_locator.items():
            if locator in user:
                return payload
        return "{}"


def test_run_extract_for_doc_threads_samples_through_to_extraction(tmp_path):
    # samples=2 must fan the extraction out into two passes (doc-a is a
    # single batch, so exactly two backend calls) and still dedup the
    # identical claims down to one written row under min_support=2.
    _build_tiny_corpus(tmp_path)
    config = _tiny_corpus_config(tmp_path)
    backend = _CountingExtractBackend({"doc-a#c001": _ALPHA_PAYLOAD})
    out = eval_pipeline.run_extract_for_doc(
        tmp_path / "doc-a", config, backend, samples=2, min_support=2, parallel=1)
    assert backend.calls == 2
    assert out["written"] == 1


def test_run_prose_role_reports_coverage_rule_counts(tmp_path):
    # Each tiny-corpus chunk carries a single citable claim (< the ratio
    # floor's minimum), so both attempts score under the "absolute" rule --
    # and the rule must be visible in the role result / report.
    _build_tiny_corpus(tmp_path)
    claims_by_chunk = eval_pipeline.load_claims_by_chunk(tmp_path)
    backend = _KeyedProseBackend({
        "a_c1": "Alpha holds [claim:a_c1].",
        "a_c2": "Beta holds [claim:a_c2].",
    })
    out = eval_pipeline.run_prose_role_with_backend(
        "wiki_write", ["doc-a#c001", "doc-a#c002"], claims_by_chunk, backend, runs=1)
    assert out["coverage_rules"] == {"absolute": 2}


def test_arg_parser_new_flag_defaults_reproduce_single_pass_behavior():
    args = eval_pipeline.build_arg_parser().parse_args([])
    assert args.samples == 1
    assert args.min_support == 1
    assert args.parallel == 1
    assert args.pooled_gold is False


def test_make_embedder_returns_callable_or_none():
    from types import SimpleNamespace
    # An Ollama-tag embedding model builds lazily (no network at construction).
    cfg = SimpleNamespace(embedding_model="qwen3-embedding:4b",
                          llm_local={"base_url": "http://localhost:11434/v1"})
    embedder = eval_pipeline.make_embedder(cfg)
    assert callable(embedder)
    # A config the route can't be built from degrades to None, never raises.
    assert eval_pipeline.make_embedder(SimpleNamespace()) is None


# ---------------------------------------------------------------------------
# report assembly / history / compare
# ---------------------------------------------------------------------------

def test_build_report_shape():
    report = eval_pipeline.build_report(
        corpus_dir="tests/fixtures/eval-corpus",
        join_keys={"prompt_version": "sha256:x", "corpus_version": "sha256:y", "ollama_version": "0.31.1"},
        role_results={"extract": {"models": {"m1": {"gate_pass_rate": 0.9, "recall": 0.8}}}},
        ts="2026-07-06T00:00:00+00:00",
    )
    assert report["ts"] == "2026-07-06T00:00:00+00:00"
    assert report["join_keys"]["ollama_version"] == "0.31.1"
    assert report["roles"]["extract"]["models"]["m1"]["recall"] == 0.8


def test_history_lines_one_row_per_model_and_role():
    report = eval_pipeline.build_report(
        corpus_dir="c", join_keys={"prompt_version": "p", "corpus_version": "c", "ollama_version": "o"},
        role_results={
            "extract": {"models": {"m1": {"gate_pass_rate": 0.9, "recall": 0.8},
                                    "m2": {"gate_pass_rate": 0.7, "recall": 0.6}}},
            "wiki_write": {"model": "m3", "mean_coverage": 0.5},
        },
        ts="2026-07-06T00:00:00+00:00",
    )
    lines = eval_pipeline.history_lines(report)
    keys = {(row["role"], row["model"]) for row in lines}
    assert keys == {("extract", "m1"), ("extract", "m2"), ("wiki_write", "m3")}
    for row in lines:
        assert row["ts"] == "2026-07-06T00:00:00+00:00"
        assert row["prompt_version"] == "p"


def test_history_lines_strip_bulky_detail_keys():
    # per_doc / missed_claim_ids / per_chunk live in the run report only --
    # history.jsonl is the flat summary time series.
    report = eval_pipeline.build_report(
        corpus_dir="c", join_keys={"prompt_version": "p"},
        role_results={
            "extract": {"models": {"m1": {
                "gate_pass_rate": 0.9, "recall": 0.8,
                "per_doc": {"doc-a": {"recall": 0.8}},
                "missed_claim_ids": ["r9"],
            }}},
            "wiki_write": {"model": "m3", "mean_coverage": 0.5,
                           "per_chunk": {"doc-a#c001": {"rate": 1.0}}},
        },
        ts="2026-07-06T00:00:00+00:00",
    )
    lines = eval_pipeline.history_lines(report)
    for row in lines:
        assert "per_doc" not in row
        assert "missed_claim_ids" not in row
        assert "per_chunk" not in row
    extract_row = next(r for r in lines if r["role"] == "extract")
    assert extract_row["recall"] == 0.8  # summary metrics survive the strip


def test_compare_reports_flags_gate_pass_rate_regression():
    baseline = {"roles": {"extract": {"models": {"m1": {"gate_pass_rate": 0.9, "recall": 0.8}}}}}
    current = {"roles": {"extract": {"models": {"m1": {"gate_pass_rate": 0.80, "recall": 0.8}}}}}
    regressions = eval_pipeline.compare_reports(current, baseline, tolerance=0.03)
    assert len(regressions) == 1
    assert "gate_pass_rate" in regressions[0]


def test_compare_reports_flags_recall_regression():
    baseline = {"roles": {"extract": {"models": {"m1": {"gate_pass_rate": 0.9, "recall": 0.8}}}}}
    current = {"roles": {"extract": {"models": {"m1": {"gate_pass_rate": 0.9, "recall": 0.5}}}}}
    regressions = eval_pipeline.compare_reports(current, baseline, tolerance=0.03)
    assert len(regressions) == 1
    assert "recall" in regressions[0]


def test_compare_reports_within_tolerance_is_clean():
    baseline = {"roles": {"extract": {"models": {"m1": {"gate_pass_rate": 0.90, "recall": 0.80}}}}}
    current = {"roles": {"extract": {"models": {"m1": {"gate_pass_rate": 0.88, "recall": 0.79}}}}}
    regressions = eval_pipeline.compare_reports(current, baseline, tolerance=0.03)
    assert regressions == []


def test_compare_reports_improvement_is_clean():
    baseline = {"roles": {"extract": {"models": {"m1": {"gate_pass_rate": 0.80, "recall": 0.70}}}}}
    current = {"roles": {"extract": {"models": {"m1": {"gate_pass_rate": 0.95, "recall": 0.90}}}}}
    regressions = eval_pipeline.compare_reports(current, baseline, tolerance=0.03)
    assert regressions == []


def test_compare_reports_ignores_model_absent_from_baseline():
    baseline = {"roles": {"extract": {"models": {}}}}
    current = {"roles": {"extract": {"models": {"m1": {"gate_pass_rate": 0.1, "recall": 0.1}}}}}
    regressions = eval_pipeline.compare_reports(current, baseline, tolerance=0.03)
    assert regressions == []


def test_compare_reports_handles_missing_extract_role_gracefully():
    assert eval_pipeline.compare_reports({"roles": {}}, {"roles": {}}, tolerance=0.03) == []


def test_compare_reports_flags_metric_becoming_none():
    # A metric that was numeric in the baseline but is None in the current
    # run silently degraded to unmeasurable -- that is a regression, not a
    # skip.
    baseline = {"roles": {"extract": {"models": {"m1": {"gate_pass_rate": 0.9, "recall": 0.8}}}}}
    current = {"roles": {"extract": {"models": {"m1": {"gate_pass_rate": 0.9, "recall": None}}}}}
    regressions = eval_pipeline.compare_reports(current, baseline, tolerance=0.03)
    assert len(regressions) == 1
    assert "recall" in regressions[0] and "unmeasurable" in regressions[0]


# ---------------------------------------------------------------------------
# join keys: prompt_version / ollama_version
# ---------------------------------------------------------------------------

def test_prompt_version_is_stable_sha256_hex():
    v1 = eval_pipeline.prompt_version()
    v2 = eval_pipeline.prompt_version()
    assert v1 == v2
    assert v1.startswith("sha256:")
    assert len(v1) == len("sha256:") + 64


def test_get_ollama_version_returns_unknown_on_connection_failure():
    # Nothing is listening on this port -- must degrade to "unknown", not raise.
    version = eval_pipeline.get_ollama_version("http://127.0.0.1:1/v1", timeout=0.2)
    assert version == "unknown"


def test_get_ollama_version_parses_version_field(monkeypatch):
    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"version": "0.31.1"}).encode("utf-8")

    def _fake_urlopen(url, timeout=None):
        assert url.endswith("/api/version")
        assert "/v1" not in url
        return _FakeResp()

    monkeypatch.setattr(eval_pipeline.urllib.request, "urlopen", _fake_urlopen)
    assert eval_pipeline.get_ollama_version("http://localhost:11434/v1") == "0.31.1"


# ---------------------------------------------------------------------------
# CLI: --help must work with no live endpoint
# ---------------------------------------------------------------------------

def test_help_works_without_a_live_endpoint():
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "eval-pipeline.py"), "--help"],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert "--corpus" in result.stdout
    assert "--compare" in result.stdout
    assert "--samples" in result.stdout
    assert "--min-support" in result.stdout
    assert "--parallel" in result.stdout
    assert "--pooled-gold" in result.stdout
