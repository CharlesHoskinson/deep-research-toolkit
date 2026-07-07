"""Unit tests for scripts/eval-pipeline.py's importable pure logic: corpus
loading/limiting, stratified sampling, the adjudicate synthetic-candidate
protocol ("pair-claims-v1"), report assembly, history lines, and baseline
comparison. Extraction/prose-role wiring is exercised with fake in-process
backends (never a live model) so the whole pipeline is drivable in the fast
suite; only the CLI's actual `get_backend(config, ...)` construction against
a live endpoint is left for the live tier / Task 8 runbook."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
            {"a": "doc-a#c001", "b": "doc-b#c001", "verdict": "contradiction",
             "note": "widget count: 10 vs 20"},
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


# ---------------------------------------------------------------------------
# adjudicate synthetic-candidate protocol ("pair-claims-v1")
# ---------------------------------------------------------------------------

def test_slugify_basic():
    assert eval_pipeline.slugify("MCB v2 release year") == "mcb-v2-release-year"


def test_pick_claim_for_pair_prefers_keyword_overlap():
    claims_by_chunk = {
        "doc#c001": [
            _claim("c1", "doc#c001", "unrelated filler text"),
            _claim("c2", "doc#c001", "the widget count was measured carefully"),
        ]
    }
    picked = eval_pipeline.pick_claim_for_pair("doc#c001", "widget count", claims_by_chunk)
    assert picked["claim_id"] == "c2"


def test_pick_claim_for_pair_returns_none_when_chunk_has_no_claims():
    assert eval_pipeline.pick_claim_for_pair("doc#c999", "topic", {}) is None


def test_build_adjudicate_candidates_shape_and_unique_subjects():
    claims_by_chunk = {
        "doc-a#c001": [_claim("a1", "doc-a#c001", "widget count was 10")],
        "doc-b#c001": [_claim("b1", "doc-b#c001", "widget count was 20")],
    }
    pairs = [
        {"a": "doc-a#c001", "b": "doc-b#c001", "verdict": "contradiction", "note": "widget count: 10 vs 20"},
    ]
    candidates, meta = eval_pipeline.build_adjudicate_candidates(pairs, claims_by_chunk)
    assert len(candidates) == 1 and len(meta) == 1
    cand = candidates[0]
    assert cand["kind"] == "relation"
    assert cand["predicate"] == "asserts"
    assert cand["subject"] == "widget-count"
    assert cand["relation_ids"] == ["a1", "b1"]
    assert cand["objects"] == ["claim about widget count was 10", "claim about widget count was 20"]
    assert meta[0]["gold_verdict"] == "contradiction"
    assert meta[0]["subject"] == "widget-count"
    assert meta[0]["predicate"] == "asserts"


def test_build_adjudicate_candidates_dedupes_colliding_subjects():
    claims_by_chunk = {
        "doc-a#c001": [_claim("a1", "doc-a#c001", "x")],
        "doc-b#c001": [_claim("b1", "doc-b#c001", "y")],
        "doc-c#c001": [_claim("c1", "doc-c#c001", "z")],
        "doc-d#c001": [_claim("d1", "doc-d#c001", "w")],
    }
    pairs = [
        {"a": "doc-a#c001", "b": "doc-b#c001", "verdict": "contradiction", "note": "same topic: x vs y"},
        {"a": "doc-c#c001", "b": "doc-d#c001", "verdict": "not_contradiction", "note": "same topic: z vs w"},
    ]
    candidates, meta = eval_pipeline.build_adjudicate_candidates(pairs, claims_by_chunk)
    subjects = [c["subject"] for c in candidates]
    assert len(subjects) == len(set(subjects))  # never collide


def test_score_adjudicate_exact_match_is_full_credit():
    meta = [{"subject": "s1", "predicate": "asserts", "gold_verdict": "contradiction", "note": "n"}]
    result = {"verdicts": [{"subject": "s1", "predicate": "asserts", "verdict": "contradiction",
                            "relation_ids": ["a1", "b1"], "rationale": "x"}],
              "invalid": [], "parse_failures": 0}
    out = eval_pipeline.score_adjudicate(result, meta)
    assert out["accuracy"] == 1.0
    assert out["adjudicate_protocol"] == "pair-claims-v1"


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
        {"claim_id": "p1", "claim": "x", "supporting_evidence":
         [{"locator": "doc-a#c001", "quote": "Alpha fact one", "url": None}]},
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
