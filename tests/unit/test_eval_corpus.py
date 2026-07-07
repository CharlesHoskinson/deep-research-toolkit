"""Unit tests for scripts/check-eval-corpus.py: the mechanical gate over the
stratified eval corpus (tests/fixtures/eval-corpus/). Drives validate() against
tiny synthetic mini-corpora built in tmp_path -- one test per violation class
-- plus a final check that the committed corpus itself passes with the real,
full-scale default quotas (fast, no model).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "check_eval_corpus", REPO_ROOT / "scripts" / "check-eval-corpus.py")
check_eval_corpus = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_eval_corpus)
validate = check_eval_corpus.validate
compute_corpus_version = check_eval_corpus.compute_corpus_version

DEFAULT_CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "eval-corpus"

ALL_SLICES = ("prose", "dense-facts", "table", "list", "unicode", "math", "long", "bait")


# ---------------------------------------------------------------------------
# Mini-corpus builders
# ---------------------------------------------------------------------------

def _with_spans(claim: dict, chunks: dict[str, str]) -> dict:
    """Fill start_char/end_char on evidence rows that lack them by locating the
    quote in its chunk (the span shape the corpus now carries). A quote that
    does not occur verbatim gets a deliberate (0, len(quote)) span so the
    checker's span gate flags it -- used by the negative test."""
    out = dict(claim)
    evs = []
    for ev in claim.get("supporting_evidence") or []:
        ev = dict(ev)
        if "start_char" not in ev:
            text = chunks.get(ev.get("locator")) or ""
            quote = ev.get("quote") or ""
            start = text.find(quote)
            if start < 0:
                start = 0
            ev["start_char"] = start
            ev["end_char"] = start + len(quote)
        evs.append(ev)
    out["supporting_evidence"] = evs
    return out


def _write_doc(root: Path, doc_id: str, chunks: dict[str, str], claims: list[dict]) -> None:
    doc_dir = root / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0", "producer": "web", "document_id": doc_id,
        "title": doc_id, "source_url": f"https://example.org/eval-corpus/{doc_id}",
        "content_hash": "sha256:deadbeef", "fetched_at": "2026-01-01T00:00:00Z",
        "chunk_count": len(chunks),
    }), encoding="utf-8")
    with open(doc_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for locator, text in chunks.items():
            f.write(json.dumps({
                "schema_version": "1.0", "locator": locator, "source_id": doc_id,
                "type": "section", "title": locator, "text": text,
                "content_hash": "sha256:deadbeef",
            }, ensure_ascii=False) + "\n")
    with open(doc_dir / "reference-claims.jsonl", "w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(_with_spans(c, chunks), ensure_ascii=False) + "\n")


def _claim(claim_id: str, doc_id: str, locator: str, quote: str) -> dict:
    return {
        "claim_id": claim_id, "source_id": doc_id,
        "claim": f"A claim grounded in: {quote[:40]}",
        "claim_type": "empirical", "confidence": "high",
        "supporting_evidence": [{"locator": locator, "quote": quote, "url": None}],
        "entities": [], "relations": [],
    }


def _two_claims(doc_id: str, locator: str, quote1: str, quote2: str, prefix: str) -> list[dict]:
    return [
        _claim(f"{prefix}_1", doc_id, locator, quote1),
        _claim(f"{prefix}_2", doc_id, locator, quote2),
    ]


def _write_index(root: Path, chunks_meta: dict[str, dict], chunk_texts: dict[str, str],
                 contradiction_pairs: list | None = None, bait_sources: dict | None = None,
                 corpus_version: str | None = "AUTO") -> None:
    index = {
        "chunks": chunks_meta,
        "contradiction_pairs": contradiction_pairs or [],
        "bait_sources": bait_sources or {},
        "corpus_version": (compute_corpus_version(chunk_texts) if corpus_version == "AUTO" else corpus_version),
    }
    (root / "corpus-index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


#: A single clean document exercising every slice tag plus the bait mechanism.
#: doc-a#c002 (bait) near-copies one sentence of doc-a#c001 ("two-second" ->
#: "three-second"), recorded in bait_sources.
_DOC_A_CHUNKS = {
    "doc-a#c001": (
        "The Nimbus consensus protocol coordinates block production across a rotating "
        "committee of validators. Each epoch, the committee selects a leader through a "
        "verifiable random function seeded by the previous epoch's beacon value. The leader "
        "proposes a block, and the remaining committee members vote within a fixed two-second "
        "window. Once two-thirds of the committee has voted, the block is considered final and "
        "no honest validator will build on a competing branch. This design borrows from classical "
        "Byzantine fault tolerant protocols but adapts committee selection to operate efficiently "
        "at a scale of tens of thousands of validators across the network."
    ),
    "doc-a#c002": (
        "Nimbus validators exchange votes over a dedicated gossip overlay that separates "
        "consensus traffic from general block propagation entirely. The leader proposes a "
        "block, and the remaining committee members vote within a fixed three-second window. "
        "This overlay uses UDP-based erasure coding to tolerate up to twenty percent packet loss "
        "without triggering a view change. Bandwidth accounting caps each validator at eight "
        "megabits per second of consensus traffic, leaving headroom for ordinary block and "
        "transaction propagation on a separate channel. Peers outside the current committee "
        "still relay these messages onward for redundancy, even though they cast no vote."
    ),
    "doc-a#c003": (
        "The Solace protocol, first released in 2019, introduced sub-second finality across "
        "four geographic regions. Version 1.2 shipped in March 2020 with support for 12,000 "
        "validators, and version 2.0 arrived in November 2021 adding threshold signatures. By "
        "2023, Solace supported over 40,000 active validators spread across nine regions, with "
        "average block times of 350 milliseconds and a peak throughput of 4,200 transactions per "
        "second measured during the June 2023 load test. These figures come from the protocol's "
        "quarterly transparency report, published each January by the foundation that stewards it."
    ),
    "doc-a#c004": (
        "The following table summarizes Nimbus release history and validator counts at each "
        "milestone, drawn from the project's public changelog for external auditors.\n\n"
        "| Version | Released | Validators | Notes |\n"
        "|---------|----------|------------|-------|\n"
        "| 0.9 | 2018 | 512 | Testnet only |\n"
        "| 1.0 | 2019 | 4800 | Mainnet launch |\n"
        "| 1.5 | 2020 | 18300 | Sharding enabled |\n"
        "| 2.0 | 2021 | 33900 | Threshold signatures |\n\n"
        "Each row reflects the validator count measured thirty days after the corresponding "
        "release, not the count at the moment of the release itself, since adoption typically "
        "ramps over several weeks following any announcement."
    ),
    "doc-a#c005": (
        "Nimbus validator operators must satisfy several requirements before joining the active "
        "set, enumerated below for clarity and for onboarding documentation purposes.\n\n"
        "1. Stake at least 32,000 NIM tokens in a bonded deposit contract.\n"
        "2. Run client software within two minor versions of the current release.\n"
        "3. Maintain measured uptime above ninety-eight percent over any rolling thirty-day window.\n"
        "4. Publish a valid attestation key rotation schedule to the registry contract.\n"
        "5. Pass a one-time hardware attestation proving a trusted execution environment.\n\n"
        "Operators failing any of these checks are moved to an inactive queue and must requalify "
        "before rejoining the committee rotation."
    ),
    "doc-a#c006": (
        "Early formal treatments of committee-based finality appear outside the English-language "
        "literature. Zhang and Wu's 共识协议的理论基础 (2019) formalizes the leader-rotation "
        "argument later adopted by Nimbus, while Petrov's Устойчивость консенсуса при частичной "
        "синхронности (2020) proves a tightened bound on the fraction of Byzantine validators "
        "tolerable under partial synchrony. The Greek-language literature, notably Papadopoulos's "
        "study of ασφάλεια και ζωντάνια, reframes the tolerance bound as a function of network "
        "delay, connecting the two traditions under a single shared parameter. Both papers remain "
        "widely cited in later English-language surveys of committee-based consensus."
    ),
    "doc-a#c007": (
        "Nimbus's safety argument reduces to a quorum-intersection bound. Let n be the committee "
        "size and f the number of Byzantine members; safety holds whenever any two quorums of "
        "size q = ceil((n+f+1)/2) intersect in at least one honest member, i.e. 2q - n > f. "
        "Substituting the standard assumption n >= 3f + 1 gives 2q - n >= f + 1, so the bound "
        "holds with one member of slack. Liveness additionally requires q <= n - f, which further "
        "constrains the committee size relative to the worst-case Byzantine fraction the "
        "deployment is meant to tolerate in production."
    ),
    "doc-a#c008": (
        "Nimbus's design evolved through three distinct phases before reaching its current form. "
        "The first phase, prototyped in 2017, used a fixed validator set with no rotation at all, "
        "which simplified the implementation but concentrated liveness risk on a small committee. "
        "The second phase introduced epoch-based rotation but kept committee size fixed regardless "
        "of total stake, which under-utilized the validator set as participation grew. The third "
        "and current phase scales committee size logarithmically with total active stake, "
        "balancing communication overhead against decentralization as the network grows larger."
    ),
}

_DOC_A_CLAIMS = (
    _two_claims("doc-a", "doc-a#c001",
                "The Nimbus consensus protocol coordinates block production across a rotating "
                "committee of validators.",
                "Once two-thirds of the committee has voted, the block is considered final",
                "a_c001")
    + _two_claims("doc-a", "doc-a#c002",
                  "Nimbus validators exchange votes over a dedicated gossip overlay that separates "
                  "consensus traffic from general block propagation entirely.",
                  "This overlay uses UDP-based erasure coding to tolerate up to twenty percent "
                  "packet loss without triggering a view change.",
                  "a_c002")
    + _two_claims("doc-a", "doc-a#c003",
                  "The Solace protocol, first released in 2019, introduced sub-second finality "
                  "across four geographic regions.",
                  "By 2023, Solace supported over 40,000 active validators spread across nine "
                  "regions",
                  "a_c003")
    + _two_claims("doc-a", "doc-a#c004",
                  "Sharding enabled",
                  "Each row reflects the validator count measured thirty days after the "
                  "corresponding release",
                  "a_c004")
    + _two_claims("doc-a", "doc-a#c005",
                  "Stake at least 32,000 NIM tokens in a bonded deposit contract.",
                  "Maintain measured uptime above ninety-eight percent over any rolling thirty-day "
                  "window.",
                  "a_c005")
    + _two_claims("doc-a", "doc-a#c006",
                  "Zhang and Wu's 共识协议的理论基础 (2019) formalizes the leader-rotation argument "
                  "later adopted by Nimbus",
                  "Petrov's Устойчивость консенсуса при частичной синхронности (2020) proves a "
                  "tightened bound on the fraction of Byzantine validators tolerable under partial "
                  "synchrony",
                  "a_c006")
    + _two_claims("doc-a", "doc-a#c007",
                  "Nimbus's safety argument reduces to a quorum-intersection bound.",
                  "Liveness additionally requires q <= n - f",
                  "a_c007")
    + _two_claims("doc-a", "doc-a#c008",
                  "The first phase, prototyped in 2017, used a fixed validator set with no "
                  "rotation at all",
                  "The third and current phase scales committee size logarithmically with total "
                  "active stake",
                  "a_c008")
)

_DOC_A_META = {
    "doc-a#c001": {"slices": ["prose"], "doc": "doc-a"},
    "doc-a#c002": {"slices": ["bait"], "doc": "doc-a"},
    "doc-a#c003": {"slices": ["dense-facts"], "doc": "doc-a"},
    "doc-a#c004": {"slices": ["table"], "doc": "doc-a"},
    "doc-a#c005": {"slices": ["list"], "doc": "doc-a"},
    "doc-a#c006": {"slices": ["unicode"], "doc": "doc-a"},
    "doc-a#c007": {"slices": ["math"], "doc": "doc-a"},
    "doc-a#c008": {"slices": ["long"], "doc": "doc-a"},
}

_TINY_QUOTAS = {s: 1 for s in ALL_SLICES}
_TINY_TOTAL_RANGE = (0, 100)


def _build_doc_a(root: Path) -> None:
    _write_doc(root, "doc-a", _DOC_A_CHUNKS, list(_DOC_A_CLAIMS))


# ---------------------------------------------------------------------------
# validate_claim_evidence: the per-claim span gate (span shape, Task 3)
# ---------------------------------------------------------------------------

_SPAN_CHUNK = _DOC_A_CHUNKS["doc-a#c001"]


def _span_claim(start: int, end: int, quote: str) -> dict:
    return {
        "claim_id": "span_claim", "source_id": "doc-a",
        "claim": "A span-shaped claim.",
        "supporting_evidence": [
            {"locator": "doc-a#c001", "start_char": start, "end_char": end,
             "quote": quote, "url": None},
        ],
    }


def test_validate_claim_evidence_accepts_span_that_slices_to_its_quote():
    quote = "The Nimbus consensus protocol"
    start = _SPAN_CHUNK.find(quote)
    claim = _span_claim(start, start + len(quote), quote)
    assert check_eval_corpus.validate_claim_evidence(claim, _SPAN_CHUNK) is True


def test_validate_claim_evidence_rejects_out_of_bounds_span():
    quote = "The Nimbus consensus protocol"
    claim = _span_claim(len(_SPAN_CHUNK) + 1, len(_SPAN_CHUNK) + 1 + len(quote), quote)
    assert check_eval_corpus.validate_claim_evidence(claim, _SPAN_CHUNK) is False


def test_validate_claim_evidence_rejects_span_whose_slice_differs_from_quote():
    quote = "The Nimbus consensus protocol"
    start = _SPAN_CHUNK.find(quote)
    claim = _span_claim(start + 1, start + 1 + len(quote), quote)  # off by one
    assert check_eval_corpus.validate_claim_evidence(claim, _SPAN_CHUNK) is False


def test_validate_claim_evidence_rejects_missing_span_fields():
    # Old-shape evidence (quote only, no offsets) must be rejected outright.
    claim = {
        "claim_id": "span_claim", "source_id": "doc-a",
        "claim": "An old-shape claim.",
        "supporting_evidence": [
            {"locator": "doc-a#c001", "quote": "The Nimbus consensus protocol", "url": None},
        ],
    }
    assert check_eval_corpus.validate_claim_evidence(claim, _SPAN_CHUNK) is False


def test_validate_claim_evidence_rejects_empty_evidence():
    claim = {"claim_id": "span_claim", "source_id": "doc-a",
             "claim": "No evidence.", "supporting_evidence": []}
    assert check_eval_corpus.validate_claim_evidence(claim, _SPAN_CHUNK) is False


# ---------------------------------------------------------------------------
# Baseline: one clean document passes
# ---------------------------------------------------------------------------

def test_clean_single_doc_corpus_passes(tmp_path):
    _build_doc_a(tmp_path)
    _write_index(tmp_path, _DOC_A_META, _DOC_A_CHUNKS,
                 contradiction_pairs=[], bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=_TINY_TOTAL_RANGE,
                      min_contradiction_pairs=0)
    assert errors == []


# ---------------------------------------------------------------------------
# One test per violation class
# ---------------------------------------------------------------------------

def test_missing_required_file_is_flagged(tmp_path):
    _build_doc_a(tmp_path)
    (tmp_path / "doc-a" / "reference-claims.jsonl").unlink()
    _write_index(tmp_path, _DOC_A_META, _DOC_A_CHUNKS,
                 bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=_TINY_TOTAL_RANGE,
                      min_contradiction_pairs=0)
    assert any("missing required file" in e and "reference-claims.jsonl" in e for e in errors)


def test_bad_locator_format_is_flagged(tmp_path):
    chunks = dict(_DOC_A_CHUNKS)
    text = chunks.pop("doc-a#c001")
    chunks["doc-a_bad_001"] = text
    claims = [c for c in _DOC_A_CLAIMS if c["claim_id"] not in ("a_c001_1", "a_c001_2")]
    claims += _two_claims("doc-a", "doc-a_bad_001",
                          "The Nimbus consensus protocol coordinates block production across a "
                          "rotating committee of validators.",
                          "Once two-thirds of the committee has voted, the block is considered "
                          "final", "a_bad")
    _write_doc(tmp_path, "doc-a", chunks, claims)
    meta = dict(_DOC_A_META)
    meta.pop("doc-a#c001")
    meta["doc-a_bad_001"] = {"slices": ["prose"], "doc": "doc-a"}
    _write_index(tmp_path, meta, chunks, bait_sources={"doc-a#c002": "doc-a_bad_001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=_TINY_TOTAL_RANGE,
                      min_contradiction_pairs=0)
    assert any("does not match" in e and "doc-a_bad_001" in e for e in errors)


def test_chunk_word_count_out_of_range_is_flagged(tmp_path):
    chunks = dict(_DOC_A_CHUNKS)
    chunks["doc-a#c001"] = "Far too short a chunk to pass the word-count gate."
    _write_doc(tmp_path, "doc-a", chunks, list(_DOC_A_CLAIMS))
    _write_index(tmp_path, _DOC_A_META, chunks, bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=_TINY_TOTAL_RANGE,
                      min_contradiction_pairs=0)
    assert any("doc-a#c001" in e and "words" in e and "outside" in e for e in errors)


def test_total_chunk_count_out_of_range_is_flagged(tmp_path):
    _build_doc_a(tmp_path)
    _write_index(tmp_path, _DOC_A_META, _DOC_A_CHUNKS,
                 bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=(100, 200),
                      min_contradiction_pairs=0)
    assert any("total chunk count" in e for e in errors)


def test_slice_quota_below_minimum_is_flagged(tmp_path):
    _build_doc_a(tmp_path)
    _write_index(tmp_path, _DOC_A_META, _DOC_A_CHUNKS,
                 bait_sources={"doc-a#c002": "doc-a#c001"})
    quotas = dict(_TINY_QUOTAS)
    quotas["prose"] = 5
    errors = validate(tmp_path, quotas=quotas, total_range=_TINY_TOTAL_RANGE,
                      min_contradiction_pairs=0)
    assert any("'prose'" in e and "needs >= 5" in e for e in errors)


def test_corpus_index_missing_chunk_entry_is_flagged(tmp_path):
    _build_doc_a(tmp_path)
    meta = dict(_DOC_A_META)
    meta.pop("doc-a#c008")
    _write_index(tmp_path, meta, _DOC_A_CHUNKS, bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=_TINY_TOTAL_RANGE,
                      min_contradiction_pairs=0)
    assert any("doc-a#c008" in e and "not tagged in corpus-index" in e for e in errors)


def test_corpus_index_references_unknown_chunk_is_flagged(tmp_path):
    _build_doc_a(tmp_path)
    meta = dict(_DOC_A_META)
    meta["doc-a#c999"] = {"slices": ["prose"], "doc": "doc-a"}
    _write_index(tmp_path, meta, _DOC_A_CHUNKS, bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=_TINY_TOTAL_RANGE,
                      min_contradiction_pairs=0)
    assert any("doc-a#c999" in e and "no matching chunk on disk" in e for e in errors)


def test_claims_per_chunk_out_of_range_is_flagged(tmp_path):
    claims = [c for c in _DOC_A_CLAIMS if c["claim_id"] != "a_c001_2"]  # leave c001 with only 1
    _write_doc(tmp_path, "doc-a", _DOC_A_CHUNKS, claims)
    _write_index(tmp_path, _DOC_A_META, _DOC_A_CHUNKS, bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=_TINY_TOTAL_RANGE,
                      min_contradiction_pairs=0)
    assert any("doc-a#c001" in e and "cited by 1 reference claim" in e for e in errors)


def test_claim_quote_not_verbatim_is_flagged(tmp_path):
    claims = [c for c in _DOC_A_CLAIMS if c["claim_id"] != "a_c001_1"]
    bad = _claim("a_c001_1_bad", "doc-a", "doc-a#c001", "This exact phrase is not in the chunk.")
    claims.append(bad)
    _write_doc(tmp_path, "doc-a", _DOC_A_CHUNKS, claims)
    _write_index(tmp_path, _DOC_A_META, _DOC_A_CHUNKS, bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=_TINY_TOTAL_RANGE,
                      min_contradiction_pairs=0)
    assert any("not verbatim" in e and "a_c001_1_bad" in e for e in errors)


def test_bait_chunk_missing_from_bait_sources_is_flagged(tmp_path):
    _build_doc_a(tmp_path)
    _write_index(tmp_path, _DOC_A_META, _DOC_A_CHUNKS, bait_sources={})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=_TINY_TOTAL_RANGE,
                      min_contradiction_pairs=0)
    assert any("doc-a#c002" in e and "no entry in corpus-index.json 'bait_sources'" in e for e in errors)


def test_bait_source_in_different_doc_is_flagged(tmp_path):
    _build_doc_a(tmp_path)
    other_chunks = {"doc-b#c001": _DOC_A_CHUNKS["doc-a#c001"]}
    other_claims = _two_claims("doc-b", "doc-b#c001",
                               "The Nimbus consensus protocol coordinates block production across "
                               "a rotating committee of validators.",
                               "Once two-thirds of the committee has voted, the block is "
                               "considered final", "b_c001")
    _write_doc(tmp_path, "doc-b", other_chunks, other_claims)
    meta = dict(_DOC_A_META)
    meta["doc-b#c001"] = {"slices": ["prose"], "doc": "doc-b"}
    all_texts = {**_DOC_A_CHUNKS, **other_chunks}
    _write_index(tmp_path, meta, all_texts, bait_sources={"doc-a#c002": "doc-b#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=(0, 100), min_contradiction_pairs=0)
    assert any("doc-a#c002" in e and "not in the same document" in e for e in errors)


def test_bait_chunk_without_near_copy_sentence_is_flagged(tmp_path):
    chunks = dict(_DOC_A_CHUNKS)
    chunks["doc-a#c002"] = (
        "This replacement bait chunk shares no sentence at all with any other chunk in the "
        "document, so the near-copy check must fail cleanly and report the problem. It talks "
        "about an entirely unrelated subject, namely the migratory habits of Arctic terns, "
        "which have nothing to do with consensus protocols or validator committees whatsoever."
    )
    claims = [c for c in _DOC_A_CLAIMS if c["claim_id"] not in ("a_c002_1", "a_c002_2")]
    claims += _two_claims("doc-a", "doc-a#c002",
                          "This replacement bait chunk shares no sentence at all with any other "
                          "chunk in the document, so the near-copy check must fail cleanly and "
                          "report the problem.",
                          "It talks about an entirely unrelated subject, namely the migratory "
                          "habits of Arctic terns, which have nothing to do with consensus "
                          "protocols or validator committees whatsoever.",
                          "a_c002_repl")
    _write_doc(tmp_path, "doc-a", chunks, claims)
    _write_index(tmp_path, _DOC_A_META, chunks, bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=_TINY_TOTAL_RANGE,
                      min_contradiction_pairs=0)
    assert any("doc-a#c002" in e and "no near-copy sentence" in e for e in errors)


def _build_two_doc_corpus_for_contradiction_tests(root: Path):
    _build_doc_a(root)
    doc_b_chunks = {
        "doc-b#c001": (
            "LSM-tree storage engines batch random writes into an in-memory memtable before "
            "flushing sorted runs to disk, trading read amplification for high write throughput. "
            "Background compaction later merges overlapping runs to bound the number of files a "
            "read must consult, at the cost of extra disk bandwidth spent rewriting data that was "
            "already durable."
        ),
        "doc-b#c002": (
            "The Solace protocol launched in 2021 as a research fork of an earlier committee-"
            "based design, targeting lower validator hardware requirements than its predecessor. "
            "Independent audits published in 2022 confirmed the fork's safety proof carried over "
            "unchanged from the original specification, despite the substantially smaller "
            "committee size used in production deployments."
        ),
    }
    doc_b_claims = (
        _two_claims("doc-b", "doc-b#c001",
                   "LSM-tree storage engines batch random writes into an in-memory memtable "
                   "before flushing sorted runs to disk",
                   "Background compaction later merges overlapping runs to bound the number of "
                   "files a read must consult",
                   "b_c001")
        + _two_claims("doc-b", "doc-b#c002",
                     "The Solace protocol launched in 2021 as a research fork of an earlier "
                     "committee-based design",
                     "Independent audits published in 2022 confirmed the fork's safety proof "
                     "carried over unchanged from the original specification",
                     "b_c002")
    )
    _write_doc(root, "doc-b", doc_b_chunks, list(doc_b_claims))
    meta = dict(_DOC_A_META)
    meta["doc-b#c001"] = {"slices": ["prose"], "doc": "doc-b"}
    meta["doc-b#c002"] = {"slices": ["dense-facts"], "doc": "doc-b"}
    all_texts = {**_DOC_A_CHUNKS, **doc_b_chunks}
    return meta, all_texts


def test_contradiction_pairs_below_minimum_counts_contradiction_verdicts_only(tmp_path):
    meta, all_texts = _build_two_doc_corpus_for_contradiction_tests(tmp_path)
    _write_index(tmp_path, meta, all_texts,
                contradiction_pairs=[
                    {"a": "doc-a#c003", "b": "doc-b#c002",
                     "claim_a": "a_c003_1", "claim_b": "b_c002_1",
                     "verdict": "contradiction", "note": "release year"},
                    {"a": "doc-a#c001", "b": "doc-b#c001",
                     "claim_a": "a_c001_1", "claim_b": "b_c001_1",
                     "verdict": "not_contradiction", "note": "version-scoped, reconcilable"},
                ],
                bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=(0, 100), min_contradiction_pairs=2)
    assert any("only 1 pair(s) with verdict 'contradiction'" in e and "needs >= 2" in e
               for e in errors)


def test_contradiction_pair_unknown_locator_is_flagged(tmp_path):
    meta, all_texts = _build_two_doc_corpus_for_contradiction_tests(tmp_path)
    _write_index(tmp_path, meta, all_texts,
                contradiction_pairs=[
                    {"a": "doc-a#c003", "b": "doc-b#c999",
                     "claim_a": "a_c003_1", "claim_b": "b_c002_1",
                     "verdict": "contradiction", "note": "x"}],
                bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=(0, 100), min_contradiction_pairs=1)
    assert any("doc-b#c999" in e and "unknown locator" in e for e in errors)


def test_contradiction_pair_same_document_is_flagged(tmp_path):
    meta, all_texts = _build_two_doc_corpus_for_contradiction_tests(tmp_path)
    _write_index(tmp_path, meta, all_texts,
                contradiction_pairs=[
                    {"a": "doc-a#c003", "b": "doc-a#c001",
                     "claim_a": "a_c003_1", "claim_b": "a_c001_1",
                     "verdict": "contradiction", "note": "x"}],
                bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=(0, 100), min_contradiction_pairs=1)
    assert any("within the same document" in e for e in errors)


def test_contradiction_pair_invalid_verdict_is_flagged(tmp_path):
    meta, all_texts = _build_two_doc_corpus_for_contradiction_tests(tmp_path)
    _write_index(tmp_path, meta, all_texts,
                contradiction_pairs=[
                    {"a": "doc-a#c003", "b": "doc-b#c002",
                     "claim_a": "a_c003_1", "claim_b": "b_c002_1",
                     "verdict": "maybe", "note": "x"}],
                bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=(0, 100), min_contradiction_pairs=0)
    assert any("invalid verdict" in e and "'maybe'" in e for e in errors)


def test_contradiction_pair_missing_claim_ids_is_malformed(tmp_path):
    meta, all_texts = _build_two_doc_corpus_for_contradiction_tests(tmp_path)
    _write_index(tmp_path, meta, all_texts,
                contradiction_pairs=[
                    {"a": "doc-a#c003", "b": "doc-b#c002",
                     "verdict": "contradiction", "note": "x"}],  # no claim_a/claim_b
                bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=(0, 100), min_contradiction_pairs=0)
    assert any("malformed contradiction pair" in e and "claim_a" in e for e in errors)


def test_contradiction_pair_unknown_claim_id_is_flagged(tmp_path):
    meta, all_texts = _build_two_doc_corpus_for_contradiction_tests(tmp_path)
    _write_index(tmp_path, meta, all_texts,
                contradiction_pairs=[
                    {"a": "doc-a#c003", "b": "doc-b#c002",
                     "claim_a": "a_ghost_claim", "claim_b": "b_c002_1",
                     "verdict": "contradiction", "note": "x"}],
                bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=(0, 100), min_contradiction_pairs=1)
    assert any("claim_a='a_ghost_claim'" in e and "not found" in e for e in errors)


def test_contradiction_pair_claim_not_citing_pair_chunk_is_flagged(tmp_path):
    meta, all_texts = _build_two_doc_corpus_for_contradiction_tests(tmp_path)
    # a_c001_1 is a real doc-a claim, but it cites doc-a#c001 -- not the
    # pair's a-side chunk doc-a#c003 -- so the gold pin points at evidence
    # the model would never be shown for this pair.
    _write_index(tmp_path, meta, all_texts,
                contradiction_pairs=[
                    {"a": "doc-a#c003", "b": "doc-b#c002",
                     "claim_a": "a_c001_1", "claim_b": "b_c002_1",
                     "verdict": "contradiction", "note": "x"}],
                bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=(0, 100), min_contradiction_pairs=1)
    assert any("claim_a='a_c001_1'" in e and "does not cite the pair's chunk" in e for e in errors)


def test_contradiction_pair_bare_list_is_malformed(tmp_path):
    meta, all_texts = _build_two_doc_corpus_for_contradiction_tests(tmp_path)
    _write_index(tmp_path, meta, all_texts,
                contradiction_pairs=[["doc-a#c003", "doc-b#c002"]],
                bait_sources={"doc-a#c002": "doc-a#c001"})
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=(0, 100), min_contradiction_pairs=0)
    assert any("malformed contradiction pair" in e for e in errors)


def test_corpus_version_missing_is_flagged(tmp_path):
    _build_doc_a(tmp_path)
    _write_index(tmp_path, _DOC_A_META, _DOC_A_CHUNKS,
                bait_sources={"doc-a#c002": "doc-a#c001"}, corpus_version=None)
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=_TINY_TOTAL_RANGE,
                      min_contradiction_pairs=0)
    assert any("missing 'corpus_version'" in e for e in errors)


def test_corpus_version_mismatch_is_flagged(tmp_path):
    _build_doc_a(tmp_path)
    _write_index(tmp_path, _DOC_A_META, _DOC_A_CHUNKS,
                bait_sources={"doc-a#c002": "doc-a#c001"}, corpus_version="sha256:not-the-real-hash")
    errors = validate(tmp_path, quotas=_TINY_QUOTAS, total_range=_TINY_TOTAL_RANGE,
                      min_contradiction_pairs=0)
    assert any("corpus_version mismatch" in e for e in errors)


# ---------------------------------------------------------------------------
# The committed corpus itself, validated at full production scale.
# ---------------------------------------------------------------------------

def test_committed_corpus_passes_with_default_quotas():
    errors = validate(DEFAULT_CORPUS_DIR)
    assert errors == [], "\n".join(errors)

