# Phase 1 — Pipeline Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the extract recall, bait-rejection, throughput, and two prose-failure gaps with prompt/contract/flow/serving changes only (no weight updates), and re-establish a trustworthy, correctly-measured baseline.

**Architecture:** Reshape the extract evidence contract to character spans so the gate is an O(1) slice-compare and near-quote bait is structurally impossible; formalize union + self-consistency for recall; fix the recall metric (retire precision_proxy, add an embedding entailment matcher, pooled gold); constrain and validate citations for the two prose failures; and land the deferred tier-2 response cache + threaded extraction. Every change is guarded by a deterministic unit test; live-model behavior is re-measured with the existing eval harness at the end.

**Tech Stack:** Python ≥3.10, pytest, Ollama 0.31.1 (OpenAI-compatible `/v1`), Gemma-4 role map, `sentence-transformers`/Ollama embeddings (qwen3-embedding:4b), DuckDB (compiler extra). No new heavy dependencies.

## Global Constraints

- **Gemma-4-first**, single-GPU (RTX 5090 / 32 GB). No model swaps into non-Gemma stacks for judgment roles.
- **Serving:** send `reasoning_effort:"none"` for non-thinking roles (NOT `think:false` — ignored for Gemma-4 on `/v1`, ollama#15288). Keep any single prompt under ~16k tokens (context truncates ~16,387 processed tokens).
- **Determinism preserved:** first attempts stay at temperature 0.0; only retries raise temperature (existing pattern: 0.25).
- **The verbatim gate is the one gate** (`common/verbatim.py`): do not add a per-stage definition of "the source text." Extend it; don't fork it.
- **Back-compat:** downstream consumers read a `quote` string on evidence and `[claim:<id>]` markers on prose. New shapes MUST keep populating those so the compiler/dossier/eval stages are unchanged.
- **Commit style:** imperative, capitalized, no `feat:`/`fix:` prefix, **no AI/Claude attribution or Co-Authored-By trailers** (repo convention).
- **Tests:** unit tests in `tests/unit/`, pure/no-backend; fast suite must stay green (`.venv/Scripts/python.exe -m pytest -q`). Live re-measurement uses `-m live_model` / `scripts/eval-pipeline.py` and is out of the fast suite.
- **schema_version:** bump `extract.SCHEMA_VERSION` when the persisted evidence shape changes.

---

## File Structure

- `src/deep_research_toolkit/common/verbatim.py` — add `span_ok()` + `slice_span()`; keep `verbatim_ok()` (WS1, WS6-cache-safe).
- `src/deep_research_toolkit/llm/extract.py` — span evidence shape in `_SYSTEM`/`_WEB_EVIDENCE`/`_PDF_EVIDENCE`; gate via spans in `extract_claims_to_run`; derive+store `quote`; `citable` flag; union/self-consistency orchestration; `parallel` fan-out (WS1, WS2, WS5-citable, WS6).
- `src/deep_research_toolkit/llm/selfconsistency.py` — new: union + N-sample + ≥k-of-N + dedup helpers (WS2).
- `src/deep_research_toolkit/llm/response.py` — coverage-gate redesign (absolute rule under a citable-count floor) in `generate_cited` (WS5).
- `src/deep_research_toolkit/llm/cache.py` — new: content-hash response cache (WS6).
- `src/deep_research_toolkit/llm/local.py` — accept a JSON-schema `format` passthrough + `finish_reason` capture (WS4).
- `src/deep_research_toolkit/evalkit/metrics.py` — retire `precision_proxy` label; add embedding-matched recall + self-faithfulness + `f_fact` (WS3).
- `src/deep_research_toolkit/evalkit/embed_match.py` — new: cosine entailment matcher (WS3).
- `scripts/build-pooled-gold.py` — new: pooled-gold generator (WS3).
- `tests/unit/test_*` — one test module per new/changed unit.

---

## Task 1 — Span-offset gate primitives (`common/verbatim.py`)

**Files:**
- Modify: `src/deep_research_toolkit/common/verbatim.py`
- Test: `tests/unit/test_verbatim_span.py`

**Interfaces:**
- Produces:
  - `slice_span(source_text: str, start: int, end: int) -> str | None` — returns `source_text[start:end]` when `0 <= start < end <= len(source_text)`, else `None`.
  - `span_ok(start, end, source_text: str, claimed_quote: str | None = None) -> bool` — True when the span is in-bounds, non-empty, and (if `claimed_quote` given) the slice equals it exactly.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_verbatim_span.py
from deep_research_toolkit.common.verbatim import slice_span, span_ok

SRC = "The mempool orders transactions by fee density before inclusion."

def test_slice_span_in_bounds():
    assert slice_span(SRC, 4, 11) == "mempool"

def test_slice_span_out_of_bounds_or_empty():
    assert slice_span(SRC, -1, 5) is None
    assert slice_span(SRC, 5, 5) is None
    assert slice_span(SRC, 5, 999) is None
    assert slice_span(SRC, 11, 4) is None

def test_span_ok_bounds_only():
    assert span_ok(4, 11, SRC) is True
    assert span_ok(5, 5, SRC) is False
    assert span_ok(5, 999, SRC) is False

def test_span_ok_matches_claimed_quote():
    assert span_ok(4, 11, SRC, "mempool") is True
    # near-quote bait: claimed text is NOT what the span actually covers
    assert span_ok(4, 11, SRC, "mempols") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_verbatim_span.py -v`
Expected: FAIL — `ImportError: cannot import name 'slice_span'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/deep_research_toolkit/common/verbatim.py`:

```python
def slice_span(source_text: str, start: int, end: int) -> str | None:
    """Return source_text[start:end] iff the span is a valid, non-empty,
    in-bounds forward slice; else None. The span replaces the free-text quote:
    a claim points AT its evidence by character offsets instead of copying it,
    so 'almost a quote' (near-quote bait) cannot be represented at all."""
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 0 or end > len(source_text) or start >= end:
        return None
    return source_text[start:end]


def span_ok(start: int, end: int, source_text: str,
            claimed_quote: str | None = None) -> bool:
    """The span-contract analogue of verbatim_ok: the span must be in-bounds and
    non-empty, and if the model also echoed a `quote`, the slice it points at
    must equal that quote exactly (no near-copy). This is an O(1) slice-compare,
    not a substring search."""
    sliced = slice_span(source_text, start, end)
    if sliced is None:
        return False
    return claimed_quote is None or sliced == claimed_quote
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_verbatim_span.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/common/verbatim.py tests/unit/test_verbatim_span.py
git commit -m "Add span-offset gate primitives (slice_span, span_ok)"
```

---

## Task 2 — Span evidence contract in extraction (`llm/extract.py`)

**Files:**
- Modify: `src/deep_research_toolkit/llm/extract.py` (`_WEB_EVIDENCE`, `_PDF_EVIDENCE`, `_SYSTEM` HARD INVARIANT, the gate loop in `extract_claims_to_run`, `SCHEMA_VERSION`)
- Test: `tests/unit/test_extract_span_contract.py`

**Interfaces:**
- Consumes: `span_ok`, `slice_span` (Task 1).
- Produces: persisted claim evidence rows now carry `start_char`/`end_char` and a **derived** `quote` (`slice_span(chunk_text, start, end)`), so downstream `quote` readers and the recall metric are unchanged. Each kept claim gains `"citable": true` (has ≥1 gate-passing span); dropped claims are `citable: false`. `extract_claims_to_run` return dict unchanged in shape.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_extract_span_contract.py
import json
from pathlib import Path
from deep_research_toolkit.llm import extract

CHUNK = "Validators gossip block headers before the body arrives on the wire."

class FakeBackend:
    thinking = False
    def __init__(self, payload): self._payload = payload
    def complete(self, system, user, **kw): return self._payload

def _run(tmp_path, payload):
    (tmp_path / "chunks.jsonl").write_text(
        json.dumps({"locator": "doc#c1", "text": CHUNK}) + "\n", encoding="utf-8")
    return extract.extract_claims_to_run(tmp_path, "web", config=None,
                                         backend=FakeBackend(payload))

def test_span_claim_kept_and_quote_derived(tmp_path):
    # "Validators gossip block headers" == CHUNK[0:31]
    payload = json.dumps({"claims": [{
        "claim_id": "c1", "claim": "Validators gossip headers first.",
        "claim_type": "architectural", "confidence": "high",
        "supporting_evidence": [{"locator": "doc#c1", "start_char": 0, "end_char": 31}],
    }], "entities": [], "relations": []})
    summary = _run(tmp_path, payload)
    assert summary["written"] == 1
    row = json.loads((tmp_path / "claims.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ev = row["supporting_evidence"][0]
    assert ev["quote"] == "Validators gossip block headers"  # derived from the span
    assert row["citable"] is True

def test_out_of_bounds_span_is_dropped(tmp_path):
    payload = json.dumps({"claims": [{
        "claim_id": "c2", "claim": "bogus", "confidence": "low",
        "supporting_evidence": [{"locator": "doc#c1", "start_char": 0, "end_char": 9999}],
    }], "entities": [], "relations": []})
    summary = _run(tmp_path, payload)
    assert summary["written"] == 0
    assert "c2" in summary["dropped"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_extract_span_contract.py -v`
Expected: FAIL — quote is not derived (evidence has no `quote`), `citable` missing.

- [ ] **Step 3: Update the evidence shape + invariant text**

In `src/deep_research_toolkit/llm/extract.py`, bump the version and reshape:

```python
SCHEMA_VERSION = "2.0"  # evidence now carries character spans, not a copied quote
```

Replace the two evidence constants:

```python
_PDF_EVIDENCE = '{"node_id": "<chunk_id>", "start_char": <int>, "end_char": <int>, "page": <int>}'
_WEB_EVIDENCE = '{"locator": "<chunk_id>", "start_char": <int>, "end_char": <int>, "url": "<source url or null>"}'
```

Replace the HARD INVARIANT block in `_SYSTEM` with the span contract:

```python
HARD INVARIANT (a precondition, checked mechanically downstream):
Every supporting_evidence points at its source by CHARACTER OFFSETS into the
cited chunk's text: start_char/end_char such that chunk_text[start_char:end_char]
IS the supporting span, copied by reference not by hand. Offsets are 0-based,
end-exclusive, and must satisfy 0 <= start_char < end_char <= len(chunk_text).
A deterministic gate DROPS any claim whose span is out of bounds or empty.
Point at the SHORTEST contiguous span that supports the claim. Under-produce
rather than approximate: if no single contiguous span supports the claim, drop it.
```

- [ ] **Step 4: Rewrite the gate loop to use spans and derive the quote**

In `extract_claims_to_run`, replace the per-evidence check block (the `for ev in evidence:` body that currently calls `verbatim_ok`) with:

```python
from ..common.verbatim import span_ok, slice_span  # add to imports at top of file
```

```python
            evidence = claim.get("supporting_evidence") or []
            ok = bool(evidence)
            for ev in evidence:
                if not isinstance(ev, dict):
                    ok = False
                    continue
                real = _resolve(str(ev.get(id_key) or ev.get("node_id") or ev.get("locator") or ""))
                start, end = ev.get("start_char"), ev.get("end_char")
                if real and span_ok(start, end, chunk_text_by_id[real],
                                    ev.get("quote")):
                    ev[id_key] = real
                    ev["quote"] = slice_span(chunk_text_by_id[real], start, end)  # derived, canonical
                else:
                    ok = False
            claim["citable"] = ok
            (kept if ok else dropped).append(claim)
```

(The `claim["citable"] = ok` line replaces the bare `(kept if ok else dropped).append(claim)`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_extract_span_contract.py tests/unit/test_llm_extract.py -v`
Expected: PASS. If `tests/unit/test_llm_extract.py` asserts the old `quote`-shaped evidence, update those fixtures to the span shape (offsets → derived quote) — the derived `quote` value is unchanged, only the input shape moves to offsets.

- [ ] **Step 6: Commit**

```bash
git add src/deep_research_toolkit/llm/extract.py tests/unit/test_extract_span_contract.py tests/unit/test_llm_extract.py
git commit -m "Move extract evidence to character-span contract; derive quote, add citable flag"
```

---

## Task 3 — Update the eval corpus + validate-local-llm to the span shape

**Files:**
- Modify: `scripts/check-eval-corpus.py` (accept span evidence; keep quote-derived gold), `tests/fixtures/eval-corpus/*/reference-claims.jsonl` (regenerate with spans), `scripts/validate-local-llm.py` (span-aware recovery)
- Test: `tests/unit/test_eval_corpus.py` (existing checker tests)

**Interfaces:**
- Consumes: `slice_span`, `span_ok` (Task 1).
- Produces: reference claims carry `start_char`/`end_char` + derived `quote`; the corpus checker validates spans; `corpus_version` hash changes (expected — record it).

- [ ] **Step 1: Write the failing test** — extend `tests/unit/test_eval_corpus.py`:

```python
def test_checker_accepts_span_evidence(tmp_path):
    from importlib import import_module
    checker = import_module("scripts.check-eval-corpus".replace("-", "_"))  # or the checker's public fn
    # a reference claim whose span slices to its quote must validate;
    # an out-of-bounds span must be rejected. (Use the checker's public entry point.)
```

If `scripts/check-eval-corpus.py` is not importable as a module, add a thin `validate_claim_evidence(claim, chunk_text) -> bool` function to it and test that directly.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_eval_corpus.py -k span -v`
Expected: FAIL.

- [ ] **Step 3: Add span validation to the checker**

In `scripts/check-eval-corpus.py`, add and use:

```python
from src.deep_research_toolkit.common.verbatim import span_ok, slice_span  # adjust import to the script's path setup

def validate_claim_evidence(claim: dict, chunk_text: str) -> bool:
    evs = claim.get("supporting_evidence") or []
    if not evs:
        return False
    for ev in evs:
        if not span_ok(ev.get("start_char"), ev.get("end_char"), chunk_text, ev.get("quote")):
            return False
    return True
```

- [ ] **Step 4: Regenerate the gold reference with spans**

For each `tests/fixtures/eval-corpus/<doc>/reference-claims.jsonl`, convert each evidence `{quote}` to `{start_char, end_char, quote}` where `start_char = chunk_text.find(quote)`, `end_char = start_char + len(quote)`, and keep the derived `quote`. Write a one-off `scripts/_migrate_gold_to_spans.py` that reads each doc's `chunks.jsonl`, locates each existing quote, and rewrites the reference file; fail loudly if any quote is not found (that means the gold was already non-verbatim — fix it by hand).

- [ ] **Step 5: Run the corpus checker + tests**

Run: `.venv/Scripts/python.exe scripts/check-eval-corpus.py` then `.venv/Scripts/python.exe -m pytest tests/unit/test_eval_corpus.py -v`
Expected: checker clean; tests PASS. Record the new `corpus_version` hash printed by the eval runner.

- [ ] **Step 6: Commit**

```bash
git add scripts/check-eval-corpus.py scripts/validate-local-llm.py tests/fixtures/eval-corpus tests/unit/test_eval_corpus.py
git commit -m "Migrate eval-corpus gold and checker to the span evidence shape"
```

---

## Task 4 — Union + self-consistency orchestration (`llm/selfconsistency.py`)

**Files:**
- Create: `src/deep_research_toolkit/llm/selfconsistency.py`
- Test: `tests/unit/test_selfconsistency.py`

**Interfaces:**
- Produces:
  - `union_claims(candidate_lists: list[list[dict]], min_support: int = 1) -> list[dict]` — dedups claims across N sample passes by a normalized key, keeps those appearing in ≥ `min_support` passes, merges evidence spans.
  - `claim_key(claim: dict) -> str` — normalized dedup key: lowercased, whitespace/punct-collapsed claim text + sorted `(locator, start_char, end_char)` spans.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_selfconsistency.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# src/deep_research_toolkit/llm/selfconsistency.py
"""Union-over-samples + support-count filtering for extraction recall.

Running the extractor N times at varied temperature and UNIONing gate-passing
claims raises recall; keeping only claims that recur in >= k of N passes is a
cheap precision/bait cut. One knob (min_support) trades the two. Dedup is by a
normalized claim key, NOT by claim_id (ids are per-pass)."""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def _norm(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub("", (text or "").lower())).strip()


def claim_key(claim: dict) -> str:
    spans = sorted(
        (str(ev.get("locator") or ev.get("node_id") or ""), ev.get("start_char"), ev.get("end_char"))
        for ev in (claim.get("supporting_evidence") or [])
    )
    return _norm(claim.get("claim", "")) + "||" + repr(spans)


def union_claims(candidate_lists: list[list[dict]], min_support: int = 1) -> list[dict]:
    first: dict[str, dict] = {}
    support: dict[str, int] = {}
    for claims in candidate_lists:
        seen_this_pass: set[str] = set()
        for c in claims:
            k = claim_key(c)
            first.setdefault(k, c)
            if k not in seen_this_pass:      # one pass contributes at most 1 to support
                support[k] = support.get(k, 0) + 1
                seen_this_pass.add(k)
    return [first[k] for k, n in support.items() if n >= min_support]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_selfconsistency.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/llm/selfconsistency.py tests/unit/test_selfconsistency.py
git commit -m "Add union + support-count self-consistency helper for extraction recall"
```

---

## Task 5 — Wire self-consistency + bounded coverage loop into extraction

**Files:**
- Modify: `src/deep_research_toolkit/llm/extract.py` (`extract_claims_to_run` gains `samples`, `min_support`, `coverage_passes`)
- Test: `tests/unit/test_extract_selfconsistency.py`

**Interfaces:**
- Consumes: `union_claims` (Task 4).
- Produces: `extract_claims_to_run(run_dir, producer, config, backend, batch_size=DEFAULT_BATCH_SIZE, samples: int = 1, min_support: int = 1, coverage_passes: int = 0) -> dict`. `samples=1, coverage_passes=0` reproduces current behavior exactly (default). Summary dict gains `"samples"` and `"support_filtered"` counts.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_extract_selfconsistency.py
import json
from deep_research_toolkit.llm import extract

CHUNK = "Leaders rotate every epoch; followers verify the epoch signature."

class SeqBackend:
    """Returns a different payload per complete() call, to simulate N samples."""
    thinking = False
    def __init__(self, payloads): self._p = list(payloads); self._i = 0
    def complete(self, system, user, **kw):
        r = self._p[min(self._i, len(self._p) - 1)]; self._i += 1; return r

def _ev(s, e): return [{"locator": "d#c1", "start_char": s, "end_char": e}]
def _payload(claims): return json.dumps({"claims": claims, "entities": [], "relations": []})

def test_samples_union_recovers_more_claims(tmp_path):
    (tmp_path / "chunks.jsonl").write_text(json.dumps({"locator": "d#c1", "text": CHUNK}) + "\n", encoding="utf-8")
    # pass 1 finds claim A (span 0-7), pass 2 finds claim B (span 21-46)
    a = {"claim_id": "a", "claim": "Leaders rotate.", "confidence": "high", "supporting_evidence": _ev(0, 7)}
    b = {"claim_id": "b", "claim": "Followers verify.", "confidence": "high", "supporting_evidence": _ev(21, 46)}
    backend = SeqBackend([_payload([a]), _payload([b])])
    summary = extract.extract_claims_to_run(tmp_path, "web", None, backend, samples=2, min_support=1)
    assert summary["written"] == 2  # union of both passes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_extract_selfconsistency.py -v`
Expected: FAIL — `extract_claims_to_run` has no `samples` parameter.

- [ ] **Step 3: Implement the sample loop**

Refactor the per-batch dispatch in `extract_claims_to_run` so the existing single-pass batch/parse/gate logic becomes an inner function producing a `list[claim]` (gate-passed) for one pass, then run it `samples` times (varying temperature for passes >0: `sampling={"temperature": 0.2 * pass_no}`), collect the per-pass gate-passed claim lists, and `union_claims(lists, min_support)` before writing. Add:

```python
from .selfconsistency import union_claims  # top of file
```

Signature and the union point:

```python
def extract_claims_to_run(run_dir, producer, config, backend,
                          batch_size: int = DEFAULT_BATCH_SIZE,
                          samples: int = 1, min_support: int = 1,
                          coverage_passes: int = 0) -> dict:
    ...
    passes: list[list[dict]] = []
    for pass_no in range(max(1, samples)):
        pass_sampling = {"temperature": round(0.2 * pass_no, 3)} if pass_no else {}
        passes.append(_extract_one_pass(chunks, pass_sampling))  # existing batch/gate logic, returns kept claims
    kept = union_claims(passes, min_support=min_support) if samples > 1 else passes[0]
    ...
```

The bounded coverage loop (`coverage_passes`): after the union, for up to `coverage_passes` iterations, re-prompt the model with the already-found claim texts and "extract only ADDITIONAL atomic claims not already listed; return an empty claims array if none," gate + union the result; stop early on an empty return. Keep `_MAX_RETRY_DEPTH` halving inside each pass unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_extract_selfconsistency.py tests/unit/test_llm_extract.py tests/unit/test_extract_span_contract.py -v`
Expected: PASS; the default-args path (`samples=1`) leaves existing tests green.

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/llm/extract.py tests/unit/test_extract_selfconsistency.py
git commit -m "Wire N-sample union + bounded coverage loop into extraction (defaults preserve behavior)"
```

---

## Task 6 — Coverage-gate redesign for tiny inputs (`llm/response.py`)

**Files:**
- Modify: `src/deep_research_toolkit/llm/response.py` (`validate_citations`, `generate_cited`)
- Test: `tests/unit/test_response_coverage_gate.py`

**Interfaces:**
- Produces: `validate_citations(text, allowed_ids, *, min_citable_for_ratio: int = 4)` unchanged return keys plus `"coverage_ok"` and `"rule"` (`"ratio"` or `"absolute"`). Under `min_citable_for_ratio` allowed ids, coverage passes iff **every** allowed id is cited (absolute rule); at or above it, the existing ratio floor applies. Fixes the `compiler-ir-design#c015` 2-claim false failure.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_response_coverage_gate.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_response_coverage_gate.py -v`
Expected: FAIL — no `rule`/`coverage_ok` keys, no `min_citable_for_ratio` param.

- [ ] **Step 3: Implement the redesign**

Replace `validate_citations`:

```python
def validate_citations(text: str, allowed_ids: list[str], *,
                       min_citable_for_ratio: int = 4,
                       min_coverage: float = 0.3) -> dict:
    """Closed-set citation check + coverage rule. With few citable claims the
    ratio floor is statistically meaningless (2 claims, one uncited = 0.5 <
    0.3-floor false alarm), so below `min_citable_for_ratio` we require EVERY
    allowed id to be cited (absolute rule); at/above it the ratio floor holds."""
    allowed = set(allowed_ids)
    cited_all = extract_claim_ids(text)
    cited = [c for c in cited_all if c in allowed]
    unknown = [c for c in cited_all if c not in allowed]
    coverage = (len(cited) / len(allowed)) if allowed else 0.0
    if len(allowed) < min_citable_for_ratio:
        rule = "absolute"
        coverage_ok = set(cited) >= allowed  # every allowed id cited
    else:
        rule = "ratio"
        coverage_ok = coverage >= min_coverage
    return {"cited": cited, "unknown": unknown, "coverage": coverage,
            "coverage_ok": coverage_ok, "rule": rule}
```

In `generate_cited`, replace the two `report["coverage"] < min_coverage` conditionals with `not report["coverage_ok"]`, and pass `min_coverage=min_coverage` into the `validate_citations` calls inside `_attempt`. The final `raise ValueError` message stays but reads the rule:

```python
        if not report["coverage_ok"]:
            raise ValueError(
                f"{kind} fails citation coverage ({report['rule']}): "
                f"cited {len(report['cited'])}/{len(allowed_ids)}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_response_coverage_gate.py tests/unit/test_llm_response.py tests/unit/test_llm_synthesize.py tests/unit/test_llm_wiki.py -v`
Expected: PASS; adjust any existing synthesize/wiki test that asserted the old coverage-float raise to the new rule-aware message.

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/llm/response.py tests/unit/test_response_coverage_gate.py
git commit -m "Redesign citation coverage gate: absolute rule under a citable-count floor"
```

---

## Task 7 — Enum-constrained citations + closed-set repair (prose roles)

**Files:**
- Modify: `src/deep_research_toolkit/llm/local.py` (accept a `format` JSON-schema passthrough on `complete`), `src/deep_research_toolkit/llm/wiki.py` + `llm/synthesize.py` (build the allowed-id enum note; the closed-set validator already lives in `generate_cited`)
- Test: `tests/unit/test_local_format_passthrough.py`, `tests/unit/test_llm_wiki.py`

**Interfaces:**
- Produces: `Backend.complete(system, user, *, response_format=None, **sampling)` forwards a caller-supplied JSON-schema `format` to Ollama unchanged (None = today's behavior). Wiki/synthesize prompts gain a fenced "Valid claim ids: [...]" block with one positive and one decline exemplar. The unknown-id hallucination on `mempool-design#c005` is caught by the existing closed-set retry in `generate_cited`; this task hardens it and adds the prompt-side lever.

- [ ] **Step 1: Write the failing test** (format passthrough is the mechanical, unit-testable part)

```python
# tests/unit/test_local_format_passthrough.py
from deep_research_toolkit.llm import local

def test_complete_forwards_response_format(monkeypatch):
    captured = {}
    class FakeResp:
        def __init__(self): self.choices = [type("C", (), {"message": type("M", (), {"content": "{}", "reasoning": None})()})()]; self.usage = None
    def fake_create(**kwargs):
        captured.update(kwargs); return FakeResp()
    b = local.LocalBackend(model="gemma4:12b", base_url="http://x/v1", api_key="x",
                           thinking=False, temperature=0.0, top_p=0.95, top_k=20, max_tokens=100)
    monkeypatch.setattr(b._client.chat.completions, "create", fake_create)
    schema = {"type": "object", "properties": {"cite": {"enum": ["c1", "c2"]}}}
    b.complete("sys", "user", response_format=schema)
    assert captured.get("extra_body", {}).get("format") == schema or captured.get("format") == schema
```

(Adjust the fake to `local.py`'s actual client-call shape once read; the assertion is: a caller-supplied schema reaches the Ollama request as `format`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_local_format_passthrough.py -v`
Expected: FAIL — `complete` drops `response_format` / does not forward `format`.

- [ ] **Step 3: Implement the passthrough + prompt block**

In `llm/local.py`'s `complete`, thread an optional `response_format` (a JSON-schema dict) into the request as Ollama's `format` field (via `extra_body={"format": schema}` on the OpenAI client, matching how role `response_format: "json"` is already sent). In `llm/wiki.py` and `llm/synthesize.py`, add to the task prompt a fenced block:

```
Valid claim ids (cite ONLY these, in [claim:<id>] form):
  c1, c2, c3
Example — cite an id that supports the sentence: "Fees rise under load [claim:c1]."
Example — if no listed id supports a sentence, write it WITHOUT a citation; never invent an id.
```

Keep the `generate_cited` closed-set validator + bounded repair retry as the model-independent backstop (already implemented) — the enum/prompt reduces how often it fires; the validator guarantees correctness.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_local_format_passthrough.py tests/unit/test_llm_wiki.py tests/unit/test_llm_synthesize.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/llm/local.py src/deep_research_toolkit/llm/wiki.py src/deep_research_toolkit/llm/synthesize.py tests/unit/test_local_format_passthrough.py
git commit -m "Add JSON-schema format passthrough and enum/exemplar citation prompts"
```

---

## Task 8 — Truncation telemetry + right-sized `num_predict` (`llm/local.py`)

**Files:**
- Modify: `src/deep_research_toolkit/llm/local.py` (capture `finish_reason`; expose per-call stat), `src/deep_research_toolkit/llm/extract.py` (count `finish_reason=="length"` across calls into the summary)
- Test: `tests/unit/test_local_finish_reason.py`, `tests/unit/test_llm_local_stats.py`

**Interfaces:**
- Produces: the backend records `last_finish_reason` (and appends to its per-call stats already used by `test_llm_local_stats.py`); `extract_claims_to_run` summary gains `"truncated_calls"`. This turns the silent 3–5× truncation amplification into a measured `<5%` SLO signal.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_local_finish_reason.py
from deep_research_toolkit.llm import local

def test_finish_reason_captured(monkeypatch):
    class Msg:  content = '{"ok": 1}'; reasoning = None
    class Choice: message = Msg(); finish_reason = "length"
    class Resp: choices = [Choice()]; usage = None
    b = local.LocalBackend(model="gemma4:e4b", base_url="http://x/v1", api_key="x",
                           thinking=False, temperature=0.0, top_p=0.95, top_k=20, max_tokens=50)
    monkeypatch.setattr(b._client.chat.completions, "create", lambda **k: Resp())
    b.complete("s", "u")
    assert b.last_finish_reason == "length"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_local_finish_reason.py -v`
Expected: FAIL — no `last_finish_reason` attribute.

- [ ] **Step 3: Implement capture + counting**

In `local.py`, after reading the response, set `self.last_finish_reason = resp.choices[0].finish_reason` and include it in the trace record already written when `llm.trace` is on. In `extract.py`, after each `backend.complete(...)` in the batch loop, increment a local `truncated_calls` counter when `getattr(backend, "last_finish_reason", None) == "length"`, and add `"truncated_calls": truncated_calls` to the returned summary. Document in the extract docstring that a batch should be sized so a full extract fits under `max_tokens` (right-size `num_predict` to batch, don't rely on the 3000 cap).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_local_finish_reason.py tests/unit/test_llm_local_stats.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/llm/local.py src/deep_research_toolkit/llm/extract.py tests/unit/test_local_finish_reason.py
git commit -m "Capture finish_reason and count truncated extract calls (truncation SLO signal)"
```

---

## Task 9 — Content-hash response cache (`llm/cache.py`)

**Files:**
- Create: `src/deep_research_toolkit/llm/cache.py`
- Test: `tests/unit/test_llm_cache.py`

**Interfaces:**
- Produces:
  - `cache_key(model, role, system, user, params: dict, schema: dict | None) -> str` — `sha256` hex of the canonical-JSON of all inputs.
  - `CachingBackend(inner, cache_dir: Path, enabled: bool)` — wraps a backend; on `complete`, returns a cached reply for a known key, else calls `inner.complete` and appends `{key, reply}` to a JSONL cache. Disabled → passthrough.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_llm_cache.py
from deep_research_toolkit.llm.cache import cache_key, CachingBackend

class CountingBackend:
    thinking = False
    def __init__(self): self.calls = 0
    def complete(self, system, user, **kw): self.calls += 1; return "REPLY"

def test_cache_key_stable_and_param_sensitive():
    k1 = cache_key("m", "extract", "s", "u", {"temperature": 0.0}, None)
    k2 = cache_key("m", "extract", "s", "u", {"temperature": 0.0}, None)
    k3 = cache_key("m", "extract", "s", "u", {"temperature": 0.25}, None)
    assert k1 == k2 and k1 != k3

def test_caching_backend_hits_disk(tmp_path):
    inner = CountingBackend()
    cb = CachingBackend(inner, cache_dir=tmp_path, enabled=True)
    a = cb.complete("s", "u", temperature=0.0)
    b = cb.complete("s", "u", temperature=0.0)
    assert a == b == "REPLY" and inner.calls == 1  # second call served from cache

def test_disabled_is_passthrough(tmp_path):
    inner = CountingBackend()
    cb = CachingBackend(inner, cache_dir=tmp_path, enabled=False)
    cb.complete("s", "u"); cb.complete("s", "u")
    assert inner.calls == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_llm_cache.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# src/deep_research_toolkit/llm/cache.py
"""Content-hash response cache: deterministic pipeline (reasoning_effort=none)
means an identical (model, role, prompt, params, schema) always yields the same
reply, so a sha256-keyed JSONL cache makes --runs N and halved-batch re-runs
nearly free. Opt-in via llm.cache: true."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def cache_key(model, role, system, user, params: dict, schema: dict | None) -> str:
    blob = json.dumps({"model": model, "role": role, "system": system, "user": user,
                       "params": params or {}, "schema": schema},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CachingBackend:
    def __init__(self, inner, cache_dir: Path, enabled: bool = True,
                 role: str = "", model: str | None = None):
        self.inner = inner
        self.enabled = enabled
        self.role = role
        self.model = model or getattr(inner, "model", "")
        self.thinking = getattr(inner, "thinking", True)
        self._path = Path(cache_dir) / "llm-cache.jsonl"
        self._mem: dict[str, str] = {}
        if enabled and self._path.is_file():
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self._mem[row["key"]] = row["reply"]

    def complete(self, system, user, **sampling) -> str:
        if not self.enabled:
            return self.inner.complete(system, user, **sampling)
        schema = sampling.get("response_format")
        key = cache_key(self.model, self.role, system, user,
                        {k: v for k, v in sampling.items() if k != "response_format"}, schema)
        if key in self._mem:
            return self._mem[key]
        reply = self.inner.complete(system, user, **sampling)
        self._mem[key] = reply
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "reply": reply}, ensure_ascii=False) + "\n")
        return reply
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_llm_cache.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Wire config + backend factory**

In `config.py` add `llm.cache` (bool, default False) to the parsed config; in `llm/backend.py`'s `get_backend`, when `config.raw.get("llm", {}).get("cache")`, wrap the constructed local backend in `CachingBackend(inner, cache_dir=config.index_dir.parent / "llm-cache", enabled=True, role=role)`. Add a unit test in `tests/unit/test_llm_cache.py` asserting `get_backend` returns a `CachingBackend` when the flag is on.

- [ ] **Step 6: Commit**

```bash
git add src/deep_research_toolkit/llm/cache.py src/deep_research_toolkit/config.py src/deep_research_toolkit/llm/backend.py tests/unit/test_llm_cache.py
git commit -m "Add opt-in content-hash response cache (llm.cache)"
```

---

## Task 10 — Threaded extraction fan-out (`llm/extract.py`)

**Files:**
- Modify: `src/deep_research_toolkit/llm/extract.py` (`parallel` param; ThreadPoolExecutor over passes/batches)
- Test: `tests/unit/test_extract_parallel.py`

**Interfaces:**
- Produces: `extract_claims_to_run(..., parallel: int = 1)`. `parallel=1` = current sequential behavior (default). With `parallel>1`, independent sample-passes/batches run on a bounded `ThreadPoolExecutor`; results merged deterministically by (pass, batch) index so output is identical to sequential. Worker count is clamped to `parallel`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_extract_parallel.py
import json, threading
from deep_research_toolkit.llm import extract

CHUNK = "A checkpoint commits when two-thirds of validators sign the same height."

class ConcurrentBackend:
    thinking = False
    def __init__(self): self.max_concurrent = 0; self._n = 0; self._lock = threading.Lock()
    def complete(self, system, user, **kw):
        with self._lock:
            self._n += 1; self.max_concurrent = max(self.max_concurrent, self._n)
        try:
            return json.dumps({"claims": [{"claim_id": "c", "claim": "Checkpoint commits at 2/3.",
                    "confidence": "high", "supporting_evidence": [{"locator": "d#c1", "start_char": 0, "end_char": 12}]}],
                    "entities": [], "relations": []})
        finally:
            with self._lock: self._n -= 1

def test_parallel_runs_concurrently_same_result(tmp_path):
    (tmp_path / "chunks.jsonl").write_text(json.dumps({"locator": "d#c1", "text": CHUNK}) + "\n", encoding="utf-8")
    backend = ConcurrentBackend()
    summary = extract.extract_claims_to_run(tmp_path, "web", None, backend, samples=4, parallel=4)
    assert summary["written"] >= 1
    assert backend.max_concurrent >= 2  # actually overlapped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_extract_parallel.py -v`
Expected: FAIL — no `parallel` parameter.

- [ ] **Step 3: Implement bounded fan-out**

Wrap the per-pass extraction (from Task 5's `_extract_one_pass`) in a `concurrent.futures.ThreadPoolExecutor(max_workers=max(1, parallel))` when `parallel > 1`, submitting one future per pass, collecting results in submission order (`[f.result() for f in futures]`) so the union input is deterministic. The `openai` client is thread-safe; guard shared mutable state (entities merge, relations list) by doing all merging AFTER the parallel passes return, not inside the workers. Add a module note: worker count should be set from `OLLAMA_NUM_PARALLEL` (2–4 for e4b), never high for 31B.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_extract_parallel.py tests/unit/test_extract_selfconsistency.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/llm/extract.py tests/unit/test_extract_parallel.py
git commit -m "Add bounded ThreadPoolExecutor fan-out for extraction sample-passes"
```

---

## Task 11 — Embedding entailment matcher + retire precision_proxy (`evalkit`)

**Files:**
- Create: `src/deep_research_toolkit/evalkit/embed_match.py`
- Modify: `src/deep_research_toolkit/evalkit/metrics.py` (`extract_metrics` — rename `precision_proxy` → `gold_match_rate` with a doc note, add optional `recall_entailment`, `self_faithfulness`, `f_fact`)
- Test: `tests/unit/test_embed_match.py`, `tests/unit/test_evalkit_metrics.py`

**Interfaces:**
- Produces:
  - `embed_recall(produced: list[dict], reference: list[dict], embedder, threshold: float = 0.9) -> dict` — a reference claim is recalled when any produced claim's text has cosine ≥ threshold to it; returns `{recalled, missed, recall}`. `embedder(list[str]) -> list[list[float]]` is injected (Ollama qwen3-embedding:4b in prod, a fake in tests).
  - `extract_metrics(...)` gains `gold_match_rate` (the old `precision_proxy`, renamed + documented as gold-coverage-limited) and, when an embedder is supplied, `recall_entailment` + `f_fact`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_embed_match.py
from deep_research_toolkit.evalkit.embed_match import embed_recall

def fake_embedder(texts):
    # 2-d toy embeddings: map by keyword so cosine is deterministic
    def vec(t):
        t = t.lower()
        if "rotate" in t: return [1.0, 0.0]
        if "verify" in t: return [0.0, 1.0]
        return [0.7, 0.7]
    return [vec(t) for t in texts]

def _c(text): return {"claim": text, "supporting_evidence": []}

def test_embed_recall_matches_paraphrase_above_threshold():
    produced = [_c("Leaders are rotated each epoch")]
    reference = [_c("Rotate the leader every epoch")]  # both -> [1,0]
    r = embed_recall(produced, reference, fake_embedder, threshold=0.9)
    assert r["recall"] == 1.0

def test_embed_recall_misses_unrelated():
    produced = [_c("Followers verify signatures")]   # [0,1]
    reference = [_c("Leaders rotate")]               # [1,0], cosine 0
    r = embed_recall(produced, reference, fake_embedder, threshold=0.9)
    assert r["recall"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_embed_match.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# src/deep_research_toolkit/evalkit/embed_match.py
"""Embedding-based recall: authored gold is non-exhaustive and exact-substring
matching undercounts paraphrased recall. Cosine similarity between produced and
reference claim TEXT (not quotes) via an injected embedder gives a recall signal
that does not conflate model quality with gold incompleteness."""
from __future__ import annotations

import math


def _cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embed_recall(produced, reference, embedder, threshold: float = 0.9) -> dict:
    if not reference:
        return {"recalled": [], "missed": [], "recall": None}
    ref_texts = [r.get("claim", "") for r in reference]
    prod_texts = [p.get("claim", "") for p in produced]
    ref_vecs = embedder(ref_texts) if ref_texts else []
    prod_vecs = embedder(prod_texts) if prod_texts else []
    recalled, missed = [], []
    for ref, rv in zip(reference, ref_vecs):
        if any(_cos(rv, pv) >= threshold for pv in prod_vecs):
            recalled.append(ref)
        else:
            missed.append(ref)
    return {"recalled": recalled, "missed": missed, "recall": len(recalled) / len(reference)}
```

- [ ] **Step 4: Retire precision_proxy label in metrics.py**

In `extract_metrics`, rename the returned `"precision_proxy"` key to `"gold_match_rate"` and update its docstring line to: "share of produced claims that matched some reference claim — bounded by gold non-exhaustiveness, NOT model precision; do not gate on it." Add optional params `embedder=None, self_faithfulness=None` and, when `embedder` is provided, include `recall_entailment` (from `embed_recall`) and `f_fact` = harmonic mean of `recall_entailment` and `self_faithfulness` when both present. Update `tests/unit/test_evalkit_metrics.py` to assert the new key name and that the entailment fields appear only when an embedder is passed.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_embed_match.py tests/unit/test_evalkit_metrics.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/deep_research_toolkit/evalkit/embed_match.py src/deep_research_toolkit/evalkit/metrics.py tests/unit/test_embed_match.py tests/unit/test_evalkit_metrics.py
git commit -m "Add embedding entailment recall and retire precision_proxy label"
```

---

## Task 12 — Pooled-gold generator (`scripts/build-pooled-gold.py`)

**Files:**
- Create: `scripts/build-pooled-gold.py`
- Modify: `src/deep_research_toolkit/evalkit/__init__.py` (export a pure `pool_gold` helper the script wraps)
- Test: `tests/unit/test_pool_gold.py`

**Interfaces:**
- Produces: `pool_gold(claim_lists: list[list[dict]]) -> list[dict]` — union + dedup (reuse `selfconsistency.claim_key`) of gate-passing claims from multiple models into one deduped gold set; the script runs extract with `e4b` and `qwen3:30b-a3b` over the corpus, gate-filters, pools, and writes `tests/fixtures/eval-corpus/<doc>/pooled-gold.jsonl`. This set is both the fixed eval denominator and the Phase-2 SFT target.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pool_gold.py
from deep_research_toolkit.evalkit import pool_gold

def _c(text, s, e):
    return {"claim": text, "supporting_evidence": [{"locator": "d#c1", "start_char": s, "end_char": e}]}

def test_pool_dedups_across_models():
    e4b = [_c("Leaders rotate each epoch", 0, 10)]
    qwen = [_c("leaders  rotate each epoch", 0, 10), _c("Followers verify", 20, 30)]
    pooled = pool_gold([e4b, qwen])
    texts = sorted(c["claim"].lower().strip() for c in pooled)
    assert len(pooled) == 2  # the rotate claim collapses; verify is added
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_pool_gold.py -v`
Expected: FAIL — `pool_gold` not exported.

- [ ] **Step 3: Implement `pool_gold` + the script**

In `evalkit/__init__.py`:

```python
from ..llm.selfconsistency import union_claims

def pool_gold(claim_lists):
    """Union + dedup gate-passing claims from multiple models into one gold set
    (min_support=1: any model's gate-passed claim is gold). Dedup by claim_key."""
    return union_claims(claim_lists, min_support=1)
```

`scripts/build-pooled-gold.py`: load config, for each corpus doc run `extract_claims_to_run` with `backend=get_backend(config, "extract")` forced to `gemma4:e4b` and then to `qwen3:30b-a3b` (override the role model), read back each `claims.jsonl`, call `pool_gold([e4b_claims, qwen_claims])`, and write `pooled-gold.jsonl` per doc. Print the new `corpus_version`/pooled-gold hash. Requires both models pulled and a `provider: local` config (skip-with-reason otherwise, mirroring the live tier).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_pool_gold.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build-pooled-gold.py src/deep_research_toolkit/evalkit/__init__.py tests/unit/test_pool_gold.py
git commit -m "Add pooled-gold generator (e4b union qwen3:30b-a3b) for a fixed recall denominator"
```

---

## Task 13 — Live re-measurement + new baseline

**Files:**
- Modify: `scripts/eval-pipeline.py` (feed `samples`/`min_support`/`parallel` through to extract; add the embedder + pooled gold to metric assembly; add `truncated_calls` and rule-aware coverage to the report), `eval-results/baseline.json` (promote a new run)
- Test: `tests/unit/test_eval_pipeline.py` (pure-logic paths)

**Interfaces:**
- Consumes: everything above.
- Produces: an eval report scored against pooled gold with the entailment matcher, a truncation-SLO line, and a promoted `baseline.json`. This is the Gate-1→2 measurement.

- [ ] **Step 1: Extend the pure-logic tests**

In `tests/unit/test_eval_pipeline.py`, add assertions that the report carries `truncated_calls` per model and `recall_entailment` when an embedder is configured, using the existing in-process fake backend (no live model).

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_eval_pipeline.py -v`
Expected: FAIL on the new assertions.

- [ ] **Step 3: Wire the runner**

Thread `--samples`, `--min-support`, `--parallel`, and `--pooled-gold` flags through `scripts/eval-pipeline.py` into `extract_claims_to_run` and the metric calls; load pooled-gold per doc when present (else fall back to `reference-claims.jsonl`); construct the Ollama embedder (qwen3-embedding:4b) for `recall_entailment`; include `truncated_calls` and the coverage `rule` in `build_report`.

- [ ] **Step 4: Run the fast suite green**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS (all unit tests, live tier excluded).

- [ ] **Step 5: Live canaries + full eval (manual, needs the local stack)**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/live -m live_model -q
.venv/Scripts/python.exe scripts/build-pooled-gold.py
.venv/Scripts/python.exe scripts/eval-pipeline.py --samples 5 --min-support 2 --runs 5 --pooled-gold
.venv/Scripts/python.exe scripts/eval-pipeline.py --compare eval-results/baseline.json
```
Expected: 6/6 canaries; a full report; record bait_rejection (target ≥0.95 from the span contract), the entailment recall + its bootstrap CI lower bound, and the truncation rate (<5% SLO). Promote with `--write-baseline` only after review.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval-pipeline.py tests/unit/test_eval_pipeline.py eval-results/baseline.json
git commit -m "Re-measure Phase-1 baseline against pooled gold with entailment recall and truncation SLO"
```

---

## Gate 1→2 evaluation (end of Phase 1)

After Task 13, record in the design doc §8:
1. bait_rejection with the span contract (expect ≥0.95 → bait half closed with no training);
2. entailment recall on pooled gold with its **bootstrap 95% CI lower bound**;
3. truncation rate (<5% SLO), both prose failures fixed on their chunks.

**Launch trigger:** if the recall CI lower bound still misses 0.90, open the Phase-2 plan and begin e4b Recipe B (env provisioning proceeds in parallel throughout, GPU-idle). **Early-exit:** if N=5 union stops adding recall, do not wait for the full stack.

---

## Self-Review

**Spec coverage:** §5.1 → Tasks 1–3; §5.2 → Tasks 4–5; §5.3 → Tasks 11–12 (+13); §5.4 → Tasks 7 (format), 8 (truncation); §5.5 → Tasks 6 (coverage), 7 (enum/validator/citable via Task 2); §5.6 → Tasks 9 (cache), 10 (threaded), residency/serving flags are ops notes recorded in README (fold into Task 8's docstring + a README edit in Task 13). Gate 1→2 → the final section.

**Placeholder scan:** every code step carries real code; the two places that say "adjust to the script's actual client-call shape" (Task 7/8 fakes) are explicitly bounded by a concrete assertion (the schema reaches the request as `format`; `finish_reason` is captured) — implementer confirms the exact attribute path against `local.py` when read.

**Type consistency:** span evidence `{locator|node_id, start_char:int, end_char:int, quote:str(derived)}` is used identically in Tasks 1,2,3,4,5,12; `citable:bool` set in Task 2, consumed in Task 6's coverage rule; `union_claims(lists, min_support)` defined in Task 4, reused in Tasks 5 and 12; `validate_citations(..., min_citable_for_ratio, min_coverage)` defined in Task 6, called in Task 7's flow; `CachingBackend`/`cache_key` signatures match Task 9 throughout.
