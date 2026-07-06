# Eval Suite + Flow Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Implement docs/superpowers/specs/2026-07-06-comprehensive-eval-and-flow-hardening-design.md — a live-model eval tier (canaries, ~200-chunk stratified corpus, deterministic metrics, baseline diffing) plus tier-1 flow hardening (retry mutation, trace logging, repetition gate, prompt exemplars, env guidance).

**Architecture:** Everything extends the existing pytest + plain-Python-gates design. New live tests live in `tests/live/` behind a `live_model` marker (excluded by default). The eval runner is a standalone script producing JSON reports + a JSONL time series. Corpus is synthetic, frontier-authored, mechanically gate-verified at build time. Flow hardening touches the three judgment callers and the local backend only.

**Conventions (every task):** repo `C:\deep-research-toolkit`, branch `feat/eval-suite`, venv `.venv/Scripts/python`, run from repo root. TDD for library code. Imperative commits, NO Co-Authored-By trailer. Never touch `src/deep_research_toolkit/skill_templates/` (synced at the end) or `.deepresearch.yml`. Live tests must SKIP with a clear reason (not fail) when config/provider/endpoint is absent: use `pytest.skip` in a module-level or fixture check.

---

### Task 1: Repetition-loop gate — `has_repetition_loop` in `llm/response.py`

**Files:** Modify `src/deep_research_toolkit/llm/response.py`, `src/deep_research_toolkit/llm/adjudicate.py`, `src/deep_research_toolkit/llm/synthesize.py`; Test `tests/unit/test_llm_response.py`, `tests/unit/test_llm_adjudicate.py`.

- [ ] 1.1 Failing tests (append to `tests/unit/test_llm_response.py`; update its import line):

```python
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
```

- [ ] 1.2 Implement in `response.py` (word-level trailing-pattern detector mirroring vLLM's shipped mitigation for the Gemma 4 constrained-JSON loop bug, ollama#15502):

```python
def has_repetition_loop(text: str, max_pattern: int = 20, min_repeats: int = 4,
                        min_words: int = 40) -> bool:
    """True when the text's tail is one phrase repeated over and over --
    the constrained-decoding failure mode Gemma 4 exhibits (ollama#15502).
    Word-level check: for pattern lengths 1..max_pattern, test whether the
    last (pattern * min_repeats) words are the same pattern repeated."""
    words = text.split()
    if len(words) < min_words:
        return False
    for size in range(1, max_pattern + 1):
        window = size * min_repeats
        if window > len(words):
            break
        tail = words[-window:]
        pattern = tail[:size]
        if all(tail[i] == pattern[i % size] for i in range(window)):
            return True
    return False
```

- [ ] 1.3 Wire into the 31B-role callers: in `adjudicate.py`, after `reply = backend.complete(...)`, treat a looping reply as a parse failure for that batch (`if has_repetition_loop(reply): parse_failures += 1; continue`) BEFORE `parse_json_block`. In `synthesize.py`, check the completion before `validate_citations`; a looping reply takes the retry path with the failure reason "your previous reply degenerated into repetition" (Task 2 adds the mutation machinery — if Task 2 is not yet merged, raise `ValueError("model reply degenerated into repetition")` after the existing single retry). Add one stub-backend test per caller (looping reply → parse_failure counted / retry then error).
- [ ] 1.4 Run `tests/unit/test_llm_response.py test_llm_adjudicate.py test_llm_synthesize.py` → green. Commit: `"Add a repetition-loop gate for constrained Gemma 4 replies"`.

---

### Task 2: Retry mutation — failure-reason feedback + temperature bump

**Files:** Modify `src/deep_research_toolkit/llm/extract.py`, `llm/wiki.py`, `llm/synthesize.py`; Test the three matching unit files.

Contract (from design §4.1, research-backed): first attempts stay at the role's configured sampling; ANY retry (a) appends a concrete failure description to the user prompt and (b) passes `temperature=0.25` as a sampling override (`backend.complete(system, user, temperature=0.25)` — the Backend protocol already forwards `**sampling`; AgentBackend raises before sampling matters).

- [ ] 2.1 `wiki.py` + `synthesize.py`: the existing unknown-marker retry already appends `_CORRECTION`; add `temperature=0.25` to that retry call. Add ONE low-coverage retry: when coverage < min_coverage on the accepted (marker-clean) text, retry once with an appended note — `"Your previous reply cited only {n}/{total} of the supplied claims. Rewrite the full {kind}, grounding every factual sentence in a supplied claim marker."` — also at `temperature=0.25`; if coverage still fails, raise the existing ValueError. Total backend calls are bounded by 3 (initial + marker retry + coverage retry) — assert that bound in a test with a StubBackend counting calls. Update existing coverage tests: `test_low_coverage_body_is_rejected` / `test_zero_citation_thesis_is_rejected` now need TWO low-coverage replies queued before the raise.
- [ ] 2.2 `extract.py`: locate the halving-retry path (`_MAX_RETRY_DEPTH`, batch split on parse failure). On each retry call, append to the user prompt: `"NOTE: a previous attempt on these chunks failed to parse as the required JSON. Emit ONLY the contract JSON."` and pass `temperature=0.25`. Do not change batch-halving logic. Unit-test via the existing extract test file's fake-backend pattern (read how `test_llm_extract.py` fakes completions first; follow it).
- [ ] 2.3 Run the four unit files → green; full `tests/unit` → green. Commit: `"Mutate retries: failure feedback plus temperature bump"`.

---

### Task 3: Per-call JSONL trace logging

**Files:** Modify `src/deep_research_toolkit/llm/local.py`, `src/deep_research_toolkit/config.py` (add `llm.trace` bool, default False, parsed like `llm.provider`), `src/deep_research_toolkit/llm/backend.py` (pass trace flag/role); Test `tests/unit/test_llm_trace.py` (new).

Contract: when `llm.trace: true`, `LocalOpenAIBackend.complete()` appends one JSON line per call to `llm-trace.jsonl` in the CWD, using OTel GenAI field names: `{"ts": <iso8601 UTC>, "gen_ai.request.model": ..., "gen_ai.usage.input_tokens": ..., "gen_ai.usage.output_tokens": ..., "latency_s": ..., "role": <role or null>, "ok": <bool: content non-empty>}`. Implementation: `get_backend` passes `role` and `trace=getattr(config, "llm_trace", False)` into the constructor; `complete()` writes after stats update, wrapped in try/except (a trace-write failure must never break a call — swallow and continue). Timestamps: use `datetime.now(timezone.utc).isoformat()`. Gate-failure detail rows are NOT in scope here (the eval runner records those); this is the always-available call ledger.

- [ ] 3.1 Failing test: construct backend with `trace=True` + fake client, `complete()` twice in a tmp cwd (monkeypatch cwd or pass explicit trace path — prefer an explicit `trace_path` constructor arg defaulting to `Path("llm-trace.jsonl")`, tests pass a tmp path), assert two parseable lines with the required keys; a third test asserts trace=False writes nothing.
- [ ] 3.2 Implement; keep `Backend` protocol unchanged. `config.py`: add `llm_trace` to the config dataclass + parser default False; do NOT add it to the init template.
- [ ] 3.3 Full `tests/unit` green. Commit: `"Add opt-in per-call JSONL tracing to the local backend"`.

---

### Task 4: Prompt exemplars + operational env guidance

**Files:** Modify `llm/wiki.py`, `llm/synthesize.py` (prompts only), `README.md`; Tests: existing suites must stay green (prompt text changes may not break stub tests — verify).

- [ ] 4.1 Append one worked exemplar line to each `_SYSTEM` OUTPUT CONTRACT block: `Example of a correctly cited sentence: "The Head settles instantly among participants [claim:b00_c_0002]."` (wiki) / same shape for synthesize. Keep dash conventions.
- [ ] 4.2 README "Running local models": add a short "Serving knobs that matter" paragraph block covering: `keep_alive -1` for batch runs (prefix-cache retention); `OLLAMA_MAX_LOADED_MODELS` sizing vs 32 GB when role-routing across e4b/12b/31b; `OLLAMA_FLASH_ATTENTION=0` as the 31B long-prompt hang workaround (ollama#15368) at a ~15 tok/s cost; `OLLAMA_NUM_PARALLEL=2-4` only with VRAM headroom (each slot multiplies KV cache); measured context note — Ollama 0.31.1 processed 20.7k-token gemma4 prompts at defaults but truncated ~41k prompts, so keep single calls under ~16k tokens (cite the canary test as the guard). Match ` -- ` dash style and wrap width.
- [ ] 4.3 `pytest tests/unit -q` green. Commit: `"Add marker exemplars and serving-knob guidance"`.

---

### Task 5: Live tier scaffolding — marker, skip logic, canaries, flake helper

**Files:** Modify `pyproject.toml`; Create `tests/live/__init__.py`, `tests/live/conftest.py`, `tests/live/test_canaries.py`, `src/deep_research_toolkit/evalkit/__init__.py`, `src/deep_research_toolkit/evalkit/flake.py`; Test `tests/unit/test_evalkit_flake.py`.

- [ ] 5.1 `pyproject.toml`: register marker `live_model` under `[tool.pytest.ini_options]` `markers`, and add `addopts = "-m \"not live_model\""`. VERIFY the fast suite still collects/passes and that `pytest -m live_model` selects only live tests.
- [ ] 5.2 `tests/live/conftest.py`: session fixture `live_backend_config` that loads config via `load_config()`; `pytest.skip("provider is not 'local' -- live tests need a serving endpoint", allow_module_level=...)`-style guard (per-fixture skip is fine) when provider != local OR a 2s TCP/HTTP probe of `llm.local.base_url` fails. Also `pytestmark = pytest.mark.live_model` in every `tests/live/*.py`.
- [ ] 5.3 `evalkit/flake.py` (pure, unit-testable):

```python
"""N-run pass-rate helpers for irreducibly stochastic live-model tests."""
from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def run_flaky(fn, runs: int = 5) -> dict:
    """Run fn() `runs` times; fn returns truthy on pass. Never raises --
    exceptions count as failures and are recorded."""
    passes, errors = 0, []
    for _ in range(runs):
        try:
            if fn():
                passes += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")
    lo, hi = wilson_interval(passes, runs)
    return {"runs": runs, "passes": passes, "rate": passes / runs,
            "ci95": (round(lo, 3), round(hi, 3)), "errors": errors[:3]}
```

Unit tests: wilson bounds for (5,5), (0,5), (3,5) sanity (monotonic, within [0,1], contain p); run_flaky counts exceptions as failures.
- [ ] 5.4 `tests/live/test_canaries.py` — six canaries per design §3.2. Each uses `get_backend(config, role=...)` where sensible or raw `urllib` against `llm.local.base_url` for endpoint-shape probes. Concrete assertions:
  1. `test_reasoning_suppression`: extract-role backend `complete("Reply with the word OK only.", "Say OK.")` → non-empty; then a raw /v1 call with `reasoning_effort:"none"` asserting the response `message` has no non-empty `reasoning` field.
  2. `test_context_ceiling`: sentinel probe (reuse the design's codeword pattern — filler sentence repeated, codeword first, question last) at ~6k tokens must pass; then probe 12k/20k/40k and RECORD the first failing size via the report fixture (assert only the 6k case).
  3. `test_structured_output_with_think_off`: extract-role call demanding a tiny JSON object → `json.loads` succeeds.
  4. `test_marker_fidelity_smoke`: wiki-role style call with 2 fake claims; compute bare-marker rate = bare_markers/(bare+prefixed); record it; assert the reply cites at least one known id in either form.
  5. `test_long_prompt_liveness_31b`: skip unless a role maps to a `:31b` model; ~8k-token prompt, `max_tokens=64`, wall-clock timeout 180s enforced via a worker thread (`concurrent.futures` + `future.result(timeout=180)`); timeout → fail with the FA-hang reference.
  6. `test_determinism_smoke`: two temp-0 seeded calls (seed via `extra_body`/options passthrough — if the /v1 route ignores seed, that is itself the recorded result); RECORD identical/not, assert both non-empty.
  Recording: a session-scoped `canary_report` fixture collecting dicts, dumped by conftest at teardown to `eval-results/canaries-<UTCstamp>.json` (create dir; gitignore `eval-results/` except baseline — Task 7 handles gitignore).
- [ ] 5.5 Fast suite still green (live excluded); `pytest -m live_model --collect-only` lists the canaries. Commit: `"Add the live-model test tier: skip logic, canaries, flake helpers"`.

---

### Task 6: Eval corpus — checker + authored corpus

**Files:** Create `scripts/check-eval-corpus.py`, `tests/fixtures/eval-corpus/corpus-index.json`, `tests/fixtures/eval-corpus/<doc-id>/chunks.jsonl` + `manifest.json` + `reference-claims.jsonl` for ~10 docs; Test `tests/unit/test_eval_corpus.py` (new — runs the checker over the committed corpus in fast CI, no model).

Corpus contract:
- Each doc dir: `manifest.json` `{"document_id": "<dir name>", "producer": "web", "title": ..., "slices": [...]}`; `chunks.jsonl` rows `{"locator": "<doc>#c<NNN>", "text": "..."}` (web-producer shape — verify against `tests/fixtures/reference-run-web-ows` and match exactly, including any extra required fields found there); `reference-claims.jsonl` rows matching the reference-run claim shape (verify against the web fixture: claim_id, claim, supporting_evidence[{locator, quote}], entities, relations optional).
- ~10 docs, **180–220 chunks total**, each chunk 80–450 words, tagged in `corpus-index.json`: `{"chunks": {"<locator>": {"slices": ["prose"|"dense-facts"|"table"|"list"|"unicode"|"math"|"long"|"bait"], "doc": ...}}, "contradiction_pairs": [[locatorA, locatorB], ...], "corpus_version": "<sha256 of sorted chunk texts, computed by the checker --stamp mode>"}`.
- Slice quotas (min): prose 60, dense-facts 30, table 15, list 15, unicode 12, math 12, long 15, bait 15.
- **Bait chunks**: each contains ≥1 sentence that is a near-copy (one word/punctuation changed) of a sentence in a DIFFERENT chunk of the same doc; `corpus-index.json` lists each bait chunk under `"bait_sources"`: `{"<bait locator>": "<source locator>"}`. Purpose: extraction from the bait chunk citing the *source* chunk's exact sentence (or vice versa) must fail the chunk-scoped gate.
- **Contradiction pairs**: ≥8 pairs of chunks across different docs asserting incompatible values for the same (subject, predicate) — e.g. one doc says a protocol launched 2019, another says 2021 — recorded in `contradiction_pairs` so adjudication accuracy is measurable.
- Reference claims: 2–6 per chunk, atomic, every quote an exact substring of its chunk (the checker enforces via `verbatim_ok`), authored to be extraction gold (what a careful reader would extract), NOT exhaustive trivia.
- Topics: 10 distinct realistic technical subjects (distributed consensus, zk proofs, mempool design, LSM storage engines, BFT networking, formal verification, GPU scheduling, embedded RTOS, compiler IR design, oceanographic sensor networks) — synthetic, no real-world-source copying, so licensing is clean by construction.

- [ ] 6.1 Write `scripts/check-eval-corpus.py` FIRST (argparse; default corpus path; `--stamp` recomputes and writes corpus_version): validates every contract bullet above (quotas, shapes, verbatim quotes via `common.verbatim`, bait near-copy actually differs from source sentence but shares ≥80% of words, contradiction pairs exist and reference real locators); exits 1 with a failure list. Unit test drives it against a tmp mini-corpus (one good doc, one violation per check).
- [ ] 6.2 Author the corpus: generate the 10 docs (write coherent multi-section technical prose, then chunk it yourself at paragraph boundaries into the quota'd slices), reference claims, index. Run the checker until clean. This is deliberate frontier-agent authorship — vary sentence structure and vocabulary across docs; tables as markdown tables inside chunk text; unicode slices should mix scripts (CJK, Cyrillic, Greek letters in prose) naturally; math slices use plain-text/LaTeX-ish notation.
- [ ] 6.3 `tests/unit/test_eval_corpus.py`: one test invoking the checker's validate function over the committed corpus (fast, no model) → clean; plus the checker unit tests from 6.1.
- [ ] 6.4 Full fast suite green. Commit (may be two commits: checker, then corpus): `"Add the stratified eval corpus and its mechanical checker"`.

---

### Task 7: Eval runner — `scripts/eval-pipeline.py`

**Files:** Create `scripts/eval-pipeline.py`, `src/deep_research_toolkit/evalkit/metrics.py`, `src/deep_research_toolkit/evalkit/bootstrap.py`; Modify `.gitignore` (`eval-results/` ignored, `!eval-results/baseline.json` kept); Test `tests/unit/test_evalkit_metrics.py`, `tests/unit/test_evalkit_bootstrap.py`.

- [ ] 7.1 `evalkit/metrics.py` (pure functions, unit-tested with synthetic claim lists):
  - `quote_overlap_match(produced, reference) -> {"recalled": [...], "missed": [...], "matched_produced": set}` — a reference claim is recalled when any produced evidence quote overlaps any of its quotes (substring either direction; same rule as validate-local-llm `_recovered`).
  - `extract_metrics(produced, reference, dropped, parse_failures) -> dict` with keys `gate_pass_rate` (written/(written+dropped), 1.0 when nothing produced-and-dropped... define as `written/(written+len(dropped))` guarding zero-division to None), `recall`, `precision_proxy` (share of produced claims in matched_produced), `atomicity` (len(produced)/len(reference), None when reference empty), `parse_failures`.
  - `bait_rejection(produced_bait_chunk_claims, bait_source_text) -> rate` — share of bait-chunk claims whose quotes do NOT appear verbatim in the bait *source* chunk text while passing their own chunk (i.e., the gate held chunk-scope; compute from claims + both texts).
- [ ] 7.2 `evalkit/bootstrap.py`: `paired_bootstrap(deltas: list[float], b: int = 2000, seed: int = 7) -> {"mean": ..., "ci95": (lo, hi), "significant": bool}` using `random.Random(seed)` resampling; unit tests: all-positive deltas → significant, mixed-zero-mean → not.
- [ ] 7.3 `scripts/eval-pipeline.py` — argparse: `--corpus tests/fixtures/eval-corpus`, `--roles extract[,wiki_write,synthesize,conflict_adjudicate]` (default all), `--models` (default: whatever `.deepresearch.yml` roles map to; `--models` overrides the extract-role model list for A/B, comma-separated), `--runs N` (prose-role flake runs, default 3), `--limit N` (chunk cap for smoke runs), `--compare <baseline.json>` (exit 1 on gate_pass_rate or recall dropping > `--tolerance` [default 0.03]), `--write-baseline`. Flow per extract model: for each corpus doc, copy to a temp run dir (claims removed), call `extract_claims_to_run` (role config honored via a config constructed with `dataclasses.replace` pointing research_runs_path at the temp dir — mirror `scripts/validate-local-llm.py`'s copy/repoint pattern exactly), gate + metric per chunk-slice; wiki/synthesize: sample K=10 stratified chunks' reference claims → `write_wiki_body`/`synthesize_thesis` with `run_flaky(runs=N)`, record coverage/bare-marker/retry stats (bare-marker rate needs the raw reply — capture via a wrapping backend that records the last raw completion before normalization: implement a thin `RecordingBackend` wrapper in evalkit); adjudicate: build candidates from `corpus-index.json` contradiction_pairs (+ equal count of non-contradictory distractor pairs), call `adjudicate_candidates`, score verdict accuracy (contradiction pairs should get "contradiction", distractors "not_contradiction"; "insufficient_evidence" counts half). Report: one JSON to `eval-results/run-<UTCstamp>.json` with join keys (model, ollama version via GET /api/version, prompt_version = sha256 of the role `_SYSTEM` strings, corpus_version from index) + append one summary line per (model, role) to `eval-results/history.jsonl`; `--compare` implements the tolerance diff.
- [ ] 7.4 Unit tests for metrics/bootstrap only (runner itself is exercised live in Task 8); `python scripts/eval-pipeline.py --help` works without a live endpoint. Fast suite green. Commit: `"Add the eval runner: metrics, bootstrap comparison, baseline diffing"`.

---

### Task 8: Live run + baseline + wrap-up

**Runbook (coordinator or agent with live endpoint):**
- [ ] 8.1 Configure `.deepresearch.yml` (untracked) with the validated role map + `llm.trace: true`. Run canaries: `pytest -m live_model tests/live/ -q` → all pass/skip-with-reason; inspect `eval-results/canaries-*.json`.
- [ ] 8.2 Smoke: `python scripts/eval-pipeline.py --limit 20 --runs 2` → completes, report sane.
- [ ] 8.3 Full: `python scripts/eval-pipeline.py --runs 5` for the default role map; then extract A/B: `--roles extract --models gemma4:e4b,gemma4:12b,gemma4:31b,qwen3:30b-a3b-instruct-2507-q4_K_M`. Record wall time.
- [ ] 8.4 `--write-baseline` on the accepted run; commit `eval-results/baseline.json`.
- [ ] 8.5 Sync templates if any `skills/` files changed (none expected); run `scripts/check-skill-templates-in-sync.py`; full fast suite; ruff on changed files. Update design doc §6 success criteria with actuals + a results table. Final commit: `"Record the first full eval baseline"`.

**Deferred (explicit):** Tier-2 response cache + threaded concurrency — implement only if Tasks 1–8 land cleanly with session budget left; otherwise note as deferred in the design doc.
