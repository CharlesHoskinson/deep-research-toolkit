# Comprehensive Eval Suite + Local-Pipeline Flow Hardening

**Date:** 2026-07-06
**Status:** Approved design — implementation plan to follow
**Depends on:** merged two-track implementation (master @ 28a996d)

## 1. Goal

Two deliverables, one design:

1. A **comprehensive test suite** for the local-LLM pipeline: a stratified
   eval corpus (~200 chunks), per-role live-model tests, serving-stack
   canaries, flake-aware metrics, and a baseline-diff protocol — so model,
   prompt, and Ollama-version changes are measured, not vibed.
2. **Flow improvements** to the pipeline, drawn from a three-track research
   sweep (Gemma 4 operational practices; local-pipeline design patterns;
   LLM-pipeline test methodology), filtered for a single-GPU CLI toolkit.

## 2. Research conclusions adopted (with rejections)

### Test methodology
- **Plain pytest, extended — no eval framework.** promptfoo/DeepEval/
  inspect-ai/ragas each optimize for shapes we don't have (LLM-judge metric
  libraries, YAML red-teaming, second test runner). Our gates ARE the
  metrics: deterministic Python. inspect-ai's Dataset/Sample/Scorer split is
  borrowed as internal structure only.
- **Live tests are irreducibly stochastic** even at temp 0 + seed on Ollama
  (documented upstream). Therefore: N-run pass-rates with Wilson CIs, not
  binary pass/fail; paired bootstrap over per-chunk deltas for any
  model-vs-model claim; gate on rate-drop vs a stored baseline.
- **Canary tests for serving-stack drift** are mandatory: the `think:false`
  incident (ollama#15288) and the shifting `num_ctx` default are both
  classes of silent behavior change an upgrade can introduce. Pin and pass
  decoding params explicitly; never rely on server defaults.

### Gemma 4 operational findings
- **System role is real in Gemma 4** (unlike Gemma 2/3) — our system+user
  contract placement is architecturally correct; keep it.
- **Effective context measured on Ollama 0.31.1 + gemma4:e4b** (sentinel
  probes, 2026-07-06): 5.8k and 20.7k-token prompts process fully at
  default settings; a ~41k prompt silently truncates (usage reported 16,387
  processed tokens, start-of-prompt sentinel lost). Rules adopted: keep any
  single call's prompt under ~16k tokens; canary the ceiling per upgrade.
  The researched "4096 default" clamp does NOT reproduce on 0.31.1.
- **`reasoning_effort: "none"` works on our build** (verified 2026-07-05:
  0 reasoning tokens, 4–5× faster) but adjacent upstream issues (#15293,
  #15635) show the /v1 fix is partial across endpoints/model sizes —
  exactly what the reasoning-suppression canary guards.
- **31B risks:** repetition loops under schema-constrained JSON
  (ollama#15502, still open; repeat_penalty documented ineffective) → we
  add a client-side repeated-n-gram detector as a mechanical gate.
  Flash-attention hangs on long 31B prompts (#15368/#15350) → long-prompt
  timeout canary; document `OLLAMA_FLASH_ATTENTION=0` fallback.
- **Format adherence:** few-shot exemplars are the strongest community
  lever for literal-marker fidelity → add one worked `[claim:<id>]`
  exemplar to the wiki/synthesize prompts (normalization stays as the
  safety net). JSON-Schema `format` upgrade and temp 0 vs 0.1–0.3 are
  filed as A/B experiments for the suite, not adopted blind.

### Pipeline flow patterns
- **Adopted (tier 1, this plan):** failure-reason retry mutation with a
  temperature bump (same-prompt resampling is the documented anti-pattern);
  per-call JSONL trace logging with gate-failure samples (fixes
  "failures surface as counts"); repetition-loop gate; few-shot exemplar;
  explicit `keep_alive` guidance.
- **Adopted (tier 2, this plan if time allows):** content-hash response
  cache for unchanged-chunk reruns; threaded concurrency for extraction
  batches (bounded by OLLAMA_NUM_PARALLEL).
- **Filed as suite experiments:** e4b→12b escalation cascade on gate
  failure; JSON-Schema format; sampling A/B.
- **Rejected:** vLLM/llama.cpp migration (loses Ollama's multi-model
  ergonomics that per-role routing depends on); Langfuse/OTel stack
  (operational surface disproportionate to a CLI tool — we adopt the OTel
  GenAI *field names* in our JSONL only); agent frameworks (LangGraph/
  smolagents — our linear stages + mechanical gates are already the right
  shape); span-offset extraction (structurally elegant but offset-counting
  is likely weaker than verbatim-copy for sub-31B models — pilot someday,
  don't bet the gate on it).
- **Validation from literature:** a published production pipeline
  (arxiv 2606.00994) uses our exact extract-then-verbatim-verify design;
  the attribution survey rates quote-then-verify above write-then-cite.

## 3. Eval suite architecture

### 3.1 Tiering
- New pytest marker `live_model`, registered in pyproject, excluded by
  default via `addopts = -m "not live_model"`. Fast suite stays <5s.
- Live tests live in `tests/live/`; run via `pytest -m live_model` or
  `drt`-independent `scripts/eval-pipeline.py` for the full corpus run.
- Live tests require `.deepresearch.yml` with `provider: local` + a
  serving endpoint; they skip (not fail) with a clear reason when absent.

### 3.2 Canaries (`tests/live/test_canaries.py`) — run first, fail loud
1. Reasoning suppression: extract-role call must return 0 reasoning tokens
   and non-empty content (guards #15288-class regressions).
2. Context ceiling: sentinel probe at ~6k tokens must pass; measured
   ceiling (first failure among ~6k/12k/20k/40k probes) is recorded in the
   run report so upgrades that move it are visible.
3. Structured output: `format=json_object` + thinking off yields valid
   JSON (guards #15260-class regressions).
4. Marker fidelity smoke: one wiki-style call; report bare-marker rate
   (pre-normalization) as a tracked metric.
5. Long-prompt liveness: 31B call with an ~8k prompt under a hard timeout
   (guards the FA-hang class; skip-with-reason if 31B not pulled).
6. Determinism smoke: two seeded temp-0 calls; report (not assert)
   whether outputs are identical — drift is data, not failure.

### 3.3 Corpus (`tests/fixtures/eval-corpus/`)
- ~10 synthetic documents authored by the frontier agent (license-clean by
  construction), each a realistic technical/research text, chunked to
  **~200 total chunks** in web-producer shape (`chunks.jsonl`, locator+text)
  with a `manifest.json` per doc.
- Stratified slices, tagged per chunk in a `corpus-index.json`:
  plain prose · dense facts (numbers/dates) · markdown tables · lists ·
  unicode/multilingual · math notation · long paragraphs (>400 words) ·
  **adversarial near-quote bait** (passages containing text that is
  *almost* a quote from a neighboring chunk — single-word/punctuation
  deltas — to prove the gate rejects false positives).
- **Reference claims** per chunk, frontier-authored, mechanically
  gate-verified at build time (quotes must pass `verbatim_ok`); a
  build-time checker (`scripts/check-eval-corpus.py`) re-validates the
  whole corpus and its slice tags in fast CI (no model needed).

### 3.4 Metrics (deterministic; no LLM judge)
Per model × role, written as one JSON per run + appended JSONL time series
under `eval-results/` (gitignored except a committed `baseline.json`):
- **extract:** gate-pass rate; recall vs reference (quote-overlap matching,
  both-directions substring — same rule as validate-local-llm); precision
  proxy (share of produced claims whose quote overlaps any reference
  quote); atomicity ratio (claims per reference claim, flags over/under
  splitting); parse failures; bait-rejection rate on adversarial slices;
  latency + tokens per call (backend stats).
- **wiki_write / synthesize:** citation coverage; bare-marker rate before
  normalization; unknown-marker rate; retry rate; gate outcomes.
- **conflict_adjudicate:** schema-valid rate; verdict accuracy vs seeded
  gold contradictions (the corpus builder plants known contradiction pairs
  across docs); invalid/parse counts.
- **Flake protocol:** prose-role tests run N=5, reported as pass-rate with
  Wilson 95% CI; comparisons across models/versions use paired bootstrap
  (B=2000) over per-chunk deltas; "improved" requires CI excluding zero.
- Every row carries `model`, `ollama_version`, `prompt_version` (hash of
  the role prompt), `corpus_version` (hash of corpus files) as join keys.

### 3.5 Baseline protocol
`scripts/eval-pipeline.py --compare eval-results/baseline.json` exits
nonzero when a gate-pass/recall metric drops beyond tolerance (default 3
points) vs baseline; `--write-baseline` promotes a run. Ollama upgrades and
prompt edits require a full run + explicit baseline promotion.

## 4. Flow hardening (code changes)

1. **Retry mutation** (`llm/extract.py`, `llm/wiki.py`, `llm/synthesize.py`):
   retries append the concrete failure reason (dropped quotes / unknown or
   missing markers / parse failure) to the prompt and raise temperature to
   0.25 for the retry only. First attempts stay deterministic.
2. **Per-call trace logging** (`llm/local.py` + callers): optional JSONL
   trace (`llm-trace.jsonl` in the run dir or cwd) with OTel GenAI field
   names — model, tokens, latency, role, gate outcome, truncated sample of
   gate-failing output. Enabled via `llm.trace: true` in config.
3. **Repetition-loop gate** (`llm/response.py`): `has_repetition_loop(text,
   max_pattern=20, min_repeats=4)` — mirrors vLLM's shipped detector;
   applied to 31B-role outputs (adjudicate/synthesize) before parsing, so
   a looped reply counts as a parse failure (and now retries with
   mutation) instead of garbage-in.
4. **Few-shot marker exemplar** in wiki/synthesize `_SYSTEM` prompts (one
   worked example line with the literal `[claim:<id>]` form).
5. **`keep_alive`/env guidance** in README Running-local-models: keep_alive
   -1 for batch runs, OLLAMA_MAX_LOADED_MODELS sizing vs 32 GB, explicit
   OLLAMA_FLASH_ATTENTION note for 31B, OLLAMA_NUM_PARALLEL guidance.
6. *(Tier 2)* **Response cache**: sha256(model+system+user+params)-keyed
   JSONL cache under `.deepresearch/llm-cache/`, `llm.cache: true` opt-in,
   extraction-role only at first.
7. *(Tier 2)* **Concurrent extraction batches**: `ThreadPoolExecutor`
   fan-out of chunk-batches (the openai client is thread-safe), worker
   count from `llm.parallel` (default 1 = current behavior), results
   merged deterministically by batch index.

## 5. Out of scope
Fine-tuning (still gated on the 200-chunk corpus results — this suite IS
the gate), vLLM serving configs, span-offset extraction pilot, escalation
cascade implementation (measured first as a suite experiment), dashboards.

## 6. Success criteria — actuals (2026-07-07)
- Fast suite unchanged and green — **met**: 329 unit tests pass in ~3.8s
  with the live tier excluded by default; live tier is one-command runnable
  (`pytest -m live_model tests/live/` and `scripts/eval-pipeline.py`).
- Corpus: ~200 gate-verified chunks across all slices, committed — **met**:
  186 chunks / 10 docs / 8 slices (prose 70, dense-facts 32, table 16,
  list 16, long 16, bait 16, unicode 13, math 13), 12 contradiction pairs,
  16 bait chunks; checker-clean (`scripts/check-eval-corpus.py`).
- One full eval run across the validated role map completes and produces a
  comparable JSON report + committed baseline — **met**: full `--runs 5`
  run 2026-07-07 (58 min wall), report `run-20260707T063207Z.json` promoted
  to `eval-results/baseline.json`; self-compare clean, doctored-drop
  detection verified. Results in §7.
- Flow items 1–5 landed with unit tests — **met** (repetition gate, retry
  mutation, JSONL tracing, marker exemplars, README serving knobs). Items
  6–7 (response cache, threaded extraction concurrency) — **deferred**: the
  live baseline consumed the session budget (two multi-hour runs plus a
  crash-fix rerun, §7.5); both remain tier-2 candidates unblocked by
  anything in this design.

## 7. Baseline results (2026-07-07)

Stack: Ollama 0.31.1, single RTX 5090, role map extract=gemma4:e4b,
wiki_write=gemma4:12b, synthesize=gemma4:31b, conflict_adjudicate=gemma4:31b.
Join keys: prompt_version `9c7eb327…`, corpus_version `d42ff714…`. Session
trace ledger: 1,367 calls, 1.40M input + 2.71M output tokens, 294.6 min
GPU-busy, 0 empty replies.

### 7.1 Canaries — 6/6 pass
- Reasoning suppression: `reasoning_effort:"none"` leaks through none of the
  three channels; positive control confirms the `reasoning` field is the
  channel this stack populates when thinking is on.
- Context ceiling: sentinel recovered at 5,503 / 10,926 / 18,176 measured
  prompt tokens; the ~40k nominal probe silently truncated (usage reported
  16,387 processed tokens, sentinel lost) — same ceiling as the 2026-07-06
  probes, no drift.
- Structured output (JSON mode + thinking off): valid JSON.
- Marker fidelity smoke: 0 bare / 2 prefixed markers (bare rate 0.0).
- 31B long-prompt liveness: ~8k-token prompt returned in ~14s, no FA hang.
  (First engagement of this canary found a canary bug, not a serving bug:
  with reasoning unsuppressed, the default thinking pass exhausted the
  probe's max_tokens=64 entirely on `reasoning`, leaving empty content —
  fixed by suppressing reasoning in the probe payload.)
- Determinism smoke: two temp-0 calls byte-identical on this stack. The
  aggregate ledger agrees: the gemma4:12b extract leg emitted an identical
  call count and total output tokens (36 / 61,151) in two separate runs.

### 7.2 Full role-map run (`--runs 5`, 58 min wall)

| Role (model) | Metric | Value |
| --- | --- | --- |
| extract (gemma4:e4b) | gate_pass_rate | 0.997 |
| | recall | 0.685 |
| | precision_proxy | 0.180 |
| | atomicity | 1.326 |
| | bait_rejection | 0.861 |
| | parse_failures | 12 |
| | latency | 166 calls, 35.8 min (med ~14.6s/call) |
| wiki_write (gemma4:12b) | pass rate (Wilson N=5) | 0.90 (9/10 chunks at 1.0) |
| | mean coverage | 0.963 |
| | bare-marker rate | 0.0 (0 bare / 167 prefixed) |
| | latency | med ~1.6s/call |
| synthesize (gemma4:31b) | pass rate (Wilson N=5) | 0.96 |
| | mean coverage | 0.993 |
| | bare-marker rate | 0.0 (0 bare / 153 prefixed) |
| | latency | med ~12.2s, p95 ~51.3s/call |
| conflict_adjudicate (gemma4:31b) | verdict accuracy | 1.00 (12/12 pairs) |
| | schema valid / invalid / parse fails | 12 / 0 / 0 |
| | latency | 1 batch call, ~59s |

The two prose-role failures are informative, not noise: wiki_write fails
`mempool-design#c005` on all 5 runs by citing the same unknown id
(`mempool_design_c010`) — a deterministic, chunk-specific temptation — and
synthesize drops to 0.6 on `compiler-ir-design#c015`, a 2-claim input where
one uncited claim already breaches the 0.3 coverage floor.

### 7.3 Extract A/B (5 models, full corpus, 112 min wall)

| Model | gate_pass | recall | prec_proxy | atomicity | bait_rej | parse_fails | Δrecall vs e4b (95% CI) | sig |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma4:e4b | 0.999 | 0.736 | 0.177 | 1.47 | 0.865 | 6 | — (baseline) | — |
| gemma4:12b | 0.984 | 0.392 | 0.160 | 0.57 | 0.718 | 0 | −0.346 (−0.380, −0.309) | yes |
| gemma4:26b | 0.954 | 0.401 | 0.146 | 0.58 | 0.694 | 0 | −0.335 (−0.369, −0.295) | yes |
| gemma4:31b | 0.998 | 0.531 | 0.149 | 0.92 | 0.750 | 0 | −0.200 (−0.252, −0.143) | yes |
| qwen3:30b-a3b-instr-q4 | 0.995 | 0.746 | 0.155 | 1.43 | 0.959 | 14 | +0.000 (−0.052, +0.055) | no |

Leg wall times: e4b 35.8 min (166 calls), 12b 8.4 min (36), 26b 6.0 min
(36), 31b 30.6 min (48), qwen3-30b 28.5 min (188). Paired bootstrap is over
per-doc recall deltas (n=10 docs, B=2000).

### 7.4 Interpretation

The suite discriminates where it was designed to: recall separates the
models into an eager tier (e4b, qwen3-30b — atomicity ~1.4, recall ~0.74)
and a terse tier (12b/26b — atomicity ~0.57, recall ~0.40, both
significantly below e4b), so mid-size Gemmas lose recall by under-producing
claims, not by failing the gate. Every model clears the 95% gate-pass
fine-tune bar (min: 26b at 0.954); gate_pass_rate is near-saturated and is
the wrong axis for model choice — recall, atomicity, and bait rejection are
where models actually differ. qwen3-30b is a statistical tie with e4b on
recall while posting the best bait rejection (0.959 vs 0.865) at ~4× the
weights; e4b remains the right extract default on value-per-GB, with
qwen3-30b the strongest candidate if bait discipline becomes the binding
constraint. precision_proxy sits at 0.15–0.18 for every model and mostly
measures gold-set non-exhaustiveness rather than model quality — expected
for authored-gold recall corpora, but worth renaming or reweighting before
anyone reads it as precision. Run-to-run recall variance on the same model
and corpus was ~0.05 (e4b 0.685 full run vs 0.736 A/B), so single-run
deltas under ~5 points should not be over-read — the paired-bootstrap CI is
the decision rule. Extract's max_tokens=3000 is the main throughput lever:
~44% of e4b extract calls truncated at the cap and re-ran as halved batches
(166 calls for 31 base batches), a silent 3–5× call amplification that a
larger cap or smaller default batch would largely remove.

### 7.5 Bug found live (fixed during the run)

The first A/B attempt crashed mid-run: gemma4:26b emitted a claim whose
`supporting_evidence` array held a bare string instead of an evidence
object, and the extraction gate's evidence loop called `.get()` on it
(AttributeError). Malformed evidence rows now fail the gate — claim
dropped and counted — instead of crashing the pipeline
(`test_extract_tolerates_bare_string_evidence_rows`). This is precisely
the class of live-model shape drift the eval tier exists to surface.
