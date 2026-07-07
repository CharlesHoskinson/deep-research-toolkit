# Pipeline Hardening + Fine-Tune Meta-Infrastructure

**Date:** 2026-07-07
**Status:** Approved design — implementation plan to follow
**Depends on:** merged eval suite (master @ 83d7061): `scripts/eval-pipeline.py`,
`tests/live/`, `tests/fixtures/eval-corpus/`, `eval-results/baseline.json`,
`evalkit/` (bootstrap, metrics), per-call JSONL trace, repetition gate.
**Provenance:** derived from a six-stream web-research sweep and three adversarial
design debates (2026-07-07, workflow `wf_c94290bb-36c`), grounded in the
2026-07-07 baseline in `docs/superpowers/specs/2026-07-06-comprehensive-eval-and-flow-hardening-design.md` §7.

## 1. Goal

Two deliverables under one phased roadmap, Gemma-4-first, single-GPU
(RTX 5090 / Blackwell sm_120 / 32 GB), CLI-first, no heavy MLOps platform:

1. **Phase 1 — pipeline hardening.** Close the measured recall, bait-rejection,
   throughput, and prose-failure gaps with prompt, output-contract, flow, and
   serving changes only. No weight updates. Re-establish a trustworthy baseline.
2. **Phase 2 — fine-tune meta-infrastructure.** Stand up the four capabilities
   that let us fine-tune the Gemma-4 stack repeatably and safely — dataset
   generation, training orchestration, eval-gated promotion + registry, and
   experiment tracking — as a minimal file-based loop that reuses the existing
   eval / trace / DuckDB machinery.

The organizing insight from the debates: **"only fine-tune a role if it misses
its eval bar" is vacuous unless "misses" means "misses after cheap levers AND
correct measurement."** Phase 1 is therefore not just hardening; it is the
precondition that makes the Phase-2 tuning *decision* legitimate.

## 2. Framing — the two gaps that actually matter

Only two eval bars are genuinely missed, and one is largely a *schema* problem,
not a *model* problem:

| Metric | Baseline | Bar | Real nature of the gap |
| --- | --- | --- | --- |
| extract gate_pass | 0.997 | 0.95 | **saturated** — do not tune for this |
| extract recall | 0.685 full / 0.736 A/B | 0.90 | real, but measured against a **non-exhaustive gold denominator** |
| extract bait_rejection | 0.861 | 0.95 | real, but **closable by a span-offset contract, not training** |
| extract precision_proxy | 0.18 | — | measures gold non-exhaustiveness — **retire it** |
| wiki_write pass | 0.90 | — | one deterministic failure: hallucinated claim id |
| synthesize pass | 0.96 | — | one deterministic failure: coverage floor on a 2-claim input |

Both prose failures reproduced exactly on an independent capped run
(`run-20260707T154851Z`): wiki_write invents `mempool_design_c010` on
`mempool-design#c005`; synthesize breaches the 0.3 floor on the 2-claim
`compiler-ir-design#c015`. They are deterministic and chunk-specific, not noise.

## 3. Research conclusions adopted (with rejections)

### Gemma 4 fine-tuning on Blackwell (sourced)
- **Unsloth "Studio" is required** for sm_120 (the PyPI wheel lacks Gemma-4
  hybrid-attention kernels). Working stack: Studio py3.13 venv + torch
  **2.11.0+cu129** + transformers **≥5.5.3** + trl **≥0.23.1** + bitsandbytes
  **0.49.2**. `torch.compile` off; clear stale Triton cache. Do **not** chase
  cu130 (breaks the bnb ABI, `libnvJitLink.so.13`). (unsloth#5154; Unsloth +
  NVIDIA Blackwell docs.)
- **`attn_implementation="sdpa"` is mandatory** — FA2/FA4 reject Gemma 4's
  `head_dim=512` global-attention layers. Add a preflight assert that FA is not
  silently selected, and a cache-enabled post-train smoke test (KV-shared layers
  emit garbage under `use_cache=False`).
- **SFT-then-GRPO with a verifiable reward beats SFT and DPO** for structured
  output (RL-Struct, arXiv 2512.00319: GRPO 89.7% vs DPO 82.5% structural
  accuracy; zero inference cost). Our mechanical verbatim/span gate is a
  near-perfect shaped reward. Adopt SFT-first, then optional GRPO.
- **31B GRPO is a hard wall on 32 GB** — the 262k-vocab chunked log-softmax
  workspace requests >44 GB; every documented workaround fails (unsloth#4985).
  Recipe A stays SFT-only; route **all** RL to e4b (also the top-recall
  extractor, so RL targets the right model).
- **26B-A4B MoE is untrainable locally** — fused 3D experts load ~43.7 GB
  because bitsandbytes only 4-bit-quantizes 2D `nn.Linear`. Recipe C is
  reclassified **BLOCKED** for local training (serve-only via GGUF; cloud
  bf16/B200 if ever needed). We do **not** substitute Qwen3.5-35B-A3B — it
  violates Gemma-4-first.
- **EOS/BOS token corruption on GGUF export** (unsloth#5386/#5070) silently
  makes serving never stop. A control-token-table validator is a hard gate.

### Extraction recall + bait (sourced)
- Union-over-small-batches plus multi-sample self-consistency raise recall
  materially (span/passage decomposition literature; ~50% → ~78–80% reported);
  iterative re-prompting does **not** raise hallucination when capped.
- **Near-quote bait is best defeated structurally**, by a span-offset contract,
  not by training a model to be more discriminating.
- Recall on authored-gold corpora must be measured with an entailment/embedding
  matcher and a pooled gold denominator; exact-substring vs non-exhaustive gold
  conflates model quality with gold incompleteness.

### Synthetic data, MLOps, throughput (sourced)
- Rejection-sampled SFT (RFT/RSFT) on gate-passers is the canonical recipe;
  sample at T≈1.0 for mode coverage, curate ~1–3k deduped examples (LIMA:
  1k curated > 52k unfiltered), mix 10–20% general data against forgetting,
  never train on the eval corpus.
- A minimal file-based loop (append-only JSONL + DuckDB view + git SHAs)
  is sufficient and correct for a single-dev single-GPU toolkit; a persisted
  promotion state-machine is deferred to the first real training run.
- JSON-Schema grammar (`format`) is faster and less truncation-prone than
  json_object mode once the FSM is cached; right-sizing `num_predict` to the
  batch removes the truncation amplification.

### Rejected
- vLLM/llama.cpp migration (loses Ollama's multi-model per-role ergonomics);
  heavy MLOps platforms (MLflow/W&B servers, Kubeflow); 31B GRPO; local 26B-A4B
  training; Qwen substitution for the tuning target; DPO (GRPO wins on structure);
  parallel weight-update training during Phase-1 eval runs (Debate #1 rejected).

## 4. Debate verdicts (load-bearing decisions)

- **Debate #1 — fix-first vs tune-now (verdict: Position A, high confidence).**
  Do not start the e4b weight-update run in parallel. Adopt one carve-out from
  B: provision and validate the Blackwell/Unsloth-Studio environment now as a
  decoupled, GPU-idle de-risking task. Run the flow/contract stack, fix the
  recall metric, re-measure with a bootstrap CI, and launch Recipe B only if a
  residual recall gap survives. Holds only under the single-GPU constraint.
- **Debate #2 — infra weight (verdict: Position A, minimal file-based).**
  Ship an append-only `registry.jsonl` + DuckDB view + a `promote` verb reusing
  the existing `paired_bootstrap`; fold in two of B's disciplines as
  correctness properties (schema-validated appends, immutable anchors, the CI
  gate); defer the persisted promotion state-machine to the first Recipe B run.
- **Debate #3 — first tune target (judge did not return a valid verdict; the
  synthesis resolves it).** e4b-first for RL: it is the production extract
  default, ties top recall, has the clearest closable gap under a shaped reward
  (bait), is the cheapest to train, and is the only viable RL target (31B OOM,
  26B untrainable). This is recorded as a synthesis decision, not a debate
  verdict, and should be revisited if a second GPU or off-box capacity appears.

## 5. Phase 1 — Near-Term Pipeline Hardening

No weight updates. Ordered by leverage.

### 5.1 Span-offset extract contract (do first)
Change extract's output contract from a verbatim quote string to
`[start_char, end_char]` offsets into the chunk (optionally sentence-id + span).
- The gate becomes an O(1) slice-and-compare; `gate_pass` → ~1.0 by construction.
- Near-quote bait becomes structurally impossible to emit (cannot map to a
  contiguous source span). This is the strongest bait lever
  (0.861 → target 0.95).
- **Build:** new extract schema + gate rewrite. **Reuse:** the verbatim-gate
  code path, repurposed as slice-compare. Producers/consumers that read the
  `quote` field derive it from the span (`chunk[start:end]`) so downstream
  citation display is unchanged.

### 5.2 Formalized union + self-consistency (recall)
Replace today's opportunistic "truncate-then-halve" (undeduplicated, fires on
~44% of calls) with a deliberate strategy:
- Split each chunk into ≤2-passage / ≤~1000-char batches; **union** candidates.
- Run **N=3–5 temperature-varied samples** per chunk (reuse the temperature-bump
  retry machinery). **Union** gate-passing claims for recall; keep claims
  supported by **≥2 of N** for a precision/bait cut — one knob yields both.
- **Bounded coverage loop:** re-prompt "extract only ADDITIONAL atomic claims;
  return empty if none," max 2–3 iterations.
- **Dedup** the union with normalized n-gram / MinHash-Jaccard (not exact).
- **Atomicity is a guardrail, not a target:** keep ~1.33; flag drift >1.8 or
  <1.1. Do not push e4b to split harder (over-splitting shortens quotes and
  hurts bait discrimination).
- **Build:** union/dedup orchestrator, coverage-loop controller. **Reuse:**
  temperature-bump retry, repeated-n-gram gate.

### 5.3 Fix the recall metric (the measurement is a deliverable)
- **Retire `precision_proxy`** as a model-precision signal (rename/reweight or
  drop from the leaderboard).
- **Pooled recall gold:** union the gate-passing claims of **e4b ∪
  qwen3:30b-a3b** (the two recall co-leaders), dedup + adjudicate. Built once,
  used twice — as the eval denominator **and** the Phase-2 SFT target.
- Match recall with an **entailment/embedding matcher** (cosine ≈0.9 via
  qwen3-embedding:4b), not exact-substring vs non-exhaustive gold.
- Add a **gold-free self-faithfulness precision** (is each claim entailed by its
  own source chunk?) and report an **F_fact** harmonic mean on the A/B board.
- Re-measure with a **bootstrap 95% CI** on per-chunk recall.
- **Build:** pooled-gold generator, entailment matcher. **Reuse:**
  `evalkit/bootstrap.py` (`paired_bootstrap`, CI-excludes-zero), the embedding
  role, `scripts/check-eval-corpus.py` contract for the corpus builder.

### 5.4 Throughput / truncation amplification
- **JSON-Schema grammar** (Ollama `format`) instead of json_object mode; keep
  the schema **stable** so the FSM stays cached.
- **Right-size `num_predict` to the batch** (or shrink the batch to the cap) so a
  full extract fits in one call. Stop treating 3000 as a fixed safety cap.
- Emit **`finish_reason=length` telemetry** with a **<5% truncation SLO**.
- Keep the client-side **repeated-n-gram gate + temperature-bump retry** active
  (grammar does not cure the 31B repetition loop, ollama#15502, and can amplify
  it).
- **Build:** schema-per-role, batch right-sizer, finish_reason telemetry.
  **Reuse:** repetition gate.

### 5.5 The two prose failures
**Hallucinated claim id (wiki_write, `mempool-design#c005`):**
- **Primary:** per-request JSON schema with an `enum` of exactly the allowed
  claim ids (+ a `"NONE"` sentinel), constraining **only the citation field**,
  not the prose (limits the quality/latency cost of grammar constraint).
- **Second layer (model-independent):** deterministic closed-set validation —
  regex-extract every `[id]`, set-membership check, bounded 1–2 repair retry
  re-listing valid ids, then drop + flag. Catches truncation-corrupted ids the
  grammar cannot.
- **Prompt block:** fenced "Valid claim ids: [...]" + one positive and one
  decline exemplar.
- **Optional third layer:** entailment gate (cosine via qwen3-embedding, or 31B
  NLI) so a cited id actually supports the sentence (ALCE precision semantics).

**Coverage floor on tiny inputs (synthesize, `compiler-ir-design#c015`):**
- Gate on the ratio floor **only when citable-claim count ≥ ~4**; below that use
  an absolute rule ("every citable claim has ≥1 valid citation") and count
  violations.
- **Mark claims citable vs non-citable at extract time** so the coverage
  denominator excludes structurally uncitable claims.
- **Build:** enum-schema constructor, closed-set validator, coverage-gate
  redesign, citable flag. **Reuse:** the verbatim-gate pattern, mirrored for
  citations.

### 5.6 Deferred tier-2 (now unblocked)
- **Content-hash response cache** keyed on
  `sha256(model + role + rendered-prompt + params + schema)`; safe because the
  pipeline is deterministic (`reasoning_effort:"none"`). Makes `--runs 2..5` and
  halved-batch re-runs nearly free. Keep the shared system-prompt prefix
  byte-stable (no timestamps) so Ollama's prefix-KV cache also hits; pin
  `keep_alive`.
- **Threaded extraction:** bounded `ThreadPoolExecutor`, worker count tied to
  `OLLAMA_NUM_PARALLEL` (start 2–4 for e4b), semaphore + retry-on-503 backoff.
  Benchmark for the knee; do not crank parallelism on 31B (large per-slot KV).
- **Residency plan:** pin extract (e4b) + embeddings (qwen3-4b) resident
  (`keep_alive=-1`); time-share 12B and 31B as sequential stages.
- **Serving flags:** `OLLAMA_FLASH_ATTENTION=1` + `OLLAMA_KV_CACHE_TYPE=q8_0`
  for the small extract/embedding roles; gate flash-attention on 31B behind a
  long-prompt hang benchmark (ollama#15368) — if it hangs, run 31B FA-off / f16
  KV (it runs alone in its stage anyway).
- **Build:** content-hash cache, bounded executor. **Reuse:** config/role
  routing.

### Phase-1 serving reminder
Send `reasoning_effort:"none"` (not `think:false`, ignored for Gemma 4 on /v1,
ollama#15288); context truncates ~16k processed tokens — keep prompts +
exemplars short.

## 6. Phase 2 — Fine-Tune Meta-Infrastructure

A minimal file-based loop, not an MLOps platform. Four capabilities.

### 6.1 Dataset generation (gate-filtered rejection sampling)
- **RFT/RSFT:** sample k candidates, keep gate-passers, SFT on survivors. The
  verbatim/span gate **is** the RFT verifier — reuse the exact code path.
- **Teacher-per-slice by gate-filtered Pass@k, not by size:** e4b
  self-distillation for bulk; **qwen3:30b-a3b** for the recall slice; a
  **frontier API teacher** (Claude/GPT) for the bait slice (every local model
  caps ~0.86–0.96 under the 0.95 bar). **Do not** use gemma4:31b as a teacher
  (under-produces on extraction, recall ~0.40).
- **Sample at T≈1.0** for mode coverage; the cheap gate removes garbage. Keep
  the repeated-n-gram gate + temp-bump retry on to contain 31B repetition; cap
  heavy generation to e4b / qwen3:30b-a3b.
- **DART-style difficulty-aware allocation:** track per-chunk accepted-claim
  yield; escalate k (4 → 16 → 64) on low-yield / bait / high-density chunks.
- **Target ~1–3k diverse, deduplicated** gate-passed examples; curate for spread
  across domain, length, claim density, and bait patterns; mix 10–20% general
  instruction data; hold out 10% for validation. **Never** train on
  `tests/fixtures/eval-corpus` (test-set contamination).
- **Dataset provenance manifest:** content-hash the final JSONL; store
  `{dataset_hash, generator_model_digest, verbatim_gate_version,
  source_corpus_hash, n_accepted, n_rejected, acceptance_rate}`. `dataset_hash`
  is the immutable join key linking every run and promoted model to its exact
  examples.
- **Reuse:** the Phase-1 pooled-gold set IS the SFT target; the gate. **Build:**
  k-sampler with DART loop, teacher router, manifest hasher.

### 6.2 Training orchestration (scripted, resumable)
- **Lock the environment now** (longest lead, GPU-idle): Unsloth Studio + torch
  2.11.0+cu129 + transformers ≥5.5.3 + trl ≥0.23.1 + bnb 0.49.2. Snapshot the
  exact stack into every run's provenance.
- **Hard preflight asserts:** `attn_implementation="sdpa"` (mandatory);
  `torch.compile` off; stale-Triton-cache clear; assert FA2/FA4 not selected.
- **Recipe defaults** (narrow structured-JSON, small-data regime): LoRA r=16,
  α=32, LR 2e-4, 2 epochs (sweep, don't fix), eff-batch 16 (bs2×ga8),
  adamw_8bit, weight_decay 0.01, dropout 0.1, target all linear modules, val
  early-stop. Remove `<bos>` in data prep (the Gemma processor re-adds it).
- **Recipe routing (hard constraints):**
  - **Recipe A = 31B QLoRA SFT-ONLY.** Orchestrator forbids 31B GRPO
    (unsloth#4985, vocab-softmax workspace wall).
  - **Recipe B = E4B QLoRA, SFT-first THEN optional GRPO.** Route all RL to E4B.
    Composite shaped reward: format (valid schema) + accuracy (verbatim/span
    gate) + bait-rejection term. GRPO adds zero inference cost; enable standby
    mode.
  - **Recipe C = 26B-A4B MoE — BLOCKED for local training** (fused 3D experts
    ~43.7 GB). Serve-only via GGUF; cloud bf16 if ever needed.
- **Ship gate-filtered SFT first, GRPO as an optional second stage.**
- **Incremental save:** write `run.json` before weights.
- **Build:** resumable train script per recipe, preflight assert suite.
  **Reuse:** the eval harness for in-loop validation.

### 6.3 Eval-gated promotion + registry
- **Registry = one git-tracked append-only `registry.jsonl`** (source of truth)
  + a **DuckDB view** for SQL queries. One row per promoted version.
- **Two immutable anchors + provenance triple + eval block:**
  `{ollama_manifest_digest, hf_commit_sha (40-char), config_sha256,
  dataset_hash, git_commit, eval{recall, bait_rejection, gate_pass, atomicity},
  corpus_hash, prompt_hash, timestamp, status}`. Never store an eval number
  without the corpus_hash / prompt_hash it was measured on. Tags
  (`gemma4-extract:v3`) are mutable pointers only.
- **`promote <candidate> --baseline <current> --corpus <hash>`** verb: runs
  **both** models on the **same** corpus; bootstrap 95% CI (1000 resamples) on
  the per-chunk delta; **pre-registered assert gates:**
  1. recall lower-CI-bound clears **0.90** AND beats baseline;
  2. bait_rejection does **not** regress (any regression blocks);
  3. gate_pass stays ≥ 0.95;
  4. capability drift: wiki_write / synthesize / conflict roles run unchanged
     and must not regress.
  Require the CI **lower bound** (not a point estimate) to clear a calibrated
  floor — at n=186 a 0.3-pt delta is judge noise. Emit CI-distinct exit codes
  (0 = promote, non-zero = reject).
- **GGUF export validator (hard gate):** parse the token_type table, assert
  `<start_of_turn>` / `<end_of_turn>` / EOS / BOS are CONTROL not NORMAL (patch
  or fail — unsloth#5070/#5386); run the eval harness through the exact
  `gemma-4` chat template recorded at train time; cache-enabled smoke test.
- **Serving:** one merged, validated GGUF per role (tune → merge_and_unload →
  UD-Q4_K_XL / q4_k_m → `ollama create`). No hot-swap LoRA for Ollama.
- **Reuse (verified to exist):** `evalkit/bootstrap.py` `paired_bootstrap`;
  `scripts/eval-pipeline.py` `--compare` / `--write-baseline` / `history.jsonl`
  / `join_keys`. **Build:** the `promote` verb, registry writer, GGUF
  token-table validator.

### 6.4 Experiment tracking / provenance
- **Per-run `run.json` + `runs.jsonl` row written before weights:**
  `{git_commit, config_sha256 (hash the RESOLVED config after defaults/overrides
  merge), dataset_hash, seed=42, base_model_digest, recipe_id, gpu/driver/
  unsloth/ollama/torch/transformers/trl/bnb versions}`.
- **Key runs by `(config_sha256, dataset_hash)`** → idempotent skip-if-exists;
  any eval number reproducible from the triple `{config_sha, dataset_hash,
  git_commit}`.
- **DuckDB view** over `runs.jsonl` + `registry.jsonl` for queries like "which
  tag beats baseline bait at corpus_hash X?"
- **Build:** run.json writer, DuckDB view DDL. **Reuse:** the existing DuckDB
  machinery and `history.jsonl` pattern.

## 7. Overall approach (selected)

Three approaches were weighed:

- **A — strict sequential** (Phase 1 fully, then Phase 2): cleanest measurement
  discipline; slowest to a tuned model; risks discovering the need to train late.
- **B — fully parallel** (start e4b training now): fastest to an artifact, but
  trains against a broken denominator and soon-discarded pre-flow data, contends
  the single GPU, and optimizes a bait gap the span contract eliminates —
  **rejected by Debate #1.**
- **C — sequenced with a decoupled env carve-out (SELECTED):** run the full
  Phase-1 flow/contract/metric stack; provision and validate the
  Blackwell/Unsloth-Studio environment now in parallel (longest-lead,
  highest-failure-surface, GPU-idle — de-risking, not a training run); build the
  dataset harness + registry scaffolding in parallel too; launch the weight-update
  run only if a residual recall gap survives re-measurement.

Approach C honors Debate #1's verdict (A's sequencing hardened by B's one valid
point — env lead-time), keeps the single GPU free during eval runs, and ensures
that if training *is* needed, the pooled dataset, fixed metric, and validated
environment already exist so it can be done well immediately.

## 8. Gates — what must be TRUE to advance

### Gate 1→2: Phase 1 complete → begin the training decision
In order:
1. **Span-offset contract shipped.** If it alone drives bait_rejection ≥ 0.95
   and gate_pass ~1.0, the bait half is closed with zero training.
2. **Metric fixed.** precision_proxy retired; recall re-measured against pooled
   gold (e4b ∪ qwen3:30b-a3b, deduped/adjudicated) with an entailment matcher;
   F_fact reported.
3. **Full flow stack live:** formalized dedup union + N=3–5 self-consistency
   (≥2-of-N) + bounded coverage loop.
4. **Bootstrap 95% CI computed** on per-chunk recall over the eval corpus.
5. **Throughput/prose fixes shipped:** JSON-Schema grammar, truncation <5% SLO,
   enum-constrained citations + closed-set validator, tiny-input coverage gate.

**Launch trigger for Recipe B:** after (1)–(4), if the recall CI lower bound
still misses 0.90, launch Recipe B (SFT-first, GRPO optional). **Early-exit:**
if N=5 union self-consistency stops adding recall, launch without waiting for the
full stack. Env provisioning proceeds in parallel throughout but must not contend
the GPU during eval runs.

### Gate: trained adapter → production adoption
`promote <candidate> --baseline <current> --corpus <hash>` must pass ALL, on the
same corpus:
1. recall lower-CI-bound ≥ 0.90 AND beats baseline;
2. bait_rejection does not regress;
3. gate_pass ≥ 0.95;
4. downstream roles (wiki_write / synthesize / conflict) unchanged;
5. GGUF control-token table validated + eval through the exact recorded
   `gemma-4` chat template + cache-enabled smoke test clean;
6. registry row bound to Ollama digest + HF commit SHA +
   (config_sha256, dataset_hash, git_commit), corpus_hash/prompt_hash stamped.

Only on all-green (exit 0) does the merged GGUF get `ollama create`'d for its
role.

## 9. Top risks & mitigations

| Risk | Mitigation (designed-in) |
| --- | --- |
| Training against a broken recall denominator | Fix the metric first (pooled gold + entailment matcher + bootstrap CI); retire precision_proxy; launch trigger defined on the fixed metric only. |
| Flow plateaus ~0.80 < 0.90 bar | Early-exit trigger: if N=5 union stops adding recall, move to Recipe B without waiting for the full stack. |
| 31B GRPO OOM (>44 GB vocab-softmax wall) | Orchestrator hard-forbids 31B GRPO; route all RL to E4B; Recipe A SFT-only. |
| 26B-A4B untrainable locally (~43.7 GB) | Reclassify Recipe C BLOCKED; serve via GGUF only. |
| Silent EOS/BOS corruption on GGUF export | Hard promotion-gate validator asserts control-token types; patch-or-fail. |
| Blackwell env fragility | Lock the exact version stack; sdpa-mandatory + torch.compile-off + Triton-cache-clear preflight; snapshot into provenance; provision early as a decoupled task. |
| 31B repetition loop / FA hang | Keep repeated-n-gram gate + temp-bump retry always on; gate FA on 31B behind a long-prompt benchmark; grammar-constrain only the citation field. |
| Truncation amplification (44% / 3–5×) | JSON-Schema grammar + right-size num_predict to a one-pass batch + finish_reason <5% SLO. |
| Eval noise at n=186 → false promotion | Require the bootstrap-CI lower bound (not point estimate) to clear a calibrated floor; score candidate AND baseline on the same corpus. |
| Tuned extractor silently breaks downstream | Capability-drift gate runs wiki_write/synthesize/conflict unchanged; any regression blocks. |
| Provenance rot (eval unbound from weights) | Bind every eval to two immutable anchors + the provenance triple; pull with `revision=<sha>`. |
| Work lost on context refresh | Incremental save: run.json before weights; dataset harness writes survivors as it goes. |

## 10. Out of scope
vLLM/llama.cpp serving migration; local 26B-A4B training; heavy MLOps platforms
(MLflow/W&B servers, dashboards); a persisted promotion state-machine (deferred
to the first Recipe B run); span-offset extraction for languages without char
offsets; multi-GPU / off-box training (would reopen Debate #1's parallel option).

## 11. One-line sequencing summary
Ship the span contract + metric fix + flow stack + prose/throughput fixes now;
provision the Unsloth-Studio env + dataset harness + registry scaffolding in
parallel (GPU-idle); launch E4B Recipe B only if the recall CI lower bound still
misses 0.90 after re-measurement — and let the paired-bootstrap promotion gate,
not a point estimate or an epoch count, decide production adoption.
