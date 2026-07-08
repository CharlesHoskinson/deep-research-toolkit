# Gemma 4 Fine-Tuning Runbook (Future-Session Handoff)

**Written:** 2026-07-07 · **For:** a future Claude Code session with time and GPU budget
**Prereqs on disk:** merged eval suite (`scripts/eval-pipeline.py`, `tests/live/`,
`tests/fixtures/eval-corpus/`, `eval-results/baseline.json`), RTX 5090 (32 GB,
Blackwell sm_120), WSL2 Ubuntu-26.04, Docker, Ollama ≥0.31.1.
**Read first:** `docs/superpowers/specs/2026-07-06-comprehensive-eval-and-flow-hardening-design.md`
(§7 baseline results), `docs/superpowers/specs/2026-07-05-two-track-pipeline-gemma4-design.md` (§7),
and `docs/superpowers/specs/2026-07-07-pipeline-hardening-and-finetune-meta-infra-design.md`
(§6 dataset/train/promote/provenance, §8 gates) — this later spec supersedes several
specifics below; every superseded passage is marked inline.

This document is executable by an agent session with no other context. Every phase
states its goal, exact commands, decision gate, and — where a sub-agent should do
the work — a ready-to-paste prompt in a blockquote.

---

## Phase 0 — Decide whether to tune at all

The suite exists precisely so this is a measurement, not a vibe.

> **Baseline actuals (2026-07-07, baseline.json = run-20260707T063207Z):**
> gate_pass_rate is saturated everywhere (min 0.954, e4b 0.997-0.999) — the
> verbatim gate does NOT motivate tuning. What trips the gates below is
> extract **recall** (e4b 0.685 full / 0.736 A/B, vs the 0.90 bar; mid
> Gemmas 0.39-0.53 by under-producing — atomicity ~0.57) and e4b
> **bait rejection** (0.861-0.865 vs the 0.95 bar; qwen3-30b hits 0.959).
> wiki (0.90 pass, cov 0.963, 0 bare markers), synthesize (0.96, cov
> 0.993) and adjudicate (12/12) all clear. **Indicated action: Recipe B on
> e4b targeting recall on hard slices + near-quote discrimination**, with
> training data weighted toward table/math/dense/bait slices. Note
> precision_proxy (~0.18) mostly measures gold non-exhaustiveness — do not
> gate on it; and same-model run-to-run recall varies ~0.05, so only
> bootstrap-significant deltas count (the suite computes them).

1. Read `eval-results/baseline.json` and the design doc §7 table.
2. **Gate:** fine-tune a role's model only if, on the eval corpus:
   - extract: `gate_pass_rate < 0.95` OR `recall < 0.90` overall, or < 0.85 on any
     hard slice (table / math / unicode / dense-facts), or bait rejection < 0.95; or
   - wiki_write / synthesize: `mean_pass_rate < 0.8` at N=5, or bare-marker rate
     persistently > 0.10 despite the prompt exemplar; or
   - conflict_adjudicate: verdict accuracy < 0.75 on the 12 labeled pairs.
3. If every number clears its gate: **stop — record "no tune needed" in the design
   doc and close.** Prompt/sampling A/B (`--models`, temp variants) is cheaper than
   training and should be exhausted first for near-misses (within 2 points of gate).
4. If a gate trips, choose the recipe by which model/role failed:
   - extract (e4b) failed → **Recipe B** (E4B QLoRA, optional GRPO).
   - synthesize/adjudicate (31b) failed → **Recipe A** (31B QLoRA — SFT only;
     GRPO on 31B is confirmed OOM on 32 GB, see Phase 3 warnings).
   - want max quality-per-latency for extract/wiki → Recipe C (26B-A4B MoE) is
     **BLOCKED for local training** (superseded by spec §6.2/§3 — fused 3D
     experts ~43.7 GB, does not fit 32 GB; serve-only via GGUF, cloud training
     only if ever needed). Use Recipe B instead.

---

## Phase 1 — Environment (WSL2, verified pins as of 2026-07)

> Superseded by spec §6.2: the env pin below is torch **2.11.0+cu129**, NOT
> cu130 — cu130 breaks the bitsandbytes ABI (`libnvJitLink.so.13`). If you
> provisioned against an older copy of this runbook that said cu130, rebuild
> the env.

Unsloth "Studio" is REQUIRED for Blackwell (plain `pip install unsloth` lacks the
sm_120 hybrid-attention kernels for Gemma 4's mixed 256/512 head dims):

```bash
# inside WSL2 Ubuntu (>=16 GB WSL memory configured)
curl -fsSL https://unsloth.ai/install.sh | sh
# Verified-working combo (unslothai/unsloth#5154, design doc §3/§6.2): python 3.13,
# torch 2.11.0+cu129 (do NOT chase cu130 -- breaks the bnb ABI), transformers
# >=5.5.3 (gemma4 arch floor; latest 5.13.0 fine), trl >=0.23.1,
# bitsandbytes 0.49.2 (Studio-matched build only), triton >=3.3.1, unsloth >=2026.4.4.
```

Docker alternative: `unsloth/unsloth` (Blackwell now covered; also
`unsloth/unsloth-blackwell` tags — check hub for the current cu129/torch2.11 tag).

**Known-broken (do not fight these):** torch 2.10+cu128 + torch.compile on sm_120;
torch **+cu130** (breaks the bitsandbytes ABI — `libnvJitLink.so.13` errors, design
doc §3); mismatched bitsandbytes/torch cu builds generally; stale Triton cache
after torch upgrades (clear `~/.triton`); transformers < 5.5.3.

Snapshot the exact resolved versions (torch/transformers/trl/bnb/unsloth/ollama/
driver/GPU) into every run's provenance — `deep_research_toolkit.tunekit.provenance
.write_run_record(..., versions={...})` writes this alongside `config_sha256`
before any weights are touched (design doc §6.4).

**Mandatory model-load flag everywhere:** `attn_implementation="sdpa"` (FA2 rejects
Gemma 4's 512-dim global heads).

Sanity gate before proceeding: run a 20-step smoke SFT on `unsloth/gemma-4-E2B-it`
with 10 dummy examples; loss must fall and no CUDA errors.

---

## Phase 2 — Dataset: DART-escalated rejection sampling, filtered with the span gate

> Superseded by spec §6.1 and implemented in
> `deep_research_toolkit.tunekit.dataset` + `scripts/build-sft-dataset.py`
> (design/finetune-meta-infra worktree, `feat/finetune-meta-infra`): this
> phase no longer hand-rolls rejection sampling or a verbatim-quote gate.
> Use the tools below; the narrative steps that follow describe what they do,
> not new work to write from scratch.

Target: **2,000–5,000 gate-verified examples** for extract (the narrow JSON task);
quality-first (a small filtered set at up to ~10 epochs beats 10× unfiltered at 1).

1. **Sources:** the 186-chunk eval corpus is the TEST set — never train on it, and
   never quote it in training data. Generate a fresh training corpus the same way
   (the corpus generator pattern in `scripts/check-eval-corpus.py`'s contract): 30–50
   synthetic docs across the same slice taxonomy (`prose`/`dense-facts`/`table`/
   `list`/`unicode`/`math`/`long`/`bait`), weighted toward whatever slices failed in
   Phase 0. `scripts/build-sft-dataset.py`'s HARD contamination guard
   (`tunekit.dataset.load_contamination_index` /
   `assert_not_contaminated`) refuses any chunk whose locator OR text hash matches
   `tests/fixtures/eval-corpus/corpus-index.json` — a training doc that
   accidentally reuses eval-corpus text raises, it does not warn.
2. **Output contract is the SPAN contract (design doc §5.1), not a copied quote.**
   Every training example's assistant turn cites evidence by
   `start_char`/`end_char` offsets into the chunk, exactly like production
   `llm/extract.py` — `tunekit.dataset.gate_claim` mirrors
   `extract_claims_to_run`'s per-claim gate verbatim (same
   `common.verbatim.span_ok`/`slice_span` check), so a training example that
   wouldn't pass the production gate can't be written to `train.jsonl` either.
3. **Teacher-per-slice router, not one teacher for everything**
   (`tunekit.dataset.DEFAULT_ROUTER_TABLE`/`route_for_chunk`): e4b
   self-distillation for the bulk of chunks, **qwen3:30b-a3b** for the
   `dense-facts`/recall slice, a **frontier API teacher** (Claude/GPT) for the
   `bait` slice (every local model caps ~0.86–0.96 under the 0.95 bait bar).
   `gemma4:31b` is a hard-banned teacher (`BannedTeacherError` —
   under-produces on extraction, recall ~0.40). The frontier route is NOT
   wired to a real backend in `scripts/build-sft-dataset.py` yet — pass
   `--teachers-module` naming a module exposing
   `get_teachers(config, producer) -> dict` to supply one.
4. **DART-style difficulty-aware k escalation**
   (`tunekit.dataset.escalating_k_sample`, ladder `4 → 16 → 64`): sample at
   T≈1.0 for mode coverage (the gate removes garbage), escalate to the next k
   when a chunk's accepted yield is below the floor; a chunk pre-tagged
   `bait`/`dense-facts`/`table`/`long` skips the cheapest rung and starts
   sampling harder immediately rather than waiting to observe low yield.
5. **Dedup** the accepted set with `tunekit.dataset.dedup_claims` — the SAME
   normalized-text + source-locator key `llm/selfconsistency.py`'s
   cross-sample union uses, applied across the whole dataset.
6. **Reuse the Phase-1 pooled gold as the SFT target where available**
   (design doc §6.1): the pooled gate-passing union of e4b ∪ qwen3:30b-a3b
   built for the fixed eval denominator (`scripts/build-pooled-gold.py`) is
   already a gate-verified, cross-model-agreed claim set — feed it through
   `tunekit.dataset.to_conversation_record` directly for that slice of the
   corpus instead of re-sampling it from scratch.
7. **Format:** conversation JSONL — `{"role": "system"|"user"|"assistant"}`
   messages through the plain **`gemma-4`** chat template (NOT
   `gemma-4-thinking`; never mix thinking/non-thinking formats).
   `tunekit.dataset.to_conversation_record` renders system/user with
   `llm/extract.py`'s OWN `build_extraction_prompt` (byte-identical to
   production) and the assistant turn as the bare `{"claims": [...],
   "entities": [], "relations": []}` contract object (default
   `thinking=False` — no `<output>` wrapper; pass `thinking=True` only if
   training a reasoning-trace variant). Use `train_on_responses_only` masking.
8. Mix **10–20% general instruction data** (e.g., a slice of a permissively-licensed
   chat set) to guard against forgetting wiki-prose ability — `build_sft_dataset`
   does not do this mixing itself; blend it into `train.jsonl` afterward.
9. **Provenance manifest is written automatically**
   (`tunekit.dataset.build_manifest`/`dataset_hash`): `manifest.json` records
   `dataset_hash`, `generator_model_digests`, `verbatim_gate_version`,
   `source_corpus_hash`, and acceptance/rejection/dedup counts —
   `dataset_hash` is the immutable key that later binds a training run
   (`tunekit.provenance.write_run_record`) and a promoted registry row
   (`tunekit.registry`) back to these exact examples.
10. Hold out 10% as a validation split — deterministic and seeded
    (`tunekit.dataset.split_train_val`, default `seed=42`), never the eval corpus.

```bash
# from the repo root, with a training-corpus dir of <doc>/chunks.jsonl (NOT
# tests/fixtures/eval-corpus):
.venv/Scripts/python scripts/build-sft-dataset.py path/to/training-corpus \
    --out-dir sft-dataset --k-ladder 4,16,64
```

> **Sub-agent prompt (dataset build):**
> "Build the Gemma 4 extraction SFT dataset per Phase 2 of
> docs/superpowers/plans/2026-07-07-gemma4-finetune-runbook.md in
> C:\deep-research-toolkit using `scripts/build-sft-dataset.py` (backed by
> `deep_research_toolkit.tunekit.dataset`). Generate the training corpus docs
> (30–50, slice-weighted per the Phase 0 failures listed here: <paste>) under a
> NEW directory (never tests/fixtures/eval-corpus), run
> `scripts/build-sft-dataset.py <training-corpus-dir> --out-dir sft-dataset`, and
> report: docs generated, `manifest.json`'s acceptance/rejection/dedup counts,
> final train/val example counts, and confirmation the contamination guard
> raised zero errors. Do not touch tests/fixtures/eval-corpus."

---

## Phase 3 — Train

### Recipe A — 31B QLoRA (quality ceiling; SFT only)
- Model: `unsloth/gemma-4-31B-it-unsloth-bnb-4bit` (pre-quantized, fixed template).
- `FastModel.from_pretrained(..., max_seq_length=4096, load_in_4bit=True,
  attn_implementation="sdpa")`; LoRA r=16 α=32 dropout=0; lr 2e-4 cosine; 2–3 epochs
  on 2–5k examples; `use_gradient_checkpointing="unsloth"`. ~22 GB train VRAM.
- **Never GRPO on 31B:** the 262k-vocab softmax workspace OOMs 32 GB
  (unslothai/unsloth#4985, unresolved). SFT only.

### Recipe B — E4B QLoRA + optional GRPO (throughput play)
- Model: `unsloth/gemma-4-E4B-it` (4-bit). Same hyperparams; ~10–17 GB.
- If SFT alone leaves residual gate failures: GRPO with the gate as binary reward —
  `def reward(prompts, completions, **kw): return [1.0 if gate_passes(c) else 0.0 ...]`
  where gate_passes = parse JSON + every quote verbatim in the prompt's chunks.
  Watch reward-fn VRAM spikes (known GRPO pattern); smoke-test rollout quality for
  the first 10 steps before a long run (garbage-rollout reports exist on E4B in
  other frameworks). Skip DPO (GRPO beats it on structure per RL-Struct).

### Recipe C — 26B-A4B MoE via Axolotl — **BLOCKED for local training**

> Superseded by spec §6.2/§3: reclassified BLOCKED. Fused 3D experts load
> ~43.7 GB because bitsandbytes only 4-bit-quantizes 2D `nn.Linear` — this
> does not fit the 32 GB budget and every documented workaround fails. Do
> **not** attempt local Recipe C training; the notes below are kept for the
> serve-only / cloud path record, not as a runnable local recipe.

- **Local training is BLOCKED** on the 32 GB RTX 5090: fused 3D MoE expert
  tensors load ~43.7 GB unquantized. **Serve-only via GGUF** (a merged GGUF
  produced elsewhere/upstream); if training is ever needed, it requires cloud
  bf16/B200-class capacity, not this box. Do **not** substitute
  Qwen3.5-35B-A3B as a workaround — that violates the Gemma-4-first
  constraint (design doc §3).
- Axolotl **≥0.16.1** (prefer 0.17.0) reference config, for the cloud path only:
  start from the repo example `examples/gemma4/26b-a4b-moe-qlora.yaml`:
  `chat_template: gemma4`, `use_kernels: true`, `use_scattermoe: true`, LoRA on
  attention+MLP regex plus `lora_target_parameters: experts.gate_up_proj /
  experts.down_proj`.
- **REQUIRED fix (cloud path):** `fsdp_state_dict_type: SHARDED_STATE_DICT`
  (FULL_STATE_DICT leaks VRAM on the first checkpoint save — axolotl#3596);
  budget a post-training re-shard/merge step.

All recipes: adapters per role (one tune per role, not one do-everything model).

> **Sub-agent prompt (training run):**
> "Execute Recipe <A|B|C> per Phase 3 of the runbook against
> <train.jsonl/val.jsonl>. Use the exact pins/flags in Phases 1+3 (Studio install,
> sdpa, the recipe's model id). Log train/val loss every 50 steps; stop early if
> val loss rises for 3 consecutive evals. Save the adapter AND a merged checkpoint.
> Report: final losses, wall time, VRAM peak, artifact paths."

---

## Phase 4 — Export and patch

1. Merge + GGUF: `model.save_pretrained_gguf(dir, tokenizer,
   quantization_method="q4_k_xl")` (UD-Q4_K_XL dynamic quant — preferred over
   q4_0/q4_k_m for Gemma 4).
2. **EOS patch check (required):** verify `tokenizer_config.json` in the merged
   output has `"eos_token": "<turn|>"` (id 106), NOT `<eos>` (id 1) —
   `save_pretrained_merged` has corrupted this (unsloth#5386), which makes serving
   never stop. Patch manually if wrong.
3. **GGUF control-token validator (required, hard gate — design doc §6.2/§6.3):**

   ```bash
   .venv/Scripts/python scripts/validate-gguf-tokens.py path/to/merged.gguf
   ```

   Backed by `deep_research_toolkit.tunekit.gguf_check`: parses the GGUF
   metadata KV section (no tensor parsing) and asserts `<start_of_turn>`,
   `<end_of_turn>`, BOS, and EOS are token_type **CONTROL**, not NORMAL — the
   unsloth#5070/#5386 corruption class step 2's manual EOS check guards
   against, checked here mechanically and exhaustively. Exit 0 = clean, exit
   1 = at least one control token was demoted; patch (or re-export) before
   proceeding — do NOT `ollama create` a GGUF that fails this check.
4. Ollama Modelfile: start from `ollama show --modelfile gemma4:<size>` and copy its
   TEMPLATE/PARAMETER block verbatim; change only `FROM ./<tuned>.gguf`. A
   hand-written template is the #1 cause of degenerate output. `ollama create
   drt-<role>-gemma4-<size>:v1-<evaltag> -f Modelfile`.
5. **vLLM route:** merged safetensors only. Do NOT rely on LoRA-adapter serving for
   Gemma 4 — adapters currently load-but-no-op in vLLM (vllm#41754, open). If using
   vLLM: ≥0.24.0, upgrade transformers past vLLM's own `<5` pin.

---

## Phase 5 — Measure against the baseline (the only accept gate)

> Superseded by spec §8: the accept/reject decision is no longer a manual
> read of `--compare`'s regression list. `scripts/promote.py` runs the full
> §8 gate set as pure functions over two eval reports and returns a
> CI-distinct exit code — that exit code IS the decision.

```bash
# point the tuned model at the role under test in .deepresearch.yml, then:
.venv/Scripts/python -m pytest tests/live -m live_model -q          # canaries first
.venv/Scripts/python scripts/eval-pipeline.py --roles extract,wiki_write,synthesize,conflict_adjudicate \
    --models gemma4:e4b,drt-extract-gemma4-e4b:v1-<evaltag> --runs 5
# writes eval-results/run-<ts>.json (the candidate report) — run the SAME
# corpus/roles against the stock model too if eval-results/baseline.json
# doesn't already cover it.

.venv/Scripts/python scripts/promote.py eval-results/run-<candidate-ts>.json \
    --baseline eval-results/baseline.json --model drt-extract-gemma4-e4b:v1-<evaltag>
```

`scripts/promote.py` (backed by `deep_research_toolkit.tunekit.registry`/
`gguf_check`) checks, in order, and STOPS at the first failure:

| Exit | Gate | Meaning |
| --- | --- | --- |
| 6 | provenance | candidate/baseline share the same `corpus_hash`/`prompt_hash` — checked FIRST, or nothing below is meaningful |
| 2 | recall | candidate's per-doc recall bootstrap-CI lower bound clears 0.90 AND beats baseline (paired-bootstrap delta CI excludes zero, in the candidate's favor) |
| 3 | bait | bait_rejection does not regress vs baseline (any regression blocks) |
| 4 | gate_pass | candidate gate_pass_rate stays ≥ 0.95 |
| 5 | drift | wiki_write/synthesize/conflict_adjudicate (the forgetting check) do not regress |
| 0 | — | **promote** |

**Accept** only on exit 0. **Reject** on any nonzero exit — keep the stock model,
record the run in `eval-results/history.jsonl` (already automatic via
`scripts/eval-pipeline.py`), and note the negative result (and which gate/exit code
tripped) in the design doc. Before `ollama create`-ing the merged GGUF for the role,
also re-run Phase 4's `scripts/validate-gguf-tokens.py` — `promote.py` gates the
EVAL numbers, not the export artifact itself. Promote a new baseline
(`--write-baseline` on `scripts/eval-pipeline.py`) and append a
`deep_research_toolkit.tunekit.registry` row (anchored to the Ollama manifest
digest + 40-char HF commit SHA + the provenance triple) only after accepting.

---

## Phase 6 — Publish (verified flows, 2026-07)

**Private cross-machine (primary, free):** push GGUF to a PRIVATE Hugging Face repo
(`model.push_to_hub_gguf("you/drt-extract-gemma4-e4b-GGUF", tokenizer,
quantization_method="q4_k_xl", private=True, token=...)`), add your Ollama SSH key
(`~/.ollama/id_ed25519.pub`) at huggingface.co/settings/keys, then on any machine:
`ollama pull hf.co/you/drt-extract-gemma4-e4b-GGUF:Q4_K_XL`.

**Public sharing:** same HF repo public (set `base_model: google/gemma-4-...`
metadata + model card noting "derivative of Google's Gemma 4; not affiliated with
or endorsed by Google"; Apache 2.0 text + changes notice) — pullable via the same
`hf.co/` syntax; optionally ALSO `ollama signin && ollama cp <local> you/<name>:<tag>
&& ollama push you/<name>:<tag>` for browsability (ollama.com private hosting
requires the paid Pro tier — HF private is the free path).

**Versioning discipline:** tags are mutable on both registries — never reuse a tag
that has eval numbers recorded against it (`:v2-<evaltag>` next iteration); on HF,
record the commit SHA next to the eval entry (the SHA is the only immutable ref).
**Do not** build on Ollama↔OCI/ghcr pushes (fragile/unsupported, ollama#2745/#7244);
the zero-account fallback is syncing GGUF+Modelfile files and `ollama create`
per machine.

---

## Phase 7 — Adopt

Update `.deepresearch.yml` roles to the tuned tag; re-run the FULL eval
(`--runs 5`, all roles) + canaries; `--write-baseline`; update README's validated
role map + the design docs; commit. Add a memory note with the published repo/tag
and its baseline SHA.

---

## Standing cautions
- The eval corpus is a test set: never train on it, never quote it in training data.
- One adapter per role; 10–20% general data in every mix; low rank first (r=16).
- Never mix thinking/non-thinking chat formats in one dataset.
- Re-run canaries after ANY Ollama/vLLM upgrade before trusting new eval numbers.
- Trace every training-data generation run (`llm.trace: true`) so dataset provenance
  is reconstructible.
