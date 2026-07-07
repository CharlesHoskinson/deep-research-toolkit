# Gemma 4 Fine-Tuning Runbook (Future-Session Handoff)

**Written:** 2026-07-07 · **For:** a future Claude Code session with time and GPU budget
**Prereqs on disk:** merged eval suite (`scripts/eval-pipeline.py`, `tests/live/`,
`tests/fixtures/eval-corpus/`, `eval-results/baseline.json`), RTX 5090 (32 GB,
Blackwell sm_120), WSL2 Ubuntu-26.04, Docker, Ollama ≥0.31.1.
**Read first:** `docs/superpowers/specs/2026-07-06-comprehensive-eval-and-flow-hardening-design.md`
(§7 baseline results) and `docs/superpowers/specs/2026-07-05-two-track-pipeline-gemma4-design.md` (§7).

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
   - want max quality-per-latency for extract/wiki → **Recipe C** (26B-A4B MoE via
     Axolotl) as the experimental alternative.

---

## Phase 1 — Environment (WSL2, verified pins as of 2026-07)

Unsloth "Studio" is REQUIRED for Blackwell (plain `pip install unsloth` lacks the
sm_120 hybrid-attention kernels for Gemma 4's mixed 256/512 head dims):

```bash
# inside WSL2 Ubuntu (>=16 GB WSL memory configured)
curl -fsSL https://unsloth.ai/install.sh | sh
# Verified-working combo (unslothai/unsloth#5154): python 3.13, torch 2.10.0+cu130,
# transformers >=5.5.3 (gemma4 arch floor; latest 5.13.0 fine), trl 0.23.1,
# bitsandbytes 0.49.2 (Studio-matched build only), triton >=3.3.1, unsloth >=2026.4.4.
```

Docker alternative: `unsloth/unsloth` (Blackwell now covered; also
`unsloth/unsloth-blackwell` tags — check hub for the current cu130/torch2.10 tag).

**Known-broken (do not fight these):** torch 2.10+cu128 + torch.compile on sm_120;
mismatched bitsandbytes/torch cu builds (`libnvJitLink.so.13` errors); stale Triton
cache after torch upgrades (clear `~/.triton`); transformers < 5.5.3.

**Mandatory model-load flag everywhere:** `attn_implementation="sdpa"` (FA2 rejects
Gemma 4's 512-dim global heads).

Sanity gate before proceeding: run a 20-step smoke SFT on `unsloth/gemma-4-E2B-it`
with 10 dummy examples; loss must fall and no CUDA errors.

---

## Phase 2 — Dataset: distill from the frontier track, filter with the gate

Target: **2,000–5,000 gate-verified examples** for extract (the narrow JSON task);
quality-first (a small filtered set at up to ~10 epochs beats 10× unfiltered at 1).

1. **Sources:** the 186-chunk eval corpus is the TEST set — never train on it.
   Generate a fresh training corpus the same way (the corpus generator pattern in
   `scripts/check-eval-corpus.py`'s contract + the Task 6 approach): 30–50 synthetic
   docs across the same slice taxonomy, weighted toward whatever slices failed in
   Phase 0 (e.g., extra tables/unicode if those tripped the gate).
2. **Teacher:** the in-session frontier agent (Track A) extracts claims per chunk
   following `skills/knowledge-extraction` rules; OR programmatic generation via a
   large local model if frontier budget is short.
3. **Rejection sampling:** k=8 candidates per chunk (temp 0.9/top_p 0.95 for
   diversity), keep only candidates where EVERY quote passes
   `deep_research_toolkit.common.claims_check.check_claims_file` (the same verbatim
   gate as production — zero-cost, deterministic filtering).
4. **Format:** conversation JSONL — `{"role": "system"|"user"|"assistant"}` messages
   through the plain **`gemma-4`** chat template (NOT `gemma-4-thinking`; never mix
   thinking/non-thinking formats). System = the extract task brief (reuse
   `llm/extract.py`'s `_SYSTEM`); user = the chunk batch exactly as
   `build_extraction_prompt` renders it; assistant = the RAW JSON (no `<output>`
   wrapper — enforce structure via gates, not markup). Use
   `train_on_responses_only` masking.
5. Mix **10–20% general instruction data** (e.g., a slice of a permissively-licensed
   chat set) to guard against forgetting wiki-prose ability.
6. Hold out 10% as a validation split (never the eval corpus).

> **Sub-agent prompt (dataset build):**
> "Build the Gemma 4 extraction SFT dataset per Phase 2 of
> docs/superpowers/plans/2026-07-07-gemma4-finetune-runbook.md in
> C:\deep-research-toolkit. Generate the training corpus docs (30–50, slice-weighted
> per the Phase 0 failures listed here: <paste>), run k=8 rejection sampling with
> the claims_check gate as filter, emit train.jsonl/val.jsonl in the conversation
> format specified, and report: docs generated, candidates sampled, gate pass-rate
> of candidates, final example counts per slice. Do not touch
> tests/fixtures/eval-corpus (test-set contamination)."

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

### Recipe C — 26B-A4B MoE via Axolotl (experimental alternative)
- Axolotl **≥0.16.1** (prefer 0.17.0); start from the repo example
  `examples/gemma4/26b-a4b-moe-qlora.yaml`: `chat_template: gemma4`,
  `use_kernels: true`, `use_scattermoe: true`, LoRA on attention+MLP regex plus
  `lora_target_parameters: experts.gate_up_proj / experts.down_proj`.
- **REQUIRED fix:** `fsdp_state_dict_type: SHARDED_STATE_DICT` (FULL_STATE_DICT
  leaks VRAM on the first checkpoint save — axolotl#3596); budget a post-training
  re-shard/merge step.

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
3. Ollama Modelfile: start from `ollama show --modelfile gemma4:<size>` and copy its
   TEMPLATE/PARAMETER block verbatim; change only `FROM ./<tuned>.gguf`. A
   hand-written template is the #1 cause of degenerate output. `ollama create
   drt-<role>-gemma4-<size>:v1-<evaltag> -f Modelfile`.
4. **vLLM route:** merged safetensors only. Do NOT rely on LoRA-adapter serving for
   Gemma 4 — adapters currently load-but-no-op in vLLM (vllm#41754, open). If using
   vLLM: ≥0.24.0, upgrade transformers past vLLM's own `<5` pin.

---

## Phase 5 — Measure against the baseline (the only accept gate)

```bash
# point the tuned model at the role under test in .deepresearch.yml, then:
.venv/Scripts/python -m pytest tests/live -m live_model -q          # canaries first
.venv/Scripts/python scripts/eval-pipeline.py --roles extract \
    --models gemma4:e4b,drt-extract-gemma4-e4b:v1-<evaltag> --runs 5
.venv/Scripts/python scripts/eval-pipeline.py --compare eval-results/baseline.json
```

**Accept** only if the paired bootstrap shows the tuned model's per-doc recall delta
CI excluding zero in its favor AND gate_pass_rate/bait-rejection do not regress AND
the wiki/synthesize roles (run them too — forgetting check) hold their baseline.
**Reject** = keep the stock model, record the run in `eval-results/history.jsonl`,
and note the negative result in the design doc. Promote a new baseline
(`--write-baseline`) only after accepting.

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
