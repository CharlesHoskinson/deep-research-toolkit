# Two-Track Analysis Pipeline: Frontier Pure-Skill vs Local Gemma 4 Stack

**Date:** 2026-07-05
**Status:** Draft — awaiting review
**Branch:** `study/two-track-gemma`

## 1. Goal

Formalize the analysis pipeline's split into two first-class tracks:

- **Track A — pure skill.** A frontier model running in-session in the Claude
  Code or Codex terminal does all judgment work by reading SKILL.md files.
  No Python LLM backend, no API keys, no local server.
- **Track B — local Gemma 4 stack.** A composition of Gemma 4 models behind
  an OpenAI-compatible endpoint (Ollama today, vLLM optionally), routed
  per-role via `llm.roles`, runs the pipeline programmatically for bulk
  throughput.

The seam already exists (`llm.provider: agent | local`,
`get_backend(config, role)`); this design hardens each side and selects and
validates the Gemma 4 composition for Track B.

## 2. Current state (repo audit)

- The **only** production `Backend.complete()` call site is claim extraction
  (`llm/extract.py:234`, called by `extract_claims.py` in both extraction
  skills with `role="extract"`). Wiki synthesis, contradiction adjudication,
  synthesis, and planning are agent-only: their roles exist in
  `ROLE_DEFAULTS` and resolve through `get_backend`, but no code passes them.
- The verbatim gate (`common/verbatim.py`) runs **pre-write** on Track B
  (extraction drops non-verbatim claims before anything lands in the run
  dir) but only **post-hoc** on Track A (compile/dossier/eval time). A pure
  skill run that never compiles or evals skips the mechanical check.
- The eval harness (`scripts/validate-local-llm.py`) called
  `get_backend(config)` with no role, so it never exercised the
  `roles.extract` config that production uses. Fixed on this branch.
- Packaging is already dual-target: identical `.claude-plugin/plugin.json`
  and `.codex-plugin/plugin.json` pointing at one canonical `skills/` tree,
  synced to `src/deep_research_toolkit/skill_templates/` for wheel installs,
  with CI parity checks. `drt init` copies skills into both `.claude/skills/`
  and `.agents/skills/`.

## 3. Gemma 4 landscape (researched 2026-07-05)

Gemma 4 released 2026-03-31 (E2B, E4B, 26B-A4B MoE, 31B dense; 12B Unified
added 2026-06-03). Key facts driving this design:

| Model | Params (active) | Ctx | Ollama tag / size | Notes |
|---|---|---|---|---|
| E4B | 8B (4.5B eff.) | 128K | `gemma4:e4b` 9.6 GB | throughput champion |
| 12B Unified | 12B | 256K | `gemma4:12b` 7.6 GB | encoder-free multimodal |
| 26B-A4B | 25.2B (3.8B) | 256K | `gemma4:26b` 18 GB | MoE, near-31B quality at ~4B decode cost |
| 31B | 30.7B | 256K | `gemma4:31b` 20 GB | LMArena #3 open (Elo 1452); Google's designated fine-tune target |

- **License: Apache 2.0, ungated** — the headline change from Gemma 3's
  custom ToU. No use-policy flow-down, no derivative restrictions. Safe to
  reference, bundle configs for, and fine-tune within this MIT toolkit.
  (Caveat: EmbeddingGemma and FunctionGemma are Gemma-3-era and remain under
  the old Gemma ToU — excluded from defaults here.)
- **All sizes have toggleable thinking**, and the toggle is a serving-stack
  property, not a model property (consistent with the Ornith/qwen3.5 finding
  in the README): vLLM disables thinking by default; **Ollama enables it by
  default**, and on Ollama's OpenAI-compatible endpoint `think: false` is
  ignored for Gemma 4 (ollama/ollama#15288) while `reasoning_effort: "none"`
  works. Verified locally — see §6.
- Known serve-time bugs relevant to extraction: Ollama #15502 (31B word-loop
  under schema-constrained JSON with long string fields, unconfirmed fixed)
  and vLLM #40080 (same family of repetition loops under grammar masks).
  Mitigations: prefer json_object mode over full schema-grammar for long
  quote fields, temp 0, repetition penalty if needed.
- Official QAT Q4_0 GGUFs exist for all sizes; Unsloth dynamic quants
  recommended over naive Q4_0 for fine-tuned exports.

## 4. Track A design (pure skill, both terminals)

Track A is ~90% built; the changes are surgical:

1. **De-Claude the skill bodies.** `knowledge-extraction/SKILL.md` and the
   README say "Claude does this"; replace with "the agent"/"you" so the same
   text is native in Codex (GPT) sessions. Skill frontmatter already sticks
   to the agentskills.io lowest common denominator (`name` + `description`)
   — keep it that way; both platforms ignore unknown manifest fields.
2. **Move the gate to write-time.** Add a `check_claims.py` entry point
   (thin shim over `common/verbatim.py` + schema checks) to both extraction
   skills, and make the SKILL.md loop mandatory: extract batch → append →
   run gate script → fix/drop rejects → next batch. Instructed script runs
   are portable; hooks are not (Claude hooks are rich, Codex hooks thin).
3. **Volume discipline.** Sequential baseline: batches of 10–20 chunks with
   progress recorded in `manifest.json` so compaction/crash resumes instead
   of restarting (Codex auto-compacts; this is the main Track A hazard).
   Fan-out (Claude Task tool / Codex `spawn_agent`, depth 1, 6 threads) is a
   **degradable optimization** — never a requirement — with contiguous chunk
   ranges, batch-prefixed ids (the `b00_c_0001` scheme `extract.py` already
   uses), and a deterministic merge script.
4. **Metadata budget.** Trim skill descriptions toward ≤500 chars: Codex
   preloads all skills' metadata into max(2% of context, 8,000 chars);
   ten verbose descriptions can crowd that floor.

**Track A limits (accepted):** frontier-token cost bounds bulk throughput
("forty sources overnight" stays Track B's job); no mechanical retry
harness — the write-time gate script is the backstop; cross-model claim-set
variance between Claude and GPT is fidelity-normalized by the gate but not
coverage-normalized.

## 5. Track B design (local Gemma 4 composition)

### 5.1 Role → model mapping (32 GB RTX 5090, one role resident at a time)

```yaml
llm:
  provider: local
  embedding_model: qwen3-embedding:4b     # Qwen3 still tops MTEB; EmbeddingGemma is old-ToU + weaker
  roles:
    extract:     {model: gemma4:e4b}      # validated: 100% gate pass, 5/5 recall, fastest
    wiki_write:  {model: gemma4:12b}      # 256K ctx, strong prose, 7.6 GB
    conflict_adjudicate: {model: gemma4:31b, thinking: true}
    synthesize:  {model: gemma4:31b, thinking: true}   # Arena-1452-class reasoning
    code_agent:  {model: qwen3.6:27b}     # honesty: Qwen still owns the coding niche
  local:
    base_url: http://localhost:11434/v1
    model: gemma4:12b                     # fallback
```

- `extract` alternates: `gemma4:12b` (higher quality, still cheap) or
  `gemma4:31b` (validated highest claim yield, 21 vs 14, at ~3× latency).
  `gemma4:26b` (MoE) is the untested wildcard: near-31B quality at ~E4B
  decode speed; worth pulling and running through the harness next.
- Everything fits 32 GB solo with ≥10 GB KV headroom; e4b+12b can co-reside.

### 5.2 Serving-stack requirements (found empirically — see §6)

- **Pin Ollama ≥ 0.31.x** (structured-outputs-with-think-off fix, #15260)
  and send `reasoning_effort: "none"` for non-thinking roles — `think:
  false` alone is silently ignored for Gemma 4 on the `/v1` endpoint.
  Patched in `llm/local.py` on this branch: when a role has
  `thinking: false`, the backend now sends both.
- vLLM alternative for the reasoning roles: `--reasoning-parser gemma4`,
  QAT w4a16 checkpoints; note LoRA support requires a current release
  (post-#39291) and sm_120 wants community wheels / torch-cu128.

### 5.3 Wiring the remaining roles

Extraction is wired; `wiki_write`, `conflict_adjudicate`, and `synthesize`
have config defaults but no callers. Full-local runs need three small
programmatic callers mirroring `extract_claims_to_run`'s shape (task brief
prompt, bounded inputs, mechanical validation where possible: wiki pages
re-checked by lint + citation gate, adjudications constrained to verdicts
over gate-passed claims). This is the main Track B build item and should be
its own implementation plan.

## 6. Prototype results (2026-07-05, RTX 5090, Ollama 0.31.1)

### 6.1 Think-toggle behavior, OpenAI-compatible endpoint (`/v1`)

Extraction-style probe, `response_format: json_object`, temp 0:

| Model | Variant | Reasoning tokens | Latency | Gate |
|---|---|---|---|---|
| gemma4:31b | default | ~670 | 37.2 s | 5/6 |
| gemma4:31b | `think: false` (old backend behavior) | ~670 — **ignored** | 30.1 s | 5/6 |
| gemma4:31b | `reasoning_effort: "none"` | 0 | **7.2 s** | **6/6** |
| gemma4:e4b | default | ~316 | 13.8 s | 5/5 |
| gemma4:e4b | `reasoning_effort: "none"` | 0 | **3.8 s** | 5/5 |

Suppressing thinking is 4–5× faster and did not hurt (slightly helped)
verbatim exactness. Native `/api/chat` honors `think: false` correctly;
only the `/v1` path — the one the toolkit uses — needs `reasoning_effort`.

### 6.2 Pipeline eval harness (`scripts/validate-local-llm.py`, patched to `role="extract"`, hydra fixture: 9 chunks, 5 reference claims)

| Model | Claims written | Gate drops | Parse failures | Reference recall |
|---|---|---|---|---|
| gemma4:e4b | 14 | 0 | 0 | 5/5 |
| gemma4:12b | 11 | 0 | 0 | 5/5 |
| gemma4:31b | 21 | 0 | 0 | 5/5 |
| qwen3:30b-a3b-instruct (baseline) | 14 | 0 | 0 | 5/5 |

All Gemma 4 candidates match or beat the Qwen baseline on this fixture, at
100% gate integrity, within the extract role's 3,000-token budget (which
itself confirms reasoning suppression worked end-to-end).

Also probed: `diffusiongemma-26B-A4B` GGUF — HTTP 500 under Ollama 0.31.1
(arch unsupported); dropped from candidacy until Ollama supports it.

## 7. Fine-tuning: decision rule and recipes

**Rule:** constrained decoding guarantees JSON shape but cannot guarantee
verbatim-substring exactness — only tuning (or resampling) moves that.
Baseline stock models first on a real corpus (~200 chunks); **fine-tune only
if gate pass-rate < ~95%.** The 9-chunk fixture shows 100%; the decision
gate is a larger-corpus run, not this fixture.

If tuning is warranted (or for the quality ceiling):

- **Recipe A (primary):** Unsloth + `gemma4:31b` QLoRA (~22 GB, fits the
  5090), r=8–16, lr 2e-4, 1–2 epochs on 2–5K gate-verified examples
  distilled from Claude via rejection sampling (generate k≈4–8 per chunk,
  keep gate-passers — the verbatim gate is a free, deterministic filter).
  Export `save_pretrained_gguf` (UD-Q4_K_XL), serve via Ollama with the
  exact gemma-4 chat template (template mismatch is the #1 post-export
  failure).
- **Recipe B (throughput):** same pipeline on `gemma4:e4b` QLoRA (~10 GB);
  if residual gate failures persist, add GRPO with the gate as the binary
  reward (fits easily at this size; skip DPO — GRPO beats it on structure).
- **Recipe C (MoE):** Axolotl ≥0.16.1 fused MoE+LoRA is the only framework
  fitting `gemma4:26b` on a 5090; serve via vLLM with repetition penalty
  ~1.1 against the constrained-JSON loop bug.
- Blackwell stack status: solved — torch ≥2.7 cu128 wheels, bnb/triton
  ≥3.3.1 fine, FA2/Triton paths (FA3/FA4 don't target sm_120), Unsloth's
  Docker image covers Blackwell on WSL2.
- Keep LoRA rank low and mix 10–20% general instruction data to dodge
  forgetting on wiki-prose/adjudication duties; or one adapter per role.

## 8. Changes on this branch

- `src/deep_research_toolkit/llm/local.py` — send `reasoning_effort: "none"`
  alongside `think: false` when a role disables thinking (Gemma 4 on
  Ollama's `/v1` ignores the latter).
- `scripts/validate-local-llm.py` — resolve the backend with
  `role="extract"` (mirrors production; previously exercised the flat
  config), and print `parse_failures`.

## 9. Open questions / next steps

1. Pull and harness-test `gemma4:26b` (MoE) — likely the best
   quality-per-token extract/wiki model if the constrained-JSON loop bug
   doesn't bite at json_object level.
2. Scale the eval: build a ~200-chunk reference corpus (the tuning
   go/no-go gate) and add per-model latency/tokens to the harness output.
3. Implement Track A hardening (de-Claude wording, `check_claims.py`
   write-time gate, batching/resume discipline) — own plan.
4. Wire `wiki_write` / `conflict_adjudicate` / `synthesize` programmatic
   callers for full-local Track B — own plan.
5. Add a Codex-produced reference run to quantify cross-model coverage
   variance on Track A.
6. Decide default extract model (`e4b` speed vs `12b`/`31b` yield) after
   the 200-chunk run.
