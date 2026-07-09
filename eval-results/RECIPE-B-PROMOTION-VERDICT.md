# Recipe-B fine-tune — promotion verdict

**Date:** 2026-07-08
**Candidate:** tuned LoRA adapter `recipe-b-v1` on `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`
(`/root/gemma4-tune/runs/recipe-b-v1/adapter`), served in WSL via
`scripts/serve_adapter.py` (unsloth `FastModel` + adapter, OpenAI-compatible).
**Baseline:** stock `gemma4:e4b` on Ollama (report `run-20260708T023459Z.json`).
**Corpus/prompt (join keys, identical on both sides):**
`corpus_version sha256:d42ff714…`, `prompt_version sha256:beb0f703…`.

## VERDICT: NOT PROMOTED — `scripts/promote.py` exit code **2** (recall gate)

The tuned model must NOT become the production extract model. It is a
catastrophic regression on the single decisive gate, extraction recall.

### promote.py per-gate detail
| gate | result | detail |
|---|---|---|
| provenance (exit 6) | **PASS** | corpus_hash + prompt_hash identical on candidate & baseline |
| **recall (exit 2)** | **FAIL** | `clears_floor=false`, `beats_baseline=false` |
| bait (exit 3) | not reached | recall failed first |
| gate_pass (exit 4) | not reached | — |
| capability-drift (exit 5) | not reached | — |

- Candidate pooled per-claim recall 95% CI: **[0.000, 0.0019]** — lower bound far
  below the 0.90 floor (`n_claims = 1591`).
- Paired per-claim delta (candidate − baseline) 95% CI: **[−0.863, −0.827]**
  (`n_paired_claims = 1591`) — the tuned model loses to stock by ~0.85 recall.

## Full metrics table (same corpus, same settings, single code path)
Both models are scored by the identical production path
(`extract_claims_to_run` → `span_ok` gate, `scripts/eval-pipeline.py` loops one
`run_extract_for_model` over both; the gate call is `extract.py:391`). Pooled
gold + quote-overlap recall, single-pass, all 10 eval-corpus docs.

| metric | TUNED recipe-b-v1 | STOCK gemma4:e4b |
|---|---|---|
| pooled claim-recall | **0.0006** (1 / 1591) | **0.9617** (1530 / 1591) |
| recall 95% CI (per-claim bootstrap) | [0.000, 0.0019] | ≈[0.95, 0.97] |
| gate_pass_rate | **0.0018** | **0.9926** |
| bait_rejection | n/a (0 bait claims survived gate) | 0.9044 |
| atomicity | 0.0006 | 0.4199 |
| recall_entailment | 0.0 | (embedder degraded → null) |
| truncated_calls (non-termination) | **8** (of 44 batches) | 0 |
| parse_failures | 3 | 0 |

Per-doc: recall = 0 on 9 of 10 docs; a single claim recalled in `zk-proofs`
(0.005). 1590 of 1591 reference claims missed.

## Why it fails — the actionable finding

**The span-OFFSET contract does not survive SFT distillation into a 4B model.**
The model can copy a supporting span verbatim, but it cannot reproduce the exact
integer character offsets that slice to that span — offset-counting is a strictly
harder capability than verbatim-copy for a small model. This is the risk the
finetune design doc deferred, now confirmed empirically.

Concretely, the Recipe-B SFT targets attach a derived `quote` field to every
piece of evidence — **21,688 / 21,688 evidence rows (100%) carry a `quote`, and
in training every one is perfectly offset-consistent** (`chunk_text[start:end]
== quote`). The model faithfully learned to emit that `quote`. At inference it
writes a *correct* quote but *wrong* offsets, so `span_ok`'s strict
`slice(start,end) == quote` check drops the claim. The OUTPUT CONTRACT shown to
the model never asked for a `quote` — the teacher targets added it — so the model
was distilled onto a field that the production gate then uses to reject it.

### Fairness / serving-bug rule-out (raw completion sample, one chunk)
The 0.0006 recall is a **real model regression, not a serving-format bug**:
- The adapter server emits the exact JSON the production parser expects; the
  parser parses both models' output cleanly (tuned → 7 claims, stock → 8).
- Evidence field layout is identical **except** the tuned model adds `quote`:
  - TUNED evidence keys: `end_char, locator, quote, start_char, url` → span_ok **0/7 pass**
  - STOCK evidence keys: `end_char, locator, start_char, url` → span_ok **7/8 pass**
- Example (`bft-networking#c001`): tuned emits
  `{start_char:544, end_char:719, quote:"e every participant to send to every other partic…"}`
  — the quote is real text but `chunk[544:719]` is a *different* span, so the
  strict check fails. Stock emits `{start_char:2, end_char:34}` with **no**
  quote, so the gate only bounds-checks and passes.
- Stock passes this SAME gate at 0.99 gate_pass / 0.96 recall → the comparison is
  fair; the tuned adapter genuinely produces claims production would reject.

### Second defect: non-termination
On the tail docs the tuned model frequently fails to emit EOS and generates a
runaway ~4,145-token completion (8 of 44 batches hit `finish_reason=length`),
whereas stock e4b never truncated. This is an independent quality regression and
required a wall-clock generation cap in the server (see below) to keep the eval
from crashing on the 600 s client timeout.

## Next iteration — concrete options
1. **Retrain on a verbatim-quote-copy target (recommended, plays to the model's
   strength).** Change the SFT target so evidence carries the quoted span text
   (which a 4B copies reliably) and NOT emitted offsets; then have the gate
   *locate* the span with `str.find`/`verbatim_ok` and derive the offsets. This
   matches the model's copy capability instead of demanding offset-counting.
2. **Or change the production gate to derive quote-from-span and drop the
   emitted-offset/quote check** — i.e. keep the offset contract but score on
   `slice(start,end)` only (bounds-checked), ignoring any emitted `quote`. This
   is the smallest change and makes the tuned model's approximate-but-in-bounds
   offsets scorable, exactly as stock is scored today. (Weakens the near-quote
   bait defense the strict check provides — weigh against option 1.)
3. **Minimum data fix regardless of 1/2:** strip the non-contract `quote` (and
   `citable`) fields from the SFT targets so the model is distilled onto the
   published OUTPUT CONTRACT only.
4. Add an EOS/termination signal to training (and/or response-masked loss) to
   fix the runaway non-termination.

## Serving & infra notes (for reproduction)
- Merge to GGUF remains blocked in this env (see
  `runs/recipe-b-v1/MERGE_STATUS.txt`); serving the adapter via `FastModel`
  inference is the eval mechanism. Since the model did NOT promote, the GGUF
  export question is moot for now.
- `scripts/serve_adapter.py`: loads base+adapter via unsloth-native
  `FastModel.from_pretrained(adapter_dir)` (PEFT-wrap fallback; byte-identical
  greedy output), patches `config.architectures=["Gemma4ForCausalLM"]`,
  `FastModel.for_inference`, honors temperature/top_p/top_k/max_tokens, treats
  `reasoning_effort`/`think` as no-ops. Throughput ≈10 tok/s (4-bit Gemma-4,
  mandatory sdpa on Blackwell — flash-attn forbidden for head_dim=512).
- A 420 s wall-clock generation cap (StoppingCriteria, `finish_reason=length`)
  was added because the runaway generations otherwise exceeded the OpenAI
  client's 600 s read timeout and crashed the eval mid-run. The cap sits above
  the slowest legitimate batch (~349 s), so it only bites the tuned model's
  pathological non-termination.
