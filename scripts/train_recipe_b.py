#!/usr/bin/env python3
"""Recipe-B SFT training run (design doc §6.2, canonical copy -- the file
actually EXECUTED lives at /root/gemma4-tune/scripts/train_recipe_b.py in the
WSL Unsloth-Studio environment; this repo copy is the version-controlled
source of truth, kept byte-identical to what was run).

Trains a QLoRA adapter on `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` against
the assembled `sft-final/{train,val}.jsonl` (see `scripts/assemble-sft.py`),
incorporating all five mandatory fixes validated by the prior GPU smoke test
(`/root/gemma4-tune/scripts/smoke_sft.py`, `runs/smoke-001/run.json`):

1. `FastModel.from_pretrained(..., text_only=True, return_logits=True)` --
   `text_only=True` skips the audio/vision towers (this is a text-only span-
   extraction target) and makes unsloth return a plain tokenizer instead of a
   multimodal `Processor`, so TRL's `SFTTrainer` takes the standard language-
   modeling path (`self._is_vlm=False`) instead of the vision-language one.
   `return_logits=True` counters unsloth's default fused-loss patch that
   replaces `outputs.logits` with a callable placeholder, which breaks TRL
   0.23.1's per-token-entropy training metric (reads `.logits.shape`
   unconditionally). **Eval-boundary reassertion (found live in the first
   full run):** `return_logits=True` only sets `UNSLOTH_RETURN_LOGITS=1` at
   LOAD; the compiled Gemma-4 forward re-reads that env var on EVERY pass
   (`unsloth_zoo/compiler.py`: `logits = lm_head(h) if
   os.environ['UNSLOTH_RETURN_LOGITS']=='1' else EMPTY_LOGITS`). Running the
   per-epoch validation calls unsloth's `for_inference`/`for_training`
   toggling (`unsloth/models/vision.py` 2112/2180), and `for_training` sets
   the var back to `"0"` on the way out of eval -- so epoch 1 trains fine,
   the epoch-1 eval succeeds, then epoch 2's first forward returns the
   `EMPTY_LOGITS` placeholder and TRL's `entropy_from_logits(outputs.logits)`
   dies with `TypeError: 'function' object is not subscriptable`. The smoke
   test never caught this (no eval). Fix: a `TrainerCallback`
   (`_KeepReturnLogits` below) re-asserts `UNSLOTH_RETURN_LOGITS="1"` at
   `on_step_begin`/`on_epoch_begin`/`on_evaluate`, the tightest hooks before
   each forward -- unsloth's own documented env-var contract
   (`unsloth/models/_utils.py`: "set UNSLOTH_RETURN_LOGITS=1 before
   trainer.train()"), overriding the post-eval clobber. Real-logits training
   is memory-safe at bs1 (epoch 1 ran the whole way at 24 GB peak).
2. `SFTConfig(remove_unused_columns=False)` -- keeps the dataset's
   non-signature columns (this run passes bare `{"messages": [...]}` rows,
   so mostly moot here, but mandated by the smoke test regardless).
3. `model.config.architectures = ["Gemma4ForCausalLM"]` patched before any
   `generate()` call -- `text_only=True`'s derived text-decoder config loses
   its `architectures` list (`None`), and unsloth's own
   `unsloth_base_fast_generate()` unconditionally iterates
   `self.config.architectures` to pick the VLM-vs-LM generate path, which
   TypeErrors on `None`.
4. `HF_HUB_DISABLE_XET=1` -- set before any HF import (also set by
   `unsloth_zoo/__init__.py` internally, and moot here since the run is
   fully offline against the already-cached checkpoint, but set explicitly
   per the validated smoke-test/fetch-script precedent).
5. Base model `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` (pre-quantized
   4-bit, Studio-validated for Blackwell sm_120).

## BOS handling (design doc §6.2: "remove <bos> in data prep -- the Gemma
processor re-adds it")

Investigated directly against this Studio install's `trl==0.23.1`
(`trl/trainer/sft_trainer.py`): a dataset of PRE-RENDERED "text" strings
(`tokenizer.apply_chat_template(messages, tokenize=False)`, the smoke
test's own approach) hits `_prepare_dataset`'s NON-conversational branch,
which (a) blindly appends the literal string `tokenizer.eos_token`
(`"<eos>"`) to every example -- wrong for Gemma 4's actual template, whose
real turn-close marker is `<turn|>`, not `<eos>` -- and (b) re-tokenizes
that string with `add_special_tokens=True` (default), DOUBLE-ADDING the
`<bos>` the chat template already emitted (`{{- bos_token -}}` in
`chat_template.jinja`). This is exactly the gotcha the design doc's note
warns about, and it is why a naive copy of the smoke script's per-string
`text` field for real training would be wrong.

The correct-by-construction fix (verified against `trl`'s own source, not
guessed): pass raw `{"messages": [...]}` records instead of pre-rendered
text. This makes `SFTTrainer._prepare_dataset` take the CONVERSATIONAL
branch instead (`is_conversational(example) == True`), which calls
`processing_class.apply_chat_template(example["messages"], return_dict=True,
...)` directly -- HF's own canonical chat-template tokenization path, which
emits the template's one `<bos>` exactly once and never appends a manual
`<eos>` string. This IS "removing <bos> from data prep": we never render or
handle it ourselves at all, so there is nothing to double. `train_on_responses_only`
(loss masking, below) then finds `input_ids` already present on the dataset
and skips its own (naive, `add_special_tokens=True`) fallback tokenizer,
which would otherwise reintroduce the same double-bos bug.

## Assistant-only loss masking

Gemma 4's chat template (verified directly, `chat_template.jinja`) uses
`<|turn>ROLE\n ... <turn|>\n` markers, NOT the Gemma-2/3
`<start_of_turn>ROLE\n...<end_of_turn>\n` convention. `instruction_part`/
`response_part` for `unsloth_zoo.dataset_utils.train_on_responses_only` are
set explicitly below to `"<|turn>user\n"` / `"<|turn>model\n"` (not
auto-detected) to match.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

# ---------------------------------------------------------------------------
# Env vars: MUST be set before any heavy import (fix #4).
# ---------------------------------------------------------------------------
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, "/root/gemma4-tune/scripts")
import preflight  # noqa: E402

STUDIO_HOME = "/root/gemma4-tune"
DATA_DIR = os.path.join(STUDIO_HOME, "data", "sft-final")
RUN_DIR = os.path.join(STUDIO_HOME, "runs", "recipe-b-v1")
ADAPTER_DIR = os.path.join(RUN_DIR, "adapter")
MERGED_DIR = os.path.join(RUN_DIR, "merged")
RUN_JSON_PATH = os.path.join(RUN_DIR, "run.json")

BASE_MODEL = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 4096  # design doc §6.2: "sized to the data"; measured longest
                       # rendered record across the 802-record sft-final pool is
                       # 2963 tokens (see scratch-recipe-b/measure_lengths.py) --
                       # 4096 covers every record with margin, zero truncation
                       # (2048 would truncate the assistant JSON on 62/802 = 7.7%
                       # of records, corrupting the training target).
SEED = 42

# Gemma-4's own chat template (verified in chat_template.jinja, NOT the
# gemma-2/3 <start_of_turn> convention):
INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"

LORA_CONFIG = dict(r=16, lora_alpha=32, lora_dropout=0.1, target_modules="all-linear", bias="none")
SFT_HPARAMS = dict(
    # eff-batch 16 held EXACTLY (design doc §6.2), but as bs1 x ga16 rather than
    # the doc's nominal bs2 x ga8. Reason (found live, dry_run2 OOM at step 2):
    # the mandatory `return_logits=True` smoke fix forces the full
    # [bsz x seqlen x vocab=262144] logits tensor to be materialized (it defeats
    # unsloth's memory-efficient fused-loss path, which is exactly why TRL
    # 0.23.1's compute_loss can read .logits.shape). The GPU smoke never hit
    # this because its synthetic sequences were ~100 tokens; the REAL sft-final
    # records run up to 2963 tokens, so at bs2 x 4096 the logits tensor alone is
    # ~4.3 GB and the fused-CE workspace + activations blow the 32 GB budget
    # (31957/32607 MiB, "No or negligible GPU memory available for fused cross
    # entropy"). bs1 halves per-step activation/logits memory and, with a
    # per-example (not per-batch-max) pad, keeps short bait/general records
    # cheap. Effective batch size, LR, epochs, and every other §6.2 default are
    # unchanged -- this is a data-shape fit, not a recipe change.
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    learning_rate=2e-4,
    optim="adamw_8bit",
    weight_decay=0.01,
    seed=SEED,
)

os.makedirs(RUN_DIR, exist_ok=True)


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _config_sha256(config: dict) -> str:
    import hashlib
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> int:
    t_start = time.time()

    # -----------------------------------------------------------------
    # Preflight (hard gates) -- BEFORE any model/trainer construction.
    # -----------------------------------------------------------------
    version_snapshot = preflight.run_all(
        attn_implementation="sdpa", use_torch_compile=False, triton_auto_clear=True,
    )
    print("[train_recipe_b] preflight gates passed")

    random.seed(SEED)

    # -----------------------------------------------------------------
    # Provenance: dataset_hash from the assembled manifest, config_sha256
    # over the resolved recipe config, env snapshot, git commit of the
    # assembling repo -- written to run.json BEFORE any weights exist.
    # -----------------------------------------------------------------
    manifest = json.loads(open(os.path.join(DATA_DIR, "manifest.json"), encoding="utf-8").read())
    dataset_hash = manifest["dataset_hash"]

    git_commit = None
    git_commit_path = os.path.join(DATA_DIR, "GIT_COMMIT.txt")
    if os.path.isfile(git_commit_path):
        git_commit = open(git_commit_path, encoding="utf-8").read().strip()

    env_snapshot_path = os.path.join(STUDIO_HOME, "provenance", "env_snapshot.json")
    env_snapshot = json.loads(open(env_snapshot_path, encoding="utf-8").read()) if os.path.isfile(env_snapshot_path) else None

    resolved_config = {
        "base_model": BASE_MODEL,
        "max_seq_length": MAX_SEQ_LENGTH,
        "load_in_4bit": True,
        "attn_implementation": "sdpa",
        "text_only": True,
        "return_logits": True,
        "lora": LORA_CONFIG,
        "sft": SFT_HPARAMS,
        "instruction_part": INSTRUCTION_PART,
        "response_part": RESPONSE_PART,
        "dataset_hash": dataset_hash,
        "seed": SEED,
    }
    config_sha256 = _config_sha256(resolved_config)

    run_record: dict = {
        "run_id": "recipe-b-v1",
        "recipe_id": "recipe-b",
        "purpose": "Recipe-B QLoRA SFT on gemma-4-E4B-it against the assembled sft-final dataset",
        "config_sha256": config_sha256,
        "dataset_hash": dataset_hash,
        "seed": SEED,
        "git_commit": git_commit,
        "config": resolved_config,
        "preflight_version_snapshot": version_snapshot,
        "env_snapshot": env_snapshot,
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(RUN_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2, ensure_ascii=False)
    print(f"[train_recipe_b] wrote provenance run.json (pre-weights) to {RUN_JSON_PATH}")

    # -----------------------------------------------------------------
    # Data: raw {"messages": [...]} records only -- see module docstring
    # for why we do NOT pre-render to a "text" field ourselves.
    # -----------------------------------------------------------------
    train_records_raw = _read_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val_records_raw = _read_jsonl(os.path.join(DATA_DIR, "val.jsonl"))
    print(f"[train_recipe_b] loaded {len(train_records_raw)} train / {len(val_records_raw)} val records")

    import torch  # noqa: E402
    from datasets import Dataset  # noqa: E402
    from transformers import EarlyStoppingCallback, TrainerCallback  # noqa: E402
    from trl import SFTConfig, SFTTrainer  # noqa: E402
    from unsloth import FastModel  # noqa: E402
    from unsloth_zoo.dataset_utils import train_on_responses_only  # noqa: E402

    class _KeepReturnLogits(TrainerCallback):
        """Re-assert UNSLOTH_RETURN_LOGITS=1 before every training step/epoch
        and after every eval -- see module docstring, fix #1. unsloth's
        for_training (run when the trainer exits an evaluation loop) sets it
        to "0", which makes the compiled forward return the EMPTY_LOGITS
        placeholder and crashes TRL's entropy_from_logits on the next
        training forward. These hooks fire immediately before the forward
        reads the env var, so they are the last word."""

        def on_step_begin(self, args, state, control, **kwargs):
            os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

        def on_epoch_begin(self, args, state, control, **kwargs):
            os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

        def on_evaluate(self, args, state, control, **kwargs):
            os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    train_ds = Dataset.from_list([{"messages": r["messages"]} for r in train_records_raw])
    val_ds = Dataset.from_list([{"messages": r["messages"]} for r in val_records_raw])

    torch.cuda.reset_peak_memory_stats()
    timings: dict = {}

    # -----------------------------------------------------------------
    # 1. Load model + tokenizer (fixes #1, #4, #5)
    # -----------------------------------------------------------------
    t0 = time.time()
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        attn_implementation="sdpa",
        random_state=SEED,
        text_only=True,
        return_logits=True,
    )
    timings["load_model_s"] = time.time() - t0

    # -----------------------------------------------------------------
    # 2. Attach LoRA (design doc §6.2 recipe defaults)
    # -----------------------------------------------------------------
    t0 = time.time()
    model = FastModel.get_peft_model(
        model,
        r=LORA_CONFIG["r"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    timings["get_peft_model_s"] = time.time() - t0

    # -----------------------------------------------------------------
    # 3. SFTTrainer + assistant-only-loss masking
    # -----------------------------------------------------------------
    # RECIPE_B_MAX_STEPS: dev/dry-run escape hatch only (unset for the real
    # recipe run) -- caps steps instead of training the full 2 epochs, so the
    # new-vs-smoke-test mechanics (conversational "messages" dataset,
    # train_on_responses_only marker matching, eval/save/early-stop wiring)
    # can be validated end-to-end in a couple of minutes before committing to
    # the full run.
    _dry_max_steps = os.environ.get("RECIPE_B_MAX_STEPS")
    sft_kwargs = dict(
        per_device_train_batch_size=SFT_HPARAMS["per_device_train_batch_size"],
        per_device_eval_batch_size=SFT_HPARAMS["per_device_eval_batch_size"],
        gradient_accumulation_steps=SFT_HPARAMS["gradient_accumulation_steps"],
        num_train_epochs=SFT_HPARAMS["num_train_epochs"],
        learning_rate=SFT_HPARAMS["learning_rate"],
        optim=SFT_HPARAMS["optim"],
        weight_decay=SFT_HPARAMS["weight_decay"],
        seed=SFT_HPARAMS["seed"],
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        logging_steps=1,
        eval_strategy="epoch" if not _dry_max_steps else "steps",
        eval_steps=None if not _dry_max_steps else max(1, int(_dry_max_steps) // 2),
        save_strategy="epoch" if not _dry_max_steps else "steps",
        save_steps=None if not _dry_max_steps else max(1, int(_dry_max_steps) // 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        output_dir=os.path.join(RUN_DIR, "trainer_output"),
        report_to=[],
        remove_unused_columns=False,  # fix #2
    )
    if _dry_max_steps:
        sft_kwargs["max_steps"] = int(_dry_max_steps)
    sft_config = SFTConfig(**sft_kwargs)

    n_train_before = len(train_ds)
    n_val_before = len(val_ds)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1), _KeepReturnLogits()],
    )
    trainer = train_on_responses_only(
        trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART,
    )

    # SFTTrainer's own conversational _prepare_dataset tokenizes "messages" into
    # "input_ids"/"labels" but does NOT drop the original "messages" column
    # (dataset.map keeps source columns unless told to remove them). With
    # remove_unused_columns=False (mandatory fix #2) that raw column survives
    # into the DataCollator, which then tries to tensorize EVERY feature key
    # per batch -- "messages" is a list of role/content dicts, not tensorizable
    # ("RuntimeError: Could not infer dtype of dict"), and training crashes at
    # the very first batch. Confirmed live in a 4-step dry run before the real
    # run. Fix: drop the now-redundant raw column ourselves (input_ids/labels
    # already carry everything the collator needs) -- remove_unused_columns
    # stays False as mandated, this is a targeted removal of one column, not a
    # blanket re-enable.
    for attr in ("train_dataset", "eval_dataset"):
        ds = getattr(trainer, attr)
        if "messages" in ds.column_names:
            setattr(trainer, attr, ds.remove_columns("messages"))

    n_train_after = len(trainer.train_dataset)
    n_val_after = len(trainer.eval_dataset)
    if n_train_after < n_train_before or n_val_after < n_val_before:
        print(
            f"[train_recipe_b] WARNING: train_on_responses_only dropped rows with no "
            f"response marker found (train {n_train_before}->{n_train_after}, "
            f"val {n_val_before}->{n_val_after}) -- check instruction_part/response_part "
            "against the chat template."
        )
    else:
        print(f"[train_recipe_b] response-marker masking OK: 0 rows dropped "
              f"(train={n_train_after}, val={n_val_after})")

    # -----------------------------------------------------------------
    # 4. Train
    # -----------------------------------------------------------------
    t0 = time.time()
    trainer.train()
    timings["train_s"] = time.time() - t0

    vram_peak_bytes = torch.cuda.max_memory_allocated()
    vram_peak_gib = vram_peak_bytes / (1024 ** 3)

    train_loss_curve = [
        {"step": e["step"], "epoch": e.get("epoch"), "loss": e["loss"]}
        for e in trainer.state.log_history if "loss" in e
    ]
    eval_loss_curve = [
        {"step": e["step"], "epoch": e.get("epoch"), "eval_loss": e["eval_loss"]}
        for e in trainer.state.log_history if "eval_loss" in e
    ]
    print("[train_recipe_b] train loss curve (last 10):", train_loss_curve[-10:])
    print("[train_recipe_b] eval loss curve (per epoch):", eval_loss_curve)
    print(f"[train_recipe_b] VRAM peak: {vram_peak_gib:.3f} GiB")

    # Interim save (results so far) before the hard asserts below, so a
    # failed assert still leaves a useful record on disk.
    run_record["timings_s"] = timings
    run_record["vram_peak_gib"] = vram_peak_gib
    run_record["vram_peak_bytes"] = vram_peak_bytes
    run_record["train_loss_curve"] = train_loss_curve
    run_record["eval_loss_curve"] = eval_loss_curve
    run_record["response_marker_check"] = {
        "n_train_before": n_train_before, "n_train_after": n_train_after,
        "n_val_before": n_val_before, "n_val_after": n_val_after,
    }
    with open(RUN_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------
    # 5. Assertions: train loss fell, val loss didn't diverge
    # -----------------------------------------------------------------
    assert len(train_loss_curve) >= 2, f"expected >=2 logged train loss points, got {train_loss_curve}"
    for entry in train_loss_curve:
        assert entry["loss"] == entry["loss"], f"NaN train loss: {entry}"
        assert entry["loss"] not in (float("inf"), float("-inf")), f"inf train loss: {entry}"
    loss_early, loss_final = train_loss_curve[0]["loss"], train_loss_curve[-1]["loss"]
    assert loss_final < loss_early, f"train loss did not fall: {loss_early} -> {loss_final}"
    print(f"[train_recipe_b] train loss check OK: {loss_early:.4f} -> {loss_final:.4f}")

    early_stopped = trainer.state.global_step < trainer.state.max_steps
    val_diverged = False
    if len(eval_loss_curve) >= 2:
        val_diverged = eval_loss_curve[-1]["eval_loss"] > eval_loss_curve[0]["eval_loss"] * 1.10
        assert not val_diverged, (
            f"val loss diverged: epoch losses = {[e['eval_loss'] for e in eval_loss_curve]}"
        )
    print(f"[train_recipe_b] val loss check OK: {[e['eval_loss'] for e in eval_loss_curve]} "
          f"(early_stopped={early_stopped})")

    # -----------------------------------------------------------------
    # 6. Save adapter + merged model
    # -----------------------------------------------------------------
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"[train_recipe_b] adapter saved to {ADAPTER_DIR}")

    _skip_merge = os.environ.get("RECIPE_B_SKIP_MERGE_AND_GEN") == "1"
    if not _skip_merge:
        t0 = time.time()
        model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
        timings["save_merged_s"] = time.time() - t0
        print(f"[train_recipe_b] merged model saved to {MERGED_DIR}")

    # -----------------------------------------------------------------
    # 7. Cache-enabled generation smoke on 3 held-out (val) chunks (fix #3)
    # -----------------------------------------------------------------
    FastModel.for_inference(model)
    model.config.use_cache = True
    if getattr(model.config, "architectures", None) is None:
        model.config.architectures = ["Gemma4ForCausalLM"]

    gen_samples = []
    _gen_records = [] if _skip_merge else val_records_raw[:3]
    for rec in _gen_records:
        gen_messages = [m for m in rec["messages"] if m["role"] != "assistant"]
        input_ids = tokenizer.apply_chat_template(
            gen_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids, max_new_tokens=512, use_cache=True,
                do_sample=False, temperature=None, top_p=None, top_k=None,
            )
        gen_s = time.time() - t0
        gen_text = tokenizer.decode(gen_out[0][input_ids.shape[-1]:], skip_special_tokens=True)
        reference = next((m["content"] for m in rec["messages"] if m["role"] == "assistant"), None)
        gen_samples.append({
            "locator": rec.get("locator"), "component": rec.get("component"),
            "generated": gen_text, "reference": reference, "gen_s": gen_s,
        })
        print(f"[train_recipe_b] generation sample ({rec.get('locator')}, {rec.get('component')}):")
        print(gen_text[:800])
        print("--- reference ---")
        print((reference or "")[:800])
        print()

    total_wall_s = time.time() - t_start

    run_record["early_stopped"] = bool(early_stopped)
    run_record["val_diverged"] = bool(val_diverged)
    run_record["adapter_path"] = ADAPTER_DIR
    run_record["merged_path"] = MERGED_DIR
    run_record["generation_samples"] = gen_samples
    run_record["timings_s"] = timings
    run_record["total_wall_s"] = total_wall_s
    run_record["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with open(RUN_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2, ensure_ascii=False)

    print(f"[train_recipe_b] TOTAL WALL TIME: {total_wall_s:.1f}s")
    print("[train_recipe_b] RECIPE-B TRAINING RUN COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
