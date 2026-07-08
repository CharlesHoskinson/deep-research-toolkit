"""Standalone post-hoc merge for recipe-b-v1: loads base 4-bit + the saved
LoRA adapter and writes a 16-bit merged model, working around unsloth
save.py's `internal_model.model.layers` structure check that silently
no-ops `save_pretrained_merged(merged_16bit)` for a text_only-loaded Gemma-4
multimodal PEFT model. Diagnoses the module structure first, then merges via
PEFT `merge_and_unload()` on the dequantized model."""
from __future__ import annotations

import os

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import sys

sys.path.insert(0, "/root/gemma4-tune/scripts")
import preflight  # noqa: E402

preflight.run_all(attn_implementation="sdpa", use_torch_compile=False, triton_auto_clear=True)

ADAPTER_DIR = "/root/gemma4-tune/runs/recipe-b-v1/adapter"
MERGED_DIR = "/root/gemma4-tune/runs/recipe-b-v1/merged"
BASE_MODEL = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"

import torch  # noqa: E402
from unsloth import FastModel  # noqa: E402

# Load base + adapter (unsloth auto-reads adapter_config's base_model).
model, tokenizer = FastModel.from_pretrained(
    model_name=ADAPTER_DIR,
    max_seq_length=4096,
    load_in_4bit=True,
    attn_implementation="sdpa",
    text_only=True,
)

# --- diagnose the actual decoder-layer path ---
def _find_layers(m, prefix="model", depth=0):
    if depth > 6:
        return
    for name in ("layers", "language_model", "model", "text_model"):
        sub = getattr(m, name, None)
        if sub is not None:
            print(f"  {prefix}.{name}: {type(sub).__name__}")
            if name == "layers":
                return
            _find_layers(sub, f"{prefix}.{name}", depth + 1)

print("[merge] module structure probe:")
_find_layers(model)

# --- try unsloth merged_16bit first (may no-op) ---
before = os.path.isdir(MERGED_DIR) and len(os.listdir(MERGED_DIR)) > 0
try:
    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
except Exception as e:
    print(f"[merge] save_pretrained_merged raised: {type(e).__name__}: {e}")

has_weights = os.path.isdir(MERGED_DIR) and any(
    f.endswith(".safetensors") for f in os.listdir(MERGED_DIR)
)
if has_weights:
    print(f"[merge] unsloth merged_16bit wrote weights to {MERGED_DIR}")
    print("[merge] MERGE OK (unsloth path)")
    sys.exit(0)

print("[merge] unsloth merged_16bit produced no weights -- falling back to PEFT merge_and_unload")

# --- PEFT fallback: dequantize + merge + save 16-bit ---
from peft import PeftModel  # noqa: E402

# model is already a PeftModel wrapping the 4-bit base. merge_and_unload on a
# bnb-4bit base dequantizes the base, folds the LoRA delta, and returns a
# plain fp16/bf16 module.
merged = model.merge_and_unload()
os.makedirs(MERGED_DIR, exist_ok=True)
merged.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="5GB")
tokenizer.save_pretrained(MERGED_DIR)

has_weights = any(f.endswith(".safetensors") for f in os.listdir(MERGED_DIR))
print(f"[merge] PEFT merge wrote weights: {has_weights}")
print("[merge] files:", sorted(os.listdir(MERGED_DIR))[:12])
print("[merge] MERGE OK (peft path)" if has_weights else "[merge] MERGE FAILED")
sys.exit(0 if has_weights else 1)
