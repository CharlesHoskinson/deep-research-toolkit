#!/usr/bin/env python3
"""Minimal OpenAI-compatible /v1/chat/completions server wrapping an Unsloth
FastModel + a trained LoRA adapter, for the Recipe-B fine-tune eval gate.

Why this exists: the tuned Recipe-B model is a LoRA adapter on
`unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` that could NOT be merged in this env
(unsloth merged_16bit no-ops for a text_only-loaded multimodal Gemma-4, and the
PEFT merge_and_unload fallback trips transformers 5.5.4's un-reversible load-time
weight conversion -- see runs/recipe-b-v1/MERGE_STATUS.txt). vLLM LoRA serving
also no-ops for Gemma-4 (vllm#41754). So the only way to *run* the tuned model
for the A/B eval is to wrap `FastModel` inference (proven working by the training
smoke) behind the exact HTTP surface `LocalOpenAIBackend` (src/.../llm/local.py)
speaks: POST /v1/chat/completions with {model, messages:[system,user],
temperature, top_p, max_tokens} plus the extra_body fields {top_k, think,
reasoning_effort}. `think`/`reasoning_effort` are NO-OPS here (this model does
not emit a <think> trace); temperature/top_p/top_k/max_tokens are honored;
temperature==0 is greedy. Returns the OpenAI response shape
(choices[0].message.content + finish_reason, usage token counts).

Load-once at startup (mirrors runs/recipe-b-v1 training + smoke_sft.py exactly):
  FastModel.from_pretrained(base, load_in_4bit=True, attn_implementation="sdpa",
                            text_only=True)
  PeftModel.from_pretrained(model, adapter)      # attach the trained adapter
  model.config.architectures = ["Gemma4ForCausalLM"]   # unsloth VLM-vs-LM guard
  FastModel.for_inference(model)

Run (Studio venv):
  /root/gemma4-tune/unsloth_studio/bin/python scripts/serve_adapter.py \
      --host 0.0.0.0 --port 8099
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time

# Blackwell / Gemma-4 stack requirements (see scripts/preflight.py): no dynamo,
# sdpa attention only. Set before importing torch/unsloth.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

BASE_MODEL = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
DEFAULT_ADAPTER = "/root/gemma4-tune/runs/recipe-b-v1/adapter"
MAX_SEQ_LENGTH = 8192

# Wall-clock cap on a single generation. This 4-bit Gemma-4 decodes at ~10 tok/s
# on this Blackwell+sdpa stack, so an 8000-token request is ~13+ min and a
# NON-TERMINATING (runaway, no-EOS) generation would blow past the OpenAI
# client's 600s default read timeout and crash the eval mid-run. A time-based
# StoppingCriteria halts generation past this deadline and the call is reported
# with finish_reason="length" (a truncated_calls signal), exactly as a
# max_new_tokens truncation would be. Set safely BELOW the 600s client timeout
# and ABOVE the slowest legitimate extract batch observed (~349s), so it only
# ever bites pathological non-termination -- never a real extraction.
MAX_GEN_SECONDS = 420.0

# Serialize generate(): one model, not thread-safe under concurrent decodes.
_GEN_LOCK = threading.Lock()

_STATE: dict = {"model": None, "tokenizer": None, "ready": False, "adapter": None}


def load_model(adapter_path: str) -> None:
    import torch
    from unsloth import FastModel

    # Prefer unsloth's NATIVE adapter load (FastModel.from_pretrained pointed
    # straight at the saved adapter dir: it reads adapter_config.json, loads the
    # cached 4-bit base named there, and attaches the LoRA through unsloth's own
    # inference-optimized path). Vanilla `PeftModel.from_pretrained(base, dir)`
    # also works and gives byte-identical greedy output, but wrapping the base in
    # plain PEFT bypasses unsloth's fast-generate kernels -> ~10 tok/s vs the
    # native path's several-fold higher throughput. Fall back to the PEFT wrap if
    # the native load can't resolve the adapter.
    t0 = time.time()
    try:
        print(f"[serve] native load: FastModel.from_pretrained({adapter_path})...",
              flush=True)
        model, tokenizer = FastModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            attn_implementation="sdpa",  # MANDATORY for Gemma-4 head_dim=512 (preflight gate 1)
            text_only=True,
        )
        print(f"[serve] native base+adapter loaded in {time.time() - t0:.1f}s",
              flush=True)
    except Exception as e:  # noqa: BLE001 -- fall back to the explicit PEFT wrap
        from peft import PeftModel
        print(f"[serve] native load failed ({type(e).__name__}: {e}); "
              f"falling back to base + PeftModel.from_pretrained", flush=True)
        t0 = time.time()
        model, tokenizer = FastModel.from_pretrained(
            model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True,
            attn_implementation="sdpa", text_only=True,
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"[serve] base+adapter (PEFT fallback) loaded in {time.time() - t0:.1f}s",
              flush=True)

    # text_only's derived text-decoder config drops `architectures` (-> None);
    # unsloth's fast_generate iterates config.architectures to pick the VLM-vs-LM
    # path and TypeErrors on None. Restore the plain causal-LM class name.
    if getattr(model.config, "architectures", None) is None:
        model.config.architectures = ["Gemma4ForCausalLM"]

    FastModel.for_inference(model)
    model.config.use_cache = True

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    _STATE["model"] = model
    _STATE["tokenizer"] = tokenizer
    _STATE["adapter"] = adapter_path
    _STATE["ready"] = True
    print("[serve] model READY", flush=True)


def _generate(messages: list[dict], temperature: float, top_p: float,
              top_k: int, max_tokens: int) -> tuple[str, str, int, int]:
    """Returns (text, finish_reason, prompt_tokens, completion_tokens)."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _TimeLimit(StoppingCriteria):
        """Stop decoding once MAX_GEN_SECONDS elapse (checked per token). Bounds
        a runaway/non-terminating generation below the client read timeout."""
        def __init__(self, deadline: float) -> None:
            self.deadline = deadline
            self.hit = False

        def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001
            if time.time() > self.deadline:
                self.hit = True
                return True
            return False

    model = _STATE["model"]
    tokenizer = _STATE["tokenizer"]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    prompt_tokens = int(input_ids.shape[-1])

    time_limit = _TimeLimit(time.time() + MAX_GEN_SECONDS)
    greedy = temperature is None or float(temperature) <= 0.0
    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=int(max_tokens),
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=StoppingCriteriaList([time_limit]),
    )
    if greedy:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    else:
        gen_kwargs.update(
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p) if top_p is not None else 0.95,
            top_k=int(top_k) if top_k is not None else 20,
        )

    with _GEN_LOCK:
        with torch.no_grad():
            out = model.generate(**gen_kwargs)

    gen_ids = out[0][prompt_tokens:]
    completion_tokens = int(gen_ids.shape[-1])
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    # finish_reason "length" on either a max_new_tokens truncation OR a
    # wall-clock time-limit stop (runaway non-termination) -- both are the
    # truncated_calls signal the eval records.
    if completion_tokens >= int(max_tokens) or time_limit.hit:
        finish_reason = "length"
    else:
        finish_reason = "stop"
    return text, finish_reason, prompt_tokens, completion_tokens


def build_app(adapter_path: str):
    from fastapi import Body, FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="recipe-b-v1 adapter server")

    @app.get("/health")
    def health():
        return {"ready": _STATE["ready"], "adapter": _STATE["adapter"]}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": "recipe-b-v1", "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict = Body(...)):
        model_name = body.get("model", "recipe-b-v1")
        messages = body.get("messages") or []
        # Only role/content survive to the chat template.
        messages = [{"role": m.get("role"), "content": m.get("content") or ""}
                    for m in messages]
        temperature = body.get("temperature", 0.0)
        top_p = body.get("top_p", 0.95)
        # top_k rides in extra_body -> merged into the top-level JSON body by the
        # OpenAI client. reasoning_effort/think are accepted and ignored (this
        # model has no thinking pass).
        top_k = body.get("top_k", 20)
        max_tokens = body.get("max_tokens", 2048) or 2048

        if not _STATE["ready"]:
            return JSONResponse(status_code=503, content={
                "error": {"message": "model still loading", "type": "server_error"}})

        try:
            text, finish_reason, ptoks, ctoks = _generate(
                messages, temperature, top_p, top_k, max_tokens)
        except Exception as e:  # noqa: BLE001 -- surface as an OpenAI-shaped error
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={
                "error": {"message": f"{type(e).__name__}: {e}",
                          "type": "server_error"}})

        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": ptoks,
                "completion_tokens": ctoks,
                "total_tokens": ptoks + ctoks,
            },
        }

    return app


def main() -> None:
    global MAX_GEN_SECONDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--max-gen-seconds", type=float, default=MAX_GEN_SECONDS,
                        help="Wall-clock cap per generation (finish_reason=length on hit).")
    parser.add_argument("--self-smoke", action="store_true",
                        help="After load, run one extract-style generation and exit.")
    args = parser.parse_args()
    MAX_GEN_SECONDS = args.max_gen_seconds

    load_model(args.adapter)

    if args.self_smoke:
        sys_msg = ('Extract entity claims from the input as strict JSON: '
                   '{"claims": [{"text": str, "label": str}]}.')
        user_msg = "The Cardano Vasil hard fork went live on the mainnet in September 2022."
        text, fr, pt, ct = _generate(
            [{"role": "system", "content": sys_msg},
             {"role": "user", "content": user_msg}],
            temperature=0.0, top_p=0.95, top_k=20, max_tokens=256)
        print("[serve][self-smoke] finish_reason=%s prompt_tokens=%d completion_tokens=%d"
              % (fr, pt, ct), flush=True)
        print("[serve][self-smoke] output:\n" + text, flush=True)
        return

    import uvicorn
    app = build_app(args.adapter)
    print(f"[serve] serving on http://{args.host}:{args.port}/v1 "
          f"(adapter={args.adapter})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
