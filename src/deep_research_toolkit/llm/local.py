from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LocalLLMNotInstalled(RuntimeError):
    pass


def strip_think(text: str) -> str:
    """Remove a reasoning model's <think>...</think> trace.

    Ornith-style reasoning models emit a think block before the answer. With a
    chat template that primes the assistant turn with `<think>`, the opening tag
    is in the prompt and only the closing `</think>` comes back in the content,
    so match on the closing tag: take whatever follows the final `</think>`,
    then also drop any complete inline pairs.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return _THINK_RE.sub("", text).strip()


class LocalOpenAIBackend:
    """Talks to an OpenAI-compatible endpoint (Ollama :11434/v1, vLLM
    :8000/v1) serving a local model such as Ornith-1.0-9B."""

    def __init__(self, base_url: str, model: str, api_key: str,
                 temperature: float, top_p: float, top_k: int,
                 max_tokens: int = 16000, thinking: bool = True,
                 response_format: str | dict | None = None,
                 trace_path: Path | None = None, role: str | None = None) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        # Reasoning models spend thousands of tokens in <think> before answering;
        # a low cap truncates the reasoning and the model never reaches its
        # output. Budget generously for a thinking role, tightly for extract.
        self.max_tokens = max_tokens
        # thinking=False disables a Qwen/reasoning model's <think> pass (Ollama
        # `think` control) -- right for high-volume, well-specified extraction,
        # where a hidden deliberation pass is wasted time.
        self.thinking = thinking
        # "json" -> OpenAI json_object mode (grammar-constrained valid JSON); a
        # dict is passed through as a full response_format (e.g. json_schema).
        self.response_format = response_format
        # Opt-in per-call JSONL ledger (OTel GenAI field names, no OTel
        # dependency) -- None disables tracing entirely. `role` is the pipeline
        # phase (e.g. "extract"), recorded alongside each line for join-back.
        self.trace_path = trace_path
        self.role = role
        self._client = None
        # Cumulative usage over this backend's lifetime; read by the eval
        # harness to report cost/latency per model. Never reset internally.
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def _load_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise LocalLLMNotInstalled(
                    "The local LLM backend needs an OpenAI-compatible client. "
                    'Install it with: pip install "deep-research-toolkit[compiler]" '
                    "(or: pip install openai)."
                ) from e
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _client_complete(self, system: str, user: str, **kw):
        client = self._load_client()
        think = kw.get("thinking", self.thinking)
        extra_body = {"top_k": kw.get("top_k", self.top_k), "think": think}
        if not think:
            # Ollama's OpenAI-compatible endpoint ignores `think: false` for
            # some model families (Gemma 4: ollama/ollama#15288) but honors
            # `reasoning_effort: "none"`. Send both; stacks that ignore one
            # respect the other, and vLLM disables Gemma 4 thinking by default.
            extra_body["reasoning_effort"] = "none"
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=kw.get("temperature", self.temperature),
            top_p=kw.get("top_p", self.top_p),
            max_tokens=kw.get("max_tokens", self.max_tokens),
            extra_body=extra_body,
        )
        rf = kw.get("response_format", self.response_format)
        if rf == "json":
            kwargs["response_format"] = {"type": "json_object"}
        elif isinstance(rf, dict):
            kwargs["response_format"] = rf
        return client.chat.completions.create(**kwargs)

    def complete(self, system: str, user: str, **sampling) -> str:
        import time
        t0 = time.perf_counter()
        resp = self._client_complete(system, user, **sampling)
        elapsed = time.perf_counter() - t0
        self.stats["calls"] += 1
        self.stats["seconds"] += elapsed
        usage = getattr(resp, "usage", None)
        prompt_tokens = 0
        completion_tokens = 0
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            self.stats["prompt_tokens"] += prompt_tokens
            self.stats["completion_tokens"] += completion_tokens
        content = strip_think(resp.choices[0].message.content or "")
        if self.trace_path is not None:
            self._write_trace(elapsed, prompt_tokens, completion_tokens, content)
        return content

    def _write_trace(self, elapsed: float, prompt_tokens: int,
                     completion_tokens: int, content: str) -> None:
        """Append one JSONL ledger line for a completed call. A trace failure
        must never break a call, so every error is swallowed."""
        try:
            row = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "gen_ai.request.model": self.model,
                "gen_ai.usage.input_tokens": prompt_tokens,
                "gen_ai.usage.output_tokens": completion_tokens,
                "latency_s": elapsed,
                "role": self.role,
                "ok": bool(content),
            }
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:  # noqa: BLE001 -- tracing is best-effort by contract
            pass
