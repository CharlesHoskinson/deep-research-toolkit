from __future__ import annotations

import re

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
                 max_tokens: int = 16000) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        # Reasoning models spend thousands of tokens in <think> before answering;
        # a low cap truncates the reasoning and the model never reaches its
        # output. Budget generously (Ornith field guidance: >=8K).
        self.max_tokens = max_tokens
        self._client = None

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
        return client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=kw.get("temperature", self.temperature),
            top_p=kw.get("top_p", self.top_p),
            max_tokens=kw.get("max_tokens", self.max_tokens),
            extra_body={"top_k": kw.get("top_k", self.top_k)},
        )

    def complete(self, system: str, user: str, **sampling) -> str:
        resp = self._client_complete(system, user, **sampling)
        return strip_think(resp.choices[0].message.content or "")
