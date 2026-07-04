from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LocalLLMNotInstalled(RuntimeError):
    pass


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


class LocalOpenAIBackend:
    """Talks to an OpenAI-compatible endpoint (Ollama :11434/v1, vLLM
    :8000/v1) serving a local model such as Ornith-1.0-9B."""

    def __init__(self, base_url: str, model: str, api_key: str,
                 temperature: float, top_p: float, top_k: int) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
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
            extra_body={"top_k": kw.get("top_k", self.top_k)},
        )

    def complete(self, system: str, user: str, **sampling) -> str:
        resp = self._client_complete(system, user, **sampling)
        return strip_think(resp.choices[0].message.content or "")
