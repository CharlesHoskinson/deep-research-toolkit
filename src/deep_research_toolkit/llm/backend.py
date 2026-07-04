from __future__ import annotations

from typing import Protocol, runtime_checkable


class LLMBackendNotConfigured(RuntimeError):
    pass


@runtime_checkable
class Backend(Protocol):
    def complete(self, system: str, user: str, **sampling) -> str: ...


def get_backend(config) -> Backend:
    provider = getattr(config, "llm_provider", "agent")
    if provider in ("agent", "anthropic"):
        from .agent import AgentBackend
        return AgentBackend()
    if provider == "local":
        import os

        from .local import LocalOpenAIBackend
        local = config.llm_local
        return LocalOpenAIBackend(
            base_url=local["base_url"], model=local["model"],
            api_key=os.environ.get(local.get("api_key_env", "OPENAI_API_KEY"), "not-needed"),
            temperature=local["temperature"], top_p=local["top_p"], top_k=local["top_k"],
        )
    raise LLMBackendNotConfigured(
        f"unknown llm.provider: {provider!r} (use agent | anthropic | local)"
    )
