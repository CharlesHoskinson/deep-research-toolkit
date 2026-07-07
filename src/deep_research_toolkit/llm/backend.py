from __future__ import annotations

from typing import Protocol, runtime_checkable


class LLMBackendNotConfigured(RuntimeError):
    pass


@runtime_checkable
class Backend(Protocol):
    def complete(self, system: str, user: str, **sampling) -> str: ...


def get_backend(config, role: str | None = None) -> Backend:
    """Resolve the backend for a pipeline phase.

    `role` selects a per-phase model from `config.llm_roles` (e.g. "extract",
    "synthesize"); pass None for the flat single-model config. Under the local
    provider this is what routes high-volume extraction to a fast non-thinking
    model and one-shot synthesis to a reasoning model.
    """
    provider = getattr(config, "llm_provider", "agent")
    if provider in ("agent", "anthropic"):
        from .agent import AgentBackend
        return AgentBackend()
    if provider == "local":
        import os
        from pathlib import Path

        from .local import LocalOpenAIBackend
        roles = getattr(config, "llm_roles", None) or {}
        spec = roles[role] if role and role in roles else config.llm_local
        # Intentionally CWD-relative: one global ledger per invocation
        # directory, unlike the config paths resolved against project_root.
        trace_path = Path("llm-trace.jsonl") if getattr(config, "llm_trace", False) else None
        backend = LocalOpenAIBackend(
            base_url=spec["base_url"], model=spec["model"],
            api_key=os.environ.get(spec.get("api_key_env", "OPENAI_API_KEY"), "not-needed"),
            temperature=spec["temperature"], top_p=spec["top_p"], top_k=spec["top_k"],
            max_tokens=spec.get("max_tokens", 16000),
            thinking=spec.get("thinking", True),
            response_format=spec.get("response_format"),
            trace_path=trace_path,
            role=role,
        )
        if getattr(config, "llm_cache", False):
            from .cache import CachingBackend
            index_dir = Path(getattr(config, "index_dir", None) or ".deepresearch/index")
            return CachingBackend(backend, cache_dir=index_dir.parent / "llm-cache",
                                  enabled=True, role=role or "")
        return backend
    raise LLMBackendNotConfigured(
        f"unknown llm.provider: {provider!r} (use agent | anthropic | local)"
    )
