from __future__ import annotations

from .backend import LLMBackendNotConfigured


class AgentBackend:
    """Default backend: the judgment steps are done by the in-session agent
    reading files per SKILL.md (ADR 0001 decision #4). There is no
    programmatic model to call here -- invoking complete() is a usage error."""

    def complete(self, system: str, user: str, **sampling) -> str:
        raise LLMBackendNotConfigured(
            "llm.provider is 'agent': extraction/synthesis is done by the in-session agent "
            "following the skill's SKILL.md, not by a programmatic call. To automate it with a "
            "local model instead, set 'provider: local' in .deepresearch.yml and run a local "
            "OpenAI-compatible endpoint (e.g. Ollama serving Ornith-1.0-9B)."
        )
