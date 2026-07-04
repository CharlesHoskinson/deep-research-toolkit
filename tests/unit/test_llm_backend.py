from types import SimpleNamespace

import pytest

from deep_research_toolkit.llm.agent import AgentBackend
from deep_research_toolkit.llm.backend import LLMBackendNotConfigured, get_backend
from deep_research_toolkit.llm.local import LocalOpenAIBackend, strip_think


def _cfg(provider):
    return SimpleNamespace(llm_provider=provider, llm_local={
        "base_url": "http://localhost:11434/v1", "model": "Ornith-1.0-9B",
        "api_key_env": "OPENAI_API_KEY", "temperature": 0.6, "top_p": 0.95, "top_k": 20})


def test_agent_backend_is_default_and_refuses_programmatic_call():
    backend = get_backend(_cfg("anthropic"))
    assert isinstance(backend, AgentBackend)
    with pytest.raises(LLMBackendNotConfigured) as exc:
        backend.complete("sys", "user")
    assert "provider: local" in str(exc.value)


def test_local_provider_selects_local_backend():
    assert isinstance(get_backend(_cfg("local")), LocalOpenAIBackend)


def test_strip_think_removes_reasoning_blocks():
    assert strip_think("<think>reasoning here</think>\nFinal answer.") == "Final answer."
    assert strip_think("no think tags") == "no think tags"


def test_strip_think_handles_primed_close_only():
    # A chat template that primes the assistant turn with <think> means only the
    # closing </think> comes back in the content (no opening tag).
    assert strip_think("reasoning with no open tag\n</think>\nFinal answer.") == "Final answer."


def test_get_backend_local_reads_max_tokens_from_config():
    cfg = _cfg("local")
    cfg.llm_local["max_tokens"] = 9999
    backend = get_backend(cfg)
    assert backend.max_tokens == 9999


def test_local_backend_passes_max_tokens_to_client(monkeypatch):
    captured = {}
    backend = LocalOpenAIBackend(base_url="http://x/v1", model="m", api_key="k",
                                 temperature=0.6, top_p=0.95, top_k=20, max_tokens=12345)

    class _Msg:
        content = "ok"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    captured.update(kw)
                    return _Resp()

    monkeypatch.setattr(backend, "_load_client", lambda: _FakeClient())
    backend.complete("s", "u")
    assert captured["max_tokens"] == 12345


def test_local_backend_parses_response(monkeypatch):
    backend = LocalOpenAIBackend(base_url="http://x/v1", model="Ornith-1.0-9B", api_key="k",
                                 temperature=0.6, top_p=0.95, top_k=20)

    class _Msg:
        content = "<think>x</think>hello"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    monkeypatch.setattr(backend, "_client_complete", lambda system, user, **kw: _Resp())
    assert backend.complete("s", "u") == "hello"
