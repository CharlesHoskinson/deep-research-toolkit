"""LocalOpenAIBackend accumulates per-call usage stats."""
from types import SimpleNamespace

from deep_research_toolkit.llm.local import LocalOpenAIBackend


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        msg = SimpleNamespace(content="hello")
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


def _backend():
    b = LocalOpenAIBackend(
        base_url="http://localhost:11434/v1", model="m", api_key="x",
        temperature=0.0, top_p=0.95, top_k=20,
    )
    b._client = _FakeClient()
    return b


def test_stats_accumulate_across_calls(monkeypatch):
    # complete() imports time locally, so patch module-wide. The fake clock
    # yields t0/t1 for each call: 1.5s for the first, 2.5s for the second.
    ticks = iter([0.0, 1.5, 10.0, 12.5])
    monkeypatch.setattr("time.perf_counter", lambda: next(ticks))
    b = _backend()
    b.complete("sys", "user")
    b.complete("sys", "user")
    assert b.stats["calls"] == 2
    assert b.stats["prompt_tokens"] == 22
    assert b.stats["completion_tokens"] == 14
    assert b.stats["seconds"] == 4.0


def test_stats_survive_missing_usage():
    b = _backend()
    b._client.chat.completions.create = lambda **kw: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="x"))], usage=None
    )
    b.complete("sys", "user")
    assert b.stats["calls"] == 1
    assert b.stats["prompt_tokens"] == 0
