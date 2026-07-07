"""LocalOpenAIBackend accumulates per-call usage stats."""
import threading
from concurrent.futures import ThreadPoolExecutor
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


class _BarrierClient:
    """Fake client whose create() rendezvouses ``width`` callers at a barrier
    before returning, forcing threads to interleave right at the stats
    read-modify-write -- so an unguarded += loses updates deterministically."""
    def __init__(self, width):
        self._barrier = threading.Barrier(width)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        try:
            self._barrier.wait(timeout=5.0)
        except threading.BrokenBarrierError:
            pass
        msg = SimpleNamespace(content="hello")
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)


def test_stats_exact_under_concurrent_calls():
    # 8 threads x 20 calls on ONE backend. The stats increments are non-atomic
    # read-modify-write; the lock added for parallel extraction must make the
    # totals exact (no lost updates) despite the forced interleaving.
    threads, per_thread = 8, 20
    total = threads * per_thread
    b = _backend()
    b._client = _BarrierClient(width=threads)

    def _worker():
        for _ in range(per_thread):
            b.complete_with_meta("sys", "user")

    with ThreadPoolExecutor(max_workers=threads) as ex:
        for f in [ex.submit(_worker) for _ in range(threads)]:
            f.result()

    assert b.stats["calls"] == total
    assert b.stats["prompt_tokens"] == 11 * total
    assert b.stats["completion_tokens"] == 7 * total
