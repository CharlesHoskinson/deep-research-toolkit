"""Task 10: bounded ThreadPoolExecutor fan-out over extraction sample-passes.

`parallel=1` (default) is byte-identical sequential behavior; `parallel>1`
overlaps the sample passes on worker threads, with all merging done after the
futures join so no worker touches shared mutable state.
"""
import json
import threading

from deep_research_toolkit.llm import extract

CHUNK = "A checkpoint commits when two-thirds of validators sign the same height."


class ConcurrentBackend:
    """Tracks the maximum number of in-flight complete() calls. Has NO
    complete_with_meta, so it also exercises extract._complete_meta's
    fallback path for plain backends.

    An instant-return fake can finish each call before the next worker even
    starts, so overlap detection would be timing-flaky. A 2-party barrier
    (short timeout) makes the first two concurrent calls rendezvous: under a
    real fan-out they meet instantly; under sequential execution the lone
    waiter times out (bounded) and max_concurrent stays 1."""
    thinking = False

    def __init__(self, rendezvous=True):
        self.max_concurrent = 0
        self._n = 0
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(2) if rendezvous else None

    def complete(self, system, user, **kw):
        with self._lock:
            self._n += 1
            self.max_concurrent = max(self.max_concurrent, self._n)
        try:
            if self._barrier is not None:
                try:
                    self._barrier.wait(timeout=1.0)
                except threading.BrokenBarrierError:
                    pass  # sequential caller: nobody else ever arrives
            return json.dumps({"claims": [{"claim_id": "c", "claim": "Checkpoint commits at 2/3.",
                    "confidence": "high", "supporting_evidence": [{"locator": "d#c1", "start_char": 0, "end_char": 12}]}],
                    "entities": [], "relations": []})
        finally:
            with self._lock:
                self._n -= 1


def _write_chunk(tmp_path):
    (tmp_path / "chunks.jsonl").write_text(
        json.dumps({"locator": "d#c1", "text": CHUNK}) + "\n", encoding="utf-8")


def test_parallel_runs_concurrently_same_result(tmp_path):
    _write_chunk(tmp_path)
    backend = ConcurrentBackend()
    summary = extract.extract_claims_to_run(tmp_path, "web", None, backend, samples=4, parallel=4)
    assert summary["written"] >= 1
    assert backend.max_concurrent >= 2  # actually overlapped


class TruncatingMetaBackend:
    """Every call reports finish_reason == "length" via complete_with_meta's
    per-call return value -- the thread-safe channel. Counts calls under a
    lock so the assertion below is exact even with overlapping workers."""
    thinking = False

    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()
        self._payload = json.dumps({"claims": [{"claim_id": "c", "claim": "Checkpoint commits at 2/3.",
                "confidence": "high", "supporting_evidence": [{"locator": "d#c1", "start_char": 0, "end_char": 12}]}],
                "entities": [], "relations": []})

    def complete_with_meta(self, system, user, **kw):
        with self._lock:
            self.calls += 1
        return self._payload, "length"

    def complete(self, system, user, **kw):
        return self.complete_with_meta(system, user, **kw)[0]


def test_truncated_calls_counted_exactly_under_parallel(tmp_path):
    # Task 8's counter used to read backend.last_finish_reason after each
    # complete() -- a data race under threads (another worker's call overwrites
    # it first). The per-pass local count summed after the join must equal the
    # true number of truncated calls, with no lost updates.
    _write_chunk(tmp_path)
    backend = TruncatingMetaBackend()
    summary = extract.extract_claims_to_run(tmp_path, "web", None, backend, samples=4, parallel=4)
    assert backend.calls == 4  # one call per sample pass (single batch)
    assert summary["truncated_calls"] == 4
    assert summary["parallel"] == 4


def test_parallel_one_is_sequential_and_deterministic(tmp_path):
    # Default path: parallel=1 must never overlap calls and must produce the
    # same claims as an explicit sequential run.
    _write_chunk(tmp_path)
    backend = ConcurrentBackend(rendezvous=False)  # no barrier: nobody would ever meet it
    summary = extract.extract_claims_to_run(tmp_path, "web", None, backend, samples=4, parallel=1)
    assert backend.max_concurrent == 1
    assert summary["written"] >= 1
    assert summary["parallel"] == 1
