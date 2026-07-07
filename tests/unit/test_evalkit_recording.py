"""Unit tests for evalkit.recording.RecordingBackend -- a thin backend
wrapper, no model calls."""
from __future__ import annotations

from deep_research_toolkit.evalkit.recording import RecordingBackend


class _StubBackend:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user, **sampling):
        self.calls.append((system, user, sampling))
        return self.replies.pop(0)


def test_recording_backend_delegates_and_returns_reply():
    inner = _StubBackend(["hello"])
    rec = RecordingBackend(inner)
    out = rec.complete("sys", "user")
    assert out == "hello"


def test_recording_backend_records_every_raw_completion_in_order():
    inner = _StubBackend(["one", "two", "three"])
    rec = RecordingBackend(inner)
    rec.complete("sys", "u1")
    rec.complete("sys", "u2")
    rec.complete("sys", "u3")
    assert rec.raw == ["one", "two", "three"]


def test_recording_backend_forwards_sampling_kwargs():
    inner = _StubBackend(["x"])
    rec = RecordingBackend(inner)
    rec.complete("sys", "user", temperature=0.25, max_tokens=10)
    assert inner.calls[0][2] == {"temperature": 0.25, "max_tokens": 10}


def test_recording_backend_starts_with_empty_raw_list():
    rec = RecordingBackend(_StubBackend([]))
    assert rec.raw == []
