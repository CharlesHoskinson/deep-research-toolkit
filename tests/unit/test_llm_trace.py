"""Opt-in per-call JSONL tracing on LocalOpenAIBackend + config/backend plumbing."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from deep_research_toolkit.config import load_config
from deep_research_toolkit.llm.backend import get_backend
from deep_research_toolkit.llm.local import LocalOpenAIBackend

_REQUIRED_KEYS = {
    "ts",
    "gen_ai.request.model",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "latency_s",
    "role",
    "ok",
}


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        msg = SimpleNamespace(content="hello")
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


def _backend(**kw):
    b = LocalOpenAIBackend(
        base_url="http://localhost:11434/v1", model="m", api_key="x",
        temperature=0.0, top_p=0.95, top_k=20, **kw,
    )
    b._client = _FakeClient()
    return b


def test_trace_path_writes_one_line_per_call(tmp_path):
    trace_path = tmp_path / "t.jsonl"
    b = _backend(trace_path=trace_path, role="extract")
    b.complete("sys", "user")
    b.complete("sys", "user")

    lines = trace_path.read_text(encoding="utf-8").strip("\n").split("\n")
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)
        assert _REQUIRED_KEYS <= row.keys()
        assert row["role"] == "extract"
        assert row["ok"] is True
        assert row["gen_ai.usage.input_tokens"] == 11


def test_trace_path_none_writes_no_file(tmp_path):
    trace_path = tmp_path / "t.jsonl"
    b = _backend(trace_path=None)
    b.complete("sys", "user")
    assert not trace_path.exists()


def test_trace_write_failure_is_swallowed(tmp_path):
    trace_dir = tmp_path / "t.jsonl"
    trace_dir.mkdir()  # a directory at the trace path makes open() raise OSError
    b = _backend(trace_path=trace_dir, role="extract")
    result = b.complete("sys", "user")
    assert result == "hello"


def test_get_backend_plumbs_trace_path_and_role(tmp_path):
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\n"
        "llm:\n"
        "  provider: local\n"
        "  trace: true\n"
        "  local:\n"
        "    base_url: http://localhost:11434/v1\n"
        "    model: Ornith-1.0-9B\n",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    backend = get_backend(cfg, role="extract")
    assert backend.trace_path == Path("llm-trace.jsonl")
    assert backend.role == "extract"


def test_get_backend_trace_absent_gives_none_trace_path(tmp_path):
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\n"
        "llm:\n"
        "  provider: local\n"
        "  local:\n"
        "    base_url: http://localhost:11434/v1\n"
        "    model: Ornith-1.0-9B\n",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    backend = get_backend(cfg, role="extract")
    assert backend.trace_path is None
