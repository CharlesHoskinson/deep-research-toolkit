"""LocalOpenAIBackend records the last response's finish_reason so callers can
count truncated (finish_reason == "length") calls."""
import json
from types import SimpleNamespace

from deep_research_toolkit.llm.local import LocalOpenAIBackend


def _backend(**kw):
    return LocalOpenAIBackend(
        base_url="http://localhost:11434/v1", model="m", api_key="x",
        temperature=0.0, top_p=0.95, top_k=20, max_tokens=50, thinking=False,
        **kw,
    )


def _resp(finish_reason):
    msg = SimpleNamespace(content='{"ok": 1}')
    choice = SimpleNamespace(message=msg)
    if finish_reason is not ...:  # ... = attribute absent entirely
        choice.finish_reason = finish_reason
    return SimpleNamespace(choices=[choice], usage=None)


def _fake_client(resp):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: resp)))


def test_finish_reason_captured():
    b = _backend()
    b._client = _fake_client(_resp("length"))
    b.complete("s", "u")
    assert b.last_finish_reason == "length"


def test_finish_reason_updates_per_call():
    b = _backend()
    b._client = _fake_client(_resp("length"))
    b.complete("s", "u")
    b._client = _fake_client(_resp("stop"))
    b.complete("s", "u")
    assert b.last_finish_reason == "stop"


def test_finish_reason_missing_attribute_is_none():
    b = _backend()
    b._client = _fake_client(_resp(...))  # response without the attribute
    b.complete("s", "u")
    assert b.last_finish_reason is None


def test_finish_reason_defaults_to_none_before_any_call():
    assert _backend().last_finish_reason is None


def test_complete_with_meta_returns_text_and_finish_reason():
    # The thread-safe channel: callers get this call's finish_reason as a
    # return value instead of reading shared backend state after the fact.
    b = _backend()
    b._client = _fake_client(_resp("length"))
    text, reason = b.complete_with_meta("s", "u")
    assert text == '{"ok": 1}'
    assert reason == "length"
    assert b.last_finish_reason == "length"  # still set for trace/back-compat


def test_complete_with_meta_missing_attribute_is_none():
    b = _backend()
    b._client = _fake_client(_resp(...))  # response without the attribute
    text, reason = b.complete_with_meta("s", "u")
    assert text == '{"ok": 1}' and reason is None


def test_complete_delegates_to_complete_with_meta():
    b = _backend()
    b._client = _fake_client(_resp("stop"))
    assert b.complete("s", "u") == '{"ok": 1}'
    assert b.last_finish_reason == "stop"
    assert b.stats["calls"] == 1  # stats counted once, not double-counted


def test_trace_record_includes_finish_reason(tmp_path):
    trace = tmp_path / "trace.jsonl"
    b = _backend(trace_path=trace, role="extract")
    b._client = _fake_client(_resp("length"))
    b.complete("s", "u")
    row = json.loads(trace.read_text(encoding="utf-8").strip())
    assert row["finish_reason"] == "length"
