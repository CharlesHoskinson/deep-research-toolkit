"""complete() forwards a caller-supplied JSON schema to Ollama's `format`.

Ollama's OpenAI-compatible /v1 endpoint has NO top-level `format` field --
its compat layer copies response_format.json_schema.schema into the native
request's `format`. So the passthrough must ride the same response_format
plumbing the role-level "json" mode already uses, wrapped in the OpenAI
json_schema envelope; anything else (e.g. extra_body={"format": ...}) is
silently dropped by the server's request decoder.
"""
from deep_research_toolkit.llm.local import LocalOpenAIBackend

SCHEMA = {"type": "object", "properties": {"cite": {"enum": ["c1", "c2"]}}}


class _Msg:
    content = "{}"


class _Choice:
    message = _Msg()


class _Resp:
    choices = [_Choice()]


def _wire(backend, monkeypatch):
    """Monkeypatch the OpenAI client local.py actually calls
    (client.chat.completions.create) and return the captured kwargs."""
    captured = {}

    class _FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    captured.update(kw)
                    return _Resp()

    monkeypatch.setattr(backend, "_load_client", lambda: _FakeClient())
    return captured


def _backend(**overrides):
    kw = dict(base_url="http://x/v1", model="gemma4:12b", api_key="x",
              thinking=False, temperature=0.0, top_p=0.95, top_k=20,
              max_tokens=100)
    kw.update(overrides)
    return LocalOpenAIBackend(**kw)


def test_complete_forwards_schema_as_ollama_format(monkeypatch):
    # A bare JSON schema reaches the request in the one field Ollama
    # translates into its native `format`: response_format.json_schema.schema.
    b = _backend()
    captured = _wire(b, monkeypatch)
    b.complete("sys", "user", response_format=SCHEMA)
    rf = captured.get("response_format")
    assert rf == {"type": "json_schema",
                  "json_schema": {"name": "response", "schema": SCHEMA}}


def test_response_format_none_is_todays_behavior(monkeypatch):
    b = _backend()
    captured = _wire(b, monkeypatch)
    b.complete("sys", "user")
    assert "response_format" not in captured
    assert captured["extra_body"]["think"] is False


def test_role_level_json_mode_still_works(monkeypatch):
    b = _backend(response_format="json")
    captured = _wire(b, monkeypatch)
    b.complete("sys", "user")
    assert captured["response_format"] == {"type": "json_object"}


def test_full_response_format_dict_passes_through_unchanged(monkeypatch):
    # The documented pre-existing contract: a full OpenAI response_format
    # (already enveloped) is forwarded as-is, not double-wrapped.
    full = {"type": "json_schema", "json_schema": {"name": "n", "schema": SCHEMA}}
    b = _backend()
    captured = _wire(b, monkeypatch)
    b.complete("sys", "user", response_format=full)
    assert captured["response_format"] == full


def test_caller_schema_overrides_role_json_mode(monkeypatch):
    b = _backend(response_format="json")
    captured = _wire(b, monkeypatch)
    b.complete("sys", "user", response_format=SCHEMA)
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["schema"] == SCHEMA
