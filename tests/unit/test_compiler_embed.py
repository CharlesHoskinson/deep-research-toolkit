from deep_research_toolkit.compiler.embed import (
    FakeEmbedder,
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)


def test_fake_embedder_is_deterministic_and_fixed_dim():
    e = FakeEmbedder()
    a = e.embed(["hydra", "cardano"])
    b = e.embed(["hydra", "cardano"])
    assert a == b
    assert len(a) == 2 and all(len(v) == e.dim for v in a)


def test_fake_embedder_distinguishes_texts():
    e = FakeEmbedder()
    v1, v2 = e.embed(["alpha", "completely different text"])
    assert v1 != v2


def test_get_embedder_routes_ollama_tag_vs_sentence_transformers():
    # An Ollama model tag ("name:tag") -> endpoint embedder; a plain HF name ->
    # sentence-transformers. Neither loads a model at construction.
    assert isinstance(get_embedder("qwen3-embedding:4b"), OllamaEmbedder)
    assert isinstance(get_embedder("all-MiniLM-L6-v2"), SentenceTransformerEmbedder)


def test_ollama_embedder_parses_response(monkeypatch):
    e = OllamaEmbedder("qwen3-embedding:4b")

    class _D:
        def __init__(self, v):
            self.embedding = v

    class _Resp:
        data = [_D([0.1, 0.2]), _D([0.3, 0.4])]

    class _Client:
        class embeddings:
            @staticmethod
            def create(**kw):
                return _Resp()

    monkeypatch.setattr(e, "_client_", lambda: _Client())
    assert e.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]


class _D:
    def __init__(self, v):
        self.embedding = v


class _FlakyClient:
    """Fails `fail_times` times (runner-not-ready), then serves one vector."""
    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    class embeddings:  # replaced per-instance below
        pass

    def _create(self, **kw):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("dial tcp 127.0.0.1:61800: connection refused")

        class _Resp:
            data = [_D([1.0, 2.0])]
        return _Resp()


def _wire(client):
    ns = type("embeddings", (), {"create": staticmethod(client._create)})
    client.embeddings = ns
    return client


def test_ollama_embedder_retries_transient_runner_failure(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda *_: None)  # no real backoff wait
    e = OllamaEmbedder("qwen3-embedding:4b")
    client = _wire(_FlakyClient(fail_times=2))  # fails twice, succeeds on the 3rd
    monkeypatch.setattr(e, "_client_", lambda: client)
    assert e.embed(["a"]) == [[1.0, 2.0]]
    assert client.calls == 3


def test_ollama_embedder_raises_after_exhausting_retries(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    e = OllamaEmbedder("qwen3-embedding:4b")
    client = _wire(_FlakyClient(fail_times=99))  # never recovers
    monkeypatch.setattr(e, "_client_", lambda: client)
    import pytest
    with pytest.raises(ConnectionError):
        e.embed(["a"])
    assert client.calls == OllamaEmbedder._MAX_ATTEMPTS
