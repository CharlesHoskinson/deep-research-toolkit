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


# ---------------------------------------------------------------------------
# Anomaly 2: a single request over the corpus-level aggregate (2,400+ texts)
# has killed the Ollama embedding runner outright, so requests are chunked
# into batches of <= 128 texts, each with its own retry-with-backoff.
# ---------------------------------------------------------------------------

class _CountingClient:
    """Records each create() call's batch size (as the first input text, so
    tests can tell WHICH batch a call belongs to) and echoes back one vector
    per input, encoding the input's position so order is verifiable."""

    def __init__(self):
        self.batch_sizes: list[int] = []
        self.calls = 0

    def _create(self, **kw):
        self.calls += 1
        inputs = kw["input"]
        self.batch_sizes.append(len(inputs))

        class _Resp:
            data = [_D([float(t.split("t")[-1])]) for t in inputs]
        return _Resp()


def test_ollama_embedder_batches_large_input_into_multiple_requests(monkeypatch):
    e = OllamaEmbedder("qwen3-embedding:4b")
    client = _wire(_CountingClient())
    monkeypatch.setattr(e, "_client_", lambda: client)
    texts = [f"t{i}" for i in range(200)]

    vecs = e.embed(texts)

    assert client.batch_sizes == [128, 72]  # <= 128 per request
    assert len(vecs) == 200
    # order preserved across the batch boundary
    assert vecs[0] == [0.0] and vecs[127] == [127.0] and vecs[128] == [128.0] and vecs[199] == [199.0]


def test_ollama_embedder_batch_boundary_is_exactly_128(monkeypatch):
    e = OllamaEmbedder("qwen3-embedding:4b")

    client_at = _wire(_CountingClient())
    monkeypatch.setattr(e, "_client_", lambda: client_at)
    e.embed([f"t{i}" for i in range(128)])
    assert client_at.batch_sizes == [128]

    client_over = _wire(_CountingClient())
    monkeypatch.setattr(e, "_client_", lambda: client_over)
    e.embed([f"t{i}" for i in range(129)])
    assert client_over.batch_sizes == [128, 1]


def test_ollama_embedder_single_small_call_is_still_one_batch(monkeypatch):
    # Backward-compat sanity: the common case (well under 128 texts) must
    # still be exactly one request, unchanged from before batching existed.
    e = OllamaEmbedder("qwen3-embedding:4b")
    client = _wire(_CountingClient())
    monkeypatch.setattr(e, "_client_", lambda: client)
    e.embed(["t0", "t1", "t2"])
    assert client.batch_sizes == [3]


class _BatchFlakyClient:
    """Fails the first `fail_times` calls for the batch identified by its
    first input text, then serves normally -- so a retry inside ONE batch is
    exercisable without affecting any other batch's call count."""

    def __init__(self, flaky_batch_key: str, fail_times: int = 1):
        self.flaky_batch_key = flaky_batch_key
        self.fail_times = fail_times
        self.calls_per_batch: dict[str, int] = {}

    def _create(self, **kw):
        inputs = kw["input"]
        key = inputs[0]
        n = self.calls_per_batch.get(key, 0) + 1
        self.calls_per_batch[key] = n
        if key == self.flaky_batch_key and n <= self.fail_times:
            raise ConnectionError("dial tcp 127.0.0.1:61800: connection refused")

        class _Resp:
            data = [_D([1.0]) for _ in inputs]
        return _Resp()


def test_ollama_embedder_one_batch_fails_then_retry_succeeds(monkeypatch):
    import time
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    e = OllamaEmbedder("qwen3-embedding:4b")
    texts = [f"t{i}" for i in range(129)]  # two batches: t0..t127, t128
    client = _wire(_BatchFlakyClient(flaky_batch_key="t0", fail_times=1))
    monkeypatch.setattr(e, "_client_", lambda: client)

    vecs = e.embed(texts)

    assert len(vecs) == 129
    assert client.calls_per_batch["t0"] == 2   # failed once, retried, succeeded
    assert client.calls_per_batch["t128"] == 1  # the other batch never failed or retried


def test_ollama_embedder_batch_retry_budget_is_per_batch_not_shared(monkeypatch):
    # Each batch gets its OWN full retry budget: a batch that fails
    # _MAX_ATTEMPTS - 1 times still succeeds on its last attempt, even after
    # an earlier batch already used up retries of its own.
    import time
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    e = OllamaEmbedder("qwen3-embedding:4b")
    texts = [f"t{i}" for i in range(129)]
    client = _wire(_BatchFlakyClient(flaky_batch_key="t0", fail_times=OllamaEmbedder._MAX_ATTEMPTS - 1))
    monkeypatch.setattr(e, "_client_", lambda: client)

    vecs = e.embed(texts)

    assert len(vecs) == 129
    assert client.calls_per_batch["t0"] == OllamaEmbedder._MAX_ATTEMPTS
    assert client.calls_per_batch["t128"] == 1
