"""Embeddings for the vector index. Real path uses sentence-transformers;
tests inject FakeEmbedder so both index engines run without torch."""
from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable


class EmbedderNotInstalled(RuntimeError):
    pass


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """Deterministic, dependency-free embedder for tests. Not for production
    (no semantic meaning) -- production requires SentenceTransformerEmbedder."""

    dim = 16

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            vals = [((digest[i % len(digest)] / 255.0) * 2 - 1) for i in range(self.dim)]
            norm = math.sqrt(sum(v * v for v in vals)) or 1.0
            out.append([v / norm for v in vals])
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise EmbedderNotInstalled(
                    "sentence-transformers is required for the knowledge compiler. "
                    'Install it with: pip install "deep-research-toolkit[compiler]"'
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        return [list(map(float, v)) for v in model.encode(texts, normalize_embeddings=True)]


class OllamaEmbedder:
    """Embed via an OpenAI-compatible endpoint (Ollama :11434/v1) serving a local
    embedding model such as qwen3-embedding:4b -- a materially stronger retrieval
    embedding than MiniLM. Vector dimension is whatever the model returns (LanceDB
    infers it), so 4B/8B swap cleanly with no schema change."""

    #: Bounded retry for a transient runner-not-ready failure: when Ollama has to
    #: COLD-LOAD the embedding model (e.g. it was evicted while an e4b + 31B/30B
    #: role were resident during an eval), the next /v1/embeddings call can hit a
    #: connection-refused on the model's ephemeral runner port before the runner
    #: is listening. A cold load of an ~8.7 GB embedding model can take 10-30 s,
    #: so the retry budget must span that: 8 attempts with backoff capped at 10 s
    #: (~0.5+1+2+4+8+10+10 ~= 35 s) rather than dropping entailment metrics for
    #: the whole run on a one-time load delay. Applied PER BATCH (see _BATCH_SIZE
    #: below), so a later batch's cold-load hiccup doesn't spend an earlier
    #: batch's budget and vice versa.
    _MAX_ATTEMPTS = 8
    _BACKOFF_CAP_S = 10.0

    #: Max texts per /v1/embeddings request. A single request over the eval
    #: corpus-level aggregate (2,400+ claim texts pooled across every doc) has
    #: been observed to kill the Ollama embedding runner outright rather than
    #: time out cleanly -- retrying that same oversized request just repeats
    #: the failure. Chunking well under any observed failure size, with each
    #: batch retried independently, keeps one runner-killing request from
    #: taking the whole embed() call down.
    _BATCH_SIZE = 128

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1",
                 api_key: str = "not-needed") -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._client = None

    def _client_(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise EmbedderNotInstalled(
                    "An OpenAI-compatible client is required for an Ollama embedding model. "
                    'Install it with: pip install "deep-research-toolkit[compiler]" (or: pip install openai).'
                ) from e
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Chunks `texts` into batches of at most _BATCH_SIZE, embeds each
        batch independently (its own full retry budget -- see _embed_batch),
        and concatenates the results in input order. A single small call
        (the common case, well under _BATCH_SIZE) is still exactly one
        request, unchanged from before batching existed."""
        out: list[list[float]] = []
        for start in range(0, len(texts), self._BATCH_SIZE):
            out.extend(self._embed_batch(texts[start:start + self._BATCH_SIZE]))
        return out

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import time

        last_exc: Exception | None = None
        for attempt in range(self._MAX_ATTEMPTS):
            try:
                resp = self._client_().embeddings.create(model=self.model, input=list(texts))
                return [list(d.embedding) for d in resp.data]
            except Exception as exc:  # noqa: BLE001 -- connection/runner-not-ready is transient; retry
                last_exc = exc
                if attempt == self._MAX_ATTEMPTS - 1:
                    break
                time.sleep(min(0.5 * (2 ** attempt), self._BACKOFF_CAP_S))  # 0.5,1,2,4,8,10,10 -- covers a cold load
        raise last_exc  # exhausted this batch's retries -- surface so the caller's degrade path still runs


def get_embedder(model_name: str = "all-MiniLM-L6-v2",
                 base_url: str = "http://localhost:11434/v1") -> Embedder:
    # An Ollama model tag has a "name:tag" shape; a sentence-transformers model
    # name does not. Route on that so `embedding_model: qwen3-embedding:4b` uses
    # the endpoint while `all-MiniLM-L6-v2` uses sentence-transformers.
    if ":" in model_name:
        return OllamaEmbedder(model_name, base_url)
    return SentenceTransformerEmbedder(model_name)
