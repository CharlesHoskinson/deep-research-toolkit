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


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    return SentenceTransformerEmbedder(model_name)
