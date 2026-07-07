"""Unit tests for evalkit.embed_match: embedding-based recall with an
injected fake embedder -- deterministic 2-d toy vectors, no model calls."""
from __future__ import annotations

from deep_research_toolkit.evalkit.embed_match import embed_recall


def fake_embedder(texts):
    # 2-d toy embeddings: map by keyword so cosine is deterministic
    def vec(t):
        t = t.lower()
        if "rotate" in t: return [1.0, 0.0]
        if "verify" in t: return [0.0, 1.0]
        return [0.7, 0.7]
    return [vec(t) for t in texts]


def _c(text): return {"claim": text, "supporting_evidence": []}


def test_embed_recall_matches_paraphrase_above_threshold():
    produced = [_c("Leaders are rotated each epoch")]
    reference = [_c("Rotate the leader every epoch")]  # both -> [1,0]
    r = embed_recall(produced, reference, fake_embedder, threshold=0.9)
    assert r["recall"] == 1.0


def test_embed_recall_misses_unrelated():
    produced = [_c("Followers verify signatures")]   # [0,1]
    reference = [_c("Leaders rotate")]               # [1,0], cosine 0
    r = embed_recall(produced, reference, fake_embedder, threshold=0.9)
    assert r["recall"] == 0.0
