"""Embedding-based recall: authored gold is non-exhaustive and exact-substring
matching undercounts paraphrased recall. Cosine similarity between produced and
reference claim TEXT (not quotes) via an injected embedder gives a recall signal
that does not conflate model quality with gold incompleteness."""
from __future__ import annotations

import math


def _cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embed_recall(produced, reference, embedder, threshold: float = 0.9) -> dict:
    if not reference:
        return {"recalled": [], "missed": [], "recall": None}
    ref_texts = [r.get("claim", "") for r in reference]
    prod_texts = [p.get("claim", "") for p in produced]
    ref_vecs = embedder(ref_texts) if ref_texts else []
    prod_vecs = embedder(prod_texts) if prod_texts else []
    recalled, missed = [], []
    for ref, rv in zip(reference, ref_vecs):
        if any(_cos(rv, pv) >= threshold for pv in prod_vecs):
            recalled.append(ref)
        else:
            missed.append(ref)
    return {"recalled": recalled, "missed": missed, "recall": len(recalled) / len(reference)}
