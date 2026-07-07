"""The one verbatim-quote gate, shared by every stage that checks evidence.

A claim's supporting quote must be an exact substring of the *chunk* the claim
cites -- the exact text the extractor was shown, identified by the evidence's
locator (a `node_id` for PDF runs, a `locator` for web runs). Every stage that
enforces the invariant -- extraction (`llm.extract`), dossier composition
(`compiler.dossier`), and the eval harness (`pdf.eval`) -- resolves the same
chunk text and applies the same `verbatim_ok` check here, so a claim admitted at
one stage is never silently rejected at another because the stages disagreed on
what "the source text" is. Do not weaken the substring check, and do not
re-introduce a per-stage definition of the source text.
"""
from __future__ import annotations

import json
from pathlib import Path


def verbatim_ok(quote: str, source_text: str) -> bool:
    """Exact-substring check. A non-empty quote must appear verbatim (character
    for character, contiguous) in the source text. No normalization."""
    return bool(quote) and quote in source_text


def chunk_text_by_locator(run_dir) -> dict[str, str]:
    """Map every chunk id (`node_id` or `locator`) to its text, read from a run
    directory's `chunks.jsonl`. This is the canonical source text a quote is
    checked against, re-derived from disk (independent of the compiled index)."""
    path = Path(run_dir) / "chunks.jsonl"
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            cid = chunk.get("node_id") or chunk.get("locator")
            if cid:
                out[cid] = chunk.get("text", "")
    return out


def slice_span(source_text: str, start: int, end: int) -> str | None:
    """Return source_text[start:end] iff the span is a valid, non-empty,
    in-bounds forward slice; else None. The span replaces the free-text quote:
    a claim points AT its evidence by character offsets instead of copying it,
    so 'almost a quote' (near-quote bait) cannot be represented at all."""
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 0 or end > len(source_text) or start >= end:
        return None
    return source_text[start:end]


def span_ok(start: int, end: int, source_text: str,
            claimed_quote: str | None = None) -> bool:
    """The span-contract analogue of verbatim_ok: the span must be in-bounds and
    non-empty, and if the model also echoed a `quote`, the slice it points at
    must equal that quote exactly (no near-copy). This is an O(1) slice-compare,
    not a substring search."""
    sliced = slice_span(source_text, start, end)
    if sliced is None:
        return False
    return claimed_quote is None or sliced == claimed_quote
