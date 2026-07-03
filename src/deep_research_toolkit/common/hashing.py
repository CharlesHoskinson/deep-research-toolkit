"""Content-hashing helpers shared by every stage that needs a stable id.

One implementation, reused everywhere, so "does this hash match what I
computed at ingest time" questions (dedup, incremental compile, provenance
verification) never depend on two slightly different hash functions that
happen to agree most of the time.
"""
from __future__ import annotations

import hashlib


def content_hash(text: str, length: int = 16) -> str:
    """`sha256:` + the first `length` hex chars of sha256(text)."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:length]}"


def file_hash(path, chunk_size: int = 1 << 20) -> str:
    """Full (untruncated) sha256 of a file's bytes, for source-file identity
    (document_id derivation) where truncation risk matters more than
    readability.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
