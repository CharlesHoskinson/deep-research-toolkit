"""compose_dossier: assemble claims + citations into an evidence dossier,
gated by the verbatim-quote invariant. A claim whose quote is not a verbatim
substring of the chunk it cites is dropped into `rejected`, never emitted as if
verified. The gate is the single shared one in `common.verbatim` (chunk-based),
the same check extraction and the eval harness apply -- do not weaken it or
re-derive a different notion of "source text" here."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common.verbatim import chunk_text_by_locator, verbatim_ok

__all__ = ["verbatim_ok", "source_text_for", "compose_dossier"]


def _run_dir_for(producer: str, source_id: str, config) -> Path:
    root = Path(config.pdf_runs_path) if producer == "pdf" else Path(config.research_runs_path)
    return root / source_id


def source_text_for(evidence_row: dict[str, Any], config) -> str:
    """The chunk text an evidence quote must be verbatim in -- the chunk its
    locator names, read from the run's chunks.jsonl (same as extraction)."""
    run_dir = _run_dir_for(evidence_row["producer"], evidence_row["source_id"], config)
    return chunk_text_by_locator(run_dir).get(evidence_row.get("locator") or "", "")


def compose_dossier(con, config, claim_ids: list[str]) -> dict:
    included, rejected = [], []
    chunk_cache: dict[Path, dict[str, str]] = {}
    for cid in claim_ids:
        claim_row = con.execute(
            "SELECT claim_id, producer, source_id, claim, claim_type, confidence FROM claims WHERE claim_id = ?",
            [cid],
        ).fetchone()
        if claim_row is None:
            rejected.append({"claim_id": cid, "reason": "claim_id not found in index"})
            continue
        ev_rows = con.execute(
            "SELECT claim_id, producer, source_id, locator, page, url, quote FROM claim_evidence WHERE claim_id = ?",
            [cid],
        ).fetchall()
        cols = ["claim_id", "producer", "source_id", "locator", "page", "url", "quote"]
        evidence = [dict(zip(cols, r)) for r in ev_rows]

        failures = []
        for ev in evidence:
            run_dir = _run_dir_for(ev["producer"], ev["source_id"], config)
            if run_dir not in chunk_cache:
                chunk_cache[run_dir] = chunk_text_by_locator(run_dir)
            if not verbatim_ok(ev["quote"], chunk_cache[run_dir].get(ev.get("locator") or "", "")):
                failures.append(ev["quote"])

        entry = {
            "claim_id": cid, "claim": claim_row[3], "claim_type": claim_row[4],
            "confidence": claim_row[5], "evidence": evidence,
        }
        if failures or not evidence:
            entry["reason"] = "no evidence" if not evidence else f"non-verbatim quote(s): {failures}"
            rejected.append(entry)
        else:
            included.append(entry)
    return {"included": included, "rejected": rejected}
