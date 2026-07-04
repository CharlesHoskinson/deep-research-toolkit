"""compose_dossier: assemble claims + citations into an evidence dossier,
gated by the verbatim-quote invariant. A claim whose quote is not a verbatim
substring of its source text is dropped into `rejected`, never emitted as
if verified. This reuses the exact-substring semantics of
pdf.eval.check_evidence_quotes_verbatim; do not weaken it."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def verbatim_ok(quote: str, source_text: str) -> bool:
    return bool(quote) and quote in source_text


def _pdf_page_text(run_dir: Path, page: int | None) -> str:
    prov = run_dir / "provenance.jsonl"
    if not prov.is_file():
        return ""
    parts = []
    with open(prov, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            unit = json.loads(line)
            if unit.get("page") == page:
                parts.append(unit.get("text") or "")
    return "\n".join(parts)


def source_text_for(evidence_row: dict[str, Any], config) -> str:
    producer, source_id = evidence_row["producer"], evidence_row["source_id"]
    if producer == "pdf":
        return _pdf_page_text(Path(config.pdf_runs_path) / source_id, evidence_row.get("page"))
    if producer == "web":
        src = Path(config.research_runs_path) / source_id / "source.md"
        return src.read_text(encoding="utf-8") if src.is_file() else ""
    return ""


def compose_dossier(con, config, claim_ids: list[str]) -> dict:
    included, rejected = [], []
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
            if not verbatim_ok(ev["quote"], source_text_for(ev, config)):
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
