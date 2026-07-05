"""Mechanical write-time gate for a run directory's claims.jsonl.

Same invariant as extraction's inline gate and the eval harness: every
evidence quote must be a verbatim substring of the chunk its locator names.
This module exists so the pure-skill (provider: agent) track can gate
agent-authored claims at write time instead of only at compile/eval time.
Purely deterministic -- no LLM, no network (ADR 0001 decision 4)."""
from __future__ import annotations

import json
from pathlib import Path

from .verbatim import chunk_text_by_locator, verbatim_ok


def check_claims_file(run_dir: Path | str) -> dict:
    """Validate <run_dir>/claims.jsonl against <run_dir>/chunks.jsonl.

    Returns {"checked": int, "ok": int, "failures": [{"claim_id", "reason"}]}.
    Evidence rows may use `locator` (web runs) or `node_id` (pdf runs) as the
    chunk key -- the same two shapes extraction emits."""
    run_dir = Path(run_dir)
    chunks = chunk_text_by_locator(run_dir)
    claims_path = run_dir / "claims.jsonl"
    if not claims_path.is_file():
        return {"checked": 0, "ok": 0, "failures": [{"claim_id": None, "reason": f"no claims.jsonl in {run_dir}"}]}

    checked, ok, failures = 0, 0, []
    for line in claims_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        checked += 1
        try:
            claim = json.loads(line)
        except json.JSONDecodeError as e:
            failures.append({"claim_id": None, "reason": f"line {checked}: invalid JSON ({e})"})
            continue
        cid = claim.get("claim_id")
        evidence = claim.get("supporting_evidence") or []
        if not evidence:
            failures.append({"claim_id": cid, "reason": "no supporting evidence"})
            continue
        bad = []
        for ev in evidence:
            key = ev.get("node_id") or ev.get("locator") or ""
            quote = ev.get("quote") or ""
            if not quote:
                bad.append(f"empty quote (locator {key!r})")
            elif not verbatim_ok(quote, chunks.get(key, "")):
                bad.append(f"quote not a verbatim substring of chunk {key!r}")
        if bad:
            failures.append({"claim_id": cid, "reason": "; ".join(bad)})
        else:
            ok += 1
    return {"checked": checked, "ok": ok, "failures": failures}
