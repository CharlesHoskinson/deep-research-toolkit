"""Programmatic contradiction adjudication (role: conflict_adjudicate).

Consumes the mechanical candidates from compiler.contradictions.find_candidates
(via query.py find-contradictions) and asks a reasoning model for verdicts.
Verdicts are proposals: schema-validated here, stored with rationale, never
silently applied to the corpus. Under provider: agent the backend raises and
the retrieval-planner SKILL.md batched pass applies instead."""
from __future__ import annotations

import json

from .response import parse_json_block

VERDICTS = ("contradiction", "not_contradiction", "insufficient_evidence")

_SYSTEM = """You adjudicate contradiction candidates from a knowledge graph.

Each candidate groups relations sharing (subject, predicate) but disagreeing
on object. Some are real contradictions; many are benign (aliases, different
granularity, non-exclusive values, time-qualified facts).

OUTPUT CONTRACT: reply with exactly one <output>...</output> block containing
a JSON array, one element per candidate, each:
  {"subject": str, "predicate": str,
   "verdict": "contradiction" | "not_contradiction" | "insufficient_evidence",
   "rationale": str (one or two sentences),
   "relation_ids": [ids copied from that candidate]}
Judge only from the supplied data; when the objects could coexist, say
not_contradiction; when you cannot tell, say insufficient_evidence.
"""


def adjudicate_candidates(candidates: list[dict], backend) -> dict:
    """Returns {"verdicts": [...], "invalid": [...], "parse_failures": int}."""
    relation_cands = [c for c in candidates if c.get("kind") == "relation"]
    if not relation_cands:
        return {"verdicts": [], "invalid": [], "parse_failures": 0}
    user = "CANDIDATES:\n" + json.dumps(relation_cands, ensure_ascii=False, indent=1)
    reply = backend.complete(_SYSTEM, user)
    data = parse_json_block(reply)
    if not isinstance(data, list):
        return {"verdicts": [], "invalid": [], "parse_failures": 1}
    allowed_ids = {rid for c in relation_cands for rid in (c.get("relation_ids") or [])}
    verdicts, invalid = [], []
    for row in data:
        if not isinstance(row, dict):
            invalid.append({"row": row, "reason": "not an object"})
            continue
        problems = []
        if row.get("verdict") not in VERDICTS:
            problems.append(f"verdict {row.get('verdict')!r} not in {VERDICTS}")
        rids = row.get("relation_ids") or []
        foreign = [r for r in rids if r not in allowed_ids]
        if not rids or foreign:
            problems.append(f"relation_ids missing or unknown: {foreign or 'empty'}")
        if not (row.get("rationale") or "").strip():
            problems.append("empty rationale")
        if problems:
            invalid.append({"row": row, "reason": "; ".join(problems)})
        else:
            verdicts.append(row)
    return {"verdicts": verdicts, "invalid": invalid, "parse_failures": 0}
