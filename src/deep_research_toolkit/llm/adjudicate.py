"""Programmatic contradiction adjudication (role: conflict_adjudicate).

Consumes the mechanical candidates from compiler.contradictions.find_candidates
(via query.py find-contradictions) and asks a reasoning model for verdicts.
Verdicts are proposals: schema-validated here, stored with rationale, never
silently applied to the corpus. Under provider: agent the backend raises and
the retrieval-planner SKILL.md batched pass applies instead."""
from __future__ import annotations

import json

from .response import has_repetition_loop, parse_json_block

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


_BATCH_SIZE = 20


def _candidate_key(row: dict) -> tuple:
    return (row.get("subject"), row.get("predicate"))


def adjudicate_candidates(candidates: list[dict], backend,
                          batch_size: int = _BATCH_SIZE) -> dict:
    """Returns {"verdicts": [...], "invalid": [...], "parse_failures": int}.

    Verdict rows are validated against their own candidate: subject/predicate
    must name a supplied candidate, relation_ids must be a non-empty subset of
    that candidate's ids, and each candidate gets at most one verdict (first
    wins). Candidates go to the model in batches so one unparseable reply
    costs one batch, not the whole run."""
    relation_cands = [c for c in candidates if c.get("kind") == "relation"]
    if not relation_cands:
        return {"verdicts": [], "invalid": [], "parse_failures": 0}
    by_key = {_candidate_key(c): c for c in relation_cands}
    verdicts, invalid = [], []
    parse_failures = 0
    seen_keys = set()
    for i in range(0, len(relation_cands), batch_size):
        batch = relation_cands[i:i + batch_size]
        user = "CANDIDATES:\n" + json.dumps(batch, ensure_ascii=False, indent=1)
        reply = backend.complete(_SYSTEM, user)
        data = parse_json_block(reply)
        if isinstance(data, dict):
            # Reasoning models like wrapping the array in an object; accept a
            # single list-valued key, reject anything more ambiguous.
            lists = [v for v in data.values() if isinstance(v, list)]
            data = lists[0] if len(lists) == 1 else None
        if not isinstance(data, list):
            parse_failures += 1
            continue
        for row in data:
            if not isinstance(row, dict):
                invalid.append({"row": row, "reason": "not an object"})
                continue
            problems = []
            key = _candidate_key(row)
            cand = by_key.get(key)
            if cand is None:
                problems.append(f"subject/predicate {key!r} names no supplied candidate")
            if row.get("verdict") not in VERDICTS:
                problems.append(f"verdict {row.get('verdict')!r} not in {VERDICTS}")
            rids = row.get("relation_ids") or []
            if cand is not None:
                own = set(cand.get("relation_ids") or [])
                foreign = [r for r in rids if r not in own]
                if not rids or foreign:
                    problems.append(f"relation_ids missing or not this candidate's: {foreign or 'empty'}")
            elif not rids:
                problems.append("relation_ids empty")
            if not (row.get("rationale") or "").strip():
                problems.append("empty rationale")
            if has_repetition_loop(row.get("rationale") or ""):
                problems.append("rationale degenerated into repetition")
            if cand is not None and key in seen_keys and not problems:
                problems.append("duplicate verdict for this candidate")
            if problems:
                invalid.append({"row": row, "reason": "; ".join(problems)})
            else:
                verdicts.append(row)
                seen_keys.add(key)
    return {"verdicts": verdicts, "invalid": invalid, "parse_failures": parse_failures}
