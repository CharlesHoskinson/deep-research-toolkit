"""Shared mechanical validation for programmatic judgment callers.

Citation markers: prose roles (wiki_write, synthesize) must tag claim-bearing
sentences with [claim:<id>]. The marker check is the prose analogue of the
verbatim gate -- it cannot prove the prose is faithful, but it proves every
cited id exists in the gate-passed set supplied to the model, so nothing can
cite evidence the corpus does not hold."""
from __future__ import annotations

import json
import re

CLAIM_MARKER_RE = re.compile(r"\[claim:([A-Za-z0-9_\-\.]+)\]")
_OUTPUT_RE = re.compile(r"<output>(.*?)</output>", re.DOTALL)


def extract_claim_ids(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in CLAIM_MARKER_RE.finditer(text):
        seen.setdefault(m.group(1))
    return list(seen)


def validate_citations(text: str, allowed_ids: list[str]) -> dict:
    allowed = set(allowed_ids)
    cited_all = extract_claim_ids(text)
    cited = [c for c in cited_all if c in allowed]
    unknown = [c for c in cited_all if c not in allowed]
    return {
        "cited": cited,
        "unknown": unknown,
        "coverage": (len(cited) / len(allowed)) if allowed else 0.0,
    }


def parse_json_block(text: str):
    """JSON from a model reply: prefer the last <output>...</output> block,
    else the widest [...] or {...} slice. None if nothing parses."""
    blocks = _OUTPUT_RE.findall(text)
    candidates = [blocks[-1]] if blocks else []
    for opener, closer in ("[]", "{}"):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None
