"""Programmatic thesis synthesis over a composed dossier (role: synthesize).

Input is compose_dossier output: claims that already passed the verbatim
gate. The model writes an argued thesis; the mechanical check is the same
citation-marker gate as wiki_write -- every claim-bearing sentence must cite
a claim_id from the dossier's included set."""
from __future__ import annotations

import json

from .response import generate_cited
from .wiki import CitationError

_SYSTEM = """You write the synthesis section of an evidence dossier.

GOAL: answer the question using only the included claims below -- weigh
them, connect them, state what they establish and what remains open.

OUTPUT CONTRACT:
- Markdown only, no title heading (the dossier supplies one).
- Every sentence that rests on a claim MUST end with [claim:<claim_id>]
  markers referencing the supplied claims -- the literal claim: prefix is
  required; [<claim_id>] alone is wrong. Do not invent ids.
- If the claims cannot answer the question, say exactly what is missing.
- Example of a correctly cited sentence: "Settlement is final once all
  parties sign [claim:b00_c_0008]." (illustrative id -- cite only ids that
  appear in the supplied claims)
"""

_CORRECTION = (
    "Your previous synthesis cited unknown claim id(s): {bad}. Only supplied "
    "claim_ids may appear in [claim:...] markers. Rewrite the full synthesis."
)

_LOW_COVERAGE_CORRECTION = (
    "Your previous reply cited only {n}/{total} of the supplied claims. Rewrite "
    "the full {kind}, grounding every factual sentence in a supplied claim marker."
)


def synthesize_thesis(question: str, dossier: dict, backend,
                      min_coverage: float = 0.3) -> dict:
    """Returns {"thesis": str, "citations": validate_citations report}.

    Raises CitationError when the model cites unknown ids after a retry;
    ValueError on an empty dossier, when coverage still falls below
    min_coverage after one low-coverage retry, or when a reply degenerates
    into repetition (raised by the shared loop guard)."""
    included = dossier.get("included") or []
    if not included:
        raise ValueError("dossier has no included claims -- nothing to synthesize")
    allowed = [c.get("claim_id") for c in included]
    rows = [{"claim_id": c.get("claim_id"), "claim": c.get("claim"),
             "quotes": [e.get("quote") for e in (c.get("evidence") or [])]}
            for c in included]
    user = f"QUESTION: {question}\n\nINCLUDED CLAIMS:\n" + json.dumps(rows, ensure_ascii=False, indent=1)
    out = generate_cited(
        backend, _SYSTEM, user, allowed,
        min_coverage=min_coverage, kind="synthesis",
        correction_unknown=_CORRECTION,
        correction_low_coverage=_LOW_COVERAGE_CORRECTION,
        citation_error=CitationError)
    return {"thesis": out["text"], "citations": out["citations"]}
