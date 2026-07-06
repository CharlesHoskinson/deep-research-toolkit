"""Programmatic thesis synthesis over a composed dossier (role: synthesize).

Input is compose_dossier output: claims that already passed the verbatim
gate. The model writes an argued thesis; the mechanical check is the same
citation-marker gate as wiki_write -- every claim-bearing sentence must cite
a claim_id from the dossier's included set."""
from __future__ import annotations

import json

from .response import has_repetition_loop, normalize_claim_markers, unfence, validate_citations
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
"""

_CORRECTION = (
    "Your previous synthesis cited unknown claim id(s): {bad}. Only supplied "
    "claim_ids may appear in [claim:...] markers. Rewrite the full synthesis."
)

_REPETITION_CORRECTION = (
    "Your previous reply degenerated into repetition. Write the synthesis normally."
)


def _checked_complete(backend, system: str, user: str) -> str:
    reply = backend.complete(system, user)
    if has_repetition_loop(reply):
        reply = backend.complete(system, user + "\n\n" + _REPETITION_CORRECTION)
        if has_repetition_loop(reply):
            raise ValueError("model reply degenerated into repetition")
    return reply


def synthesize_thesis(question: str, dossier: dict, backend,
                      min_coverage: float = 0.3) -> dict:
    """Returns {"thesis": str, "citations": validate_citations report}.

    Raises CitationError if the model cites unknown ids twice; ValueError on
    an empty dossier or when the accepted thesis's coverage falls below
    min_coverage."""
    included = dossier.get("included") or []
    if not included:
        raise ValueError("dossier has no included claims -- nothing to synthesize")
    allowed = [c.get("claim_id") for c in included]
    rows = [{"claim_id": c.get("claim_id"), "claim": c.get("claim"),
             "quotes": [e.get("quote") for e in (c.get("evidence") or [])]}
            for c in included]
    user = f"QUESTION: {question}\n\nINCLUDED CLAIMS:\n" + json.dumps(rows, ensure_ascii=False, indent=1)
    thesis = normalize_claim_markers(unfence(_checked_complete(backend, _SYSTEM, user)), allowed)
    report = validate_citations(thesis, allowed)
    if report["unknown"]:
        thesis = normalize_claim_markers(
            unfence(_checked_complete(
                backend, _SYSTEM,
                user + "\n\n" + _CORRECTION.format(bad=", ".join(report["unknown"])))),
            allowed)
        report = validate_citations(thesis, allowed)
        if report["unknown"]:
            raise CitationError(f"model cited unknown claim ids after retry: {report['unknown']}")
    if report["coverage"] < min_coverage:
        raise ValueError(
            f"synthesis cites {len(report['cited'])}/{len(allowed)} included claims "
            f"(coverage {report['coverage']:.2f} < {min_coverage})"
        )
    return {"thesis": thesis, "citations": report}
