"""Programmatic wiki-body writer (role: wiki_write) for provider: local.

Mirrors extract.py's stance: the model proposes prose; a mechanical check
(citation markers resolving to supplied gate-passed claims) decides whether
it is accepted. Under provider: agent the backend raises -- the in-session
agent writes wiki pages itself per the llm-wiki-writer SKILL.md."""
from __future__ import annotations

import json

from .response import normalize_claim_markers, unfence, validate_citations

_SYSTEM = """You write one wiki page body for a research knowledge base.

GOAL: synthesize the supplied claims into clear, well-organized markdown
prose for the page named in the task. Use only the supplied claims -- no
outside knowledge, no speculation.

OUTPUT CONTRACT:
- Markdown body only. No frontmatter, no code fences around the whole reply.
- Every sentence that states a fact MUST end with the marker of the claim it
  came from, formatted exactly: [claim:<claim_id>] -- the literal claim:
  prefix is required; [<claim_id>] alone is wrong.
- A sentence may carry several markers. Do not invent claim ids.
- Organize with ## sections when the material warrants it; otherwise a single
  coherent body. Neutral, precise register. No filler.
"""

_CORRECTION = (
    "Your previous body cited unknown claim id(s): {bad}. Every [claim:...] "
    "marker must use one of the supplied claim_ids exactly. Rewrite the full "
    "body now, fixing or removing the offending sentences."
)

_LOW_COVERAGE_CORRECTION = (
    "Your previous reply cited only {n}/{total} of the supplied claims. Rewrite "
    "the full page body, grounding every factual sentence in a supplied claim marker."
)


class CitationError(ValueError):
    pass


def _task(title: str, page_type: str, claims: list[dict]) -> str:
    rows = [
        {"claim_id": c.get("claim_id"), "claim": c.get("claim"),
         "quotes": [ev.get("quote") for ev in (c.get("supporting_evidence") or [])]}
        for c in claims
    ]
    return (
        f"PAGE: {title} (type: {page_type})\n\nCLAIMS (the only permitted sources):\n"
        + json.dumps(rows, ensure_ascii=False, indent=1)
    )


def write_wiki_body(title: str, page_type: str, claims: list[dict], backend,
                    min_coverage: float = 0.5) -> dict:
    """Returns {"body": str, "citations": validate_citations report}.

    Raises CitationError if the model cites unknown ids twice; ValueError on
    empty claims or when the accepted body's coverage still falls below
    min_coverage after one low-coverage retry."""
    if not claims:
        raise ValueError("write_wiki_body needs at least one gate-passed claim")
    allowed = [c.get("claim_id") for c in claims]
    user = _task(title, page_type, claims)
    body = normalize_claim_markers(unfence(backend.complete(_SYSTEM, user)), allowed)
    report = validate_citations(body, allowed)
    if report["unknown"]:
        body = normalize_claim_markers(
            unfence(backend.complete(
                _SYSTEM, user + "\n\n" + _CORRECTION.format(bad=", ".join(report["unknown"])),
                temperature=0.25)),
            allowed)
        report = validate_citations(body, allowed)
        if report["unknown"]:
            raise CitationError(f"model cited unknown claim ids after retry: {report['unknown']}")
    if report["coverage"] < min_coverage:
        body = normalize_claim_markers(
            unfence(backend.complete(
                _SYSTEM,
                user + "\n\n" + _LOW_COVERAGE_CORRECTION.format(n=len(report["cited"]), total=len(allowed)),
                temperature=0.25)),
            allowed)
        report = validate_citations(body, allowed)
        if report["coverage"] < min_coverage:
            raise ValueError(
                f"body cites {len(report['cited'])}/{len(allowed)} claims "
                f"(coverage {report['coverage']:.2f} < {min_coverage}) -- refusing a page that ignores its sources"
            )
    return {"body": body, "citations": report}
