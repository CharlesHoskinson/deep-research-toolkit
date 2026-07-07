"""Union-over-samples + support-count filtering for extraction recall.

Running the extractor N times at varied temperature and UNIONing gate-passing
claims raises recall; keeping only claims that recur in >= k of N passes is a
cheap precision/bait cut. One knob (min_support) trades the two. Dedup is by a
normalized claim key, NOT by claim_id (ids are per-pass)."""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def _norm(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub("", (text or "").lower())).strip()


def claim_key(claim: dict) -> str:
    """Dedup key for the union: normalized claim text + the SOURCE LOCATORS it
    cites, but NOT the exact character offsets. Different samples routinely pick
    slightly different start/end offsets for the same supporting span, so keying
    on exact offsets treats one claim as N distinct claims and the union stops
    deduplicating (atomicity explodes ~N-fold under samples=N). Keying on the
    locator set collapses those jittered-offset duplicates while still keeping a
    claim that cites a genuinely different chunk distinct."""
    locators = sorted(
        str(ev.get("locator") or ev.get("node_id") or "")
        for ev in (claim.get("supporting_evidence") or [])
    )
    return _norm(claim.get("claim", "")) + "||" + repr(locators)


def union_claims(candidate_lists: list[list[dict]], min_support: int = 1) -> list[dict]:
    first: dict[str, dict] = {}
    support: dict[str, int] = {}
    for claims in candidate_lists:
        seen_this_pass: set[str] = set()
        for c in claims:
            k = claim_key(c)
            first.setdefault(k, c)
            if k not in seen_this_pass:      # one pass contributes at most 1 to support
                support[k] = support.get(k, 0) + 1
                seen_this_pass.add(k)
    return [first[k] for k, n in support.items() if n >= min_support]
