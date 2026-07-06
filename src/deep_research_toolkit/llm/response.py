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
_FENCE_RE = re.compile(r"\A\s*```[a-zA-Z]*\s*\n(.*?)\n?```\s*\Z", re.DOTALL)


def unfence(text: str) -> str:
    """Unwrap a whole-reply code fence -- a formatting reflex, not content.
    Prose roles apply this mechanically before the citation gate rather than
    trusting the prompt's 'no fences' rule. Mid-body fences are untouched."""
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


def normalize_claim_markers(text: str, allowed_ids: list[str]) -> str:
    """Rewrite a bare [<id>] marker to [claim:<id>] when <id> is a supplied
    claim id. Gemma 4 models were measured citing correctly by id while
    dropping the literal `claim:` prefix; the allowed-id set is closed, so
    the rewrite is unambiguous and mechanical. Unknown bracket contents are
    left alone -- semantic gating still happens in validate_citations."""
    if not allowed_ids:
        return text
    bare = re.compile(r"\[(" + "|".join(re.escape(i) for i in allowed_ids if i) + r")\]")
    return bare.sub(r"[claim:\1]", text)


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


def _fence_stripped(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`\n")
        if t[:4].lower() == "json":
            t = t[4:]
    return t.strip()


def _bracket_slices(text: str) -> list[str]:
    # Widest [...] and {...} slices, ordered by whichever bracket opens first
    # in the text -- same first-bracket-wins rule as extract._loads_lenient,
    # so a top-level object with an array field parses as the object.
    slices = []
    for opener, closer in ("[]", "{}"):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            slices.append((start, text[start:end + 1]))
    return [s for _, s in sorted(slices, key=lambda p: p[0])]


def has_repetition_loop(text: str, max_pattern: int = 20, min_repeats: int = 4,
                        min_words: int = 40) -> bool:
    """True when the text's tail is one phrase repeated over and over --
    the constrained-decoding failure mode Gemma 4 exhibits (ollama#15502).
    Word-level check: for pattern lengths 1..max_pattern, test whether the
    last (pattern * min_repeats) words are the same pattern repeated."""
    words = text.split()
    if len(words) < min_words:
        return False
    for size in range(1, max_pattern + 1):
        window = size * min_repeats
        if window > len(words):
            break
        tail = words[-window:]
        pattern = tail[:size]
        if all(tail[i] == pattern[i % size] for i in range(window)):
            return True
    return False


def parse_json_block(text: str):
    """JSON from a model reply. An <output>...</output> block, when present,
    is authoritative: only its content is considered (fences stripped, then
    bracket-sliced) -- surrounding prose can never leak into the result.
    Without one, the widest bracket slice opening earliest in the reply wins."""
    blocks = _OUTPUT_RE.findall(text)
    scope = _fence_stripped(blocks[-1]) if blocks else _fence_stripped(text)
    for cand in [scope, *_bracket_slices(scope)]:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None
