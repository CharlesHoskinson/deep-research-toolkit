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


def validate_citations(text: str, allowed_ids: list[str], *,
                       min_citable_for_ratio: int = 4,
                       min_coverage: float = 0.3) -> dict:
    """Closed-set citation check + coverage rule. With few citable claims the
    ratio floor is statistically meaningless (2 claims, one uncited = 0.5 <
    0.3-floor false alarm), so below `min_citable_for_ratio` we require EVERY
    allowed id to be cited (absolute rule); at/above it the ratio floor holds."""
    allowed = set(allowed_ids)
    cited_all = extract_claim_ids(text)
    cited = [c for c in cited_all if c in allowed]
    unknown = [c for c in cited_all if c not in allowed]
    coverage = (len(cited) / len(allowed)) if allowed else 0.0
    if len(allowed) < min_citable_for_ratio:
        rule = "absolute"
        coverage_ok = set(cited) >= allowed  # every allowed id cited
    else:
        rule = "ratio"
        coverage_ok = coverage >= min_coverage
    return {"cited": cited, "unknown": unknown, "coverage": coverage,
            "coverage_ok": coverage_ok, "rule": rule}


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


_TOKEN_TRIM = ".,;:!?|`*_-—()[]{}\"'"


def _normalized_words(text: str) -> list[str]:
    words = []
    for raw in text.split():
        w = raw.strip(_TOKEN_TRIM).lower()
        if w:
            words.append(w)
    return words


def has_repetition_loop(text: str, max_pattern: int = 20, min_repeats: int = 6,
                        min_region_words: int = 40) -> bool:
    """True when anywhere in the text one short phrase repeats consecutively
    enough to cover a long region -- the constrained-decoding degeneration
    Gemma 4 exhibits (ollama#15502), including inside JSON string values.
    Tokens are lowercased and punctuation-trimmed so sampler jitter cannot
    hide a loop, and markdown table furniture ('|', '---') normalizes away.
    Thresholds are conservative: real loops run for many dozens of words,
    while legitimate repetition (separator rows, n/a cells) stays short.
    Patterns longer than max_pattern words are out of scope by design."""
    words = _normalized_words(text)
    n = len(words)
    if n < min_region_words:
        return False
    for size in range(1, max_pattern + 1):
        if size * min_repeats > n:
            break
        i = 0
        while i + size <= n:
            j = i + size
            while j + size <= n and words[j:j + size] == words[i:i + size]:
                j += size
            repeats = (j - i) // size
            if repeats >= min_repeats and repeats * size >= min_region_words:
                return True
            i += max(1, (repeats - 1) * size)
    return False


_REPETITION_CORRECTION = (
    "Your previous reply degenerated into repetition. Write the reply normally."
)


def checked_complete(backend, system: str, user: str, **sampling) -> str:
    """complete() with the repetition-loop guard: a looping reply gets one
    corrected retry (temperature raised unless the caller already set one --
    loops are a greedy-decoding artifact); a second loop is an error."""
    reply = backend.complete(system, user, **sampling)
    if has_repetition_loop(reply):
        retry_sampling = dict(sampling)
        retry_sampling.setdefault("temperature", 0.25)
        reply = backend.complete(system, user + "\n\n" + _REPETITION_CORRECTION, **retry_sampling)
        if has_repetition_loop(reply):
            raise ValueError("model reply degenerated into repetition")
    return reply


def generate_cited(backend, system: str, user: str, allowed_ids: list[str], *,
                   min_coverage: float, kind: str, correction_unknown: str,
                   correction_low_coverage: str, citation_error: type) -> dict:
    """Shared gate-and-retry loop for prose roles: complete (loop-guarded) ->
    unfence -> normalize markers -> gate unknown ids (one corrected retry at
    temperature 0.25) -> gate coverage (one corrected retry, whose reply is
    re-gated for unknown ids before the coverage re-check). Returns
    {"text", "citations"}; raises citation_error on unknown ids surviving a
    retry, ValueError on a coverage floor miss after retry."""
    def _attempt(prompt: str, **sampling) -> tuple[str, dict]:
        text = normalize_claim_markers(
            unfence(checked_complete(backend, system, prompt, **sampling)), allowed_ids)
        return text, validate_citations(text, allowed_ids, min_coverage=min_coverage)

    text, report = _attempt(user)
    if report["unknown"]:
        text, report = _attempt(
            user + "\n\n" + correction_unknown.format(bad=", ".join(report["unknown"])),
            temperature=0.25)
        if report["unknown"]:
            raise citation_error(f"model cited unknown claim ids after retry: {report['unknown']}")
    if not report["coverage_ok"]:
        note = correction_low_coverage.format(
            n=len(report["cited"]), total=len(allowed_ids), kind=kind)
        text, report = _attempt(user + "\n\n" + note, temperature=0.25)
        if report["unknown"]:
            raise citation_error(f"model cited unknown claim ids after retry: {report['unknown']}")
        if not report["coverage_ok"]:
            raise ValueError(
                f"{kind} fails citation coverage ({report['rule']}): "
                f"cited {len(report['cited'])}/{len(allowed_ids)}")
    return {"text": text, "citations": report}


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
