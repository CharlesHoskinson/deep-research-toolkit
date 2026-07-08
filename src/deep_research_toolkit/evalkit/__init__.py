"""Eval-suite helpers: flake-rate statistics for the live-model tier (Task 5),
plus the extraction/prose/adjudication metrics, paired-bootstrap comparison,
and the raw-completion RecordingBackend wrapper consumed by
scripts/eval-pipeline.py (Task 7). Pure Python (RecordingBackend excepted,
which wraps whatever backend the runner gives it) -- no model calls of its
own; the live tier and the eval runner call into this package.

Also exports pool_gold (Task 12): the pure union+dedup that
scripts/build-pooled-gold.py wraps to pool gate-passing claims from multiple
models into one gold set (the fixed eval denominator / Phase-2 SFT target).

And the Anomaly-1 fix (pooled-gold claim_id collisions): both pooled teacher
models mint bNN_c_NNNN-style claim_ids independently per doc, and pool_gold
dedups by claim TEXT (see selfconsistency.claim_key), not id -- so two
different-content claims from two different models can carry the identical
raw id straight into the pooled set, inflating quote-overlap recall via id
conflation in evalkit.metrics.quote_overlap_match (a claim_id that matches
gets ALL same-id rows credited as recalled, not just the one whose quotes
actually matched). namespace_claim_ids tags every claim's id with its source
model's slug BEFORE pool_gold ever sees it, so pooled ids can't collide
within a doc. renamespace_duplicate_claim_ids is the matching migration for
rows pooled before this fix existed (scripts/build-pooled-gold.py
--renamespace)."""
from __future__ import annotations

import re

from ..llm.selfconsistency import union_claims

#: Characters outside evalkit's claim-id regex tolerance: CLAIM_MARKER_RE in
#: ../llm/response.py matches a marker's id against [A-Za-z0-9_\-\.]+ (no
#: colon), so an Ollama "name:tag" model id can't be used verbatim as an id
#: namespace -- its colon (and anything else outside that class) is replaced.
_ID_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_\-\.]+")


def pool_gold(claim_lists: list[list[dict]]) -> list[dict]:
    """Union + dedup gate-passing claims from multiple models into one gold set
    (min_support=1: any model's gate-passed claim is gold). Dedup by claim_key."""
    return union_claims(claim_lists, min_support=1)


def slugify_model_id(model: str) -> str:
    """Model tag -> a namespace-safe slug for a claim_id prefix: runs of
    characters outside CLAIM_MARKER_RE's allowed id charset collapse to a
    single '-' (e.g. "gemma4:e4b" -> "gemma4-e4b"), so a namespaced claim_id
    still round-trips through a [claim:<id>] marker downstream."""
    return _ID_UNSAFE_RE.sub("-", model).strip("-") or "model"


def namespace_claim_ids(claims: list[dict], model: str) -> list[dict]:
    """Prefix every claim's claim_id with its source model's slug (Anomaly-1
    fix, applied at pool time before pool_gold): returns NEW dicts (does not
    mutate `claims`), each with claim_id rewritten to "<model-slug>.<id>".
    Because this runs per model before the lists are pooled, two models that
    independently mint the same raw id can never collide in the pooled
    output -- pool_gold's own dedup logic (by claim_key, unaffected by this
    rename) still decides which claims survive.

    That guarantee rests on a premise this function now VERIFIES rather than
    assumes: one model's kept claims must already carry unique raw ids.
    extract_claims_to_run's batch/pass prefixes are designed to ensure that,
    but an LLM emitting the same claim_id twice within one batch (the
    samples=1 path has no union dedup to collapse it) would slip through as a
    within-model collision the model-slug prefix cannot fix. Raises
    ValueError naming the colliding id(s) -- fail loud, never hand pool_gold
    a list that could write a colliding pool."""
    tag = slugify_model_id(model)
    out = [{**c, "claim_id": f"{tag}.{c.get('claim_id', '')}"} for c in claims]
    dups = duplicate_claim_id_groups(out)
    if dups:
        raise ValueError(
            f"model {model!r} emitted duplicate claim_id(s) within its own claim list "
            f"(namespacing cannot disambiguate a within-model collision): "
            f"{', '.join(sorted(dups))}")
    return out


def duplicate_claim_id_groups(claims: list[dict]) -> dict[str, list[int]]:
    """claim_id -> list of row indices sharing it, in encounter order. Ids
    that appear exactly once are omitted (nothing to disambiguate) -- this is
    the before/after uniqueness signal for the --renamespace migration."""
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(claims):
        groups.setdefault(c.get("claim_id"), []).append(i)
    return {cid: idxs for cid, idxs in groups.items() if len(idxs) > 1}


def renamespace_duplicate_claim_ids(claims: list[dict], models: list[str]) -> list[dict]:
    """Migration for claims pooled BEFORE namespace_claim_ids existed
    (scripts/build-pooled-gold.py --renamespace): disambiguates every
    claim_id that repeats in `claims` by tagging each occurrence with the
    model that must have minted it.

    The committed legacy rows carry no separate per-row source-model field --
    but one isn't needed to do this correctly. pool_gold/union_claims
    processes its candidate model lists in order, one model's list fully
    before the next, and a single model's own claims.jsonl already has
    unique ids (batch-prefixed, and per-pass-tagged under samples>1) -- so
    two rows can only share a raw claim_id if they came from two DIFFERENT
    models. That means the pooled output is exactly the concatenation of
    each model's surviving claims, in `models` order, so the k-th
    file-order occurrence of a repeated id is unambiguously `models[k]`'s
    row (confirmed against the committed corpus: every doc's claim_id batch
    component ascends 0,1,2,... then resets back to 0 exactly once -- the
    model-A-then-model-B seam union_claims produces).

    Ids that appear once are left untouched, so a second pass over
    already-migrated claims (every id now unique) makes no further change --
    idempotent by construction. A duplicate group deeper than len(models)
    (not expected for a 2-model pool) falls back to a "dupN" tag for the
    occurrences beyond the known model list, rather than guessing a model.

    Residual risk, legacy path only: if a model had itself emitted a
    duplicate raw id within its own list (the case namespace_claim_ids now
    rejects at pool time, but which legacy rows were pooled without checking
    for), this occurrence-order attribution would silently assign the second
    same-model occurrence to the wrong model -- ids stay unique either way,
    but the model tag would be a misattribution."""
    groups = duplicate_claim_id_groups(claims)
    out = list(claims)
    for claim_id, idxs in groups.items():
        for occurrence, i in enumerate(idxs):
            model = models[occurrence] if occurrence < len(models) else f"dup{occurrence}"
            tag = slugify_model_id(model)
            out[i] = {**out[i], "claim_id": f"{tag}.{claim_id}"}
    return out
