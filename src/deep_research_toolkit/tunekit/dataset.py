"""SFT dataset harness (design doc §6.1): gate-filtered rejection sampling
("RFT/RSFT" -- sample k candidates, keep gate-passers) with DART-style
difficulty-aware k escalation, a teacher-per-slice router, near-dup dedup,
a hard eval-corpus contamination guard, and a provenance manifest. Every
piece here is PURE orchestration over an INJECTED teacher callable -- no
network/model access of its own; real teachers (e4b self-distillation,
qwen3:30b-a3b, a frontier API) are wired in by the caller, not this module.

This deliberately reuses THE SAME code paths production extraction uses, so
the SFT target distribution matches the served contract exactly:
- the span gate (`common.verbatim.span_ok`/`slice_span`) -- the identical
  mechanical check `llm.extract.extract_claims_to_run` applies per claim;
- `llm.extract.parse_claims_response` to parse a teacher's raw completion;
- `llm.extract.build_extraction_prompt` to render the system/user turns a
  conversation-format example's prompt is built from;
- `llm.selfconsistency.claim_key` (the same normalized-text+locator key the
  production union/dedup step uses) for near-dup dedup across the dataset.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..common.verbatim import slice_span, span_ok
from ..llm.extract import build_extraction_prompt, parse_claims_response
from ..llm.selfconsistency import claim_key

#: DART-style difficulty escalation ladder (design doc §6.1: "escalate k
#: (4 -> 16 -> 64) on low-yield / bait / high-density chunks").
DEFAULT_K_LADDER: tuple[int, ...] = (4, 16, 64)

#: A chunk accepted at least this many gate-passing claims in a round is
#: considered adequately sampled; below this, escalate to the next k.
DEFAULT_YIELD_FLOOR = 1

#: Slice tags (matching tests/fixtures/eval-corpus/corpus-index.json's
#: `chunks.<locator>.slices` vocabulary) that mark a chunk as a priori hard,
#: per design doc §6.1 ("low-yield / bait / high-density chunks"). A chunk
#: carrying one of these skips the cheapest ladder rung and starts sampling
#: at the next one, instead of waiting to observe low yield first.
DIFFICULT_SLICE_TAGS = frozenset({"bait", "dense-facts", "table", "long"})

#: Version tag for the gate this module applies -- stamped into the dataset
#: manifest so a future gate change is a visible, versioned drift, mirroring
#: how llm.extract.SCHEMA_VERSION documents the extraction contract version.
VERBATIM_GATE_VERSION = "span-v2.0"


# ---------------------------------------------------------------------------
# Teacher router (design doc §6.1: "teacher-per-slice by gate-filtered
# Pass@k, not by size" -- e4b bulk / qwen3:30b-a3b recall slice / a frontier
# API teacher for the bait slice). An interface + config: real teacher
# callables are wired in by the caller (scripts/build-sft-dataset.py), never
# constructed here.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TeacherRoute:
    """`role` names WHY this teacher was chosen (bulk/recall/bait, for the
    manifest's generator-digest breakdown); `model` is the model id/tag a
    caller resolves to an actual teacher callable."""
    role: str
    model: str


#: Do not use gemma4:31b as an SFT teacher (design doc §6.1: "under-produces
#: on extraction, recall ~0.40"). Every teacher route is checked against this
#: set; a route naming a banned model raises immediately rather than quietly
#: distilling from a bad teacher.
BANNED_TEACHER_MODELS = frozenset({"gemma4:31b", "gemma-4:31b"})

#: Slice tag -> teacher route. Only slices needing a NON-default teacher are
#: listed; anything else falls back to DEFAULT_TEACHER_ROUTE (e4b bulk).
DEFAULT_ROUTER_TABLE: dict[str, TeacherRoute] = {
    "bait": TeacherRoute("bait", "frontier"),
    "dense-facts": TeacherRoute("recall", "qwen3:30b-a3b"),
}

#: e4b self-distillation for bulk (design doc §6.1).
DEFAULT_TEACHER_ROUTE = TeacherRoute("bulk", "e4b")


class BannedTeacherError(ValueError):
    """Raised when a route names a model design doc §6.1 forbids as an SFT
    teacher (currently only gemma4:31b -- see BANNED_TEACHER_MODELS)."""


def route_for_chunk(chunk: dict, router_table: dict[str, TeacherRoute] = DEFAULT_ROUTER_TABLE,
                    fallback: TeacherRoute = DEFAULT_TEACHER_ROUTE) -> TeacherRoute:
    """Picks the first `router_table` entry whose slice tag is present on
    `chunk["slices"]` (first match in `chunk["slices"]`'s own order, so a
    caller controls precedence by how it tags a multi-slice chunk); falls
    back to `fallback` (bulk/e4b) when no tag matches. Raises
    BannedTeacherError if the resolved route names a banned model."""
    route = fallback
    for tag in chunk.get("slices") or []:
        if tag in router_table:
            route = router_table[tag]
            break
    if route.model in BANNED_TEACHER_MODELS:
        raise BannedTeacherError(
            f"{route.model!r} is banned as an SFT teacher (design doc §6.1: "
            "under-produces on extraction, recall ~0.40)")
    return route


def is_a_priori_difficult(chunk: dict) -> bool:
    """A chunk is a priori hard (design doc §6.1) if any of its slice tags
    is in DIFFICULT_SLICE_TAGS -- used to skip the cheapest DART rung
    instead of waiting to observe low yield reactively."""
    return bool(set(chunk.get("slices") or []) & DIFFICULT_SLICE_TAGS)


# ---------------------------------------------------------------------------
# Span gate -- mirrors llm.extract.extract_claims_to_run's per-claim gate
# exactly (single-chunk case; DART sampling is always one chunk at a time).
# ---------------------------------------------------------------------------

def gate_claim(claim: dict, chunk: dict, producer: str = "web") -> dict | None:
    """Applies the SAME span-gate rule `llm.extract.extract_claims_to_run`
    applies per claim: every `supporting_evidence` entry must name THIS
    chunk's locator/node_id and its start_char/end_char must
    `common.verbatim.span_ok`-slice the chunk's text (matching an echoed
    `quote`, if the teacher supplied one). A claim with no evidence, or any
    evidence failing the check, is rejected (returns None). A gate-passing
    claim is returned as a NEW dict (the input is never mutated) with
    `citable: True` and each evidence's `quote` rewritten to the canonical
    derived slice -- exactly as production does."""
    id_key = "node_id" if producer == "pdf" else "locator"
    chunk_id = chunk.get(id_key) or chunk.get("node_id") or chunk.get("locator")
    chunk_text = chunk.get("text", "")

    evidence = claim.get("supporting_evidence") or []
    if not evidence:
        return None

    ok = True
    new_evidence = []
    for ev in evidence:
        if not isinstance(ev, dict):
            ok = False
            continue
        locator = ev.get(id_key) or ev.get("node_id") or ev.get("locator")
        if locator != chunk_id:
            ok = False
            new_evidence.append(dict(ev))
            continue
        start, end = ev.get("start_char"), ev.get("end_char")
        if span_ok(start, end, chunk_text, ev.get("quote")):
            new_ev = dict(ev)
            new_ev[id_key] = chunk_id
            new_ev["quote"] = slice_span(chunk_text, start, end)
            new_evidence.append(new_ev)
        else:
            ok = False
            new_evidence.append(dict(ev))

    if not ok:
        return None

    gated = dict(claim)
    gated["supporting_evidence"] = new_evidence
    gated["citable"] = True
    return gated


# ---------------------------------------------------------------------------
# DART-style k-escalation sampler
# ---------------------------------------------------------------------------

#: teacher(chunk_batch, k, temperature) -> list[str] -- k raw completions
#: (each a full model reply, parsed with llm.extract.parse_claims_response).
TeacherFn = Callable[[list[dict], int, float], list[str]]


def escalating_k_sample(chunk: dict, teacher: TeacherFn,
                        k_ladder: tuple[int, ...] = DEFAULT_K_LADDER,
                        yield_floor: int = DEFAULT_YIELD_FLOOR,
                        temperature: float = 1.0, producer: str = "web") -> dict:
    """Samples the teacher at increasing k (design doc §6.1: "escalate k
    (4 -> 16 -> 64) on low-yield/bait/high-density chunks"), gate-filtering
    each round's candidates, and stopping at the first round whose accepted
    count clears `yield_floor` (or after exhausting the ladder). A chunk
    flagged `is_a_priori_difficult` skips the cheapest rung -- it starts at
    the SECOND ladder entry -- rather than waiting to observe a low yield at
    k=4 first.

    Sampling at T~=1.0 (design doc §6.1: "mode coverage; the cheap gate
    removes garbage") is the caller's default via `temperature`; each round's
    raw completions are parsed with `llm.extract.parse_claims_response` (the
    exact production parser -- lenient to a fenced/prose-wrapped reply) and
    every parsed claim is passed through `gate_claim`.

    Returns `{"claims": [gate-passed claim, ...], "rounds": [{"k", "raw",
    "accepted"}, ...], "escalated": bool}`. `claims` is the LAST round's
    accepted set (escalation replaces, not accumulates, across ladder rungs
    -- a higher k re-samples the whole chunk, it does not top up the
    previous round)."""
    start_idx = 1 if is_a_priori_difficult(chunk) and len(k_ladder) > 1 else 0
    rounds: list[dict] = []
    accepted: list[dict] = []

    for k in k_ladder[start_idx:]:
        raw_completions = teacher([chunk], k, temperature)
        round_accepted: list[dict] = []
        for raw in raw_completions:
            for claim in parse_claims_response(raw):
                if not isinstance(claim, dict):
                    continue
                gated = gate_claim(claim, chunk, producer=producer)
                if gated is not None:
                    round_accepted.append(gated)
        rounds.append({"k": k, "raw": len(raw_completions), "accepted": len(round_accepted)})
        accepted = round_accepted
        if len(round_accepted) >= yield_floor:
            break

    return {"claims": accepted, "rounds": rounds, "escalated": start_idx > 0 or len(rounds) > 1}


# ---------------------------------------------------------------------------
# Near-dup dedup -- reuses llm.selfconsistency.claim_key
# ---------------------------------------------------------------------------

def dedup_claims(claims: list[dict]) -> list[dict]:
    """Drops near-duplicate claims using selfconsistency.claim_key (the SAME
    normalized-claim-text + source-locator key production's cross-sample
    union uses), applied here across the WHOLE accepted dataset rather than
    across samples of one chunk. Keeps the FIRST occurrence of each key, in
    input order."""
    seen: set[str] = set()
    out: list[dict] = []
    for c in claims:
        k = claim_key(c)
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Contamination guard (HARD -- design doc §6.1: "Never train on
# tests/fixtures/eval-corpus"). Checked by locator AND by content hash, so a
# copy of an eval chunk under a different locator is still caught.
# ---------------------------------------------------------------------------

class ContaminationError(ValueError):
    """Raised when a candidate training chunk is present in the eval corpus
    -- by locator match against corpus-index.json, or by exact-text-hash
    match against the eval corpus's actual chunk text."""


def _normalized_text_hash(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()


def load_contamination_index(eval_corpus_dir) -> dict:
    """Builds the HARD contamination guard's lookup tables from
    `eval_corpus_dir` (normally `tests/fixtures/eval-corpus`): every locator
    named in `corpus-index.json`'s `chunks` map, PLUS the content hash of
    every chunk's actual text read from each doc dir's `chunks.jsonl` (when
    present) -- so a differently-locatored duplicate/copy of an eval chunk's
    text is caught too, not just an exact locator match. A missing
    `corpus-index.json` or missing per-doc `chunks.jsonl` degrades to an
    empty set for that half of the check rather than raising (the guard
    itself must never be the reason a legitimate build fails to run)."""
    eval_corpus_dir = Path(eval_corpus_dir)
    index_path = eval_corpus_dir / "corpus-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    locators = set((index.get("chunks") or {}).keys())

    text_hashes: set[str] = set()
    if eval_corpus_dir.is_dir():
        for doc_dir in sorted(p for p in eval_corpus_dir.iterdir() if p.is_dir()):
            chunks_path = doc_dir / "chunks.jsonl"
            if not chunks_path.is_file():
                continue
            with open(chunks_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    text_hashes.add(_normalized_text_hash(row.get("text", "")))

    return {"locators": locators, "text_hashes": text_hashes}


def assert_not_contaminated(chunk: dict, contamination_index: dict) -> None:
    """Raises ContaminationError if `chunk` is present in the eval corpus --
    by locator (checked first, the cheap and exact signal) or by its text's
    content hash (catches a copy/duplicate under a different locator)."""
    locator = chunk.get("locator") or chunk.get("node_id")
    if locator is not None and locator in contamination_index["locators"]:
        raise ContaminationError(
            f"chunk {locator!r} is present in the eval corpus (locator match) -- "
            "refusing to include it in the SFT dataset (test-set contamination)")
    text_hash = _normalized_text_hash(chunk.get("text", ""))
    if text_hash in contamination_index["text_hashes"]:
        raise ContaminationError(
            f"chunk {locator!r} text matches an eval-corpus chunk by content hash -- "
            "refusing to include it in the SFT dataset (test-set contamination)")


# ---------------------------------------------------------------------------
# Conversation-format output -- matches the extract contract exactly
# ---------------------------------------------------------------------------

def to_conversation_record(chunk: dict, claims: list[dict], producer: str = "web",
                           thinking: bool = False) -> dict:
    """One SFT example in conversation format. system/user turns are
    RENDERED BY `llm.extract.build_extraction_prompt` for this one chunk --
    byte-identical to what production sends the model -- so the fine-tune
    target distribution is the production contract, not a bespoke
    training-only prompt. The assistant turn is the bare
    `{"claims": [...], "entities": [], "relations": []}` contract object
    (only `claims` populated -- a DART/rejection-sampled example teaches
    claim extraction, not entity/relation extraction), wrapped in
    `<output>...</output>` when `thinking=True` (matching
    `parse_extraction_response`'s thinking-mode contract) and left bare when
    `thinking=False` (matching its direct-mode contract).

    Default `thinking=False`: a distilled SFT target should not need to
    reproduce a reasoning trace to hit the contract (design decision -- see
    the runbook amendment note; revisit if a target model's serving profile
    changes)."""
    system, user = build_extraction_prompt([chunk], producer=producer, thinking=thinking)
    assistant_obj = {"claims": claims, "entities": [], "relations": []}
    body = json.dumps(assistant_obj, ensure_ascii=False)
    assistant = f"<output>{body}</output>" if thinking else body
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "locator": chunk.get("locator") or chunk.get("node_id"),
    }


# ---------------------------------------------------------------------------
# Deterministic 10% val split
# ---------------------------------------------------------------------------

def split_train_val(records: list[dict], val_fraction: float = 0.10, seed: int = 42) -> tuple[list, list]:
    """Deterministic shuffle-then-split via `random.Random(seed)` -- the
    same (records, val_fraction, seed) triple always yields the same split.
    `val` gets `round(n * val_fraction)` records (at least 1, when
    `val_fraction > 0` and there is at least one record); everything else
    goes to `train`. An empty `records` list returns `([], [])`."""
    n = len(records)
    if n == 0:
        return [], []
    order = list(range(n))
    random.Random(seed).shuffle(order)
    n_val = (max(1, round(n * val_fraction)) if val_fraction > 0 else 0)
    n_val = min(n_val, n)
    val_idx = set(order[:n_val])
    train = [records[i] for i in range(n) if i not in val_idx]
    val = [records[i] for i in range(n) if i in val_idx]
    return train, val


# ---------------------------------------------------------------------------
# Dataset provenance manifest
# ---------------------------------------------------------------------------

def dataset_hash(records: list[dict]) -> str:
    """sha256 over the final JSONL body -- each record serialized with
    canonical (sorted) key order, one per line, in list order -- the
    immutable join key linking a dataset to every run/registry row trained
    on it (design doc §6.1/§6.4)."""
    body = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records)
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def build_manifest(train: list[dict], val: list[dict], *, generator_model_digests: dict,
                   source_corpus_hash: str | None, n_gate_rejected: int,
                   n_dedup_dropped: int, verbatim_gate_version: str = VERBATIM_GATE_VERSION) -> dict:
    """Design doc §6.1's manifest fields: `dataset_hash` is computed over
    train+val TOGETHER (train then val, in that order) so one hash covers
    the whole shipped dataset; `n_accepted`/`n_rejected`/`acceptance_rate`
    account for every claim the teacher(s) produced, gate-rejected or
    dedup-dropped alike, so the rate reflects the true yield of the whole
    pipeline, not just the post-dedup survivors."""
    all_records = train + val
    n_accepted = len(all_records)
    n_rejected = n_gate_rejected + n_dedup_dropped
    total = n_accepted + n_rejected
    return {
        "dataset_hash": dataset_hash(all_records),
        "generator_model_digests": dict(generator_model_digests),
        "verbatim_gate_version": verbatim_gate_version,
        "source_corpus_hash": source_corpus_hash,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "n_gate_rejected": n_gate_rejected,
        "n_dedup_dropped": n_dedup_dropped,
        "n_train": len(train),
        "n_val": len(val),
        "acceptance_rate": (n_accepted / total) if total else None,
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def build_sft_dataset(chunks: list[dict], teachers: dict[str, TeacherFn],
                      contamination_index: dict, *,
                      router_table: dict[str, TeacherRoute] = DEFAULT_ROUTER_TABLE,
                      fallback_route: TeacherRoute = DEFAULT_TEACHER_ROUTE,
                      k_ladder: tuple[int, ...] = DEFAULT_K_LADDER,
                      yield_floor: int = DEFAULT_YIELD_FLOOR,
                      temperature: float = 1.0, producer: str = "web",
                      val_fraction: float = 0.10, seed: int = 42,
                      source_corpus_hash: str | None = None) -> dict:
    """End-to-end (but network/model-free) dataset build: for every chunk,
    (1) raise on eval-corpus contamination, (2) route to a teacher by slice
    tag, (3) run the DART k-escalation sampler, gate-filtering as it goes.
    All ACCEPTED claims across every chunk are THEN deduped together
    (`dedup_claims`), so a near-duplicate claim surfaced by two different
    chunks/rounds is caught even though dedup couldn't happen per-round.
    Surviving claims are regrouped by their originating chunk into one
    conversation record per chunk (`to_conversation_record`), split
    train/val (`split_train_val`), and wrapped in a provenance manifest
    (`build_manifest`).

    `teachers` maps a TeacherRoute's `model` name to the actual callable
    (design doc §6.1: "an interface + config, real teachers wired later") --
    a route naming a model not present in `teachers` raises KeyError
    immediately (fail fast, not a silently-skipped chunk).

    Returns `{"train": [...], "val": [...], "manifest": {...},
    "escalation_log": [...]}`."""
    escalation_log: list[dict] = []
    claim_chunk_pairs: list[tuple[dict, dict]] = []  # (claim, its originating chunk)
    generator_digests: dict[str, int] = {}
    n_gate_rejected = 0

    for chunk in chunks:
        assert_not_contaminated(chunk, contamination_index)
        route = route_for_chunk(chunk, router_table=router_table, fallback=fallback_route)
        teacher = teachers[route.model]

        result = escalating_k_sample(chunk, teacher, k_ladder=k_ladder, yield_floor=yield_floor,
                                     temperature=temperature, producer=producer)
        n_accepted_round = len(result["claims"])
        # A per-chunk gate-rejection count is only meaningful relative to the
        # LAST round's own raw/accepted pair (a completion can carry >1
        # claim, and escalation re-samples rather than tops up) -- so the
        # final round's raw-minus-accepted is the tally, not a sum across
        # escalation rounds.
        last_round = result["rounds"][-1] if result["rounds"] else {"raw": 0, "accepted": 0}
        n_gate_rejected += max(0, last_round["raw"] - last_round["accepted"])

        generator_digests[route.model] = generator_digests.get(route.model, 0) + n_accepted_round
        escalation_log.append({
            "locator": chunk.get("locator") or chunk.get("node_id"),
            "teacher_role": route.role,
            "teacher_model": route.model,
            "rounds": result["rounds"],
            "escalated": result["escalated"],
        })
        for claim in result["claims"]:
            claim_chunk_pairs.append((claim, chunk))

    n_before_dedup = len(claim_chunk_pairs)
    deduped_keys: set[str] = set()
    deduped_pairs: list[tuple[dict, dict]] = []
    for claim, chunk in claim_chunk_pairs:
        k = claim_key(claim)
        if k in deduped_keys:
            continue
        deduped_keys.add(k)
        deduped_pairs.append((claim, chunk))
    n_dedup_dropped = n_before_dedup - len(deduped_pairs)

    claims_by_locator: dict[str, list[dict]] = {}
    chunk_by_locator: dict[str, dict] = {}
    order: list[str] = []
    for claim, chunk in deduped_pairs:
        loc = chunk.get("locator") or chunk.get("node_id")
        if loc not in claims_by_locator:
            claims_by_locator[loc] = []
            chunk_by_locator[loc] = chunk
            order.append(loc)
        claims_by_locator[loc].append(claim)

    records = [
        to_conversation_record(chunk_by_locator[loc], claims_by_locator[loc], producer=producer)
        for loc in order
    ]

    train, val = split_train_val(records, val_fraction=val_fraction, seed=seed)
    manifest = build_manifest(
        train, val, generator_model_digests=generator_digests,
        source_corpus_hash=source_corpus_hash, n_gate_rejected=n_gate_rejected,
        n_dedup_dropped=n_dedup_dropped)

    return {"train": train, "val": val, "manifest": manifest, "escalation_log": escalation_log}
