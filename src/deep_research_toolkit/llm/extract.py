"""Programmatic claim/entity/relation extraction via a Backend (meaningful
under llm.provider=local).

The prompt is a *task brief*, not a rigid schema dump: it hands a reasoning
model (e.g. Ornith-1.0) the goal, the output contract, the hard verbatim-quote
invariant, and the extraction rules, then lets the model plan and self-verify
its own approach before emitting -- playing to a self-scaffolding, coding-tuned
model's strengths. The model reasons freely, then emits the final JSON inside
<output>...</output> so parsing is robust to the reasoning trace.

The verbatim gate is still applied here mechanically as the backstop: every
supporting quote must be an exact substring of the chunk text the model was
shown, so an off-label local model can only under-produce, never corrupt the
corpus.
"""
from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path

from ..common.verbatim import verbatim_ok

SCHEMA_VERSION = "1.0"

_SYSTEM = """\
You are building an extraction harness for a research knowledge base. From the \
CHUNKS in the next message, produce atomic, evidence-backed claims plus the \
entities and relations they mention.

OUTPUT CONTRACT (a typed API -- emit exactly this shape):
{{
  "claims": [{{
    "claim_id": "c_0001",
    "claim": "<one checkable assertion>",
    "claim_type": "architectural|empirical|definitional|comparative",
    "confidence": "high|medium|low",
    "supporting_evidence": [{evidence_shape}]
  }}],
  "entities": [{{"entity_id": "<slug>", "name": "<most formal name>",
                "aliases": ["<variant>"], "type": "<kind>", "mentions": ["<chunk_id>"]}}],
  "relations": [{{"relation_id": "r_0001", "subject": "<entity_id>",
                 "predicate": "<verb_phrase>", "object": "<entity_id or literal>",
                 "supporting_claim": "<claim_id>"}}]
}}

HARD INVARIANT (a precondition, checked mechanically downstream):
For every supporting_evidence quote, `chunk_text.find(quote) != -1` MUST hold --
an exact, contiguous substring of the cited chunk's text, copied character for
character. No paraphrase, no ellipsis, no stitching spans together. A
deterministic gate DROPS any claim whose quote fails this check. Under-produce
rather than approximate: if you cannot find an exact supporting substring, drop
the claim.

RULES:
- One checkable assertion per claim (split compound sentences).
- Quote first, then write the claim around the quote you found.
- Merge mentions of the same thing under one entity_id (most formal name; other
  forms as aliases). mentions are the chunk_ids the entity appears in.
- Only emit a relation a claim actually asserts. Do not force claims, entities,
  or relations the text does not support -- a short solid set beats a long shaky one.

{tail}"""

_TAIL_THINKING = """\
METHOD (build your own approach; this is your harness):
Plan, identify the entities, draft each claim with a candidate quote, re-read
each quote against its chunk to confirm it is an exact substring, revise or drop,
then emit. Reason freely first.

FORMAT: After reasoning, emit ONLY the final JSON object inside <output> and
</output> tags -- nothing else inside those tags."""

_TAIL_DIRECT = """\
Work through the chunks and copy each quote exactly. Output ONLY the JSON object
matching the contract above -- no reasoning, no commentary, no markdown fences."""

_PDF_EVIDENCE = '{"node_id": "<chunk_id>", "quote": "<verbatim substring>", "page": <int>}'
_WEB_EVIDENCE = '{"locator": "<chunk_id>", "quote": "<verbatim substring>", "url": "<source url or null>"}'


def build_extraction_prompt(chunks: list[dict], producer: str = "web",
                            thinking: bool = True) -> tuple[str, str]:
    system = _SYSTEM.format(
        evidence_shape=_PDF_EVIDENCE if producer == "pdf" else _WEB_EVIDENCE,
        tail=_TAIL_THINKING if thinking else _TAIL_DIRECT,
    )
    lines = []
    for c in chunks:
        cid = c.get("node_id") or c.get("locator")
        if producer == "pdf":
            page = (c.get("source") or {}).get("page_start")
            lines.append(f"[{cid} | page {page}]\n{c.get('text', '')}")
        else:
            lines.append(f"[{cid}]\n{c.get('text', '')}")
    user = "CHUNKS:\n\n" + "\n\n".join(lines)
    return system, user


def _loads_lenient(body: str):
    """Parse a JSON object/array, tolerating surrounding prose or a code fence."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body.strip(), flags=re.DOTALL).strip()
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back to slicing out the outermost structure. Pick object-vs-array by
    # whichever bracket appears first, so a bare "[ {..} ]" isn't mis-sliced to
    # its inner object.
    obj_start, arr_start = body.find("{"), body.find("[")
    obj_first = obj_start != -1 and (arr_start == -1 or obj_start < arr_start)
    order = [("{", "}"), ("[", "]")] if obj_first else [("[", "]"), ("{", "}")]
    for open_ch, close_ch in order:
        candidate = _slice(body, open_ch, close_ch)
        if candidate is None:
            continue
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _slice(text: str, open_ch: str, close_ch: str) -> str | None:
    start, end = text.find(open_ch), text.rfind(close_ch)
    return text[start:end + 1] if start != -1 and end != -1 and end > start else None


def parse_extraction_response(text: str) -> dict:
    """Extract {"claims", "entities", "relations"} from the model output.

    Prefers the final JSON inside <output>...</output>; falls back to the whole
    message. Accepts either the full object contract or a bare claims array (so
    older/looser outputs still parse as claims-only)."""
    match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
    obj = _loads_lenient(match.group(1) if match else text)
    if isinstance(obj, list):
        return {"claims": obj, "entities": [], "relations": []}
    if isinstance(obj, dict):
        return {
            "claims": obj.get("claims") or [],
            "entities": obj.get("entities") or [],
            "relations": obj.get("relations") or [],
        }
    return {"claims": [], "entities": [], "relations": []}


# Back-compat alias: callers that only want claims.
def parse_claims_response(text: str) -> list[dict]:
    return parse_extraction_response(text)["claims"]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict], stamp: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"schema_version": SCHEMA_VERSION, **stamp, **row}, ensure_ascii=False) + "\n")


#: Chunks per LLM call. A reasoning model spends its whole token budget thinking
#: about a large chunk set and can run out before emitting the final JSON, so a
#: source is extracted in bounded batches rather than one giant prompt. Small
#: enough that even a verbose reasoning pass fits comfortably under max_tokens.
DEFAULT_BATCH_SIZE = 6

#: How many times a failed (unparseable) batch may be halved and retried before
#: it's counted as a parse failure. 6 -> 3 -> 1 covers the default batch size.
_MAX_RETRY_DEPTH = 2

#: Appended to the user prompt on a halved-batch retry (depth > 0) so the model
#: sees the concrete failure reason instead of an identical prompt -- an
#: identical-prompt retry reproduces the same parse failure.
_RETRY_NOTE = (
    "NOTE: a previous attempt on these chunks failed to parse as the required "
    "JSON. Emit ONLY the contract JSON."
)


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def extract_claims_to_run(run_dir, producer: str, config, backend,
                          batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    """Read chunks.jsonl, have the backend extract claims/entities/relations in
    bounded batches, drop any claim whose quote is not verbatim in its chunk, and
    write claims.jsonl / entities.jsonl / relations.jsonl into the run directory.
    Returns a summary dict (including ``parse_failures``: batches whose output
    could not be parsed, usually reasoning-token truncation).
    """
    run_dir = Path(run_dir)
    source_id = run_dir.name if producer == "web" else \
        json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")).get("document_id", run_dir.name)
    chunks = _read_jsonl(run_dir / "chunks.jsonl")

    chunk_text_by_id = {(c.get("node_id") or c.get("locator")): c.get("text", "") for c in chunks}
    chunk_ids = [cid for cid in chunk_text_by_id if cid]

    def _resolve(emitted: str) -> str | None:
        # A reasoning model often "cleans up" a long chunk id (emits "n002" for
        # "<document_id>:n002"). Accept an exact key, then a ":"-delimited suffix
        # (the real id shape), then a bare suffix only if it's unambiguous -- so a
        # short label can't silently resolve to the wrong chunk.
        if emitted in chunk_text_by_id:
            return emitted
        if len(emitted) < 2:
            return None
        delimited = [cid for cid in chunk_ids if cid.endswith(":" + emitted)]
        if delimited:
            return delimited[0] if len(delimited) == 1 else None
        bare = [cid for cid in chunk_ids if cid.endswith(emitted)]
        return bare[0] if len(bare) == 1 else None

    id_key = "node_id" if producer == "pdf" else "locator"
    kept, dropped = [], []
    entities_by_id: dict[str, dict] = {}
    relations: list[dict] = []
    parse_failures = 0
    batch_list = list(_batches(chunks, max(1, batch_size)))
    multi = len(batch_list) > 1

    thinking = getattr(backend, "thinking", True)
    # A work queue rather than a fixed loop, so a batch whose output can't be
    # parsed (usually token truncation) is retried as smaller halves instead of
    # silently lost -- bounded by depth so it terminates.
    queue: deque[tuple[list[dict], int]] = deque((b, 0) for b in batch_list)
    batch_no = 0
    while queue:
        batch, depth = queue.popleft()
        system, user = build_extraction_prompt(batch, producer, thinking=thinking)
        sampling = {}
        if depth > 0:  # a halved batch dispatched after a parse failure -- a retry
            user = user + "\n\n" + _RETRY_NOTE
            sampling = {"temperature": 0.25}
        parsed = parse_extraction_response(backend.complete(system, user, **sampling))
        if not (parsed["claims"] or parsed["entities"] or parsed["relations"]):
            if len(batch) > 1 and depth < _MAX_RETRY_DEPTH:
                mid = len(batch) // 2
                queue.appendleft((batch[mid:], depth + 1))
                queue.appendleft((batch[:mid], depth + 1))
                multi = True  # splitting a lone batch means ids now need a prefix
            else:
                parse_failures += 1
            continue

        # Give ids a batch prefix (only when batched) so they stay unique across
        # batches while keeping each relation's supporting_claim reference valid.
        prefix = f"b{batch_no:02d}_" if multi else ""
        batch_no += 1
        for claim in parsed["claims"]:
            if not isinstance(claim, dict):
                continue  # some models emit a bare-string claim -> no evidence, can't pass the gate
            claim["claim_id"] = prefix + str(claim.get("claim_id", ""))
            evidence = claim.get("supporting_evidence") or []
            ok = bool(evidence)
            for ev in evidence:
                real = _resolve(str(ev.get(id_key) or ev.get("node_id") or ev.get("locator") or ""))
                if real and verbatim_ok(ev.get("quote") or "", chunk_text_by_id[real]):
                    ev[id_key] = real  # rewrite to the canonical chunk id
                else:
                    ok = False
            (kept if ok else dropped).append(claim)

        for ent in parsed["entities"]:
            if not isinstance(ent, dict):
                continue
            eid = ent.get("entity_id")
            if not eid:
                continue
            # Entity `mentions` are abbreviated chunk ids too -- resolve them so
            # entity_mentions joins back to chunks; drop unresolvable ones.
            ment = [r for m in (ent.get("mentions") or []) if (r := _resolve(str(m)))]
            if eid in entities_by_id:  # same entity seen in an earlier batch -> merge
                cur = entities_by_id[eid]
                cur["mentions"] = sorted(set(cur.get("mentions") or []) | set(ment))
                cur["aliases"] = sorted(set(cur.get("aliases") or []) | set(ent.get("aliases") or []))
            else:
                ent["mentions"] = sorted(set(ment))
                entities_by_id[eid] = ent

        for rel in parsed["relations"]:
            if not isinstance(rel, dict):
                continue
            rel["relation_id"] = prefix + str(rel.get("relation_id", ""))
            if rel.get("supporting_claim"):
                rel["supporting_claim"] = prefix + str(rel["supporting_claim"])
            relations.append(rel)

    # Drop relations whose supporting_claim was gate-dropped (or never existed),
    # so no relation points at a claim_id that isn't in claims.jsonl. A relation
    # with no supporting_claim is kept as-is.
    kept_ids = {c["claim_id"] for c in kept}
    relations = [r for r in relations if not r.get("supporting_claim") or r["supporting_claim"] in kept_ids]

    _write_jsonl(run_dir / "claims.jsonl", kept, {"document_id": source_id})
    _write_jsonl(run_dir / "entities.jsonl", list(entities_by_id.values()), {})
    _write_jsonl(run_dir / "relations.jsonl", relations, {"document_id": source_id})

    return {
        "written": len(kept),
        "dropped": [c.get("claim_id") for c in dropped],
        "entities": len(entities_by_id),
        "relations": len(relations),
        "batches": batch_no,
        "parse_failures": parse_failures,
    }
