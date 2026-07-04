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
from pathlib import Path

from ..compiler.dossier import verbatim_ok

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

METHOD (build your own approach; this is your harness):
Plan, identify the entities, draft each claim with a candidate quote, re-read
each quote against its chunk to confirm it is an exact substring, revise or drop,
then emit. Reason freely first.

FORMAT: After reasoning, emit ONLY the final JSON object inside <output> and
</output> tags -- nothing else inside those tags."""

_PDF_EVIDENCE = '{"node_id": "<chunk_id>", "quote": "<verbatim substring>", "page": <int>}'
_WEB_EVIDENCE = '{"locator": "<chunk_id>", "quote": "<verbatim substring>", "url": "<source url or null>"}'


def build_extraction_prompt(chunks: list[dict], producer: str = "web") -> tuple[str, str]:
    system = _SYSTEM.format(evidence_shape=_PDF_EVIDENCE if producer == "pdf" else _WEB_EVIDENCE)
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


def extract_claims_to_run(run_dir, producer: str, config, backend) -> dict:
    """Read chunks.jsonl, have the backend extract claims/entities/relations,
    drop any claim whose quote is not verbatim in its chunk, and write
    claims.jsonl / entities.jsonl / relations.jsonl into the run directory.
    Returns a summary dict.
    """
    run_dir = Path(run_dir)
    source_id = run_dir.name if producer == "web" else \
        json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")).get("document_id", run_dir.name)
    chunks = _read_jsonl(run_dir / "chunks.jsonl")
    system, user = build_extraction_prompt(chunks, producer)
    parsed = parse_extraction_response(backend.complete(system, user))

    # Gate claims against the exact chunk text the model was shown (see the
    # llm.provider=local design note; the mechanical gate is the backstop).
    chunk_text_by_id = {(c.get("node_id") or c.get("locator")): c.get("text", "") for c in chunks}
    chunk_ids = [cid for cid in chunk_text_by_id if cid]

    def _resolve(emitted: str) -> str | None:
        # A reasoning model often "cleans up" a long chunk id (emits "n002" for
        # "<document_id>:n002"). Accept an exact key, else a chunk id that ends
        # with the emitted label, so the claim isn't dropped over id cosmetics.
        if emitted in chunk_text_by_id:
            return emitted
        if len(emitted) >= 2:
            for cid in chunk_ids:
                if cid.endswith(":" + emitted) or cid.endswith(emitted):
                    return cid
        return None

    id_key = "node_id" if producer == "pdf" else "locator"
    kept, dropped = [], []
    for claim in parsed["claims"]:
        evidence = claim.get("supporting_evidence") or []
        ok = bool(evidence)
        for ev in evidence:
            real = _resolve(str(ev.get(id_key) or ev.get("node_id") or ev.get("locator") or ""))
            if real and verbatim_ok(ev.get("quote") or "", chunk_text_by_id[real]):
                ev[id_key] = real  # rewrite to the canonical chunk id
            else:
                ok = False
        (kept if ok else dropped).append(claim)

    # Entity `mentions` are chunk ids too, and the model abbreviates them the same
    # way it abbreviates evidence ids -- resolve them to canonical chunk ids so
    # entity_mentions actually joins back to chunks/sources downstream. Drop any
    # mention that resolves to no real chunk rather than leave a dangling id.
    for ent in parsed["entities"]:
        ent["mentions"] = [r for m in (ent.get("mentions") or []) if (r := _resolve(str(m)))]

    _write_jsonl(run_dir / "claims.jsonl", kept, {"document_id": source_id})
    _write_jsonl(run_dir / "entities.jsonl", parsed["entities"], {})
    _write_jsonl(run_dir / "relations.jsonl", parsed["relations"], {"document_id": source_id})

    return {
        "written": len(kept),
        "dropped": [c.get("claim_id") for c in dropped],
        "entities": len(parsed["entities"]),
        "relations": len(parsed["relations"]),
    }
