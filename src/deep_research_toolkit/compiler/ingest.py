"""Walk the three producers (knowledge_base/, pdf-runs/, research-runs/)
into flat row dicts ready to insert into the DuckDB index."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..common.frontmatter import read_okf, resolve_link
from .schema import normalize_evidence


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_wiki_pages(kb_dir: Path) -> list[dict[str, Any]]:
    # Resolve once so relative_to agrees with resolve_link's resolved paths.
    kb_dir = Path(kb_dir).resolve()
    rows: list[dict[str, Any]] = []
    for md in sorted(kb_dir.rglob("*.md")):
        page = read_okf(md)
        if page is None or "type" not in page.frontmatter:
            continue
        rel = md.relative_to(kb_dir).as_posix()
        links = []
        for target in page.links:
            resolved = resolve_link(target, md, kb_dir)
            try:
                links.append(resolved.relative_to(kb_dir).as_posix())
            except ValueError:
                continue  # link points outside the kb; not a graph edge
        rows.append({
            "path": rel,
            "type": page.frontmatter.get("type"),
            "title": page.frontmatter.get("title"),
            "status": page.frontmatter.get("status"),
            "timestamp": str(page.frontmatter.get("timestamp", "")),
            "body": page.body,
            "frontmatter_json": json.dumps(page.frontmatter, default=str),
            "links": links,
        })
    return rows


def _source_id(run_dir: Path, producer: str) -> str:
    if producer == "pdf":
        manifest = run_dir / "manifest.json"
        if manifest.is_file():
            return json.loads(manifest.read_text(encoding="utf-8")).get("document_id", run_dir.name)
    return run_dir.name


def iter_run_claims(run_dir: Path, producer: str) -> tuple[list[dict], list[dict]]:
    run_dir = Path(run_dir)
    sid = _source_id(run_dir, producer)
    claim_rows, ev_rows = [], []
    for claim in _read_jsonl(run_dir / "claims.jsonl"):
        cid = claim.get("claim_id")
        claim_rows.append({
            "claim_id": cid, "producer": producer, "source_id": sid,
            "claim": claim.get("claim", ""), "claim_type": claim.get("claim_type"),
            "confidence": claim.get("confidence"),
        })
        for ref in normalize_evidence(claim, producer, sid):
            ev_rows.append({
                "claim_id": cid, "producer": ref.producer, "source_id": ref.source_id,
                "locator": ref.locator, "page": ref.page, "url": ref.url, "quote": ref.quote,
            })
    return claim_rows, ev_rows


def iter_run_entities(run_dir: Path, producer: str) -> tuple[list[dict], list[dict]]:
    run_dir = Path(run_dir)
    sid = _source_id(run_dir, producer)
    entity_rows, mention_rows = [], []
    for ent in _read_jsonl(run_dir / "entities.jsonl"):
        eid = ent.get("entity_id")
        entity_rows.append({
            "entity_id": eid, "name": ent.get("name"), "type": ent.get("type"),
            "aliases_json": json.dumps(ent.get("aliases") or []), "producer": producer, "source_id": sid,
        })
        for locator in ent.get("mentions") or []:
            mention_rows.append({"entity_id": eid, "locator": locator, "producer": producer, "source_id": sid})
    return entity_rows, mention_rows


def iter_run_relations(run_dir: Path, producer: str) -> list[dict]:
    run_dir = Path(run_dir)
    sid = _source_id(run_dir, producer)
    rows = []
    for rel in _read_jsonl(run_dir / "relations.jsonl"):
        rows.append({
            "relation_id": rel.get("relation_id"), "subject": rel.get("subject"),
            "predicate": rel.get("predicate"), "object": rel.get("object"),
            "supporting_claim": rel.get("supporting_claim"), "producer": producer, "source_id": sid,
        })
    return rows


def merge_entities(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse entity rows that share an entity_id across runs into one row.

    Different sources independently emit the same entity_id (the model picks the
    same slug), each with its own name/aliases/type. Without merging, the
    ``entities`` table has several rows per id and ``get_entity`` picks one
    arbitrarily. This unions aliases (folding in the non-canonical names), picks a
    deterministic canonical name (longest, then lexically smallest) and the most
    common type. ``entity_mentions`` already aggregate by entity_id, so only the
    ``entities`` table needs this.
    """
    from collections import Counter

    grouped: dict[str, dict[str, Any]] = {}
    for r in rows:
        eid = r.get("entity_id")
        if not eid:
            continue
        g = grouped.setdefault(eid, {"names": set(), "aliases": set(), "types": [],
                                     "producer": r.get("producer"), "source_id": r.get("source_id")})
        if r.get("name"):
            g["names"].add(r["name"])
        g["aliases"] |= set(json.loads(r.get("aliases_json") or "[]"))
        if r.get("type"):
            g["types"].append(r["type"])

    out = []
    for eid, g in grouped.items():
        names = g["names"]
        canonical = min(names, key=lambda n: (-len(n), n)) if names else ""
        aliases = sorted((g["aliases"] | names) - {canonical})
        typ = None
        if g["types"]:
            counts = Counter(g["types"])
            top = max(counts.values())
            typ = sorted(t for t, c in counts.items() if c == top)[0]
        out.append({"entity_id": eid, "name": canonical, "type": typ,
                    "aliases_json": json.dumps(aliases),
                    "producer": g["producer"], "source_id": g["source_id"]})
    return sorted(out, key=lambda r: r["entity_id"])


def discover_runs(runs_root: Path) -> list[Path]:
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    return sorted(p for p in runs_root.iterdir() if p.is_dir() and (p / "claims.jsonl").is_file())
