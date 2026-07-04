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


def discover_runs(runs_root: Path) -> list[Path]:
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    return sorted(p for p in runs_root.iterdir() if p.is_dir() and (p / "claims.jsonl").is_file())
