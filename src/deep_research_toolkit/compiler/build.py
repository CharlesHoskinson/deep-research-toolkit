"""Compile knowledge_base/ + *-runs/ into a DuckDB + LanceDB index.

v1 is a full rebuild each call (drop + recreate). Incremental compilation is
deferred (see docs/decisions/0002-knowledge-compiler.md)."""
from __future__ import annotations

import shutil
from pathlib import Path

import duckdb

from . import ingest
from .embed import Embedder, get_embedder
from .schema import create_tables


def open_duckdb(index_dir: Path):
    con = duckdb.connect(str(Path(index_dir) / "knowledge.duckdb"))
    con.execute("INSTALL fts; LOAD fts;")
    return con


def open_lancedb(index_dir: Path):
    import lancedb
    return lancedb.connect(str(Path(index_dir) / "lancedb"))


def _insert(con, table: str, rows: list[dict], columns: list[str]) -> None:
    if not rows:
        return
    placeholders = ", ".join("?" for _ in columns)
    con.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        [[r.get(c) for c in columns] for r in rows],
    )


def compile_index(config, embedder: Embedder | None = None) -> dict:
    index_dir = Path(config.index_dir)
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    embedder = embedder or get_embedder(config.embedding_model)

    con = open_duckdb(index_dir)
    create_tables(con)

    wiki = ingest.iter_wiki_pages(config.knowledge_base_path)
    _insert(con, "wiki_pages", wiki,
            ["path", "type", "title", "status", "timestamp", "body", "frontmatter_json"])
    link_rows = [{"from_path": w["path"], "to_path": t} for w in wiki for t in w["links"]]
    _insert(con, "wiki_links", link_rows, ["from_path", "to_path"])

    claims, evidence, entities, mentions, relations = [], [], [], [], []
    for producer, root in [("pdf", config.pdf_runs_path), ("web", config.research_runs_path)]:
        for run in ingest.discover_runs(root):
            c, e = ingest.iter_run_claims(run, producer)
            claims += c
            evidence += e
            en, mn = ingest.iter_run_entities(run, producer)
            entities += en
            mentions += mn
            relations += ingest.iter_run_relations(run, producer)

    _insert(con, "claims", claims, ["claim_id", "producer", "source_id", "claim", "claim_type", "confidence"])
    _insert(con, "claim_evidence", evidence,
            ["claim_id", "producer", "source_id", "locator", "page", "url", "quote"])
    _insert(con, "entities", entities, ["entity_id", "name", "type", "aliases_json", "producer", "source_id"])
    _insert(con, "entity_mentions", mentions, ["entity_id", "locator", "producer", "source_id"])
    _insert(con, "relations", relations,
            ["relation_id", "subject", "predicate", "object", "supporting_claim", "producer", "source_id"])

    if wiki:
        con.execute("PRAGMA create_fts_index('wiki_pages', 'path', 'body', overwrite=1)")
    if claims:
        con.execute("PRAGMA create_fts_index('claims', 'claim_id', 'claim', overwrite=1)")

    # LanceDB vector tables
    wiki_vecs = _build_vectors(embedder, [(w["path"], w["body"]) for w in wiki])
    claim_vecs = _build_vectors(embedder, [(c["claim_id"], c["claim"]) for c in claims])
    if wiki_vecs or claim_vecs:
        db = open_lancedb(index_dir)
        if wiki_vecs:
            db.create_table("wiki_chunks", data=wiki_vecs, mode="overwrite")
        if claim_vecs:
            db.create_table("claim_vectors", data=claim_vecs, mode="overwrite")

    con.close()
    return {
        "wiki_pages": len(wiki), "claims": len(claims), "entities": len(entities),
        "relations": len(relations), "wiki_chunks": len(wiki_vecs), "claim_vectors": len(claim_vecs),
    }


def _build_vectors(embedder: Embedder, id_text: list[tuple[str, str]]) -> list[dict]:
    if not id_text:
        return []
    vectors = embedder.embed([t for _, t in id_text])
    return [{"id": i, "text": t, "vector": v} for (i, t), v in zip(id_text, vectors)]
