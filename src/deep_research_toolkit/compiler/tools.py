"""The retrieval-planner tool surface: an Index handle plus 8 cheap,
deterministic tools. No tool makes an LLM call (ADR 0001 decision #3)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import search as search_mod
from . import graph as graph_mod
from .build import open_duckdb, open_lancedb
from .contradictions import find_candidates
from .dossier import compose_dossier as _compose_dossier
from .embed import Embedder, get_embedder


class Index:
    def __init__(self, con, lancedb_handle, embedder, config, degraded: bool = False) -> None:
        self.con = con
        self.lance = lancedb_handle
        self.embedder = embedder
        self.config = config
        self.degraded = degraded

    @classmethod
    def open(cls, config, embedder: Embedder | None = None) -> "Index":
        index_dir = Path(config.index_dir)
        if not (index_dir / "knowledge.duckdb").is_file():
            raise FileNotFoundError(
                f"No index at {index_dir}. Run the knowledge-compiler skill's compile.py first."
            )
        con = open_duckdb(index_dir)
        degraded = False
        try:
            lance = open_lancedb(index_dir)
        except Exception:
            lance = None
            degraded = True
            logging.warning(
                "Could not open LanceDB at %s; vector search is unavailable, "
                "falling back to lexical-only search.", index_dir / "lancedb",
            )
        return cls(con, lance, embedder or get_embedder(config.embedding_model), config,
                   degraded=degraded)

    def close(self) -> None:
        self.con.close()

    def search_wiki(self, query: str, k: int = 8) -> list[dict]:
        ids = search_mod.hybrid_search(self.con, self.lance, self.embedder,
                                       table="wiki_pages", id_col="path", text_col="body",
                                       vec_table="wiki_chunks", query=query, k=k)
        out = []
        for path in ids:
            row = self.con.execute("SELECT path, title, type, status, body FROM wiki_pages WHERE path = ?",
                                   [path]).fetchone()
            if row:
                out.append({"path": row[0], "title": row[1], "type": row[2], "status": row[3],
                            "snippet": (row[4] or "")[:200]})
        return out

    def read_page(self, path: str) -> dict:
        row = self.con.execute("SELECT path, body, frontmatter_json FROM wiki_pages WHERE path = ?",
                               [path]).fetchone()
        if not row:
            return {"path": path, "error": "not found"}
        return {"path": row[0], "body": row[1], "frontmatter": json.loads(row[2] or "{}")}

    def search_claims(self, query: str, k: int = 8, producer: str | None = None) -> list[dict]:
        ids = search_mod.hybrid_search(self.con, self.lance, self.embedder,
                                       table="claims", id_col="claim_id", text_col="claim",
                                       vec_table="claim_vectors", query=query, k=k)
        out = []
        for cid in ids:
            row = self.con.execute(
                "SELECT claim_id, producer, source_id, claim, claim_type, confidence FROM claims WHERE claim_id = ?",
                [cid]).fetchone()
            if not row or (producer and row[1] != producer):
                continue
            ev = self.con.execute(
                "SELECT producer, source_id, locator, page, url, quote FROM claim_evidence WHERE claim_id = ?",
                [cid]).fetchall()
            out.append({"claim_id": row[0], "producer": row[1], "source_id": row[2], "claim": row[3],
                        "claim_type": row[4], "confidence": row[5],
                        "evidence": [dict(zip(["producer", "source_id", "locator", "page", "url", "quote"], e))
                                     for e in ev]})
        return out

    def get_entity(self, name_or_id: str) -> dict:
        row = self.con.execute(
            "SELECT entity_id, name, type, aliases_json FROM entities "
            "WHERE entity_id = ? OR lower(name) = lower(?) LIMIT 1", [name_or_id, name_or_id]).fetchone()
        if not row:
            return {"query": name_or_id, "error": "entity not found"}
        eid = row[0]
        mentions = [r[0] for r in self.con.execute(
            "SELECT locator FROM entity_mentions WHERE entity_id = ?", [eid]).fetchall()]
        relations = [dict(zip(["relation_id", "subject", "predicate", "object"], r)) for r in self.con.execute(
            "SELECT relation_id, subject, predicate, object FROM relations WHERE subject = ? OR object = ?",
            [eid, eid]).fetchall()]
        return {"entity_id": eid, "name": row[1], "type": row[2],
                "aliases": json.loads(row[3] or "[]"), "mentions": mentions, "relations": relations}

    def neighbors(self, entity: str, depth: int = 1) -> list[dict]:
        return graph_mod.neighbors(self.con, entity, depth)

    def get_sources(self, page: str | None = None, claim: str | None = None) -> dict:
        if page:
            row = self.con.execute("SELECT frontmatter_json FROM wiki_pages WHERE path = ?", [page]).fetchone()
            fm = json.loads(row[0]) if row else {}
            return {"page": page, "source": fm.get("source"), "source_docs": fm.get("source_docs"),
                    "resource": fm.get("resource")}
        if claim:
            ev = self.con.execute(
                "SELECT DISTINCT producer, source_id, url FROM claim_evidence WHERE claim_id = ?",
                [claim]).fetchall()
            return {"claim": claim,
                    "sources": [dict(zip(["producer", "source_id", "url"], e)) for e in ev]}
        return {"error": "pass page= or claim="}

    def find_contradictions(self) -> list[dict]:
        return find_candidates(self.con)

    def compose_dossier(self, query: str | None = None, claim_ids: list[str] | None = None, k: int = 12) -> dict:
        if claim_ids is None:
            claim_ids = [c["claim_id"] for c in self.search_claims(query or "", k=k)]
        return _compose_dossier(self.con, self.config, claim_ids)
