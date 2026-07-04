"""Index schema: producer-agnostic evidence_ref + the DuckDB table DDL.

The compiler normalizes PDF- and web-sourced evidence into one EvidenceRef
shape at index time; on-disk producer files keep their native shapes (see
docs/contracts/knowledge-compiler.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

INDEX_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class EvidenceRef:
    producer: str          # "pdf" | "web"
    source_id: str         # document_id (pdf) or web source_id
    locator: str | None    # node_id (pdf) or research chunk_id (web)
    quote: str
    page: int | None = None
    url: str | None = None


def normalize_evidence(claim: dict[str, Any], producer: str, source_id: str) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    for ev in claim.get("supporting_evidence") or []:
        refs.append(
            EvidenceRef(
                producer=producer,
                source_id=source_id,
                locator=ev.get("node_id") if producer == "pdf" else ev.get("locator"),
                quote=ev.get("quote") or "",
                page=ev.get("page") if producer == "pdf" else None,
                url=ev.get("url") if producer == "web" else None,
            )
        )
    return refs


_DDL = """
CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR);
CREATE TABLE IF NOT EXISTS wiki_pages (
    path VARCHAR PRIMARY KEY, type VARCHAR, title VARCHAR, status VARCHAR,
    timestamp VARCHAR, body VARCHAR, frontmatter_json VARCHAR
);
CREATE TABLE IF NOT EXISTS wiki_links (from_path VARCHAR, to_path VARCHAR);
CREATE TABLE IF NOT EXISTS claims (
    claim_id VARCHAR, producer VARCHAR, source_id VARCHAR, claim VARCHAR,
    claim_type VARCHAR, confidence VARCHAR
);
CREATE TABLE IF NOT EXISTS claim_evidence (
    claim_id VARCHAR, producer VARCHAR, source_id VARCHAR, locator VARCHAR,
    page INTEGER, url VARCHAR, quote VARCHAR
);
CREATE TABLE IF NOT EXISTS entities (
    entity_id VARCHAR, name VARCHAR, type VARCHAR, aliases_json VARCHAR,
    producer VARCHAR, source_id VARCHAR
);
CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_id VARCHAR, locator VARCHAR, producer VARCHAR, source_id VARCHAR
);
CREATE TABLE IF NOT EXISTS relations (
    relation_id VARCHAR, subject VARCHAR, predicate VARCHAR, object VARCHAR,
    supporting_claim VARCHAR, producer VARCHAR, source_id VARCHAR
);
"""


def create_tables(con) -> None:
    con.execute(_DDL)
    con.execute("INSERT OR REPLACE INTO meta VALUES ('index_schema_version', ?)", [INDEX_SCHEMA_VERSION])
