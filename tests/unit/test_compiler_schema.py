import duckdb
from deep_research_toolkit.compiler.schema import (
    EvidenceRef, normalize_evidence, create_tables, INDEX_SCHEMA_VERSION,
)


def test_normalize_pdf_evidence_maps_node_id_and_page():
    claim = {"supporting_evidence": [{"node_id": "doc:n005", "quote": "Hydra can be used", "page": 1}]}
    refs = normalize_evidence(claim, producer="pdf", source_id="doc")
    assert refs == [EvidenceRef(producer="pdf", source_id="doc", locator="doc:n005",
                                quote="Hydra can be used", page=1, url=None)]


def test_normalize_web_evidence_maps_locator_and_url():
    claim = {"supporting_evidence": [{"locator": "src:c02", "quote": "OWS delegates signing", "url": "https://x/y"}]}
    refs = normalize_evidence(claim, producer="web", source_id="src")
    assert refs == [EvidenceRef(producer="web", source_id="src", locator="src:c02",
                                quote="OWS delegates signing", page=None, url="https://x/y")]


def test_create_tables_makes_all_expected_tables():
    con = duckdb.connect(":memory:")
    create_tables(con)
    names = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert {"wiki_pages", "wiki_links", "claims", "claim_evidence",
            "entities", "entity_mentions", "relations", "meta"} <= names
    assert INDEX_SCHEMA_VERSION == "1.0"
