import json

from deep_research_toolkit.common.frontmatter import write_okf
from deep_research_toolkit.compiler import ingest


def test_iter_wiki_pages_reads_frontmatter_and_links(tmp_path):
    kb = tmp_path / "kb"
    write_okf(kb / "index.md", {"type": "Index", "title": "Index", "timestamp": "t"}, "[A](/concepts/a.md)\n")
    write_okf(kb / "concepts/a.md", {"type": "Concept", "title": "A", "timestamp": "t", "status": "seed"}, "body A\n")
    rows = {r["path"]: r for r in ingest.iter_wiki_pages(kb)}
    assert set(rows) == {"index.md", "concepts/a.md"}
    assert rows["index.md"]["links"] == ["concepts/a.md"]
    assert rows["concepts/a.md"]["title"] == "A"


def test_iter_run_claims_normalizes_pdf_evidence(tmp_path):
    run = tmp_path / "doc-abc"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"document_id": "doc-abc"}), encoding="utf-8")
    (run / "claims.jsonl").write_text(json.dumps({
        "claim_id": "c1", "claim": "X", "claim_type": "architectural", "confidence": "high",
        "supporting_evidence": [{"node_id": "doc-abc:n5", "quote": "X", "page": 2}],
    }) + "\n", encoding="utf-8")
    claim_rows, ev_rows = ingest.iter_run_claims(run, producer="pdf")
    assert claim_rows[0]["source_id"] == "doc-abc"
    assert ev_rows[0]["locator"] == "doc-abc:n5" and ev_rows[0]["page"] == 2


def test_discover_runs_finds_only_dirs_with_claims(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "a" / "claims.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "b").mkdir()
    assert [p.name for p in ingest.discover_runs(tmp_path)] == ["a"]


def test_merge_entities_collapses_same_id_across_sources():
    from deep_research_toolkit.compiler.ingest import merge_entities
    rows = [
        {"entity_id": "ows", "name": "OWS", "type": "standard",
         "aliases_json": json.dumps(["Open Wallet"]), "producer": "pdf", "source_id": "d1"},
        {"entity_id": "ows", "name": "Open Wallet Standard", "type": "standard",
         "aliases_json": "[]", "producer": "web", "source_id": "d2"},
        {"entity_id": "hydra", "name": "Hydra", "type": "protocol",
         "aliases_json": "[]", "producer": "pdf", "source_id": "d1"},
    ]
    merged = {e["entity_id"]: e for e in merge_entities(rows)}
    assert set(merged) == {"ows", "hydra"}                       # ows collapsed to one row
    assert merged["ows"]["name"] == "Open Wallet Standard"       # longest name is canonical
    aliases = json.loads(merged["ows"]["aliases_json"])
    assert "OWS" in aliases and "Open Wallet" in aliases         # non-canonical name + aliases folded in
    assert merged["ows"]["type"] == "standard"
