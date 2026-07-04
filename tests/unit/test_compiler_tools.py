import json
from deep_research_toolkit.common.frontmatter import write_okf
from deep_research_toolkit.compiler.build import compile_index
from deep_research_toolkit.compiler.embed import FakeEmbedder
from deep_research_toolkit.compiler.tools import Index
from deep_research_toolkit.config import load_config


def _project(tmp_path):
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\nknowledge_base:\n  path: kb\n  pdf_runs_dir: pdf-runs\n"
        "  research_runs_dir: research-runs\n  index_dir: idx\n", encoding="utf-8")
    write_okf(tmp_path / "kb" / "concepts/hydra.md",
              {"type": "Concept", "title": "Hydra", "timestamp": "t", "status": "draft"},
              "Hydra is a settlement layer.\n")
    run = tmp_path / "pdf-runs" / "doc-abc"; run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"document_id": "doc-abc"}), encoding="utf-8")
    (run / "provenance.jsonl").write_text(json.dumps(
        {"page": 1, "text": "Hydra settles synchronously among participants."}) + "\n", encoding="utf-8")
    (run / "claims.jsonl").write_text(json.dumps({
        "claim_id": "c1", "claim": "Hydra settles synchronously", "claim_type": "architectural",
        "confidence": "high",
        "supporting_evidence": [{"node_id": "doc-abc:n5", "quote": "settles synchronously", "page": 1}],
    }) + "\n", encoding="utf-8")
    (run / "entities.jsonl").write_text(json.dumps(
        {"entity_id": "hydra", "name": "Hydra", "type": "protocol", "aliases": [], "mentions": ["doc-abc:n5"]}
    ) + "\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    compile_index(cfg, embedder=FakeEmbedder())
    return cfg


def test_search_claims_and_read_page(tmp_path):
    cfg = _project(tmp_path)
    idx = Index.open(cfg, embedder=FakeEmbedder())
    assert any(c["claim_id"] == "c1" for c in idx.search_claims("settlement"))
    page = idx.read_page("concepts/hydra.md")
    assert page["frontmatter"]["title"] == "Hydra"
    idx.close()


def test_get_entity_and_compose_dossier_gate(tmp_path):
    cfg = _project(tmp_path)
    idx = Index.open(cfg, embedder=FakeEmbedder())
    ent = idx.get_entity("hydra")
    assert ent["name"] == "Hydra"  # entity resolves by id
    dossier = idx.compose_dossier(claim_ids=["c1"])
    assert [c["claim_id"] for c in dossier["included"]] == ["c1"]  # verbatim passes
    idx.close()
