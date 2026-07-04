"""fts_search must degrade to no lexical hits when a table had zero rows at
compile time (build.py only creates the FTS index when rows exist)."""
from deep_research_toolkit.common.frontmatter import write_okf
from deep_research_toolkit.compiler.build import compile_index
from deep_research_toolkit.compiler.embed import FakeEmbedder
from deep_research_toolkit.compiler.tools import Index
from deep_research_toolkit.config import load_config


def test_search_claims_returns_empty_list_when_index_has_no_claims(tmp_path):
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\nknowledge_base:\n  path: kb\n  pdf_runs_dir: pdf-runs\n"
        "  research_runs_dir: research-runs\n  index_dir: idx\n", encoding="utf-8")
    write_okf(tmp_path / "kb" / "concepts/hydra.md",
              {"type": "Concept", "title": "Hydra", "timestamp": "t", "status": "draft"},
              "Hydra is a settlement layer.\n")
    cfg = load_config(tmp_path)
    stats = compile_index(cfg, embedder=FakeEmbedder())
    assert stats["wiki_pages"] == 1 and stats["claims"] == 0

    idx = Index.open(cfg, embedder=FakeEmbedder())
    try:
        result = idx.search_claims("anything")  # must not raise on missing fts_main_claims
        assert result == []
    finally:
        idx.close()
