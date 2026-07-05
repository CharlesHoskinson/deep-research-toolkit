"""Heavy end-to-end compiler test: assemble a project from the shipped
fixtures (wiki KB + pdf run + web run), compile the full index with REAL
sentence-transformers embeddings, and exercise the retrieval tool surface.
Marked heavy: requires the `compiler` extra (torch); not run on every push.
"""
import shutil
from pathlib import Path

import pytest

from deep_research_toolkit.config import load_config
from deep_research_toolkit.compiler.build import compile_index
from deep_research_toolkit.compiler.tools import Index

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.heavy
def test_full_pipeline_with_real_embeddings(tmp_path):
    # Assemble a project from the shipped fixtures.
    # Pin the sentence-transformers embedder so this stays hermetic -- the
    # shipped default (qwen3-embedding:8b) would route to a live Ollama endpoint.
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\nknowledge_base:\n  path: kb\n  pdf_runs_dir: pdf-runs\n"
        "  research_runs_dir: research-runs\n  index_dir: idx\n"
        "llm:\n  embedding_model: all-MiniLM-L6-v2\n", encoding="utf-8")
    shutil.copytree(FIXTURES / "reference-kb", tmp_path / "kb")
    shutil.copytree(FIXTURES / "reference-run-hydra-settlement",
                    tmp_path / "pdf-runs" / "hydra-settlement-test-fixture-4edb3c3c")
    shutil.copytree(FIXTURES / "reference-run-web-ows", tmp_path / "research-runs" / "reference-run-web-ows")

    cfg = load_config(tmp_path)
    stats = compile_index(cfg)  # real sentence-transformers embeddings
    assert stats["claims"] >= 6  # 5 pdf + >=1 web
    assert stats["wiki_chunks"] >= 1 and stats["claim_vectors"] >= 6

    idx = Index.open(cfg)
    try:
        assert idx.search_wiki("settlement")
        assert idx.search_claims("throughput")
        assert idx.get_entity("hydra-head")["name"]
        assert any(n["node"] for n in idx.neighbors("hydra-head", depth=2))
        # Every real fixture claim has a verbatim quote -> all included, none rejected on verbatim grounds.
        dossier = idx.compose_dossier(query="hydra settlement", k=12)
        assert dossier["included"]
    finally:
        idx.close()
