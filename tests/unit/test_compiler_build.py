import dataclasses
import json

import pytest

from deep_research_toolkit.common.frontmatter import write_okf
from deep_research_toolkit.compiler.build import compile_index, open_duckdb
from deep_research_toolkit.compiler.embed import FakeEmbedder
from deep_research_toolkit.config import load_config


def _project(tmp_path):
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\nknowledge_base:\n  path: kb\n  pdf_runs_dir: pdf-runs\n"
        "  research_runs_dir: research-runs\n  index_dir: idx\n", encoding="utf-8")
    kb = tmp_path / "kb"
    write_okf(kb / "concepts/hydra.md",
              {"type": "Concept", "title": "Hydra", "timestamp": "t", "status": "draft"}, "Hydra body\n")
    run = tmp_path / "pdf-runs" / "doc-abc"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"document_id": "doc-abc"}), encoding="utf-8")
    (run / "claims.jsonl").write_text(json.dumps({
        "claim_id": "c1", "claim": "Hydra settles synchronously", "claim_type": "architectural",
        "confidence": "high", "supporting_evidence": [{"node_id": "doc-abc:n5", "quote": "settles", "page": 1}],
    }) + "\n", encoding="utf-8")
    return load_config(tmp_path)


def test_compile_index_populates_both_engines(tmp_path):
    cfg = _project(tmp_path)
    stats = compile_index(cfg, embedder=FakeEmbedder())
    assert stats["wiki_pages"] == 1 and stats["claims"] == 1
    con = open_duckdb(cfg.index_dir)
    assert con.execute("SELECT count(*) FROM claims").fetchone()[0] == 1
    assert con.execute("SELECT quote FROM claim_evidence WHERE claim_id='c1'").fetchone()[0] == "settles"


def test_compile_index_is_idempotent(tmp_path):
    cfg = _project(tmp_path)
    compile_index(cfg, embedder=FakeEmbedder())
    stats = compile_index(cfg, embedder=FakeEmbedder())  # second run must not double rows
    assert stats["wiki_pages"] == 1 and stats["claims"] == 1


def test_compile_index_refuses_nonempty_dir_without_marker(tmp_path):
    cfg = _project(tmp_path)
    foreign = tmp_path / "my-notes"
    foreign.mkdir()
    (foreign / "important.txt").write_text("do not delete", encoding="utf-8")
    cfg = dataclasses.replace(cfg, index_dir=foreign)
    with pytest.raises(ValueError, match="knowledge.duckdb"):
        compile_index(cfg, embedder=FakeEmbedder())
    assert (foreign / "important.txt").is_file()  # nothing was deleted


def test_compile_index_refuses_knowledge_base_as_index_dir(tmp_path):
    cfg = _project(tmp_path)
    cfg = dataclasses.replace(cfg, index_dir=cfg.knowledge_base_path)
    with pytest.raises(ValueError, match="Refusing"):
        compile_index(cfg, embedder=FakeEmbedder())
