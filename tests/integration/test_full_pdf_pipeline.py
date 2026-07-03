"""Heavy integration test: chain all 7 PDF-ingestion stages against the real
test fixture through real Docling conversion, and confirm pass_rate == 1.0.
This is the "actually run it and check real output" layer -- the fast unit
tests cover the mockable logic; this covers what can only be verified by
really running the pipeline. Marked heavy: not run on every push (see
.github/workflows/ci.yml), only on a schedule / manual dispatch, since
Docling's first-run model download + real conversion is slow.
"""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.heavy

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "hydra-settlement-test-fixture.pdf"


def test_full_pipeline_pass_rate_1(tmp_path):
    from deep_research_toolkit.pdf.chunk import chunk_nodes
    from deep_research_toolkit.pdf.convert import convert
    from deep_research_toolkit.pdf.eval import run_eval
    from deep_research_toolkit.pdf.extract import extract_figures, extract_tables
    from deep_research_toolkit.pdf.provenance import extract_provenance
    from deep_research_toolkit.pdf.router import classify
    from deep_research_toolkit.pdf.wiki_writer import record_wiki_page
    from deep_research_toolkit.common.scaffold import scaffold_page

    assert FIXTURE.is_file(), f"test fixture missing: {FIXTURE}"

    runs_dir = tmp_path / "pdf-runs"
    run_dir = classify(FIXTURE, runs_dir)
    convert(run_dir)
    extract_provenance(run_dir)
    chunk_nodes(run_dir)
    extract_tables(run_dir)
    extract_figures(run_dir)

    # claims/entities/relations are normally written by Claude's own
    # judgment following knowledge-extraction's SKILL.md -- for this
    # automated test, reuse the real, previously-verified reference content
    # so the eval-harness stage (which checks THESE files) has something
    # real to validate rather than skipping those checks entirely. This is
    # only valid because document_id is a deterministic hash of the fixture
    # PDF's own bytes, so a fresh classify() run against the same fixture
    # file reproduces the exact same document_id the reference was built
    # with -- assert that explicitly so this test fails loudly (not
    # silently validates the wrong document) if the fixture ever changes.
    reference = Path(__file__).resolve().parent.parent / "fixtures" / "reference-run-hydra-settlement"
    document_id = run_dir.name
    reference_manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    assert document_id == reference_manifest["document_id"], (
        "fixture PDF content changed -- reference claims/entities/relations no longer apply"
    )
    for name in ["claims.jsonl", "entities.jsonl", "relations.jsonl"]:
        (run_dir / name).write_text((reference / name).read_text(encoding="utf-8"), encoding="utf-8")

    knowledge_base = tmp_path / "knowledge_base"
    page_path = knowledge_base / "concepts" / "hydra-settlement.md"
    scaffold_page(
        page_path,
        type="Concept",
        title="Hydra Head Settlement",
        source_docs=[document_id],
        status="draft",
        body="# Hydra Head Settlement\n\nSee reference run for full content.\n",
    )
    record_wiki_page(run_dir, "concepts/hydra-settlement.md")

    report = run_eval(run_dir)

    assert report["pass_rate"] == 1.0, json.dumps(report, indent=2)
