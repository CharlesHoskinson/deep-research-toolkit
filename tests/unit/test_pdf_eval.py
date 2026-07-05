"""Fast unit tests for pdf/eval.py's six mechanical checks, replayed against
the checked-in reference-run-hydra-settlement fixture (a known, internally
self-consistent run) instead of running the pipeline. Each check is exercised
both on the consistent reference (should pass) and against a copy with one
targeted defect injected (should fail), so both directions of every check are
covered.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from deep_research_toolkit.pdf.eval import (
    check_evidence_quotes_verbatim,
    check_figures_accounted_for,
    check_headings_recovered,
    check_no_ocr_garbage,
    check_page_citations_valid,
    check_tables_present,
    run_eval,
)

REFERENCE = Path(__file__).resolve().parent.parent / "fixtures" / "reference-run-hydra-settlement"

# Everything run_eval()'s six checks read from, so a full self-consistent
# copy can be made without needing docling_raw.json or a real Docling run.
EVAL_INPUT_FILES = ["canonical.md", "provenance.jsonl", "chunks.jsonl", "claims.jsonl", "manifest.json"]
EVAL_INPUT_DIRS = ["tables", "figures"]


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in EVAL_INPUT_FILES:
        shutil.copy(REFERENCE / name, run_dir / name)
    for name in EVAL_INPUT_DIRS:
        shutil.copytree(REFERENCE / name, run_dir / name)
    return run_dir


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_run_eval_passes_on_consistent_reference_run(tmp_path):
    run_dir = _make_run_dir(tmp_path)

    report = run_eval(run_dir)

    assert report["pass_rate"] == 1.0, json.dumps(report, indent=2)
    assert all(c["passed"] for c in report["checks"])
    assert (run_dir / "eval_report.json").is_file()
    assert (run_dir / "eval_report.md").is_file()


def test_run_eval_updates_manifest_stage(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    report = run_eval(run_dir)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["rag-eval-harness"]["pass_rate"] == report["pass_rate"]


def test_check_headings_recovered_fails_when_chunk_missing_a_heading(tmp_path):
    run_dir = _make_run_dir(tmp_path)

    nodes = _read_jsonl(run_dir / "chunks.jsonl")
    # drop "2. Architecture" from every node's section_path so canonical.md's
    # heading is no longer represented anywhere in chunks.jsonl
    for node in nodes:
        node["section_path"] = [s for s in node["section_path"] if s != "2. Architecture"]
        if node["title"] == "2. Architecture":
            node["title"] = "gone"
    _write_jsonl(run_dir / "chunks.jsonl", nodes)

    result = check_headings_recovered(run_dir)
    assert result["passed"] is False
    assert "2. Architecture" in result["detail"]


def test_check_tables_present_fails_when_csv_missing(tmp_path):
    run_dir = _make_run_dir(tmp_path)

    (run_dir / "tables" / "table_01.csv").unlink()

    result = check_tables_present(run_dir)
    assert result["passed"] is False


def test_check_tables_present_passes_on_reference(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    result = check_tables_present(run_dir)
    assert result["passed"] is True


def test_check_page_citations_valid_fails_on_out_of_range_page(tmp_path):
    run_dir = _make_run_dir(tmp_path)

    claims = _read_jsonl(run_dir / "claims.jsonl")
    claims[0]["supporting_evidence"][0]["page"] = 9999
    _write_jsonl(run_dir / "claims.jsonl", claims)

    result = check_page_citations_valid(run_dir)
    assert result["passed"] is False
    assert "9999" in result["detail"]


def test_check_evidence_quotes_verbatim_fails_when_quote_not_in_chunk(tmp_path):
    run_dir = _make_run_dir(tmp_path)

    claims = _read_jsonl(run_dir / "claims.jsonl")
    claims[0]["supporting_evidence"][0]["quote"] = "this text does not appear anywhere in the chunk"
    _write_jsonl(run_dir / "claims.jsonl", claims)

    result = check_evidence_quotes_verbatim(run_dir)
    assert result["passed"] is False
    assert claims[0]["claim_id"] in result["detail"]


def test_check_evidence_quotes_verbatim_is_chunk_based_not_locator_agnostic(tmp_path):
    """A quote that is verbatim text somewhere in the run but NOT in the
    specific chunk named by its own node_id must still fail -- the check
    resolves node_id -> that chunk's text and nowhere else."""
    run_dir = _make_run_dir(tmp_path)

    claims = _read_jsonl(run_dir / "claims.jsonl")
    chunks = _read_jsonl(run_dir / "chunks.jsonl")

    # c_0002's evidence cites n004 with a quote that is verbatim in n004's
    # text. Point it at a different node_id (n001) instead, whose text does
    # not contain that quote.
    target = next(c for c in claims if c["claim_id"] == "c_0002")
    ev = target["supporting_evidence"][0]
    original_node_id = ev["node_id"]
    other_node_id = next(n["node_id"] for n in chunks if n["node_id"] != original_node_id)
    ev["node_id"] = other_node_id
    _write_jsonl(run_dir / "claims.jsonl", claims)

    result = check_evidence_quotes_verbatim(run_dir)
    assert result["passed"] is False


def test_check_figures_accounted_for_fails_when_extracted_field_missing(tmp_path):
    run_dir = _make_run_dir(tmp_path)

    captions = _read_jsonl(run_dir / "figures" / "captions.jsonl")
    del captions[0]["extracted"]
    _write_jsonl(run_dir / "figures" / "captions.jsonl", captions)

    result = check_figures_accounted_for(run_dir)
    assert result["passed"] is False
    assert "no 'extracted' field" in result["detail"]


def test_check_figures_accounted_for_fails_when_marked_extracted_but_no_image_file(tmp_path):
    run_dir = _make_run_dir(tmp_path)

    captions = _read_jsonl(run_dir / "figures" / "captions.jsonl")
    captions[0]["extracted"] = True
    _write_jsonl(run_dir / "figures" / "captions.jsonl", captions)

    result = check_figures_accounted_for(run_dir)
    assert result["passed"] is False
    assert "no image file found" in result["detail"]


def test_check_no_ocr_garbage_fails_on_mojibake_page(tmp_path):
    run_dir = _make_run_dir(tmp_path)

    units = _read_jsonl(run_dir / "provenance.jsonl")
    # page 1 already carries ~1.4KB of real (printable) text from its other
    # units, so the injected garbage has to be long enough to push the
    # page's combined non-printable ratio past the 5% threshold on its own.
    units[0]["text"] = "�" * 2000
    _write_jsonl(run_dir / "provenance.jsonl", units)

    result = check_no_ocr_garbage(run_dir)
    assert result["passed"] is False


def test_checks_skip_when_upstream_files_absent(tmp_path):
    run_dir = tmp_path / "empty-run"
    run_dir.mkdir()

    result = check_tables_present(run_dir)
    assert result["passed"] is True
    assert "skipped" in result["detail"]


def test_run_eval_still_reaches_pass_rate_one_with_a_defect_fixed_afterward(tmp_path):
    # Sanity check that the injected-defect tests above are actually
    # detecting something real: break a check, confirm run_eval's overall
    # pass_rate drops below 1.0, restore it, confirm it goes back to 1.0.
    run_dir = _make_run_dir(tmp_path)

    (run_dir / "tables" / "table_01.csv").unlink()
    broken_report = run_eval(run_dir)
    assert broken_report["pass_rate"] < 1.0

    shutil.copy(REFERENCE / "tables" / "table_01.csv", run_dir / "tables" / "table_01.csv")
    fixed_report = run_eval(run_dir)
    assert fixed_report["pass_rate"] == 1.0
