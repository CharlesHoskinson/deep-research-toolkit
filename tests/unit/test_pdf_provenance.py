"""Fast unit tests for pdf/provenance.py, replayed against the checked-in
reference-run-hydra-settlement fixture instead of running Docling. Because
extract_provenance() is a pure, deterministic walk of docling_raw.json, the
provenance.jsonl it writes from the reference docling_raw.json + manifest.json
must match the reference provenance.jsonl exactly -- that equality is the
core assertion these tests lean on.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from deep_research_toolkit.pdf.provenance import extract_provenance, heading_depth

REFERENCE = Path(__file__).resolve().parent.parent / "fixtures" / "reference-run-hydra-settlement"


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _make_run_dir(tmp_path: Path) -> Path:
    """Copy just the two inputs extract_provenance() needs: docling_raw.json
    (the thing being walked) and manifest.json (source of document_id /
    source_file) -- no Docling run required."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shutil.copy(REFERENCE / "docling_raw.json", run_dir / "docling_raw.json")
    shutil.copy(REFERENCE / "manifest.json", run_dir / "manifest.json")
    return run_dir


def _drop_schema_version(units: list[dict]) -> list[dict]:
    # The checked-in reference provenance.jsonl predates the schema_version
    # field (added to build_unit() after the fixture was generated), so it
    # doesn't carry that key -- strip it from both sides before comparing
    # rather than asserting exact dict equality including it.
    return [{k: v for k, v in u.items() if k != "schema_version"} for u in units]


def test_extract_provenance_matches_reference_output(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    reference_units = _read_jsonl(REFERENCE / "provenance.jsonl")

    count = extract_provenance(run_dir)

    assert count == len(reference_units)
    produced_units = _read_jsonl(run_dir / "provenance.jsonl")
    assert _drop_schema_version(produced_units) == _drop_schema_version(reference_units)


def test_extract_provenance_units_have_expected_fields(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    extract_provenance(run_dir)

    units = _read_jsonl(run_dir / "provenance.jsonl")
    assert units, "expected at least one provenance unit"
    for unit in units:
        assert "page" in unit
        assert "section_path" in unit
        assert isinstance(unit["section_path"], list)
        assert "bbox" in unit
        assert "unit_type" in unit
        assert unit["unit_type"] in {
            "heading",
            "paragraph",
            "table",
            "figure",
            "caption",
            "list_item",
        }


def test_extract_provenance_section_path_tracks_heading_nesting(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    extract_provenance(run_dir)

    units = _read_jsonl(run_dir / "provenance.jsonl")
    by_text = {u["text"]: u for u in units if u["unit_type"] == "heading"}

    # "2.1 Head Lifecycle" is a numbered subsection of "2. Architecture" --
    # its own section_path must show the parent heading as an ancestor, not
    # include itself.
    subsection = by_text["2.1 Head Lifecycle"]
    assert subsection["section_path"] == ["2. Architecture"]

    top_level = by_text["2. Architecture"]
    assert top_level["section_path"] == []


def test_extract_provenance_updates_manifest_stage(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    count = extract_provenance(run_dir)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    stage = manifest["stages"]["pdf-layout-provenance"]
    assert stage["unit_count"] == count


def test_extract_provenance_missing_docling_raw_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shutil.copy(REFERENCE / "manifest.json", run_dir / "manifest.json")

    with pytest.raises(FileNotFoundError):
        extract_provenance(run_dir)


def test_extract_provenance_missing_manifest_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shutil.copy(REFERENCE / "docling_raw.json", run_dir / "docling_raw.json")

    with pytest.raises(FileNotFoundError):
        extract_provenance(run_dir)


@pytest.mark.parametrize(
    ("text", "level", "expected_depth"),
    [
        ("1. Introduction", 1, 1),
        ("2.1 Head Lifecycle", 1, 2),
        ("2.1.3 Nested Subsection", 1, 3),
        ("Unnumbered Heading", 1, 1),
        ("Unnumbered Heading", 2, 2),
    ],
)
def test_heading_depth_heuristic(text, level, expected_depth):
    assert heading_depth(text, level) == expected_depth
