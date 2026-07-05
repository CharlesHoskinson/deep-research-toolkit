"""Fast unit tests for pdf/extract.py (extract_tables / extract_figures),
replayed against the checked-in reference-run-hydra-settlement fixture's
docling_raw.json instead of running Docling. Both functions are pure,
deterministic transforms of docling_raw.json, so their output from the
reference input is checked against the reference tables/*.csv and
figures/captions.jsonl.
"""
from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

from deep_research_toolkit.pdf.extract import extract_figures, extract_tables

REFERENCE = Path(__file__).resolve().parent.parent / "fixtures" / "reference-run-hydra-settlement"

# A minimal 1x1 transparent PNG, base64-encoded, for the synthetic
# extracted-image test below.
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shutil.copy(REFERENCE / "docling_raw.json", run_dir / "docling_raw.json")
    return run_dir


def test_extract_tables_matches_reference_csv(tmp_path):
    run_dir = _make_run_dir(tmp_path)

    count = extract_tables(run_dir)

    assert count == 1
    produced = (run_dir / "tables" / "table_01.csv").read_text(encoding="utf-8")
    expected = (REFERENCE / "tables" / "table_01.csv").read_text(encoding="utf-8")
    assert produced == expected


def test_extract_tables_updates_manifest_without_disturbing_figure_count(tmp_path):
    run_dir = _make_run_dir(tmp_path)

    from deep_research_toolkit.common.manifest import load_manifest, update_stage

    update_stage(run_dir, "knowledge-extraction", figure_count=7)
    extract_tables(run_dir)

    stage = load_manifest(run_dir)["stages"]["knowledge-extraction"]
    assert stage["table_count"] == 1
    assert stage["figure_count"] == 7


def test_extract_figures_matches_reference_captions(tmp_path):
    run_dir = _make_run_dir(tmp_path)

    count = extract_figures(run_dir)

    assert count == 1
    produced = _read_jsonl(run_dir / "figures" / "captions.jsonl")
    expected = _read_jsonl(REFERENCE / "figures" / "captions.jsonl")
    # The checked-in reference captions.jsonl predates the schema_version
    # field, so it doesn't carry that key -- strip it before comparing.
    produced = [{k: v for k, v in r.items() if k != "schema_version"} for r in produced]
    assert produced == expected


def test_extract_figures_orphan_caption_gets_extracted_false_not_dropped(tmp_path):
    """The reference fixture has 0 real picture elements and one caption
    with no corresponding image (an 'orphan caption') -- it must still get a
    captions.jsonl row, with extracted: false and an explanatory note, never
    silently disappear."""
    run_dir = _make_run_dir(tmp_path)
    extract_figures(run_dir)

    rows = _read_jsonl(run_dir / "figures" / "captions.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["extracted"] is False
    assert row["caption"] == "Figure 1: Head Lifecycle (placeholder)"
    assert row["note"]
    assert not (run_dir / "figures" / f"{row['figure_id']}.png").exists()


def test_extract_figures_updates_manifest_without_disturbing_table_count(tmp_path):
    run_dir = _make_run_dir(tmp_path)

    from deep_research_toolkit.common.manifest import load_manifest, update_stage

    update_stage(run_dir, "knowledge-extraction", table_count=3)
    extract_figures(run_dir)

    stage = load_manifest(run_dir)["stages"]["knowledge-extraction"]
    assert stage["figure_count"] == 1
    assert stage["table_count"] == 3


def test_extract_figures_real_picture_with_embedded_image_is_extracted(tmp_path):
    """The reference fixture happens to have zero real picture elements, so
    build a minimal synthetic docling_raw.json with one to cover the
    extracted: true path (embedded data: URI image bytes materialized to
    figures/figure_01.png)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    doc = {
        "body": {"children": []},
        "texts": [{"label": "caption", "text": "Figure 1: A synthetic figure"}],
        "pictures": [
            {
                "prov": [{"page_no": 1}],
                "captions": [{"$ref": "#/texts/0"}],
                "image": {"uri": f"data:image/png;base64,{TINY_PNG_B64}"},
            }
        ],
        "tables": [],
    }
    (run_dir / "docling_raw.json").write_text(json.dumps(doc), encoding="utf-8")

    count = extract_figures(run_dir)

    assert count == 1
    rows = _read_jsonl(run_dir / "figures" / "captions.jsonl")
    assert rows[0]["extracted"] is True
    assert rows[0]["note"] is None
    png_path = run_dir / "figures" / "figure_01.png"
    assert png_path.is_file()
    assert png_path.read_bytes() == base64.b64decode(TINY_PNG_B64)


def test_extract_figures_bad_image_uri_is_reported_not_extracted(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    doc = {
        "body": {"children": []},
        "texts": [],
        "pictures": [
            {
                "prov": [{"page_no": 1}],
                "captions": [],
                "image": {"uri": "does-not-exist.png"},
            }
        ],
        "tables": [],
    }
    (run_dir / "docling_raw.json").write_text(json.dumps(doc), encoding="utf-8")

    count = extract_figures(run_dir)

    assert count == 1
    rows = _read_jsonl(run_dir / "figures" / "captions.jsonl")
    assert rows[0]["extracted"] is False
    assert "does not point at any file" in rows[0]["note"]
