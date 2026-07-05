"""Fast unit tests for pdf/chunk.py, replayed against the checked-in
reference-run-hydra-settlement fixture's provenance.jsonl instead of running
Docling + provenance extraction. chunk_nodes() is a pure, deterministic
transform of provenance.jsonl, so its output from the reference input must
match the reference chunks.jsonl exactly (field-for-field).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from deep_research_toolkit.pdf.chunk import (
    build_nodes,
    chunk_nodes,
    group_units,
    maybe_split,
    narrative_text,
    read_jsonl,
)

REFERENCE = Path(__file__).resolve().parent.parent / "fixtures" / "reference-run-hydra-settlement"


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _drop_schema_version(nodes: list[dict]) -> list[dict]:
    # The checked-in reference chunks.jsonl predates the schema_version
    # field, so it doesn't carry that key -- strip it before comparing.
    return [{k: v for k, v in n.items() if k != "schema_version"} for n in nodes]


def _make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shutil.copy(REFERENCE / "provenance.jsonl", run_dir / "provenance.jsonl")
    shutil.copy(REFERENCE / "manifest.json", run_dir / "manifest.json")
    return run_dir


def test_chunk_nodes_matches_reference_output(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    reference_nodes = _read_jsonl(REFERENCE / "chunks.jsonl")

    count = chunk_nodes(run_dir)

    assert count == len(reference_nodes)
    produced_nodes = _read_jsonl(run_dir / "chunks.jsonl")
    assert _drop_schema_version(produced_nodes) == _drop_schema_version(reference_nodes)


def test_chunk_nodes_produces_expected_node_types(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    chunk_nodes(run_dir)

    nodes = _read_jsonl(run_dir / "chunks.jsonl")
    types = [n["type"] for n in nodes]
    # The reference document has 8 headings (-> 8 section nodes) and exactly
    # one table unit (-> 1 table node); nothing long enough to split, and no
    # figure/picture unit_type in this fixture's provenance (its one figure
    # reference is an orphan caption, folded into its enclosing section).
    assert types.count("table") == 1
    assert types.count("section") == len(types) - 1
    assert "figure" not in types


def test_chunk_nodes_links_previous_next_in_document_order(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    chunk_nodes(run_dir)

    nodes = _read_jsonl(run_dir / "chunks.jsonl")
    assert nodes[0]["links"]["previous"] is None
    assert nodes[-1]["links"]["next"] is None
    for i in range(len(nodes) - 1):
        assert nodes[i]["links"]["next"] == nodes[i + 1]["node_id"]
        assert nodes[i + 1]["links"]["previous"] == nodes[i]["node_id"]


def test_chunk_nodes_section_path_reflects_heading_nesting(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    chunk_nodes(run_dir)

    nodes = _read_jsonl(run_dir / "chunks.jsonl")
    by_title = {n["title"]: n for n in nodes}
    assert by_title["2.1 Head Lifecycle"]["section_path"] == ["2. Architecture", "2.1 Head Lifecycle"]
    assert by_title["2. Architecture"]["section_path"] == ["2. Architecture"]

    table_node = next(n for n in nodes if n["type"] == "table")
    assert table_node["section_path"] == ["3. Throughput Comparison"]


def test_chunk_nodes_missing_provenance_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        chunk_nodes(run_dir)


def test_chunk_nodes_empty_provenance_raises(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "provenance.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        chunk_nodes(run_dir)


# --- helper-level tests (group_units / maybe_split / build_nodes) ---


def test_group_units_folds_body_text_into_section_and_isolates_table_figure():
    units = [
        {"unit_id": "u1", "unit_type": "heading", "text": "1. Intro", "section_path": [], "page": 1},
        {"unit_id": "u2", "unit_type": "paragraph", "text": "body one", "section_path": ["1. Intro"], "page": 1},
        {"unit_id": "u3", "unit_type": "paragraph", "text": "body two", "section_path": ["1. Intro"], "page": 1},
        {"unit_id": "u4", "unit_type": "table", "text": "a | b", "section_path": ["1. Intro"], "page": 2},
        {"unit_id": "u5", "unit_type": "figure", "text": "", "section_path": ["1. Intro"], "page": 2},
        {"unit_id": "u6", "unit_type": "caption", "text": "Figure 1", "section_path": ["1. Intro"], "page": 2},
    ]
    raw_nodes = group_units(units)

    # A figure raw node is flushed standalone (just the figure unit itself);
    # the caption that follows it doesn't belong to the figure's own
    # member_units -- with no section open, it starts a new implicit
    # section raw node instead (see group_units' `else` branch).
    assert [rn["type"] for rn in raw_nodes] == ["section", "table", "figure", "section"]
    section = raw_nodes[0]
    assert [u["unit_id"] for u in section["member_units"]] == ["u1", "u2", "u3"]
    assert narrative_text(section["member_units"]) == "1. Intro\n\nbody one\n\nbody two"

    figure = raw_nodes[2]
    assert [u["unit_id"] for u in figure["member_units"]] == ["u5"]

    trailing_caption_section = raw_nodes[3]
    assert [u["unit_id"] for u in trailing_caption_section["member_units"]] == ["u6"]
    # caption text is excluded from narrative_text -- it describes an image
    # or table rather than continuing the body prose (see TEXT_EXCLUDED_TYPES).
    assert narrative_text(trailing_caption_section["member_units"]) == ""


def test_group_units_starts_implicit_section_for_body_text_before_any_heading():
    units = [
        {"unit_id": "u1", "unit_type": "paragraph", "text": "preamble", "section_path": [], "page": 1},
    ]
    raw_nodes = group_units(units)
    assert len(raw_nodes) == 1
    assert raw_nodes[0]["type"] == "section"
    assert raw_nodes[0]["title"] == ""


def test_maybe_split_leaves_short_section_untouched():
    raw_node = {
        "type": "section",
        "title": "1. Intro",
        "section_path": ["1. Intro"],
        "member_units": [
            {"unit_id": "u1", "unit_type": "heading", "text": "1. Intro"},
            {"unit_id": "u2", "unit_type": "paragraph", "text": "short body"},
        ],
    }
    assert maybe_split(raw_node) == [raw_node]


def test_maybe_split_expands_over_long_section_into_paragraph_children():
    long_a = "a" * 900
    long_b = "b" * 900
    raw_node = {
        "type": "section",
        "title": "1. Intro",
        "section_path": ["1. Intro"],
        "member_units": [
            {"unit_id": "u1", "unit_type": "heading", "text": "1. Intro"},
            {"unit_id": "u2", "unit_type": "paragraph", "text": long_a},
            {"unit_id": "u3", "unit_type": "paragraph", "text": long_b},
        ],
    }
    result = maybe_split(raw_node)

    assert len(result) == 3
    parent, child1, child2 = result
    assert parent is raw_node
    assert parent["_has_split_children"] is True
    assert child1["_is_split_child"] is True
    assert child1["type"] == "paragraph"
    assert child1["member_units"][0]["unit_id"] == "u2"
    assert child2["member_units"][0]["unit_id"] == "u3"


def test_maybe_split_does_not_split_non_section_nodes():
    table_node = {"type": "table", "title": "Table", "section_path": [], "member_units": []}
    assert maybe_split(table_node) == [table_node]


def test_build_nodes_wires_parent_link_for_split_children():
    long_a = "a" * 900
    long_b = "b" * 900
    units = [
        {"unit_id": "u1", "unit_type": "heading", "text": "1. Intro", "section_path": [], "page": 1},
        {"unit_id": "u2", "unit_type": "paragraph", "text": long_a, "section_path": ["1. Intro"], "page": 1},
        {"unit_id": "u3", "unit_type": "paragraph", "text": long_b, "section_path": ["1. Intro"], "page": 1},
    ]
    raw_nodes = group_units(units)
    expanded = []
    for rn in raw_nodes:
        expanded.extend(maybe_split(rn))

    nodes = build_nodes("doc-1", expanded)

    assert len(nodes) == 3
    parent, child1, child2 = nodes
    assert parent["type"] == "section"
    assert child1["links"]["parent"] == parent["node_id"]
    assert child2["links"]["parent"] == parent["node_id"]
    # previous/next chain the flattened sequence including split children
    assert parent["links"]["next"] == child1["node_id"]
    assert child1["links"]["next"] == child2["node_id"]
    assert child2["links"]["next"] is None
    assert child2["links"]["previous"] == child1["node_id"]


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]
