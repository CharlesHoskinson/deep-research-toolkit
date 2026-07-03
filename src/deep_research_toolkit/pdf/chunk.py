"""Turn provenance.jsonl into chunks.jsonl: retrieval-ready, structure-aware nodes.

Reads:
    <run_dir>/provenance.jsonl   (pdf-layout-provenance output)
    <run_dir>/manifest.json      (for document_id; created if missing)

Writes:
    <run_dir>/chunks.jsonl
    <run_dir>/manifest.json      (adds/updates stages.canonical-markdown-to-llm-nodes)

Grouping rule (see skills/canonical-markdown-to-llm-nodes/references/chunking-strategy.md
for the full writeup):
  - A "heading" unit starts a new `section` node. Its section_path is the
    heading's own section_path (its ancestor chain) plus its own text.
  - Consecutive non-heading, non-table, non-figure units are folded into the
    current section node's text (joined with "\n\n") until the next heading,
    table, or figure unit ends it.
  - A "table" unit always becomes its own `table` node.
  - A "figure"/"picture" unit always becomes its own `figure` node.
  - If a section node's folded text exceeds SPLIT_THRESHOLD characters and
    was built from more than one contributing unit, it is split: the section
    node is kept (as a coarse, citable overview node) and each contributing
    non-heading unit also becomes its own `paragraph` child node, inserted
    immediately after the section node, with `links.parent` set to the
    section node's id.
  - `links.previous`/`links.next` chain every node in the final, flattened
    sequence (including split-off paragraph children) in document order.

`summary`, `keywords`, and `entities` are intentionally left as empty
placeholders ("" / [] / []) here -- populating them requires a language
model's judgment and is the job of the downstream knowledge-extraction
skill, not this deterministic function. This is expected, not a bug.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..common.hashing import content_hash
from ..common import manifest as manifest_mod

SCHEMA_VERSION = "1.0"
SPLIT_THRESHOLD = 1500  # characters; see references/chunking-strategy.md

HEADING_TYPES = {"heading"}
TABLE_TYPES = {"table"}
FIGURE_TYPES = {"figure", "picture"}
# Caption units are folded into whichever section/figure they sit under for
# provenance purposes (their unit_id/page still count) but their text is not
# spliced into the surrounding narrative -- a caption describes an image or
# table rather than continuing the body prose, so mixing it into `text`
# would read as a non-sequitur and pollute the node's embedding.
TEXT_EXCLUDED_TYPES = {"caption"}


def narrative_text(member_units: list[dict[str, Any]]) -> str:
    return "\n\n".join(u["text"] for u in member_units if u["unit_type"] not in TEXT_EXCLUDED_TYPES)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    units = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                units.append(json.loads(line))
    return units


def group_units(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """First pass: fold provenance units into a flat list of "raw nodes".

    Each raw node is a dict: {type, title, section_path, member_units}
    where member_units is the ordered list of contributing provenance units
    (for a section, this includes the heading unit itself as member_units[0]).
    """
    raw_nodes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None  # in-progress section raw node

    def flush():
        if current is not None:
            raw_nodes.append(current)

    for unit in units:
        utype = unit["unit_type"]
        if utype in HEADING_TYPES:
            flush()
            current = {
                "type": "section",
                "title": unit["text"],
                "section_path": list(unit.get("section_path") or []) + [unit["text"]],
                "member_units": [unit],
            }
        elif utype in TABLE_TYPES:
            flush()
            current = None
            raw_nodes.append({
                "type": "table",
                "title": "Table",
                "section_path": list(unit.get("section_path") or []),
                "member_units": [unit],
            })
        elif utype in FIGURE_TYPES:
            flush()
            current = None
            raw_nodes.append({
                "type": "figure",
                "title": "Figure",
                "section_path": list(unit.get("section_path") or []),
                "member_units": [unit],
            })
        else:
            # paragraph, caption, list_item, or anything else: fold into the
            # current section, or start an implicit one if none is open yet
            # (e.g. body text before any heading has appeared).
            if current is None:
                current = {
                    "type": "section",
                    "title": (unit.get("section_path") or [""])[-1] if unit.get("section_path") else "",
                    "section_path": list(unit.get("section_path") or []),
                    "member_units": [],
                }
            current["member_units"].append(unit)

    flush()
    return raw_nodes


def maybe_split(raw_node: dict[str, Any]) -> list[dict[str, Any]]:
    """Second pass: expand an over-long section into itself + paragraph children.

    Returns a list: [raw_node] normally, or [raw_node, child1, child2, ...]
    if the section's folded text exceeds SPLIT_THRESHOLD and has more than
    one contributing unit. Children are tagged `_is_split_child`; the parent
    is tagged `_has_split_children`. `build_nodes` resolves these tags into
    real `node_id`s for `links.parent` once ids have been assigned.
    """
    if raw_node["type"] != "section":
        return [raw_node]

    full_text = narrative_text(raw_node["member_units"])
    body_units = [
        u for u in raw_node["member_units"]
        if u["unit_type"] not in HEADING_TYPES and u["unit_type"] not in TEXT_EXCLUDED_TYPES
    ]

    if len(full_text) <= SPLIT_THRESHOLD or len(body_units) < 2:
        return [raw_node]

    children = []
    for u in body_units:
        children.append({
            "type": "paragraph",
            "title": raw_node["title"],
            "section_path": raw_node["section_path"],
            "member_units": [u],
            "_is_split_child": True,
        })
    raw_node["_has_split_children"] = True
    return [raw_node] + children


def build_nodes(document_id: str, raw_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    width = max(3, len(str(len(raw_nodes))))
    nodes = []
    for i, rn in enumerate(raw_nodes, start=1):
        node_id = f"{document_id}:n{str(i).zfill(width)}"
        text = narrative_text(rn["member_units"])
        pages = [u["page"] for u in rn["member_units"] if u.get("page") is not None]
        unit_ids = [u["unit_id"] for u in rn["member_units"]]
        node = {
            "schema_version": SCHEMA_VERSION,
            "node_id": node_id,
            "document_id": document_id,
            "type": rn["type"],
            "title": rn["title"],
            "section_path": rn["section_path"],
            "text": text,
            "summary": "",
            "keywords": [],
            "entities": [],
            "source": {
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
                "unit_ids": unit_ids,
            },
            "links": {"previous": None, "next": None, "parent": None},
            "content_hash": content_hash(text),
        }
        nodes.append(node)
        rn["_node_id"] = node_id

    # wire up previous/next over the flattened sequence, and parent for split children
    for i, node in enumerate(nodes):
        node["links"]["previous"] = nodes[i - 1]["node_id"] if i > 0 else None
        node["links"]["next"] = nodes[i + 1]["node_id"] if i < len(nodes) - 1 else None

    parent_id = None
    for rn, node in zip(raw_nodes, nodes):
        if rn.get("_has_split_children"):
            parent_id = node["node_id"]
        elif rn.get("_is_split_child"):
            node["links"]["parent"] = parent_id
        else:
            parent_id = None

    return nodes


def chunk_nodes(run_dir: str | Path) -> int:
    """Read <run_dir>/provenance.jsonl, write <run_dir>/chunks.jsonl, and
    update manifest.json's canonical-markdown-to-llm-nodes stage. Returns
    the number of nodes written.
    """
    run_dir = Path(run_dir)
    provenance_path = run_dir / "provenance.jsonl"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"{provenance_path} not found. Run pdf-layout-provenance first.")

    units = read_jsonl(provenance_path)
    if not units:
        raise ValueError(f"{provenance_path} is empty.")

    existing = manifest_mod.load_manifest(run_dir)
    if existing is not None:
        document_id = existing.get("document_id", run_dir.name)
    else:
        document_id = units[0].get("document_id") or run_dir.name

    raw_nodes = group_units(units)
    expanded: list[dict[str, Any]] = []
    for rn in raw_nodes:
        expanded.extend(maybe_split(rn))

    nodes = build_nodes(document_id, expanded)

    chunks_path = run_dir / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for node in nodes:
            f.write(json.dumps(node, ensure_ascii=False) + "\n")

    manifest_mod.update_stage(run_dir, "canonical-markdown-to-llm-nodes", node_count=len(nodes))

    return len(nodes)
