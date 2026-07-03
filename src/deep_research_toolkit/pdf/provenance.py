"""Walk Docling's raw document JSON in true document order and emit one
provenance record per structural unit (heading/paragraph/table/figure/
caption/list_item), tracking section_path as a stack of active headings.

See docs/contracts/pdf-ingestion-pipeline.md's `provenance.jsonl` section
for the schema, and skills/pdf-layout-provenance/references/provenance-fields.md
for the full field-by-field reference including the heading-depth heuristic
explained below.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..common.hashing import content_hash
from ..common.manifest import load_manifest, update_stage

SCHEMA_VERSION = "1.0"

LABEL_TO_UNIT_TYPE = {
    "section_header": "heading",
    "title": "heading",
    "text": "paragraph",
    "paragraph": "paragraph",
    "table": "table",
    "picture": "figure",
    "caption": "caption",
    "list_item": "list_item",
}

# Docling's own `level` field on section_header items is frequently flat
# (every heading in a document comes back as level=1, regardless of true
# nesting -- confirmed against the real output for the test fixture, where
# "2. Architecture" and "2.1 Head Lifecycle" both have level=1). To recover
# real nesting we count the dot-separated numeric components in a heading's
# own numbering: "2. Architecture" -> depth 1, "2.1 Settlement Guarantees"
# -> depth 2, a hypothetical "2.1.3 ..." -> depth 3. Headings with no
# leading number (document titles, unnumbered headings) fall back to
# Docling's `level` field (default 1) since there's no numbering to parse.
# This is a heuristic, not something Docling guarantees -- it matches every
# heading in the test fixture and is documented here so a future document
# that breaks it is easy to diagnose.
NUMBERING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")


def heading_depth(text: str, level: int | None) -> int:
    m = NUMBERING_RE.match(text.strip())
    if m:
        return len(m.group(1).split("."))
    return level or 1


def render_table(table_item: dict[str, Any]) -> str:
    """Render a Docling table item as pipe-separated rows (no header
    separator line) -- a plain-text rendering good enough for provenance
    and content-hashing; knowledge-extraction's extract_tables.py is what
    produces the real per-cell CSV."""
    data = table_item.get("data", {})
    grid = data.get("grid")
    if not grid:
        cells = data.get("table_cells", [])
        if not cells:
            return ""
        n_rows = max(c["end_row_offset_idx"] for c in cells)
        n_cols = max(c["end_col_offset_idx"] for c in cells)
        grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        for c in cells:
            grid[c["start_row_offset_idx"]][c["start_col_offset_idx"]] = c.get("text", "")
        return "\n".join(" | ".join(row) for row in grid)
    return "\n".join(" | ".join(cell.get("text", "") for cell in row) for row in grid)


def resolve_ref(doc: dict[str, Any], ref: str) -> tuple[str, dict[str, Any]]:
    """Resolve a Docling `{"$ref": "#/texts/3"}` pointer to (list_name, item)."""
    list_name, idx = ref.lstrip("#/").split("/")
    return list_name, doc[list_name][int(idx)]


def iter_body_items(doc: dict[str, Any]):
    """Yield (list_name, item) for every leaf unit in true document order.

    Walks `body.children` (which interleaves texts/tables/pictures in
    reading order -- this matters: a table dropped between two paragraphs
    must get the section_path active at that point in the document, not
    whatever section happened to be active after every text item was
    processed). Recurses into `groups`, since Docling nests constructs like
    lists inside a group wrapper rather than listing their items flat in
    body.children -- the group itself isn't a unit, only its children are.
    """
    def walk(children):
        for child_ref in children:
            list_name, item = resolve_ref(doc, child_ref["$ref"])
            if list_name == "groups":
                yield from walk(item.get("children", []))
            else:
                yield list_name, item
    yield from walk(doc.get("body", {}).get("children", []))


def build_unit(
    list_name: str,
    item: dict[str, Any],
    unit_index: int,
    document_id: str,
    source_file: str,
    section_stack: list[tuple[int, str]],
) -> dict[str, Any]:
    label = item.get("label", "text")
    unit_type = LABEL_TO_UNIT_TYPE.get(label, "paragraph")

    text = render_table(item) if list_name == "tables" else item.get("text", "")

    prov0 = (item.get("prov") or [{}])[0]
    page = prov0.get("page_no")
    bbox = prov0.get("bbox")

    if unit_type == "heading":
        depth = heading_depth(text, item.get("level"))
        # A heading's own section_path is its ancestors, not itself: pop
        # any active heading at this depth or deeper (a sibling or a
        # previous deeper subsection) before recording, then push this one.
        section_stack[:] = [s for s in section_stack if s[0] < depth]
        section_path = [t for _, t in section_stack]
        section_stack.append((depth, text))
    else:
        section_path = [t for _, t in section_stack]

    return {
        "schema_version": SCHEMA_VERSION,
        "unit_id": f"u{unit_index:04d}",
        "document_id": document_id,
        "source_file": source_file,
        "page": page,
        "section_path": section_path,
        "bbox": bbox,
        "content_hash": content_hash(text),
        "extraction_method": "docling",
        # Plain digital-text extraction only -- this pipeline doesn't run
        # OCR yet, so there's no per-element confidence score to report.
        "confidence": 1.0,
        "unit_type": unit_type,
        "text": text,
    }


def extract_provenance(run_dir: Path | str) -> int:
    """Read `<run_dir>/docling_raw.json`, walk it in true document order,
    write `<run_dir>/provenance.jsonl` (one JSON object per line), append
    the `pdf-layout-provenance` stage to manifest.json, and return the
    number of units written.
    """
    run_dir = Path(run_dir)
    docling_path = run_dir / "docling_raw.json"
    if not docling_path.is_file():
        raise FileNotFoundError(
            f"{docling_path} not found -- run pdf-to-canonical-markdown first"
        )

    manifest = load_manifest(run_dir)
    if manifest is None:
        raise FileNotFoundError(
            f"{run_dir / 'manifest.json'} not found -- run pdf-ingest-router first"
        )

    document_id = manifest["document_id"]
    source_file = manifest.get("source_file", "")

    with open(docling_path, encoding="utf-8") as f:
        doc = json.load(f)

    units = []
    section_stack: list[tuple[int, str]] = []  # (depth, heading_text) stack
    for unit_index, (list_name, item) in enumerate(iter_body_items(doc), start=1):
        units.append(
            build_unit(list_name, item, unit_index, document_id, source_file, section_stack)
        )

    out_path = run_dir / "provenance.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for unit in units:
            f.write(json.dumps(unit) + "\n")

    update_stage(run_dir, "pdf-layout-provenance", unit_count=len(units))

    return len(units)
