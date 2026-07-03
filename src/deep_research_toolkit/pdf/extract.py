"""Knowledge-extraction stage, deterministic half: tables -> CSV, figures ->
PNG + captions.jsonl.

Ported from the agentictrading prototype's extract_tables.py/extract_figures.py
(see docs/contracts/pdf-ingestion-pipeline.md) -- the grid-reconstruction and
figure/caption-matching logic is unchanged, just moved into importable
functions so it's unit-testable without a subprocess and so both extractors
can share `common.manifest.update_stage`'s stage-merge behavior safely (see
that module's docstring: two callers writing into the same
`stages.knowledge-extraction` entry -- one contributing `table_count`, the
other `figure_count` -- must not clobber each other).

This module intentionally does NOT touch claims.jsonl/entities.jsonl/
relations.jsonl -- deciding what's a real claim or the same entity is a
judgment call Claude makes directly, per this skill's SKILL.md, not a script.
"""
from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path
from typing import Any

from deep_research_toolkit.common.manifest import update_stage

CAPTION_SCHEMA_VERSION = "1.0"


def _load_docling_raw(run_dir: Path) -> dict[str, Any]:
    docling_path = run_dir / "docling_raw.json"
    if not docling_path.is_file():
        raise FileNotFoundError(
            f"no docling_raw.json in {run_dir} -- run pdf-to-canonical-markdown first"
        )
    with open(docling_path, encoding="utf-8") as f:
        return json.load(f)


def _build_grid(table: dict[str, Any]) -> list[list[str]]:
    """Reconstruct a table's row/col grid from Docling's flat cell list.

    Docling gives each cell start/end row and col offsets plus spans. Only
    the anchor (top-left, i.e. start_row/start_col) position gets the cell's
    text; spanned-over cells stay blank rather than repeating the value --
    that's how a human transcribing a merged-cell table into CSV would do
    it, and it keeps the CSV's column count meaningful.
    """
    data = table.get("data", {})
    cells = data.get("table_cells", [])

    num_rows = data.get("num_rows")
    num_cols = data.get("num_cols")
    if num_rows is None:
        num_rows = max(
            (c.get("end_row_offset_idx", c.get("start_row_offset_idx", 0) + 1) for c in cells),
            default=0,
        )
    if num_cols is None:
        num_cols = max(
            (c.get("end_col_offset_idx", c.get("start_col_offset_idx", 0) + 1) for c in cells),
            default=0,
        )

    grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    for cell in cells:
        r = cell.get("start_row_offset_idx")
        c = cell.get("start_col_offset_idx")
        if r is None or c is None or not (0 <= r < num_rows) or not (0 <= c < num_cols):
            continue
        if not grid[r][c]:
            grid[r][c] = cell.get("text", "")
    return grid


def extract_tables(run_dir: str | Path) -> int:
    """Write tables/table_NN.csv for every table in docling_raw.json.

    Returns the number of tables extracted, and merges `table_count` into
    manifest.json's `stages.knowledge-extraction` (without disturbing
    `figure_count` or any judgment-based counts already recorded there).
    """
    import csv

    run_dir = Path(run_dir)
    doc = _load_docling_raw(run_dir)

    tables = doc.get("tables", [])
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for i, table in enumerate(tables, start=1):
        grid = _build_grid(table)
        out_path = tables_dir / f"table_{i:02d}.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(grid)

    update_stage(run_dir, "knowledge-extraction", table_count=len(tables))
    return len(tables)


def _ref_index(ref: Any) -> int | None:
    """Docling refs look like '#/texts/15' -- pull out the 15."""
    if not ref:
        return None
    try:
        return int(ref.rsplit("/", 1)[-1])
    except (ValueError, TypeError):
        return None


def _page_of(item: dict[str, Any]) -> int | None:
    prov = item.get("prov") or []
    if prov and "page_no" in prov[0]:
        return prov[0]["page_no"]
    return None


def _find_node_id(chunks: list[dict[str, Any]], page: int | None, caption_text: str | None) -> str | None:
    """Best-effort match of a figure to an already-chunked node in
    chunks.jsonl. Returns None if nothing lines up, which is expected and
    fine -- the figure row is written either way, just with node_id: null."""
    if not chunks or not caption_text:
        return None
    for node in chunks:
        if node.get("type") not in ("figure", "caption"):
            continue
        src = node.get("source", {})
        if page is not None and page not in (src.get("page_start"), src.get("page_end")):
            continue
        if caption_text.strip() and caption_text.strip() in node.get("text", ""):
            return node.get("node_id")
    return None


def _try_extract_image(item: dict[str, Any], run_dir: Path, out_path: Path) -> tuple[bool, str | None]:
    """Attempt to materialize actual image bytes for a Docling picture item.

    Returns (extracted, note). Handles the two shapes Docling's own image ref
    can take: an embedded data: URI, or a path to a file it wrote alongside
    the export. Anything else (no image key, empty uri, a uri that doesn't
    resolve to bytes) is reported via `note` rather than raising -- a missing
    image is data, not a bug in this function.
    """
    image = item.get("image")
    if not image:
        return False, "no image data captured by Docling for this picture element"

    uri = image.get("uri")
    if not uri:
        return False, "picture element present but its image.uri is empty"

    if uri.startswith("data:"):
        try:
            _, b64data = uri.split(",", 1)
            raw = base64.b64decode(b64data)
        except (ValueError, binascii.Error) as e:
            return False, f"could not decode embedded image data: {e}"
        with open(out_path, "wb") as f:
            f.write(raw)
        return True, None

    # Otherwise treat uri as a filesystem path Docling wrote the image to,
    # tried relative to the run dir and as an absolute/CWD-relative path.
    for candidate in (Path(uri), run_dir / uri):
        if candidate.is_file():
            with open(candidate, "rb") as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            return True, None
    return False, f"image.uri {uri!r} does not point at any file that actually exists"


def _collect_candidates(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Every figure reference in document order: real picture elements
    first (each may point at caption texts via its own `captions` list),
    then any caption-labeled text item no picture claimed -- an "orphan"
    caption, i.e. Docling recorded caption text with no picture element
    next to it. Both kinds are real figure references and both get a row."""
    texts = doc.get("texts", [])
    pictures = doc.get("pictures", [])

    candidates = []
    claimed_caption_idxs = set()

    for pic in pictures:
        caption_text = None
        for ref in pic.get("captions", []):
            idx = _ref_index(ref.get("$ref") if isinstance(ref, dict) else ref)
            if idx is None:
                continue
            claimed_caption_idxs.add(idx)
            if 0 <= idx < len(texts):
                caption_text = texts[idx].get("text")
        candidates.append({"kind": "picture", "page": _page_of(pic), "caption": caption_text, "item": pic})

    for idx, t in enumerate(texts):
        if t.get("label") == "caption" and idx not in claimed_caption_idxs:
            candidates.append({"kind": "orphan_caption", "page": _page_of(t), "caption": t.get("text"), "item": t})

    candidates.sort(key=lambda c: c["page"] if c["page"] is not None else 10**9)
    return candidates


def extract_figures(run_dir: str | Path) -> int:
    """Write figures/figure_NN.png (where extractable) and always write a
    row per figure reference to figures/captions.jsonl, even when no image
    could be extracted (`extracted: false`, with a `note` -- never dropped).

    Returns the number of figure references recorded, and merges
    `figure_count` into manifest.json's `stages.knowledge-extraction`
    (without disturbing `table_count` or any judgment-based counts already
    recorded there).
    """
    run_dir = Path(run_dir)
    doc = _load_docling_raw(run_dir)

    chunks = []
    chunks_path = run_dir / "chunks.jsonl"
    if chunks_path.is_file():
        with open(chunks_path, encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, cand in enumerate(_collect_candidates(doc), start=1):
        figure_id = f"figure_{i:02d}"
        page = cand["page"]
        caption = cand["caption"]
        node_id = _find_node_id(chunks, page, caption)

        if cand["kind"] == "picture":
            out_path = figures_dir / f"{figure_id}.png"
            extracted, note = _try_extract_image(cand["item"], run_dir, out_path)
        else:
            extracted, note = False, (
                "caption text detected but no picture element present in the source "
                "document (Docling recorded caption text with no corresponding image) "
                "-- recorded as a figure reference, not dropped"
            )

        rows.append(
            {
                "schema_version": CAPTION_SCHEMA_VERSION,
                "figure_id": figure_id,
                "page": page,
                "caption": caption,
                "node_id": node_id,
                "extracted": extracted,
                "note": note,
            }
        )

    captions_path = figures_dir / "captions.jsonl"
    with open(captions_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    update_stage(run_dir, "knowledge-extraction", figure_count=len(rows))
    return len(rows)
