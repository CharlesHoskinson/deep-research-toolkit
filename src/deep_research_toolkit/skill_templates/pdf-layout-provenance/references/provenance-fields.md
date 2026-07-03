# provenance.jsonl field reference

One JSON object per line, one line per structural unit extracted from
`docling_raw.json`, in true document order. This reference stands alone,
but is derived from the shared contract at
`docs/contracts/pdf-ingestion-pipeline.md` — consult that doc if a
downstream stage's expectations seem to differ from what's written here.

```json
{
  "schema_version": "1.0",
  "unit_id": "u0009",
  "document_id": "hydra-settlement-test-fixture-4edb3c3c",
  "source_file": "/absolute/path/to/hydra-settlement-test-fixture.pdf",
  "page": 1,
  "section_path": ["2. Architecture", "2.2 Settlement Guarantees"],
  "bbox": {"l": 78.0, "t": 368.38, "r": 515.96, "b": 335.13, "coord_origin": "BOTTOMLEFT"},
  "content_hash": "sha256:6c3a286307040c3b",
  "extraction_method": "docling",
  "confidence": 1.0,
  "unit_type": "paragraph",
  "text": "Because every state transition inside the Head requires unanimous signatures..."
}
```

(This example is drawn from `tests/fixtures/reference-run-hydra-settlement/`,
this pipeline's verified reference run.)

## Fields

- **schema_version** — always `"1.0"` for this artifact shape. See
  `docs/contracts/schema-versions.md`.
- **unit_id** — `u` + a 4-digit, 1-based sequence number in document
  order (`u0001`, `u0002`, ...). Stable within one run of this script for
  one document; re-running the script regenerates all unit_ids, so
  downstream files that reference a `unit_id` (chunks.jsonl's
  `source.unit_ids`) must be regenerated together with it, not mixed
  across runs.
- **document_id** — copied from `manifest.json`, not recomputed here.
- **source_file** — copied from `manifest.json`'s `source_file` (the
  original PDF path). Host-machine-absolute, not portable across
  machines/teams — see the contract doc's note on this.
- **page** — 1-based page number, taken from the unit's first provenance
  entry (`item["prov"][0]["page_no"]`). `null` if the item has no `prov`.
- **section_path** — ordered list of ancestor heading texts, outermost
  first, e.g. `["2. Architecture", "2.2 Settlement Guarantees"]`. Built by
  walking the document in order and maintaining a stack of
  `(depth, heading_text)` pairs:
  - On a heading unit: pop every stack entry whose depth is `>=` the new
    heading's depth (this removes both true siblings and anything nested
    deeper than the new heading), record `section_path` as the *remaining*
    stack (the heading's own ancestors — **not** itself), then push the new
    heading.
  - On any non-heading unit: `section_path` is just the current stack,
    unmodified.
  - See "Heading depth heuristic" below for how `depth` is computed —
    Docling's own `level` field is not reliable enough to use directly.
  - Document order matters here: the walk must interleave text/table/
    picture items exactly as `body.children` lists them, not process all
    headings first and then attach every non-heading item afterward. A
    table sitting between "3. Throughput Comparison" and "4. Threat Model"
    must get `section_path: ["3. Throughput Comparison"]` — not the next
    section that happens to come later in the document — because the
    table's actual position in the document is before that heading is
    ever pushed. This is a real regression that an earlier, cruder
    implementation hit; the reference run's table unit
    (`unit_type: table`) is the regression check for it.
- **bbox** — Docling's own bbox dict (`l`/`t`/`r`/`b`/`coord_origin`),
  passed through unchanged from `item["prov"][0]["bbox"]`. `null` if the
  backend doesn't expose one for this unit. Deliberately *not* reprojected
  or flattened — Docling already gives page-relative coordinates, and
  inventing a different convention here would just be a new source of bugs
  for anything that later needs to draw a highlight box on the source PDF.
- **content_hash** — computed via `common.hashing.content_hash`:
  `"sha256:" + <first 16 hex chars of sha256(text)>`. A short fingerprint,
  not a security hash — good enough to notice when a unit's text has
  changed across re-runs, or to dedupe identical units.
- **extraction_method** — always `"docling"` in this pipeline's first
  pass (the only backend currently wired up).
- **confidence** — always `1.0`. This pipeline does plain digital-text
  extraction only; Docling doesn't expose a per-element confidence score
  outside its OCR path, and OCR isn't wired up yet. When it is, OCR'd units
  should carry Docling's own OCR confidence here instead of `1.0`.
- **unit_type** — one of `heading | paragraph | table | figure | caption |
  list_item`, derived from Docling's `label`:

  | Docling `label`   | `unit_type`  |
  |--------------------|-------------|
  | `section_header`, `title` | `heading`   |
  | `text`, `paragraph`       | `paragraph` |
  | `table`                   | `table`     |
  | `picture`                 | `figure`    |
  | `caption`                 | `caption`   |
  | `list_item`               | `list_item` |

  Any other/unrecognized label falls back to `paragraph` rather than
  raising, so an unfamiliar Docling label on a future document degrades
  gracefully instead of crashing the pipeline.
- **text** — the unit's plain text. For `table` units this is a
  pipe-separated rendering of the table grid (rows joined by `\n`, cells by
  `" | "`, no header separator row) built from `table_cells`/`grid` — a
  plain-text form good enough for provenance and hashing; the real per-cell
  CSV is produced later by `knowledge-extraction`'s `extract_tables.py`.

## Heading depth heuristic

Docling's `level` field on `section_header` items is often flat — in the
reference fixture, `"2. Architecture"` and `"2.1 Head Lifecycle"` **both**
come back as `level: 1`. To recover the true nesting implied by numbered
headings, depth is computed as:

1. If the heading's text starts with a numbering pattern like `"2."` or
   `"2.1"` or `"2.1.3"`, depth = the number of dot-separated numeric
   components (`"2."` → 1, `"2.1"` → 2, `"2.1.3"` → 3).
2. Otherwise (document titles, unnumbered headings), depth falls back to
   Docling's own `level` field, defaulting to 1 if absent.

This is a heuristic, documented here rather than hidden in code, because it
is not something Docling guarantees — it is tuned against how this
pipeline's verified reference run and typical working-paper-style
documents number their sections. A document with real (non-flat) `level`
values and non-numeric headings will fall through to case 2 and just use
`level` directly, which is the correct behavior in that case.
