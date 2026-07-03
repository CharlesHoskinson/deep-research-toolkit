---
name: pdf-layout-provenance
description: Third stage of the PDF ingestion pipeline — walks Docling's raw structured document JSON and emits provenance.jsonl, one record per structural unit (heading/paragraph/table/figure/caption/list_item) with its page, section_path, bbox, and content hash. Runs after pdf-to-canonical-markdown and before canonical-markdown-to-llm-nodes. Use when a run directory has docling_raw.json but no provenance.jsonl yet.
---

# PDF Layout Provenance

Third stage of the seven-stage PDF ingestion pipeline described in
`docs/contracts/pdf-ingestion-pipeline.md` — read that doc first if you
haven't; it defines every file shape this stage reads and writes, including
the exact `provenance.jsonl` schema.

## Why this stage exists

`pdf-to-canonical-markdown` gives downstream stages a nice markdown
rendering, but markdown throws away *where each piece of text came from*.
Everything built on top of this pipeline — chunked retrieval nodes,
extracted claims, wiki pages — eventually makes a factual assertion that
traces back to some span of the source PDF. Without a durable record of
"this text is on page 2, under section 2.2, at this bounding box" for every
extracted unit, none of those downstream claims are auditable: a reader (or
`rag-eval-harness`'s evidence-quote check) has no way to verify a claim
against the original document short of re-reading the whole PDF. This stage
produces that record once, up front, so every later stage can cite it
instead of re-deriving it.

## Usage

```
python scripts/extract_provenance.py <run_dir>
```

`<run_dir>` is `<pdf_runs_dir>/<document_id>/` (see
`docs/contracts/pdf-ingestion-pipeline.md` for how `pdf_runs_dir` resolves),
and must already contain `docling_raw.json` (written by
`pdf-to-canonical-markdown`) and `manifest.json` (for `document_id` and
`source_file` — never passed on the command line here, per the pipeline's
CLI convention: every stage after the first takes only the run directory).
The script is a thin CLI shim over
`deep_research_toolkit.pdf.provenance.extract_provenance`, which:

1. Reads `docling_raw.json` and walks `body.children` in true document
   order — not `texts` then `tables` separately — resolving each
   `{"$ref": "#/texts/N"}` / `{"$ref": "#/tables/N"}` pointer to its item.
   This ordering matters: a table sitting between two paragraphs must
   inherit the section active *at that point*, not whatever section was
   active after every text item had already been processed.
2. Maps each item's Docling `label` to a `unit_type` (`section_header`→
   `heading`, `text`/`paragraph`→`paragraph`, `table`→`table`, `picture`→
   `figure`, `caption`→`caption`, `list_item`→`list_item`), tracks a stack
   of active headings to compute `section_path`, and hashes each unit's
   text into `content_hash` via `common.hashing.content_hash`.
3. Writes one JSON object per line to `<run_dir>/provenance.jsonl`, each
   tagged with `"schema_version": "1.0"`.
4. Appends `stages["pdf-layout-provenance"]` (`completed_at`, `unit_count`)
   to `manifest.json` via `common.manifest.update_stage` — this never
   clobbers another stage's already-recorded entry.

Full field-by-field schema, including the section_path stack algorithm and
the heading-depth heuristic (Docling's own `level` field is frequently flat
— see below), is in `references/provenance-fields.md`.

## The heading-depth heuristic

Docling's `level` field on `section_header` items is often uninformative:
in the reference fixture, "2. Architecture" and "2.1 Head Lifecycle" **both**
come back as `level: 1`, even though "2.1" is clearly a subsection of "2.".
`deep_research_toolkit.pdf.provenance` disambiguates by counting the
dot-separated numeric components in a heading's own numbering — "2.
Architecture" → depth 1, "2.1 Settlement Guarantees" → depth 2 — and only
falls back to Docling's `level` field for headings with no leading number
(document titles, unnumbered headings). This is a heuristic, not a Docling
guarantee; if a future document uses non-numeric heading schemes (all-caps
headers, no numbering at all) with real nesting, `level` alone is what's
used and this stage trusts whatever Docling reports.

## What this stage does not do

`confidence` is hardcoded to `1.0` for every unit — this pipeline doesn't
run OCR yet (see `pdf-to-canonical-markdown`'s references for when a
scanned-document route would change that), so there's no per-element
confidence score to surface. Table cell structure (rows/columns) is
flattened into a pipe-separated text rendering here purely for provenance
and hashing purposes; the real per-cell CSV is `knowledge-extraction`'s
`extract_tables.py` job, not this stage's.
