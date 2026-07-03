---
name: pdf-to-canonical-markdown
description: Second stage of the PDF ingestion pipeline — converts a routed PDF into canonical markdown and Docling's raw structured JSON. Runs after pdf-ingest-router and before pdf-layout-provenance. Use when a run directory has classification.json but no canonical.md yet.
---

# PDF to Canonical Markdown

Second stage of the seven-stage PDF ingestion pipeline described in
`docs/contracts/pdf-ingestion-pipeline.md` — read that doc first if you
haven't; it defines every file shape this stage reads and writes. This
stage's job is narrow: turn the source PDF into a canonical markdown
rendering plus Docling's own structured document JSON, which downstream
stages (`pdf-layout-provenance`, chunking) parse for headings, tables, and
bounding boxes.

## Usage

```
python scripts/convert.py <run_dir>
```

Requires the `pdf` extra: `pip install "deep-research-toolkit[pdf]"`.

`<run_dir>` is `<pdf_runs_dir>/<document_id>/`, created by
`pdf-ingest-router`. The script is a thin CLI shim over
`deep_research_toolkit.pdf.convert.convert`, which:

1. Reads `manifest.json` in `<run_dir>` for `source_file` (the original PDF
   path — never passed on the command line here, per the pipeline's CLI
   convention: every stage after the first takes only the run directory).
2. Converts it with Docling, `PdfPipelineOptions(do_table_structure=True)`
   explicitly set.
3. Writes `canonical.md` (`doc.export_to_markdown()`) and
   `docling_raw.json` (`doc.export_to_dict()`, kept as Docling's own shape —
   downstream stages read `texts`/`tables`/`pictures`/`pages` directly from
   it rather than a reprocessed format).
4. Appends `stages["pdf-to-canonical-markdown"]` to `manifest.json` (via
   `common.manifest.update_stage`, which never clobbers another stage's
   already-recorded entry) with `completed_at`, `parser: "docling"`, and
   `parser_version`.

This skill doesn't need `.deepresearch.yml` itself — it takes a run
directory as its only argument and reads everything it needs
(`source_file`) back out of that directory's `manifest.json`, which
`pdf-ingest-router` already resolved against the project's configured
`pdf_runs_dir`.

## Why the retry-once behavior exists

Even for a plain digital-text PDF, `DocumentConverter()` with default
settings tries to fetch OCR/layout models on first use. On a flaky network
that download can fail mid-stream (`requests.exceptions.ChunkedEncodingError`
or a bare `urllib3.exceptions.ProtocolError`), which looks like a Docling
bug but is really a transient connectivity blip — this is a real failure
mode hit and fixed during this pipeline's original development, not a
hypothetical. `deep_research_toolkit.pdf.convert.convert_with_retry` retries
the whole `converter.convert()` call once when it sees one of these
transient exceptions before giving up. Do **not** "fix" this by disabling
OCR or stripping the model download — this pipeline routes `scanned`
documents through Docling's OCR path later, so OCR needs to stay available
and working, not silently turned off to dodge a flaky connection.

## What this stage does not do

Only Docling is wired up. `references/backend-fallbacks.md` documents when
a future document should reach for Marker, MinerU, or Unstructured instead
— none of those are installed or called here. If `classification.json`'s
`route` is `scientific-math` or `scanned` and Docling's output looks wrong
(mangled formulas, empty pages), that's the cue to consult that reference,
not to extend this script.
