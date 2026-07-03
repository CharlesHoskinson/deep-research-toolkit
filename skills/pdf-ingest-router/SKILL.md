---
name: pdf-ingest-router
description: Use this FIRST, before any other step, whenever a PDF needs to enter the knowledge pipeline — a whitepaper, spec, research paper, financial/legal document, form, slide deck, or scanned document. Computes a stable document_id, classifies the PDF (digital-text/scanned/scientific-math/form/financial-legal/slide-like) from real extractable-text/table/math/form signals, and starts the pdf-runs/<document_id>/ working directory (classification.json + manifest.json) that every later pipeline stage (pdf-to-canonical-markdown, pdf-layout-provenance, canonical-markdown-to-llm-nodes, knowledge-extraction, llm-wiki-writer, rag-eval-harness) reads from.
---

# PDF Ingest Router

First stage of the seven-stage PDF ingestion pipeline described in
`docs/contracts/pdf-ingestion-pipeline.md` (read that doc in full if you
haven't — it's the contract every stage below depends on). This skill's
only job: look at a raw PDF, decide what kind of document it is, and start
its `<pdf_runs_dir>/<document_id>/` directory so every later stage has a
stable identity to key off.

Do not hand-derive `document_id` anywhere else in the pipeline — it is
computed once, here, and every later stage reads it back out of
`manifest.json`.

## First: find this project's configuration

Before doing anything else, read `.deepresearch.yml` (walk up from the
current directory if it's not in the cwd — same discovery as `.git`). It
tells you:
- `knowledge_base.pdf_runs_dir` — where PDF run directories live (the
  command below defaults to this via `deep_research_toolkit.config`, but
  you can pass `--runs-dir` to override).
- `features.pdf_ingestion` — whether this project has PDF ingestion turned
  on at all.

If there's no `.deepresearch.yml` yet, this still works zero-config
(falls back to `pdf-runs/` under the current directory) for quick
exploration, but tell the user to run `drt init` before relying on it for
real project use.

## Quick start

```
python scripts/classify_pdf.py <pdf_path> [--runs-dir <dir>]
```

This creates `<runs_dir>/<document_id>/` (document_id = slugified filename +
first 8 hex chars of the file's sha256), writes `classification.json`, starts
`manifest.json`, and prints the run directory path to stdout. Pass that
printed path to the next stage (`pdf-to-canonical-markdown`'s
`scripts/convert.py <run_dir>`).

Requires the `pdf` extra: `pip install "deep-research-toolkit[pdf]"`.

## What it looks at

Using `pypdf` (page count, AcroForm field detection) and `pdfplumber`
(per-page text extraction, table detection, math-symbol counting), it
computes five signals and picks one `route`:

| signal | what it measures |
|---|---|
| `avg_extractable_chars_per_page` | mean characters `pdfplumber` can extract per page |
| `has_acroform_fields` | any fillable form fields present (pypdf) |
| `image_only_page_ratio` | fraction of pages with under ~10 extractable chars (i.e. probably scanned images, no text layer) |
| `table_like_page_ratio` | fraction of pages where `pdfplumber.find_tables()` finds at least one table |
| `detected_math_density` | low/medium/high, from a count of math/LaTeX-ish symbols and Greek letters per page |

See `references/routing-table.md` for the exact priority order and
thresholds used to turn those five signals into a `route`, and for the
routing table mapping each `route` to the backend that's ideal for it "in
principle" per the original research.

## Backend recommendation this pass

This toolkit only has **Docling actually wired up** as a conversion backend
(`pdf-to-canonical-markdown`). So `recommended_backend` in
`classification.json` is `"docling"` for every route right now — but
`notes` still records what the ideal backend would be per the routing
table (e.g. Docling's OCR mode for `scanned`, Marker as a scientific-math
fallback) so a human/future stage can see where a smarter backend would
help once one is wired up.

## Output

- `classification.json` — see `docs/contracts/pdf-ingestion-pipeline.md`
  for the exact schema.
- `manifest.json` — started here with `schema_version`, `document_id`,
  `source_file`, `source_hash`, `created_at`, and
  `stages.pdf-ingest-router`. Every later stage appends its own key rather
  than overwriting the file.

Both land in `<pdf_runs_dir>/<document_id>/`, which should be tracked in
git in a consuming project — this is meant to be the durable, auditable
corpus, not throwaway scratch state.
