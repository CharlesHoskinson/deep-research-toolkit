---
name: canonical-markdown-to-llm-nodes
description: Use after pdf-layout-provenance in the PDF ingestion pipeline. Turns a document's provenance-enriched units (provenance.jsonl: headings, paragraphs, tables, figures, captions) into retrieval-ready chunks.jsonl nodes for the downstream knowledge-extraction and rag-eval-harness stages.
---

# canonical-markdown-to-llm-nodes

Fourth stage of the PDF ingestion pipeline
(`docs/contracts/pdf-ingestion-pipeline.md`). Reads
`<run_dir>/provenance.jsonl`, writes `<run_dir>/chunks.jsonl` plus the
`canonical-markdown-to-llm-nodes` entry in `manifest.json`.

## Why structure-aware chunking, not fixed-token splitting

A naive pipeline would tokenize `canonical.md` and cut every N tokens. That
breaks two things this pipeline depends on:

- **Meaning boundaries.** A fixed-size window will happily split a heading
  from the paragraph it introduces, or cut a sentence in half at a table
  boundary, producing chunks that read as fragments rather than coherent
  ideas -- worse embeddings, worse retrieval.
- **Page-accurate citations.** `rag-eval-harness` later verifies that every
  claim's supporting quote is a verbatim substring on its cited page. If a
  chunk boundary doesn't line up with a real structural unit, a citation can
  span two provenance units awkwardly, or point at a page that only holds
  half of what the chunk claims to contain.

Instead, this stage chunks along the document's own structure, which
`pdf-layout-provenance` already recovered per-unit: one node per heading
section (grouping its constituent paragraphs), one node per table, one node
per figure. Nodes stay small enough to embed well, but each one is a real,
citable unit of meaning.

## Usage

```
python scripts/chunk_nodes.py <run_dir>
```

`<run_dir>` is `<pdf_runs_dir>/<document_id>`, the same directory every
other stage in this pipeline reads and writes
(`docs/contracts/pdf-ingestion-pipeline.md` for the full layout).

Reads `<run_dir>/provenance.jsonl`. If `<run_dir>/manifest.json` doesn't
exist yet (e.g. running this stage standalone against a hand-built
provenance.jsonl), it is created with `document_id` taken from the
provenance units' own `document_id` field, or the run directory's basename
as a last resort -- normally `document_id` already exists in the manifest,
written once by `pdf-ingest-router`, and this stage just reads it back out.

Writes `<run_dir>/chunks.jsonl` (each line stamped with
`"schema_version": "1.0"`) and updates `manifest.json`'s
`stages.canonical-markdown-to-llm-nodes` with `completed_at` and
`node_count`, via `common.manifest.update_stage` -- this never clobbers
another stage's already-recorded entry.

The actual logic lives in `deep_research_toolkit.pdf.chunk.chunk_nodes(run_dir)`
(returns the node count); the script is a thin CLI wrapper so the same
function is importable and unit-testable without going through a
subprocess.

## Grouping and splitting logic

Full detail, including the exact rules for what folds into a section's text
and what doesn't, is in `references/chunking-strategy.md` -- read it before
changing the grouping logic. Summary:

- A `heading` unit starts a new `section` node; consecutive `paragraph` /
  `list_item` / other prose units fold into it until the next heading,
  table, or figure.
- `table` and `figure`/`picture` units always get their own node.
- `caption` units contribute their `unit_id` and `page` to whichever
  section/node they fall under, but their text is **not** spliced into the
  surrounding narrative text (a caption describing a figure reads as a
  non-sequitur mid-paragraph).
- `links.previous`/`links.next` chain every node in final document order.

### Section splitting: implemented (not just documented as a v1 gap)

When a section's folded text exceeds **1500 characters** and was built from
more than one contributing paragraph-like unit, the section is split: the
`section` node itself is kept (as a coarse, still-citable overview node, its
`text` unchanged) and each contributing unit *also* becomes its own
`paragraph`-type child node, inserted immediately after the section node in
the sequence, with `links.parent` pointing back at the section's `node_id`.
This keeps a single flat, linearly-chained `chunks.jsonl` (easy for
`rag-eval-harness` to walk) while giving retrieval smaller, more precise
nodes for long sections. See `references/chunking-strategy.md` for why 1500
was chosen and how to tune it.

## Deliberately out of scope here

`summary`, `keywords`, and `entities` are always written as `""` / `[]` /
`[]` placeholders. Filling those in requires actual judgment about what a
chunk means and which entities it mentions -- that's an LLM task, and it's
the `knowledge-extraction` skill's job, run after this one, reading
`chunks.jsonl` directly. An empty `summary`/`keywords`/`entities` in
`chunks.jsonl` is expected output from this stage, not a bug.

## Reference example

`tests/fixtures/reference-run-hydra-settlement/provenance.jsonl` (17 units)
→ `chunks.jsonl` (9 nodes: 8 sections + 1 table) is the real, verified
example for this stage, produced by chaining all seven pipeline skills
together against `tests/fixtures/hydra-settlement-test-fixture.pdf`. Note
the table node's `section_path` is `["3. Throughput Comparison"]` -- it
sits under that heading, not a later one. See
`references/chunking-strategy.md` for the full note on why this matters and
how `content_hash` is computed.
