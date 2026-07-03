---
name: knowledge-extraction
description: Fifth stage of the PDF ingestion pipeline — pulls tables and figures out of Docling's raw export, and has Claude read chunks.jsonl to write claims.jsonl/entities.jsonl/relations.jsonl by hand. Runs after canonical-markdown-to-llm-nodes and before llm-wiki-writer. Use when a run directory has chunks.jsonl but no claims.jsonl yet.
---

# Knowledge Extraction

Fifth stage of the seven-stage PDF ingestion pipeline described in
`docs/contracts/pdf-ingestion-pipeline.md` — read that doc first if you
haven't; it defines every file shape below. This stage produces six
outputs, and they split cleanly into two kinds of work:

| Output | How it's produced |
|---|---|
| `tables/table_NN.csv` | `scripts/extract_tables.py` — deterministic |
| `figures/figure_NN.png` + `figures/captions.jsonl` | `scripts/extract_figures.py` — deterministic |
| `claims.jsonl` | **Claude, reading chunks.jsonl, following this file** |
| `entities.jsonl` | **Claude, reading chunks.jsonl, following this file** |
| `relations.jsonl` | **Claude, reading chunks.jsonl, following this file** |

Don't try to script the second half. Deciding whether a sentence is a real,
atomic claim, or whether "Hydra" and "Hydra Head" are the same entity, is a
judgment call — that's exactly the kind of thing this repo's skills call out
explicitly rather than papering over with a brittle regex (see
`research-knowledge-graph`'s `references/wiki-operations.md` for the same
pattern applied to web research instead of PDFs).

## Part 1 — deterministic extraction (run the scripts)

```
python scripts/extract_tables.py <run_dir>
python scripts/extract_figures.py <run_dir>
```

`<run_dir>` is `<pdf_runs_dir>/<document_id>/` (`pdf_runs_dir` resolves from
`.deepresearch.yml`'s `knowledge_base.pdf_runs_dir`, else a `pdf-runs/`
fallback — see the contract doc). Both scripts:

- Read `docling_raw.json` directly (Docling's own structured export, kept
  as-is by `pdf-to-canonical-markdown`) — never the original PDF.
- Are thin CLI shims over `deep_research_toolkit.pdf.extract`, so the same
  logic is unit-testable without a subprocess.
- Are idempotent: re-running overwrites their own output files and merges
  their own key into `manifest.json`'s `stages.knowledge-extraction` without
  touching the other script's keys or the claim/entity/relation counts
  Claude fills in later.

`extract_tables.py` walks `tables[].data.table_cells`, placing each cell's
text at its `start_row_offset_idx`/`start_col_offset_idx` (spanned-over
cells stay blank rather than repeating the value), and writes one
`tables/table_NN.csv` per table.

`extract_figures.py` walks `pictures[]` plus any `caption`-labeled item in
`texts[]` that no picture claimed. For each one it tries to materialize
`figures/figure_NN.png` from Docling's own image data, but **always** writes
a row to `figures/captions.jsonl` — a figure with a caption but no
extractable image (a diagram-only placeholder in a text-only test fixture,
or a picture element Docling didn't capture pixel data for) is recorded with
`extracted: false` and a `note`, never silently dropped. This matters
because `rag-eval-harness`'s `figures_accounted_for` check counts figure
*references*, not just successfully-extracted PNGs.

## Part 2 — claims, entities, relations (Claude does this directly)

Read `chunks.jsonl` in the run directory, then write three JSONL files
directly into the run directory, one JSON object per line, matching these
schemas exactly (see the contract doc for the authoritative version):

```json
// claims.jsonl
{"schema_version": "1.0", "claim_id": "c_0001", "document_id": "...", "claim": "...", "claim_type": "architectural | empirical | definitional | comparative", "confidence": "high | medium | low", "supporting_evidence": [{"node_id": "...", "quote": "...", "page": 2}], "entities": ["hydra", "cardano"], "relations": [{"subject": "hydra", "predicate": "serves_as", "object": "synchronous settlement layer"}]}

// entities.jsonl
{"schema_version": "1.0", "entity_id": "hydra-head", "name": "Hydra Head", "aliases": ["Hydra", "Head"], "type": "protocol", "mentions": ["...:n002", "...:n004"]}

// relations.jsonl
{"schema_version": "1.0", "relation_id": "r_0001", "subject": "hydra-head", "predicate": "serves_as", "object": "synchronous settlement layer", "supporting_claim": "c_0001", "document_id": "..."}
```

After writing all three, update `manifest.json` yourself (the scripts above
only ever touch `table_count`/`figure_count`): merge `claim_count`,
`entity_count`, and `relation_count` into `stages.knowledge-extraction`, and
set/refresh `completed_at`. Use `deep_research_toolkit.common.manifest.
update_stage(run_dir, "knowledge-extraction", claim_count=..., entity_count=...,
relation_count=...)` — it merges into the existing stage entry rather than
replacing it, so the table/figure counts already recorded there survive.

Read `references/claim-extraction-guide.md` before your first extraction
pass — it works through the fixture's own five claims to show what makes a
claim atomic and well-evidenced versus compound or unsupported. The rules
that most affect whether the eval harness (`rag-eval-harness`, downstream)
passes are these four:

### 1. One assertion per claim — no compound sentences

If a sentence asserts two things ("Hydra reaches 1,000 TPS *and* settles
in under a second"), and both halves matter, either split it into two
claims or pick the single load-bearing assertion — don't cram a compound
sentence into one `claim` string. A claim should be checkable as true or
false on its own.

### 2. Every `supporting_evidence[].quote` is copy-pasted, never paraphrased

The **evidence rule**, checked mechanically downstream: each quote must be
a **verbatim substring** of the text on that `page` (per
`provenance.jsonl`/`chunks.jsonl`). Select the exact span from the source
node's `text` field and paste it in — do not clean up wording, expand an
abbreviation, or fix punctuation. If you find yourself editing the quote to
make it read better, that's the sign to shorten the span instead, not to
paraphrase it. Always include `page`, matching where that text actually
appears.

### 3. Merge entity mentions that clearly refer to the same thing

A document will use a short name and a full name interchangeably ("Hydra"
and "Hydra Head", "OWS" and "Open Wallet Standard"). These become **one**
row in `entities.jsonl`, with the canonical/most-formal form as `name` and
the others in `aliases` — never two separate `entity_id`s for the same
underlying concept. Collect every `mentions` node_id the entity appears in
across the whole document before writing its row, rather than writing one
row per mention.

### 4. Don't force it

If a sentence is vague, hedged, or you can't find a page/quote that
actually backs it, leave it out rather than writing a low-confidence claim
just to hit a quota. Same for relations: only emit a `relations.jsonl` row
when a claim in `claims.jsonl` actually asserts that subject-predicate-object
triple — don't infer relations the text doesn't state. It's fine, and
expected, for `claims.jsonl` to be short (the fixture's own example is five
claims from a two-page document) — a handful of well-evidenced claims beats
twenty shaky ones.

## Test fixture

`tests/fixtures/hydra-settlement-test-fixture.pdf` is the shared regression
fixture for the whole pipeline. The real, verified reference outputs for
this stage are `tests/fixtures/reference-run-hydra-settlement/claims.jsonl`,
`entities.jsonl`, and `relations.jsonl` — produced by chaining all 7 skills
together end-to-end, not a hand-derived approximation. They show what
Claude, following this file, should produce from that document's
`chunks.jsonl`.
