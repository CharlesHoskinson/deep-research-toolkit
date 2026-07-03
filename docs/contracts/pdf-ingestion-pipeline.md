# PDF Ingestion Pipeline — Shared Data Contract

Date: 2026-07-03 (generalized from the private `agentictrading` prototype)

## Why a contract doc

The pdf-ingestion work is seven small skills, not one big one, because that's
how Claude Skills are meant to compose (progressive disclosure: each
SKILL.md stays small, only the relevant stage loads into context). Seven
independently-built skills only work together if they agree on file formats
up front. This doc is that agreement.

## What changed from the original prototype

This contract was proven end-to-end in a private repo against one
hardcoded project (`knowledge/`, `pdf-runs/` at a fixed path, one research
topic). Generalizing it for public, multi-project use changed exactly
three things — everything else below is unchanged and already verified:

1. **Paths are config-driven, not hardcoded.** `--runs-dir` and
   `--knowledge-base-dir` flags now resolve via
   `deep_research_toolkit.config.resolve_path()`: an explicit CLI flag wins,
   else `.deepresearch.yml`'s `knowledge_base.pdf_runs_dir`/`.path`, else a
   hardcoded fallback (`pdf-runs/` / `knowledge_base/`) for zero-config
   quick starts. No skill script assumes a specific project's directory
   layout anymore.
2. **Every artifact gets a `schema_version` field.** The original had none
   — see `docs/contracts/schema-versions.md` for the registry. This is
   additive; nothing below changes shape, it just gains a version tag.
3. **Script logic moved into the installable package.** Each
   `skills/<name>/scripts/*.py` is now a thin CLI shim (argparse + a call
   into `deep_research_toolkit.pdf.*`) rather than containing the logic
   directly, so the same logic is unit-testable without going through a
   subprocess and is shared with anything else that wants to import it
   (e.g. the knowledge compiler, in a later phase).

## Pipeline

```
PDF
  → pdf-ingest-router          (classify + pick a backend)
  → pdf-to-canonical-markdown  (Docling primary → canonical.md + raw doc JSON)
  → pdf-layout-provenance      (raw doc JSON → provenance-enriched units)
  → canonical-markdown-to-llm-nodes  (units → chunked, typed nodes)
  → knowledge-extraction       (nodes → claims/entities/relations + tables/figures)
  → llm-wiki-writer            (claims/entities → OKF wiki pages in knowledge_base/)
  → rag-eval-harness           (everything → eval_report)
```

## CLI convention — every stage takes a run directory

`document_id` is computed **once**, by `pdf-ingest-router`, and every later
stage reads it back out of `manifest.json` rather than recomputing it:

```
python scripts/classify_pdf.py <pdf_path> [--runs-dir <dir>]
    # --runs-dir omitted -> resolved from .deepresearch.yml, else "pdf-runs/"
    # creates <runs-dir>/<document_id>/, writes classification.json + starts manifest.json
    # prints the run dir path

python scripts/convert.py <run_dir>
python scripts/extract_provenance.py <run_dir>
python scripts/chunk_nodes.py <run_dir>
python scripts/extract_tables.py <run_dir>
python scripts/extract_figures.py <run_dir>
# claims.jsonl / entities.jsonl / relations.jsonl are written directly by
# Claude following knowledge-extraction's SKILL.md, not by a script
python scripts/scaffold_wiki_page.py <run_dir> <knowledge_base_relative_path> --type ... --title ... --status draft
    # writes into .deepresearch.yml's knowledge_base.path, not a hardcoded "knowledge/"
python scripts/run_eval.py <run_dir>
```

Every script after the first takes the run directory as its one required
argument and appends its own key to `manifest.json` on success (see
`common.manifest.update_stage` — this must never clobber another stage's
entry; that was a real bug found and fixed in the original prototype and is
now covered by a regression test in `tests/unit/`).

## Per-document working directory

`<pdf_runs_dir>/<document_id>/`, where `document_id` is a filesystem-safe
slug of the source filename plus a short hash of its content
(`hydra-settlement-test-fixture-4edb3c3c`). Tracked in git in a consuming
project — this is the durable, auditable corpus, not throwaway state. Raw
source PDFs are NOT copied in — only their path and a content hash.

```
<pdf_runs_dir>/<document_id>/
├── manifest.json          # every stage appends its own key; schema_version at top level
├── classification.json    # pdf-ingest-router
├── docling_raw.json       # pdf-to-canonical-markdown (Docling's own structured export)
├── canonical.md           # pdf-to-canonical-markdown
├── provenance.jsonl       # pdf-layout-provenance
├── chunks.jsonl           # canonical-markdown-to-llm-nodes
├── claims.jsonl           # knowledge-extraction
├── entities.jsonl         # knowledge-extraction
├── relations.jsonl        # knowledge-extraction
├── tables/*.csv           # knowledge-extraction
├── figures/*.png + figures/captions.jsonl   # knowledge-extraction
├── wiki_pages_written.json  # llm-wiki-writer: which knowledge_base/ paths it touched
├── eval_report.md
└── eval_report.json
```

## manifest.json

```json
{
  "schema_version": "1.0",
  "document_id": "hydra-settlement-test-fixture-4edb3c3c",
  "source_file": "/absolute/path/to/source.pdf",
  "source_hash": "sha256:...",
  "created_at": "2026-07-03T00:00:00Z",
  "stages": {
    "pdf-ingest-router": {"completed_at": "...", "route": "digital-text"},
    "pdf-to-canonical-markdown": {"completed_at": "...", "parser": "docling", "parser_version": "2.109.0"},
    "pdf-layout-provenance": {"completed_at": "...", "unit_count": 14},
    "canonical-markdown-to-llm-nodes": {"completed_at": "...", "node_count": 9},
    "knowledge-extraction": {"completed_at": "...", "claim_count": 5, "entity_count": 6, "relation_count": 4, "table_count": 1, "figure_count": 1},
    "llm-wiki-writer": {"completed_at": "...", "pages_written": ["concepts/hydra-settlement.md"]},
    "rag-eval-harness": {"completed_at": "...", "pass_rate": 0.9}
  }
}
```

`source_file` is host-machine-absolute and **not portable across
machines/teams** — a known, deliberately-deferred limitation (see
`docs/decisions/0001-architecture.md`'s "deferred" list). Don't build
tooling that assumes two machines' `manifest.json` files reference the
same `source_file` path.

## classification.json (pdf-ingest-router)

```json
{
  "schema_version": "1.0",
  "document_id": "...",
  "source_file": "...",
  "page_count": 3,
  "route": "digital-text | scanned | scientific-math | form | financial-legal | slide-like",
  "signals": {
    "avg_extractable_chars_per_page": 812.3,
    "has_acroform_fields": false,
    "image_only_page_ratio": 0.0,
    "table_like_page_ratio": 0.33,
    "detected_math_density": "low"
  },
  "recommended_backend": "docling",
  "notes": "free-text explanation of the routing decision"
}
```

Docling is the only backend actually wired up. Marker/MinerU/Unstructured
are documented fallback options in `pdf-to-canonical-markdown`'s
references, not implemented, until a real scanned or math-heavy document
forces the issue.

## provenance.jsonl (pdf-layout-provenance) — one line per structural unit

```json
{
  "schema_version": "1.0",
  "unit_id": "u0007",
  "document_id": "...",
  "source_file": "...",
  "page": 2,
  "section_path": ["2. Architecture", "2.2 Settlement Guarantees"],
  "bbox": {"l": 78.0, "t": 490.05, "r": 199.39, "b": 477.10, "coord_origin": "BOTTOMLEFT"},
  "content_hash": "sha256:...",
  "extraction_method": "docling",
  "confidence": 0.93,
  "unit_type": "heading | paragraph | table | figure | caption | list_item",
  "text": "..."
}
```

`bbox` is Docling's own native bbox dict, passed through as-is. `unit_type`
is derived from Docling's `label` field; `section_path` is built by
walking prior heading items in document order.

## chunks.jsonl (canonical-markdown-to-llm-nodes) — one line per retrieval node

```json
{
  "schema_version": "1.0",
  "node_id": "<document_id>:n004",
  "document_id": "...",
  "type": "section | paragraph | table | figure | definition | theorem | claim | example",
  "title": "2.2 Settlement Guarantees",
  "section_path": ["2. Architecture", "2.2 Settlement Guarantees"],
  "text": "...",
  "summary": "",
  "keywords": [],
  "entities": [],
  "source": {"page_start": 2, "page_end": 2, "unit_ids": ["u0007"]},
  "links": {"previous": "...:n003", "next": "...:n005", "parent": null},
  "content_hash": "sha256:..."
}
```

`summary`/`keywords`/`entities` are intentionally empty here — filling
them in with real judgment is `knowledge-extraction`'s job, run after this
stage. Chunking follows document structure (one node per heading section,
tables/figures as their own nodes), never a fixed token-count split.

**Note for the knowledge-compiler phase (not yet built):** `node_id` here
is PDF-pipeline-specific. When the compiler layer generalizes across web-
and PDF-sourced evidence, it wraps this into a producer-agnostic
`evidence_ref: {producer: "pdf", source_id: document_id, locator: node_id}`
rather than changing this field — existing `pdf-runs/` data stays valid
under the pipeline as documented here.

## claims.jsonl / entities.jsonl / relations.jsonl (knowledge-extraction)

Not produced by a deterministic script — extracting a genuine claim, or
deciding two mentions are the same entity, is a judgment call. The skill's
scripts handle the mechanical parts (tables → CSV, figures → PNG +
caption); its SKILL.md instructs Claude to read `chunks.jsonl` and write
these three files directly.

```json
// claims.jsonl
{
  "schema_version": "1.0",
  "claim_id": "c_0001",
  "document_id": "...",
  "claim": "Hydra can be used as a synchronous settlement layer over Cardano-style eUTXO state.",
  "claim_type": "architectural | empirical | definitional | comparative",
  "confidence": "high | medium | low",
  "supporting_evidence": [{"node_id": "...:n004", "quote": "...", "page": 2}],
  "entities": ["hydra", "cardano", "eutxo"],
  "relations": [{"subject": "hydra", "predicate": "serves_as", "object": "synchronous settlement layer"}]
}

// entities.jsonl
{"schema_version": "1.0", "entity_id": "hydra", "name": "Hydra", "aliases": ["Hydra Head"], "type": "protocol", "mentions": ["...:n002", "...:n004"]}

// relations.jsonl
{"schema_version": "1.0", "relation_id": "r_0001", "subject": "hydra", "predicate": "serves_as", "object": "synchronous settlement layer", "supporting_claim": "c_0001", "document_id": "..."}
```

**Evidence rule (checked mechanically by rag-eval-harness):** every
`supporting_evidence[].quote` must be a verbatim substring of the text on
the cited `page` in `provenance.jsonl` — no paraphrase-as-quote. This is
the single most valuable invariant in the whole pipeline; it's what keeps
claims auditable instead of plausible-sounding, and it must survive into
the knowledge-compiler phase as a hard gate on `compose_dossier`, not a
suggestion.

## tables/ and figures/ (knowledge-extraction, deterministic)

Unchanged from the original prototype — see the schema examples above.
`figures/captions.jsonl` always records a figure reference even when no
image could be extracted (`extracted: false`), never silently drops it.

## Wiki pages (llm-wiki-writer) — written into the configured knowledge base

PDF-derived wiki pages are OKF documents written into whatever directory
`.deepresearch.yml`'s `knowledge_base.path` points at — the same graph
`research-knowledge-graph` maintains from web sources, so both producers
feed one graph, lintable by the same lint logic. See
`docs/contracts/okf-frontmatter.md` for the full frontmatter schema,
including the `draft`/`conflicted` status values and `source_docs` field
this stage relies on.

`wiki_pages_written.json` records exactly which knowledge-base-relative
paths this run touched.

## eval_report.json (rag-eval-harness)

```json
{
  "schema_version": "1.0",
  "document_id": "...",
  "generated_at": "...",
  "checks": [
    {"name": "headings_recovered", "passed": true, "detail": "..."},
    {"name": "tables_present", "passed": true, "detail": "..."},
    {"name": "page_citations_valid", "passed": true, "detail": "..."},
    {"name": "evidence_quotes_verbatim", "passed": true, "detail": "..."},
    {"name": "figures_accounted_for", "passed": true, "detail": "..."},
    {"name": "no_ocr_garbage", "passed": true, "detail": "..."}
  ],
  "pass_rate": 1.0
}
```

Q/A retrieval probes are documented as an optional manual step for
high-stakes documents, not part of the automated pass rate.

## What ships in this phase

All seven skill directories, real working logic for everything
deterministic, and SKILL.md instructions for the two judgment-based stages.
Verified end-to-end against `tests/fixtures/` before considered done — see
`tests/integration/` for the reproduction of the original prototype's
pass_rate-1.0 run under the new config-driven paths.
