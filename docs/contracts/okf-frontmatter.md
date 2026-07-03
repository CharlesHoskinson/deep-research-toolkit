# OKF Frontmatter Schema

Every file under a project's configured `knowledge_base.path` (see
`.deepresearch.yml`) is a markdown document with YAML frontmatter — the
"Open Knowledge Format" convention this suite writes and reads.

## Required fields
- `type` — one of `Index`, `Tool`, `Standard`, `Pattern`, `Product`,
  `Concept`, `Entity` (extend as needed, keep it a single word/phrase).
- `title` — human-readable name.
- `timestamp` — ISO 8601 UTC, when this file was last substantively updated.

## Optional fields
- `okf_version` — schema version of this page's frontmatter shape (see
  `schema-versions.md`). Absent means "predates versioning" (treat as
  `"1"`) — this lets a migration script identify pre-versioning pages
  written by earlier suite versions.
- `description` — one sentence, shows up in graph traversal/search.
- `resource` — canonical source URL for this concept, if there is one.
- `aliases` — alternate names/spellings for this concept, distinct from
  `tags` (aliases are identity, tags are categorization) — used by the
  knowledge compiler's entity-merge step in a later phase.
- `tags` — short list of free-text tags.
- `source` — id of a row in `<knowledge_base>/sources/index.md` this
  content was ingested from (web-scraped provenance).
- `source_docs` — list of `document_id`s (from `<pdf_runs_dir>/<document_id>/`)
  this page was synthesized from, when written or extended by the
  `llm-wiki-writer` PDF-ingestion stage. A page can have both `source` and
  `source_docs` if it was built from a mix of web research and ingested PDFs.
- `status` — `seed` (placeholder, not yet researched), `researched` (has
  real content), `stale` (flagged by lint as needing a refresh), `draft`
  (written from a single ingested source, not yet cross-checked), or
  `conflicted` (two ingested sources make claims that directly disagree;
  the page keeps both claims and describes the disagreement explicitly in
  a "Conflicting claims" section rather than silently picking a side).

## Body conventions
- Cross-link related concepts with real relative markdown links — these
  are the graph's edges, and the lint tooling checks them. A link starting
  with `/` resolves from the knowledge-base root; a link without a leading
  `/` resolves relative to the current file's directory.
- One concept per file. If a file starts covering more than one concept,
  split it and link between the parts.
- Prefer updating an existing file over creating a near-duplicate — search
  the knowledge base first (the `query` operation in
  `skills/research-knowledge-graph/references/wiki-operations.md`).

## Implementation

`deep_research_toolkit.common.frontmatter` is the single parser/writer
every skill uses — `parse_okf`/`render_okf`/`read_okf`/`write_okf` plus
`validate_frontmatter` (checks required fields and the `status` enum) and
`resolve_link` (implements the `/`-vs-relative link resolution above).
