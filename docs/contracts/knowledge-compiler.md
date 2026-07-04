# Knowledge Compiler — Index and Tool Contract

Date: 2026-07-03

## Why a contract doc

The knowledge compiler sits between two independently-evolving halves of
this toolkit: the producers (the web-research and PDF-ingestion skills,
which write files) and the consumer (the `retrieval-planner` skill, whose
eight tools query an index built from those files). Neither half should
have to know how the other works internally — the producers keep writing
their native on-disk formats, and the tools keep returning stable JSON
shapes — but both have to agree on what the index in the middle contains.
This doc is that agreement: the normalized evidence shape, the index
tables, the fusion rule, the tool contracts, and the LLM-backend seam.

One distinction runs through everything below and is worth stating once,
up front: **the on-disk run files are the durable contract; the index is a
rebuildable cache.** The compiler never rewrites a producer's files. It
reads them, normalizes at index time, and writes the result into a
git-ignored index directory (`knowledge_base.index_dir`, default
`.deepresearch/index/`) that can be deleted and rebuilt from the runs at
any moment. If the index and the files ever disagree, the files win — you
recompile.

## `evidence_ref`: one evidence shape for two producers

The PDF pipeline and the web pipeline cite evidence differently on disk,
because their sources are genuinely different: a PDF claim points at a
chunk `node_id` and a page number; a web claim points at a chunk locator
and a URL, and has no pages at all. Rather than force one producer to
mimic the other's file format (which would have meant migrating every
existing `pdf-runs/` directory), the compiler normalizes both into a
single producer-agnostic shape **at index time**, defined in
`src/deep_research_toolkit/compiler/schema.py`:

```python
@dataclass(frozen=True)
class EvidenceRef:
    producer: str          # "pdf" | "web"
    source_id: str         # document_id (pdf) or web source_id
    locator: str | None    # node_id (pdf) or research chunk_id (web)
    quote: str
    page: int | None = None
    url: str | None = None
```

The mapping `normalize_evidence()` applies to each entry of a claim's
`supporting_evidence` list:

| `EvidenceRef` field | PDF run (`pdf-runs/<id>/claims.jsonl`) | Web run (`research-runs/<id>/claims.jsonl`) |
|---|---|---|
| `producer` | `"pdf"` | `"web"` |
| `source_id` | `document_id` from `manifest.json` | the run's `source_id` (directory name) |
| `locator` | the evidence entry's `node_id` | the evidence entry's `locator` (`<source_id>:cNN`) |
| `quote` | `quote`, unchanged | `quote`, unchanged |
| `page` | the evidence entry's `page` | always `null` (web runs have no pages) |
| `url` | always `null` | the evidence entry's `url` |

Existing PDF `claims.jsonl` files are **unchanged on disk** — no field was
renamed, nothing was migrated, and the `claims/entities/relations.jsonl`
schema version stays at `1.0` (see `docs/contracts/schema-versions.md`).
The asymmetry lives in one function, and everything downstream of the
index (search results, dossiers, source lookups) sees only the unified
shape.

## The web research run: `research-runs/<source_id>/`

Web research gained the same claim-extraction step PDF ingestion already
had (this was decision #1 in ADR 0001 — the asymmetry was known and is now
closed). When a fetched source is substantial enough to mine for claims,
`research-knowledge-graph`'s `start_research_run.py` scaffolds a run
directory that deliberately mirrors a PDF run's layout so the compiler can
index both uniformly:

```
research-runs/<source_id>/
├── manifest.json      # schema_version 1.0, producer: "web", document_id,
│                      # source_url, content_hash, fetched_at, chunk_count
├── source.md          # the fetched content, verbatim — the file quotes
│                      # are checked against
├── chunks.jsonl       # one node per heading section, node_id "<source_id>:cNN"
├── claims.jsonl       # written by the agent (or extract_claims.py under
├── entities.jsonl     # llm.provider=local) — same 1.0 schemas as PDF runs,
└── relations.jsonl    # web evidence shape: {locator, quote, url}
```

`source_id` is deterministic: a slug of the URL's host and path plus an
8-hex-character content hash, so re-fetching identical content lands in
the same directory. The extraction rules (one assertion per claim,
verbatim quotes against `source.md`, merged entity mentions, no forced
claims) are in `skills/research-knowledge-graph/references/web-claim-extraction.md`.

## The index: DuckDB tables

`compile_index()` (in `compiler/build.py`) is a **full rebuild** each run:
it deletes the index directory, recreates it, and walks all three
producers — `knowledge_base/` for wiki pages, `pdf-runs/` and
`research-runs/` for claims, entities, and relations. A run directory is
discovered when it contains a `claims.jsonl`. Eight DuckDB tables are
created (DDL in `compiler/schema.py`), all in
`<index_dir>/knowledge.duckdb`:

| Table | Columns | Contents |
|---|---|---|
| `meta` | `key`, `value` | index metadata, including `index_schema_version` |
| `wiki_pages` | `path` (PK), `type`, `title`, `status`, `timestamp`, `body`, `frontmatter_json` | one row per OKF page, path relative to the knowledge base |
| `wiki_links` | `from_path`, `to_path` | the wiki's link graph; links resolving outside the kb are dropped |
| `claims` | `claim_id`, `producer`, `source_id`, `claim`, `claim_type`, `confidence` | every claim from every run, both producers |
| `claim_evidence` | `claim_id`, `producer`, `source_id`, `locator`, `page`, `url`, `quote` | one row per normalized `EvidenceRef` |
| `entities` | `entity_id`, `name`, `type`, `aliases_json`, `producer`, `source_id` | extracted entities |
| `entity_mentions` | `entity_id`, `locator`, `producer`, `source_id` | one row per mention locator |
| `relations` | `relation_id`, `subject`, `predicate`, `object`, `supporting_claim`, `producer`, `source_id` | subject–predicate–object triples; the graph `neighbors` walks |

Two FTS (BM25) indexes are built with DuckDB's `fts` extension: over
`wiki_pages.body` (keyed by `path`) and over `claims.claim` (keyed by
`claim_id`). Graph queries (`neighbors`, wiki-link walks) are recursive
CTEs over `relations` and `wiki_links` — edges treated as undirected, with
`UNION` (not `UNION ALL`) so cycles terminate.

The `index_schema_version` row in `meta` is **internal**: it lets a future
compiler refuse to query an index built by an incompatible earlier build,
but it is not a portable on-disk contract like the `schema_version` fields
on run files — the index is git-ignored and rebuildable, so "migration"
is just recompiling. It does not appear in the schema-version registry's
table for exactly that reason.

## The index: LanceDB tables

Alongside the DuckDB file, `<index_dir>/lancedb/` holds two vector
tables, both with the same three-column shape `{id, text, vector}`:

- **`wiki_chunks`** — one row per wiki page (`id` = kb-relative path,
  `text` = page body).
- **`claim_vectors`** — one row per claim (`id` = `claim_id`, `text` = the
  claim sentence).

Embeddings come from sentence-transformers using the configured
`llm.embedding_model` (default `all-MiniLM-L6-v2`), normalized. Tests
inject a deterministic `FakeEmbedder` instead (16-dim, hash-derived, no
torch) so the full compile-and-query path runs in fast CI; the
`compile.py` shim exposes the same via `DRT_FAKE_EMBEDDER=1` for smoke
tests only — never for a real corpus, since fake vectors carry no
semantic meaning.

## Hybrid search: Reciprocal Rank Fusion, k=60

`search_wiki` and `search_claims` each run two retrievals — DuckDB
FTS/BM25 (lexical) and LanceDB nearest-neighbor (semantic) — and fuse the
two ranked lists with Reciprocal Rank Fusion (`compiler/search.py`):

```
score(d) = Σ over ranked lists containing d of  1 / (k + rank(d))
```

with 1-based ranks and the standard constant **k = 60**. RRF was chosen
over score-based fusion because BM25 scores and cosine distances live on
incomparable scales; rank-based fusion needs no calibration between them
and degrades gracefully when one side returns nothing (if the vector
tables are absent or empty, results are simply the lexical ranking). The
fused list is truncated to the caller's `k` results.

## The 8 retrieval-planner tools

All eight are methods on `deep_research_toolkit.compiler.tools.Index`;
the `retrieval-planner` skill's `scripts/query.py` subcommands are thin
shims that print exactly the method's return value as JSON. **None of
them makes an LLM call** (ADR 0001 decision #3 — tools stay cheap and
deterministic). `Index.open()` raises `FileNotFoundError` if
`<index_dir>/knowledge.duckdb` is missing, pointing you at the
knowledge-compiler skill. The authoritative field-by-field shapes with
examples live in
`skills/retrieval-planner/references/tool-contracts.md`; this table is
the summary:

| Tool | Input arguments | Output shape |
|---|---|---|
| `search_wiki` | `query: str`, `k: int = 8` | list of `{path, title, type, status, snippet}` — hybrid (RRF) hits, best-first; `snippet` is the first 200 chars of the body |
| `read_page` | `path: str` (kb-relative) | `{path, body, frontmatter}`; `{path, error: "not found"}` if missing |
| `search_claims` | `query: str`, `k: int = 8`, `producer: "pdf"\|"web"\|None` | list of `{claim_id, producer, source_id, claim, claim_type, confidence, evidence: [{producer, source_id, locator, page, url, quote}]}` — the producer filter applies after retrieval, so fewer than `k` may return |
| `get_entity` | `name_or_id: str` (entity id, or case-insensitive exact name) | `{entity_id, name, type, aliases, mentions, relations: [{relation_id, subject, predicate, object}]}`; `{query, error: "entity not found"}` if missing |
| `neighbors` | `entity: str` (exact entity id), `depth: int = 1` | list of `{node, depth}` — nodes reachable over undirected relation edges within `depth` hops, start node excluded, each at its minimum distance, ordered by depth then node |
| `get_sources` | exactly one of `page: str` or `claim: str` | with `page`: `{page, source, source_docs, resource}` from frontmatter; with `claim`: `{claim, sources: [{producer, source_id, url}]}` (distinct); neither: `{error: "pass page= or claim="}` |
| `find_contradictions` | none | list mixing `{kind: "relation", subject, predicate, objects, relation_ids, source_ids}` (same subject+predicate, >1 distinct object) and `{kind: "conflicted_page", path}` (`status: conflicted` pages) |
| `compose_dossier` | `query: str \| None`, `claim_ids: list[str] \| None`, `k: int = 12` (query selection via `search_claims` when `claim_ids` is not given) | `{included: [...], rejected: [...]}` — see the gate below |

Two of these deserve more than a table row.

### `compose_dossier` and the verbatim hard gate

A dossier entry reaches `included` only if the claim has at least one
evidence row **and every quote is a verbatim substring of its source
text** — the PDF page's text reassembled from `provenance.jsonl` for
`producer: pdf`, or the run's whole `source.md` for `producer: web`.
Anything else lands in `rejected` with an explicit `reason` (`"no
evidence"`, `"non-verbatim quote(s): [...]"`, or `"claim_id not found in
index"` for unknown ids, which carry only `claim_id` and `reason`). The
check is `dossier.verbatim_ok()` — an exact-substring test with the same
semantics as `rag-eval-harness`'s `check_evidence_quotes_verbatim`, and it
must never be weakened to fuzzy matching: a claim whose quote does not
literally appear in its source looks verified without being verifiable,
which is worse than no claim at all. `rejected` is a signal that an
extraction pass needs fixing, never something to silently trust.

### `find_contradictions` finds candidates, not verdicts

The tool is purely mechanical: a SQL `GROUP BY` finding relation triples
where one subject+predicate maps to more than one distinct object, plus
any wiki page `llm-wiki-writer` already marked `status: conflicted`.
Whether a candidate is a *real* contradiction ("founded in 2015" vs
"founded in 2017") or just under-specified data ("supports X" and
"supports Y" can both be true) takes judgment, and per ADR 0001 decision
#3 that judgment belongs to the agent, in one batched pass over all
candidates — not to a hidden per-candidate LLM call inside the tool.

## LLM backends

The judgment steps in this toolkit — claim extraction, wiki synthesis —
are performed by a model. Which model, and how it's invoked, is the one
pluggable seam, configured by `llm.provider` in `.deepresearch.yml` and
resolved by `deep_research_toolkit.llm.backend.get_backend()`:

- **`agent` (the default, also accepted as `anthropic`).** The in-session
  agent — Claude Code or Codex, reading the relevant SKILL.md — *is* the
  extraction step, exactly as ADR 0001 decision #4 established. There is
  no programmatic model call; `AgentBackend.complete()` raising
  `LLMBackendNotConfigured` is by design, because under this provider a
  script asking a backend to extract claims is a usage error.
- **`local` (opt-in).** An OpenAI-compatible endpoint — Ollama's
  `:11434/v1`, vLLM's `:8000/v1` — serving a local model such as
  `Ornith-1.0-9B`, configured under `llm.local` (`base_url`, `model`,
  `api_key_env`, `temperature`, `top_p`, `top_k`). Responses have any
  `<think>...</think>` reasoning block stripped before parsing.

Under `local`, claim extraction can run programmatically:
`extract_claims_to_run()` (exposed as `scripts/extract_claims.py` in both
`research-knowledge-graph` and the run-based extraction path) builds a
producer-aware prompt from a run's `chunks.jsonl`, parses the model's JSON
claims, and then applies **the same verbatim gate as `compose_dossier`**
before writing anything: every claim whose evidence quotes are not all
verbatim substrings of the run's source text is **auto-dropped**, reported
in the result's `dropped` list rather than written to `claims.jsonl`. The
consequence is the property the whole design leans on: a smaller or
off-label local model can only *under-produce* (fewer claims survive the
gate), never corrupt the corpus with plausible-sounding paraphrases.
`scripts/validate-local-llm.py` is the manual (not-CI) harness for
checking how much of the reference extraction a given local model
recovers.

Embeddings are configured separately (`llm.embedding_model`, default
`all-MiniLM-L6-v2`) and always run locally via sentence-transformers —
they are not routed through the LLM backend.
