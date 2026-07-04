# Knowledge Compiler + Retrieval Planner + Local LLM Backend — Design

Date: 2026-07-03
Status: Approved (brainstorming)
Supersedes-in-part: the "Designed, not yet built" section of the README and
ADR 0001's deferred knowledge-compiler layer.

## Summary

This is the design for finishing the deep-research-toolkit skill suite. ADR
0001 and the README both name exactly one unbuilt piece: the **knowledge
compiler** (a queryable index over everything the web + PDF pipelines have
produced) and its **retrieval-planner tools**. This spec builds that, plus
two changes the same ADR flagged as belonging to this phase:

1. **Web claim-extraction.** ADR decision #1 records that web research is
   missing the claim-extraction step PDF ingestion already has. This spec
   closes that asymmetry so web sources write
   `claims/entities/relations.jsonl` into `research-runs/<id>/`, and
   generalizes the per-producer evidence pointer into a producer-agnostic
   `evidence_ref` the compiler normalizes to.
2. **An opt-in local LLM backend.** The five generative touchpoints in the
   suite are currently always the in-session agent (ADR decision #4). This
   spec adds an optional `local` provider — an OpenAI-compatible endpoint
   serving `Ornith-1.0-9B` — so extraction/synthesis can run headless,
   offline, and without an API key, guarded by the suite's own verbatim
   gate. The in-session agent stays the default; this is additive.

Suite version bumps `0.1.0 → 0.2.0` (minor: new skills, new additive
fields, new CLI flags — nothing that makes an existing on-disk file fail to
read; see `docs/contracts/schema-versions.md`).

## Goals

- Index `knowledge_base/` (OKF wiki), `pdf-runs/`, and `research-runs/` into
  a hybrid DuckDB + LanceDB store, per ADR decision #2.
- Ship eight cheap, deterministic retrieval tools (ADR decision #3):
  `search_wiki`, `read_page`, `search_claims`, `get_entity`, `neighbors`,
  `get_sources`, `find_contradictions`, `compose_dossier`.
- Give web research a claim-extraction step that produces the same evidence
  artifacts as PDF ingestion, normalized under `evidence_ref`.
- Make the extraction/synthesis steps runnable by an optional local model
  (Ornith-1.0-9B) without weakening the verbatim-quote guarantee.

## Non-goals (explicitly deferred, consistent with ADR 0001)

- Incremental/watched-folder compilation — v1 does a full rebuild.
- GraphRAG-style community detection, a learned reranker, a hosted MCP query
  server, a multi-package PyPI split.
- Replacing the in-session agent as the default reasoning engine. The local
  backend is opt-in only.
- Running the 31B/35B/397B Ornith variants as a supported local path — the
  supported local tier is the 9B (GGUF/Ollama/vLLM). Larger variants may
  work via the same OpenAI-compatible client but are not a design target.

## Architecture overview

```
knowledge_base/  pdf-runs/<id>/  research-runs/<id>/   (three producers)
        \             |               /
         \            |              /
          v           v             v
   +-----------------------------------------+
   | knowledge-compiler skill (compile.py)   |
   |  normalize -> evidence_ref              |
   |  build DuckDB (FTS + graph)             |
   |  build LanceDB (wiki-chunk + claim vecs)|
   +-----------------------------------------+
                     |
                     v
        <index_dir>/knowledge.duckdb
        <index_dir>/lancedb/
                     |
                     v
   +-----------------------------------------+
   | retrieval-planner skill (query.py)      |
   |  8 tools; RRF fuses FTS + vectors       |
   |  compose_dossier runs the verbatim gate |
   +-----------------------------------------+

   deep_research_toolkit.llm  (agent default | local Ornith)
     used only by the optional programmatic extract_claims path
```

## Component 1 — `evidence_ref` normalization

The compiler is the single place PDF- and web-sourced evidence become one
substrate. Rather than migrate the verified PDF `claims.jsonl` (which uses
`node_id` + `page`), the compiler normalizes **at index time** into:

```
evidence_ref = {
  "producer": "pdf" | "web",
  "source_id": "<document_id or web source_id>",
  "locator":   "<node_id or research chunk_id>",
  "page":      <int, PDF only>,
  "url":       "<string, web only>"
}
```

- **PDF** `claims.jsonl` is unchanged on disk. The compiler maps each
  `supporting_evidence[]` entry `{node_id, quote, page}` →
  `{producer: "pdf", source_id: document_id, locator: node_id, page}`. This
  is exactly the wrapping predicted by the note in
  `docs/contracts/pdf-ingestion-pipeline.md` (chunks.jsonl section).
- **Web** `claims.jsonl` (new) is written with web-shaped evidence from the
  start: `{locator, quote, url}` → `{producer: "web", source_id, locator,
  url}`. No `page`.

`evidence_ref` lives in the DuckDB `claim_evidence` table, not in the
on-disk JSONL. On-disk producer files keep their native shapes; the
normalization is an index-build concern only.

## Component 2 — Web claim-extraction (research-knowledge-graph)

A web research run mirrors a PDF run so the compiler treats them uniformly:

```
research-runs/<source_id>/
├── manifest.json      # producer: "web", source_url, fetched_at, content_hash, schema_version
├── source.md          # fetched, cleaned content — the verbatim-quote target
├── chunks.jsonl       # web content chunked (same node shape as PDF chunks.jsonl)
├── claims.jsonl       # same schema as PDF; web-shaped supporting_evidence (locator/url, no page)
├── entities.jsonl     # same schema as PDF
└── relations.jsonl    # same schema as PDF
```

- `source_id` is a filesystem-safe slug of the URL host+path plus a short
  content hash (mirrors `document_id` derivation in `common.hashing`).
- New module `deep_research_toolkit.web.research_run`: `start_research_run()`
  creates the dir, writes `manifest.json` (via `common.manifest`), stores
  `source.md`, and chunks it into `chunks.jsonl` (reusing the PDF chunker's
  structure-aware logic where it applies to markdown headings; a simpler
  heading/paragraph chunk is acceptable for web text).
- New thin shim `research-knowledge-graph/scripts/start_research_run.py`.
- `research-knowledge-graph/SKILL.md` gains a "claim-extraction" section
  mirroring `knowledge-extraction`'s Part 2 rules, with one change: the
  evidence rule checks quotes as verbatim substrings of `source.md`, not of
  a `page` in `provenance.jsonl`. The four load-bearing rules (one assertion
  per claim, verbatim quotes, merge entity mentions, don't force it) carry
  over unchanged.

The existing web fetch/scaffold-page flow is unchanged. Claim extraction is
an added, optional step for sources worth turning into claims — a fetched
page can still just become an OKF wiki page as it does today.

## Component 3 — The index (knowledge-compiler skill)

Two engines, always built together (full vector stack required, per the
chosen approach):

**DuckDB** — `<index_dir>/knowledge.duckdb`, tables:

- `meta(key, value)` — includes an `index_schema_version`.
- `wiki_pages(path, type, title, status, timestamp, body, frontmatter_json)`
- `wiki_links(from_path, to_path)` — the OKF graph edges (resolved via
  `common.frontmatter.resolve_link`).
- `claims(claim_id, producer, source_id, claim, claim_type, confidence)`
- `claim_evidence(claim_id, producer, source_id, locator, page, url, quote)`
- `entities(entity_id, name, type, aliases_json, producer, source_id)`
- `entity_mentions(entity_id, locator, producer, source_id)`
- `relations(relation_id, subject, predicate, object, supporting_claim, producer, source_id)`
- FTS/BM25 indexes over `wiki_pages.body` and `claims.claim` via DuckDB's
  `fts` extension.

**LanceDB** — `<index_dir>/lancedb/`, two tables:

- `wiki_chunks` — one row per wiki page (or per section for long pages):
  `{path, chunk_id, text, vector}`.
- `claim_vectors` — one row per claim: `{claim_id, text, vector}`.

Vectors come from `deep_research_toolkit.compiler.embed` (Component 6).

`compile.py` walks the three producers, normalizes to `evidence_ref`,
populates both engines idempotently (full rebuild: drop + recreate). It
resolves `knowledge_base.path`, `pdf_runs_dir`, `research_runs_dir`, and the
new `knowledge_base.index_dir` from `.deepresearch.yml` via
`config.resolve_path`. Default `index_dir` is `.deepresearch/index/`, which
is already git-ignored (derived artifact, unlike the tracked `*-runs/`).

`knowledge-compiler/scripts/compile.py` is a thin shim over
`deep_research_toolkit.compiler.build.compile_index()`. No `drt compile`
subcommand — compile stays a skill script, honoring the CLI's "stages are
skill scripts, not subcommands" rule.

## Component 4 — Retrieval-planner tools (retrieval-planner skill)

One `query.py` with subcommands, JSON to stdout (agent-consumable), plus an
optional `--format md`. All tools are cheap and deterministic (ADR decision
#3); none makes an LLM call at query time.

| Subcommand | Behavior |
|---|---|
| `search-wiki QUERY [--k N]` | DuckDB FTS + LanceDB vector over wiki, RRF-fused; returns path, title, snippet, score |
| `read-page PATH` | full OKF page: frontmatter + body |
| `search-claims QUERY [--k N] [--producer pdf\|web]` | hybrid over claims; returns claim + evidence_ref + quote + confidence |
| `get-entity NAME_OR_ID` | entity row, aliases, mentions, and its claims/relations |
| `neighbors ENTITY_ID [--depth D]` | recursive-CTE graph walk over `relations` + wiki-link graph, depth-bounded |
| `get-sources (--page P \| --claim C)` | resolve back to source records: OKF `source`/`source_docs`, research-run URLs |
| `find-contradictions` | compile-time mechanical candidate detector (below); prints candidate pairs |
| `compose-dossier QUERY [--claims c1,c2,...]` | deterministic formatter; runs the verbatim gate as a hard filter |

**RRF fusion** (`compiler.search`): given the FTS ranked list and the vector
ranked list, score each item `sum(1 / (k_rrf + rank_i))` with `k_rrf = 60`
(standard), sort descending. Pure function, unit-tested with hand-built
rank lists.

**`find_contradictions`** (`compiler.contradictions`) is mechanical only: it
flags (a) claims sharing an entity + predicate but with conflicting objects,
and (b) OKF pages already marked `status: conflicted`. It emits candidate
pairs. Confirmation (deciding a candidate pair is a *real* contradiction) is
a batched LLM step driven by the retrieval-planner SKILL.md — never a
query-time LLM call, matching ADR decision #3's split.

**`compose_dossier`** (`compiler.dossier`) gathers the relevant claims and
their quotes + citations and emits a structured evidence dossier. It reuses
`pdf.eval`'s verbatim-substring check as a **hard gate**: any claim whose
quote is not a verbatim substring of its source text (PDF page text via
`provenance.jsonl`, or web `source.md`) is dropped and listed in a
`rejected` section — never silently included. This is the single invariant
ADR 0001 requires survive into the compiler phase.

## Component 5 — Package layout

```
src/deep_research_toolkit/compiler/
  __init__.py
  schema.py          # DuckDB DDL, index_schema_version, evidence_ref normalization + dataclasses
  embed.py           # Embedder protocol; SentenceTransformerEmbedder (lazy) + deterministic fake for tests
  ingest.py          # walk the 3 producers -> normalized rows
  build.py           # compile_index(): orchestrate DuckDB + LanceDB population
  search.py          # FTS query, vector query, RRF fusion
  graph.py           # recursive-CTE neighbors, wiki-link graph walk
  tools.py           # the 8 tool functions (pure; return dataclasses/dicts)
  contradictions.py  # mechanical candidate detector
  dossier.py         # compose_dossier + verbatim gate (reuses pdf.eval)

src/deep_research_toolkit/llm/
  __init__.py
  backend.py         # Backend protocol; get_backend(config) factory
  agent.py           # AgentBackend: no programmatic call; raise-with-guidance if invoked
  local.py           # LocalOpenAIBackend: OpenAI-compatible client, <think> stripping (lazy import)

src/deep_research_toolkit/web/
  research_run.py    # start_research_run(): dir + manifest + source.md + chunks.jsonl
```

## Component 6 — Embeddings and the fast-CI strategy

`compiler.embed` exposes an `Embedder` protocol
(`embed(texts: list[str]) -> list[list[float]]`) with two implementations:

- `SentenceTransformerEmbedder` — the real one; lazy-imports
  `sentence_transformers`, default model `all-MiniLM-L6-v2` (configurable via
  `compiler.embedding_model`), raising the standard specific "install
  `deep-research-toolkit[compiler]`" error if absent.
- A deterministic fake (in tests) — hashes each text to a small fixed-dim
  vector, no model download, no torch.

Runtime always requires the full vector stack (the product does not have a
lexical-only mode). The fake is a **test double**, not a product fallback.

CI strategy:

- **Fast tier** installs `duckdb` + `lancedb` (pip wheels, no model
  download) alongside the existing `[dev,pdf]`. Injecting the fake embedder,
  fast tests cover: `evidence_ref` normalization, DuckDB schema build + FTS,
  RRF math, recursive-CTE neighbors, contradiction candidate detection,
  `compose_dossier`'s verbatim gate, the LLM backend plumbing (via a
  monkeypatched fake OpenAI response), and web `research_run` scaffolding.
  This exercises **both** index engines end-to-end without torch.
- **Heavy tier** installs the full `[compiler]` extra and runs one
  integration test: real `all-MiniLM-L6-v2` embeddings + LanceDB + RRF over
  the fixtures, asserting each of the 8 tools returns sensible results and
  that `compose_dossier` drops a deliberately-paraphrased claim.

New fixtures: a small OKF `knowledge_base/` and a synthetic web
`research-runs/<id>/` alongside the existing hydra PDF reference run, so the
compiler indexes all three producers in tests.

## Component 7 — LLM backend + local Ornith

`deep_research_toolkit.llm` abstracts the generative touchpoints behind:

```
class Backend(Protocol):
    def complete(self, system: str, user: str, **sampling) -> str: ...
```

`get_backend(config)` selects by `.deepresearch.yml` `llm.provider`:

- **`agent`** (default; `anthropic` is an accepted alias for backward-compat)
  → `AgentBackend`, which does **not** make a programmatic call. The
  judgment steps stay agent-driven via SKILL.md (ADR decision #4 preserved).
  Calling `.complete()` on it raises a clear error: use the in-session agent
  flow, or set `provider: local`.
- **`local`** → `LocalOpenAIBackend`: an OpenAI-compatible client
  (`base_url` from config; default Ollama `http://localhost:11434/v1`, vLLM
  `http://localhost:8000/v1` documented), model `Ornith-1.0-9B`,
  `api_key_env` (a dummy is fine for local servers). It strips
  `<think>…</think>` before returning and applies the repo's recommended
  sampling defaults (`temperature=0.6, top_p=0.95, top_k=20`). Lazy client
  import with the standard specific-error pattern.

Config `llm` block extension (all optional, back-compatible):

```yaml
llm:
  provider: agent            # agent | anthropic (alias) | local
  model: claude-sonnet-4-5   # reference only for the agent path
  local:
    base_url: http://localhost:11434/v1
    model: Ornith-1.0-9B
    api_key_env: OPENAI_API_KEY
    temperature: 0.6
    top_p: 0.95
    top_k: 20
  embedding_model: all-MiniLM-L6-v2   # used by the compiler, provider-independent
```

**Optional programmatic extraction path** (active only under `local`): new
`extract_claims.py` shims in `knowledge-extraction` and
`research-knowledge-graph` read `chunks.jsonl`, call the backend with the
claim-extraction rules (prompt text in a reference file that mirrors the
existing `claim-extraction-guide.md`), write the three JSONL files, then
**auto-run the verbatim gate and drop any claim whose quote isn't a verbatim
substring** of its source. Result: an off-label coding model can only
under-produce, never corrupt the corpus. Under `agent` the script errors
with guidance rather than pretending to run.

**Ornith validation harness** — `scripts/validate-local-llm.py`: against an
already-running local endpoint, extract claims from the hydra fixture's
`chunks.jsonl`, diff against the verified
`tests/fixtures/reference-run-hydra-settlement/claims.jsonl`, run
`rag-eval-harness`, and report claim recovery + verbatim-pass rate. It needs
a live model, so it is a manual/heavy tool, not fast CI. The backend's own
plumbing (think-stripping, JSON parsing, auto-drop) is fast-CI-tested with a
monkeypatched fake endpoint.

## Data flow (end to end)

1. Producers write `knowledge_base/`, `pdf-runs/<id>/`, `research-runs/<id>/`
   (web claim-extraction is the new producer path).
2. `compile.py` normalizes to `evidence_ref` and builds DuckDB + LanceDB
   under `index_dir`.
3. `query.py` answers retrieval requests, RRF-fusing lexical + vector, and
   `compose_dossier` emits only verbatim-verified evidence.
4. Optionally, `extract_claims.py --backend local` automates step 1's
   judgment work with a local Ornith, still gated by the verbatim check.

## Error handling

- Missing heavy deps (`duckdb`, `lancedb`, `sentence_transformers`,
  local-LLM client) → the established specific-error pattern naming the exact
  extra, verified by a `test_dependency_boundary`-style test.
- `compile.py` on an empty/partial corpus → builds whatever exists; a
  producer directory that's absent is skipped, not an error (mirrors
  `rag-eval-harness`'s skip-don't-fail posture).
- `query.py` against a missing/stale index → clear "run compile first" error.
- `compose_dossier` with zero verbatim-passing claims → returns an empty
  dossier with the `rejected` list populated, never a fabricated one.

## Docs & versioning changes

- New `docs/contracts/knowledge-compiler.md`: index schema, `evidence_ref`,
  the 8 tool contracts, RRF definition, and an "LLM backends" section.
- **ADR 0002**: records (a) full-vector-stack + injectable-test-embedder,
  (b) web `evidence_ref` shape, (c) the opt-in `local` extension of ADR 0001
  decision #4.
- `docs/contracts/schema-versions.md`: add a `0.2.0` row (research-run
  claims reuse claims `1.0`; note `index_schema_version` is internal to the
  DuckDB `meta` table, not a portable on-disk contract).
- Bump `pyproject.toml`, `__init__.py`, both `plugin.json`s to `0.2.0`;
  update `CHANGELOG.md`.
- README: move the compiler from "designed, not yet built" to "built";
  redraw the diagram's dashed lower half as solid; document the two new
  skills, the web claim-extraction step, and the optional local backend.
- `drt doctor`: add a `sentence-transformers` check and an optional
  local-endpoint ping under `--warm`.
- `sync-skill-templates.py` run after adding the two skill dirs; the
  manifest/sync CI guards must stay green.

## Test plan (summary)

Fast tier (no torch, no network, no model): evidence_ref normalization;
DuckDB build + FTS; RRF; neighbors CTE; contradiction candidates; dossier
verbatim gate (verbatim passes, paraphrase dropped); LLM backend plumbing
via fake endpoint; web research_run scaffolding; dependency-boundary errors
for the new extras. Heavy tier: full-stack compile + all 8 tools over the
fixtures with the real embedding model. Manual: `validate-local-llm.py`
against a running Ornith endpoint.

## Open risks accepted

- Ornith is a coding model used off-label for extraction; the verbatim gate
  bounds the downside to under-production, and the validation harness
  measures it against ground truth before any real use.
- Full-rebuild compile is O(corpus) each run; acceptable at the
  per-project/laptop scale this toolkit targets. Incremental is a later
  phase.
