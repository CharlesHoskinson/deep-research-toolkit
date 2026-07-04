# ADR 0002: Knowledge Compiler Build Decisions

Date: 2026-07-03
Status: Accepted

## Context

ADR 0001 designed the knowledge-compiler layer (two index engines, eight
cheap deterministic tools, a producer-agnostic `evidence_ref`) but
deferred building it until the extracted web-research and PDF-ingestion
skills were verified standalone. That build has now happened: the
`knowledge-compiler` and `retrieval-planner` skills exist, web research
gained the claim-extraction step it was missing, and the end-to-end
integration test passes against the real embedding model. Building it
surfaced three decisions ADR 0001 had left open or under-specified, plus
two trade-offs accepted deliberately rather than by accident. This ADR
records them. The resulting on-disk and tool contracts are in
`docs/contracts/knowledge-compiler.md`.

## Decisions

1. **The full vector stack is required for real use; tests inject an
   embedder instead of stubbing the engines.** The `[compiler]` extra
   pulls real DuckDB, LanceDB, and sentence-transformers — no degraded
   "FTS-only" production mode, because a hybrid index that silently runs
   at half strength produces worse retrieval while looking fine. What
   keeps CI fast is not mocking the index engines (both run for real in
   every test) but making the embedder injectable: a deterministic,
   dependency-free `FakeEmbedder` (16-dim, hash-derived) replaces
   sentence-transformers in fast tests, so the compile-and-query path is
   exercised end to end without torch or a model download. Exactly one
   test, marked `heavy`, runs the real `all-MiniLM-L6-v2` model. The
   `compile.py` shim exposes the same seam as `DRT_FAKE_EMBEDDER=1` for
   smoke tests only — fake vectors carry no semantic meaning and must
   never back a real corpus.

2. **Web evidence gets its own on-disk shape; unification happens at
   index time, and existing PDF files are untouched.** Web claims cite
   `{locator, quote, url}` (no `page` — web sources have no pages); PDF
   claims keep citing `{node_id, quote, page}` exactly as before. The
   alternative — migrating every existing `pdf-runs/` directory to a
   unified format — would have broken the "files written by an older
   suite version keep reading" promise for zero retrieval benefit, since
   the compiler normalizes both shapes into one `EvidenceRef` in a single
   function (`schema.normalize_evidence`) as it indexes. Web runs live in
   `research-runs/<source_id>/` mirroring a PDF run's layout
   (`manifest.json` with `producer: web`, `source.md`, `chunks.jsonl`,
   then `claims/entities/relations.jsonl` at the same `1.0` schema
   version), so run discovery, extraction rules, and the verbatim check
   work identically for both producers.

3. **An opt-in `local` LLM backend extends ADR 0001 decision #4; the
   in-session agent stays the default.** Decision #4 held that the model
   *is* the extraction step — there is no server to call for "extract
   claims." That remains true by default (`llm.provider: agent`), and the
   default backend deliberately raises if a script asks it to complete a
   prompt. What's new is an escape hatch: `llm.provider: local` points at
   an OpenAI-compatible endpoint (e.g. Ollama serving `Ornith-1.0-9B`)
   and enables programmatic extraction via `extract_claims.py`. The
   reason this doesn't reopen the "hidden LLM call" concern from decision
   #3 is the gate: programmatic extraction applies the same
   verbatim-quote check as `compose_dossier` and auto-drops any claim
   whose evidence is not an exact substring of the source, so a weaker
   local model can only under-produce, never corrupt the corpus.
   Retrieval-planner tools still make no LLM calls under either provider.

## Accepted trade-offs

- **Full rebuild, not incremental compilation.** `compile_index` deletes
  and recreates the index every run. At the corpus sizes this toolkit
  actually serves (a per-project knowledge base on a laptop), a rebuild
  is seconds of DuckDB inserts plus one embedding pass, and it buys total
  freedom from cache-invalidation bugs — there is no state in which the
  index disagrees with the files except "stale, so recompile."
  Incremental compilation (re-embedding only changed pages) is deferred
  until a real corpus makes rebuild time hurt.
- **The index is git-ignored, not committed.** Run directories and the
  knowledge base are the auditable, durable record; the index under
  `.deepresearch/` is a derived cache any checkout can regenerate.
  Committing it would bloat history with binary DuckDB/LanceDB files
  that churn on every compile while adding no auditability. The
  corollary: the DuckDB `index_schema_version` is internal (a
  compatibility check between compiler and index), not a portable
  on-disk contract, and is intentionally absent from the
  `schema-versions.md` registry table.

## Consequences

The lower half of ADR 0001's architecture diagram is now built and
tested, and the suite version moves to 0.2.0 (additive only — every
on-disk schema stays at `1.0`). Still deferred, unchanged from ADR 0001:
the MCP query server, GraphRAG-style community detection, a learned
reranker, the multi-package split, and — added here — incremental
compilation.
