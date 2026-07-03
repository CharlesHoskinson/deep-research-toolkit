# ADR 0001: Core Architecture

Date: 2026-07-03
Status: Accepted

## Context

This repo extracts and generalizes two Claude Code skill stacks originally
built in a private research repo (`agentictrading`): web research via
Scrapling into an Open Knowledge Format (OKF) wiki, and a 7-stage PDF
ingestion pipeline producing the same kind of OKF pages plus
`claims.jsonl`/`entities.jsonl`/`relations.jsonl` with page-cited,
verbatim-quote-verified evidence (verified end-to-end, pass_rate 1.0 on a
real test fixture). The goal is a public, project-agnostic suite plus a new
"knowledge compiler" layer that turns the accumulated wiki into an
agent-queryable retrieval substrate.

Three independent expert proposals were solicited (AI/knowledge-systems
architecture, principal-engineer repo/packaging, Claude/Codex platform
loading) and converged strongly. This ADR is the synthesis.

## Decisions

1. **Extend the proven claims/entities/relations schema rather than
   redesigning it.** Generalize `node_id` into a producer-agnostic
   `evidence_ref`. Give web research the same claim-extraction step PDF
   ingestion already has (currently missing — this asymmetry is fixed
   here). Both producers write into their own `*-runs/<id>/` directory.

2. **Two physical index engines, not five.** DuckDB (FTS/BM25, claims
   table, wiki-link graph, entity/relation graph via recursive CTEs) +
   LanceDB (vector search, two tables: wiki chunks + claims). Reciprocal
   rank fusion between them. No graph database server. No GraphRAG-style
   community detection — flagged by all three experts as premature at
   per-project/laptop scale.

3. **Tools stay cheap and deterministic; judgment happens at compile
   time.** `search_wiki`, `read_page`, `search_claims`, `get_entity`,
   `neighbors`, `get_sources` are plain lookups. `find_contradictions` is a
   mechanical candidate-detector at compile time plus a batched LLM
   confirmation pass (formalizing what `llm-wiki-writer` already does), not
   a query-time tool. `compose_dossier` is a deterministic formatter/
   validator (reusing the eval-harness's verbatim-quote check as a hard
   gate), never a tool that hides an internal LLM call.

4. **File-based contracts stay; CLI-first; no MCP server in v1.** The two
   genuinely judgment-based pipeline stages (claim extraction, wiki
   synthesis) *are* the agent reasoning over files — there's no server to
   call for "extract claims," the model is the extraction step. An
   optional, read-only MCP query layer over the *finished* knowledge base
   is a reasonable v2 addition once a corpus outgrows file-scan search.

5. **One shared `skills/` tree for Claude Code and Codex.** Thin manifests
   (`.claude-plugin/plugin.json`, `.codex-plugin/plugin.json`) point at the
   same directory, mirroring the real dual-platform pattern already in use
   by the `superpowers` plugin. Codex now has native skill discovery
   (implicit invocation on `description` match, progressive disclosure,
   `.agents/skills` directory-walk-up) — none of the older hook-bootstrap
   machinery some plugins use is needed for ordinary task-triggered skills
   like these.

6. **Per-project parameterization via `.deepresearch.yml`**, discovered by
   walking up from cwd (like `.git`). Configures `knowledge_base` paths,
   `topic.scope_hint`, `features.*` flags, and `llm.api_key_env` (never a
   raw key). This is what removes the original repo's Midnight.city/OWS
   hardcoding from the generalized skills.

7. **`pip install "deep-research-toolkit[web|pdf|compiler|full]"` +
   `drt init`/`drt upgrade`.** Extras-based dependency tiers. `drt upgrade`
   diffs installed-file hashes against what was last installed, skips
   user-modified skill files, and never touches `.deepresearch.yml` or the
   knowledge base itself.

8. **Schema versioning added now**, before there's anything to migrate:
   `schema_version` fields on `manifest.json`, `classification.json`, and
   OKF frontmatter, a `docs/contracts/schema-versions.md` registry, and a
   `drt migrate` stub (detect + report for v1; real field-rewriting
   migrations get built when an actual breaking change happens).

## Explicitly deferred past v1

Multi-package split (independent `core`/`web`/`pdf`/`compiler` PyPI
packages), GraphRAG-style community detection, a 10-file semantic-object
schema (definitions/procedures/constraints/etc. as separate files/indexes),
an MCP query server, a learned reranker, an autonomous error-book
correction loop (the write path + a read-side filter ship now; the
autonomous "detect and re-correct" loop does not), a real-time/watched-
folder compilation daemon, and Codex marketplace listing polish.

## Consequences

The knowledge-compiler layer (index building, retrieval-planner tools) is
designed here but built in a later phase, after the extracted/generalized
web-research and PDF-ingestion skills are verified working standalone in a
fresh consuming project.
