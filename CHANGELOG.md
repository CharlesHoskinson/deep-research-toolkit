# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).
Versioning: one semver number for the whole suite (see `docs/contracts/schema-versions.md`
for how suite versions map to on-disk schema versions).

## [Unreleased]

### 0.2.0 — the knowledge compiler

#### Added
- `knowledge-compiler` skill: compiles `knowledge_base/`, `pdf-runs/`, and
  `research-runs/` into a hybrid index — DuckDB (FTS/BM25, claims,
  wiki-link and entity/relation graph) plus LanceDB (vectors over wiki
  pages and claims), fused with Reciprocal Rank Fusion (k=60). Full
  rebuild per run; the index is a git-ignored, rebuildable cache
  (`.deepresearch/index/` by default). See
  `docs/contracts/knowledge-compiler.md` and
  `docs/decisions/0002-knowledge-compiler.md`.
- `retrieval-planner` skill: 8 deterministic, LLM-free query tools over
  the compiled index (`search_wiki`, `read_page`, `search_claims`,
  `get_entity`, `neighbors`, `get_sources`, `find_contradictions`,
  `compose_dossier`). `compose_dossier` enforces the verbatim-quote hard
  gate (`included`/`rejected`); `find_contradictions` reports mechanical
  candidates only — confirmation stays agent-driven.
- Web claim extraction in `research-knowledge-graph`:
  `start_research_run.py` scaffolds `research-runs/<source_id>/`
  (`manifest.json` with `producer: web`, `source.md`, `chunks.jsonl`),
  and the agent (or the local backend) writes
  `claims/entities/relations.jsonl` at the same `1.0` schemas the PDF
  pipeline produces. Producer-agnostic `evidence_ref` normalization
  happens at index time; existing PDF `claims.jsonl` files are unchanged
  on disk.
- Pluggable LLM backend (`llm.provider` in `.deepresearch.yml`): `agent`
  stays the default (the in-session agent does extraction/synthesis, per
  ADR 0001 decision #4); opt-in `local` targets an OpenAI-compatible
  endpoint (e.g. Ollama serving `Ornith-1.0-9B`) and enables programmatic
  `extract_claims.py`, which auto-drops any claim whose quotes are not
  verbatim substrings of the source.
- New `[compiler]` extra: `duckdb`, `lancedb`, `sentence-transformers`
  (embeddings default `all-MiniLM-L6-v2`), and `openai` (client for the
  local backend).

### 0.1.0 — initial extraction

#### Added
- Initial extraction and generalization of the web-research and PDF-ingestion
  skill stacks from the private `agentictrading` research repo.
- `.deepresearch.yml` project-level configuration.
- `drt` CLI (`init`, `upgrade`, `doctor`, `migrate`).
- Dual Claude Code / Codex plugin manifests over one shared `skills/` tree.
- `schema_version` fields on `manifest.json`, `classification.json`, and OKF
  frontmatter (new -- the original repo had none).
