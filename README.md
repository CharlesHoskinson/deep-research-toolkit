# deep-research-toolkit

A deep-research skill suite for **Claude Code** and **Codex**: web
retrieval, PDF ingestion, and knowledge compilation into an evidence-backed,
agent-queryable knowledge base — built so an LLM can search, traverse,
verify, and cite instead of re-deriving knowledge from raw sources every
time.

```
Scrapling / PDF ingestion → canonical wiki (OKF) → claims/entities/relations
  → knowledge compiler (incremental) → hybrid index (DuckDB + LanceDB)
  → retrieval-planner tools → evidence dossier
```

## Status

Extracted and generalized from a private research pipeline that was built
and verified end-to-end (7-stage PDF ingestion, real Docling conversion,
pass_rate 1.0 against a test fixture). This repo is the public,
project-agnostic version. The knowledge-compiler layer (indexing +
retrieval-planner tools) is designed but not yet built — see
`docs/decisions/` for the architecture writeups this repo is being built
from.

## Install

```
pip install "deep-research-toolkit[pdf]"   # or [web] / [compiler] / [full]
drt init                                    # scaffold .deepresearch.yml + knowledge_base/ in your project
```

See `docs/environment.md` for setup details, and `CONTRIBUTING.md` if
you're developing this repo itself rather than consuming it.

## Skills

- `research-knowledge-graph` — web research via Scrapling, written into an
  Open Knowledge Format (OKF) knowledge base.
- `pdf-ingest-router` → `pdf-to-canonical-markdown` → `pdf-layout-provenance`
  → `canonical-markdown-to-llm-nodes` → `knowledge-extraction` →
  `llm-wiki-writer` → `rag-eval-harness` — a 7-stage pipeline turning PDFs
  into the same knowledge base, with page-cited, verbatim-quote-verified
  claims.

Each project configures its own scope via `.deepresearch.yml` (see
`docs/contracts/` for the full schema) — nothing in the skills themselves
is hardcoded to a particular research topic.

## License

MIT — see `LICENSE`.
