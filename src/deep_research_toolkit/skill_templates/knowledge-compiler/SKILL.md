---
name: knowledge-compiler
description: Build or refresh the queryable knowledge index (DuckDB FTS + graph, LanceDB vectors) over everything the web and PDF pipelines have produced. Use after ingesting new sources, or before a research session that will query the accumulated knowledge base with the retrieval-planner skill. Requires the compiler extra.
---

# Knowledge Compiler

Compiles `knowledge_base/`, `pdf-runs/`, and `research-runs/` into a hybrid
index the `retrieval-planner` skill queries. See
`docs/contracts/knowledge-compiler.md` for the index schema and
`docs/decisions/0002-knowledge-compiler.md` for the design.

## First: configuration and dependencies

Read `.deepresearch.yml` (walk up from cwd, like `.git`). The index lives at
`knowledge_base.index_dir` (default `.deepresearch/index/`). Install the
extra once: `pip install "deep-research-toolkit[compiler]"`.

## Build the index

```
python scripts/compile.py [--index-dir DIR]
```

Full rebuild each run (idempotent). It walks all three producers, normalizes
their evidence into a producer-agnostic `evidence_ref`, builds the DuckDB
FTS + graph tables, and embeds wiki pages and claims into LanceDB with the
configured `embedding_model` (`all-MiniLM-L6-v2` by default). Prints row
counts on success.

The first run downloads the sentence-transformers model (a one-time,
offline-after cost, like Docling's models). Everything after is local.
