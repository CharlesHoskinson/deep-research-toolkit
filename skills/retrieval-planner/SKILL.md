---
name: retrieval-planner
description: Query the compiled knowledge index to answer research questions from what's already been gathered, before scraping or re-reading anything. Provides 8 tools (search_wiki, read_page, search_claims, get_entity, neighbors, get_sources, find_contradictions, compose_dossier). Use when answering a question the knowledge base may already cover, or when assembling a cited evidence dossier. Requires a built index (run knowledge-compiler first).
---

# Retrieval Planner

Deterministic, LLM-free tools over the index built by `knowledge-compiler`.
Full tool contracts are in `references/tool-contracts.md`. Every command
prints JSON to stdout.

## First

Ensure the index exists (`knowledge-compiler`'s `compile.py`), then read
`.deepresearch.yml` for paths. All commands:

```
python scripts/query.py search-wiki "<query>" [--k N]
python scripts/query.py read-page <kb-relative-path>
python scripts/query.py search-claims "<query>" [--k N] [--producer pdf|web]
python scripts/query.py get-entity <name-or-id>
python scripts/query.py neighbors <entity-id> [--depth D]
python scripts/query.py get-sources (--page P | --claim C)
python scripts/query.py find-contradictions
python scripts/query.py compose-dossier "<query>" [--claims c1,c2] [--k N]
```

## compose-dossier and the verbatim gate

`compose-dossier` returns `{included, rejected}`. A claim reaches `included`
only if **every** supporting quote is a verbatim substring of its source
(PDF page text or web `source.md`). Non-verbatim claims land in `rejected`
with a reason -- never silently included. Trust `included`; treat `rejected`
as a signal that an extraction pass needs fixing.

## find-contradictions is candidates, not verdicts

`find-contradictions` lists *mechanical* candidates (same subject+predicate
with different objects; `status: conflicted` pages). Confirming a candidate
is a real contradiction is your job as the agent: read the two claims and
their quotes via `search-claims`/`compose-dossier` and decide. Do this in a
single batched pass over all candidates, not one model call each.
