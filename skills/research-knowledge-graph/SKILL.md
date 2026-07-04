---
name: research-knowledge-graph
description: Use when researching a topic for this project (check .deepresearch.yml's topic.scope_hint for what's in-scope) or any time new material should be added to the project's knowledge base. Fetches sources with Scrapling, stores findings as Open Knowledge Format documents, and maintains the graph via ingest/query/lint so research compounds instead of being re-derived each time.
---

# Research Knowledge Graph

Builds and maintains this project's OKF-format knowledge base by combining
[Scrapling](https://github.com/d4vinci/Scrapling) for retrieval, the
[Open Knowledge Format](../../docs/contracts/okf-frontmatter.md) for
storage, and [Karpathy's LLM-wiki pattern](references/wiki-operations.md)
for the maintenance loop.

## First: find this project's configuration

Before doing anything else, read `.deepresearch.yml` (walk up from the
current directory if it's not in the cwd — same discovery as `.git`). It
tells you:
- `topic.scope_hint` — what this project's research is actually about;
  don't guess a topic from this skill's own generic description.
- `knowledge_base.path` — where the knowledge base lives (all the
  commands below default to this via `deep_research_toolkit.config`, but
  you can pass `--knowledge-base-dir` to override).

If there's no `.deepresearch.yml` yet, tell the user to run `drt init`
before proceeding — don't invent a scope or a directory to write into.

## Quick start

For any research request, run the appropriate operation from
`references/wiki-operations.md`:

- **New topic to research** → `ingest`: query the knowledge base first,
  fetch only what's missing (`scripts/fetch.py`), then write/update OKF
  docs (`scripts/scaffold_page.py`).
- **Question about something already researched** → `query`: search the
  knowledge base and traverse its links before fetching anything new.
- **Housekeeping / before starting a new research batch** → `lint`: run
  `scripts/lint_graph.py` and fix anything it flags.

Full operational detail (including the merge-not-append rule for ingest)
is in `references/wiki-operations.md` — read it before doing an ingest,
it's short.

## Fetching sources

```
python scripts/fetch.py <url> [--mode http|stealth] [--css SELECTOR] [--out PATH]
```

Requires the `web` extra: `pip install "deep-research-toolkit[web]" && scrapling install`.
Defaults to plain HTTP; use `--mode stealth` only when a plain fetch gets
blocked (anti-bot challenge, 403) — stealth mode gets past things a plain
web-fetch tool cannot.

## Turning a source into claims

When a fetched source is substantial enough to mine for claims (not just
a fact to merge into a wiki page), scaffold a web research run from the
cleaned content:

```
python scripts/start_research_run.py <url> --content-file PATH [--research-runs-dir DIR]
```

This writes `research-runs/<source_id>/` containing `source.md` (the
content as fetched), `chunks.jsonl` (one node per heading section), and a
`manifest.json` with `producer: web`. The runs dir defaults from
`.deepresearch.yml` via `deep_research_toolkit.config`.

Then do the extraction yourself: read the run's `chunks.jsonl` and write
`claims.jsonl`, `entities.jsonl`, and `relations.jsonl` into the same run
directory, following `references/web-claim-extraction.md` — it gives the
exact schemas (web evidence is `{"locator": "<source_id>:cNN", "quote":
"...", "url": "..."}`) and the rules that matter, chiefly that every
quote must be a verbatim substring of `source.md`. Don't script this
part; deciding what counts as an atomic, well-evidenced claim is a
judgment call.

These runs are indexed by the `knowledge-compiler` skill alongside PDF
runs, so web-sourced and PDF-sourced claims about the same entity end up
in one queryable graph.

## Writing to the knowledge base

Every page follows the frontmatter schema in
`../../docs/contracts/okf-frontmatter.md` (required: `type`, `title`,
`timestamp`). Create a new page:

```
python scripts/scaffold_page.py standards/ows.md \
  --type Standard --title "OWS" --resource "<source url>" \
  --tags standard,delegated-trading --source src-0004 --status researched
```

The path is relative to the configured knowledge base unless you pass an
absolute path. If the concept already has a page, edit it directly instead
— merge new findings into the existing body and bump its `timestamp`,
don't create a second file for the same concept.

## Checking knowledge-base health

```
python scripts/lint_graph.py [--knowledge-base-dir PATH] [--stale-days 180]
```

Reports orphaned pages, invalid/missing frontmatter, broken relative links,
and stale `researched` pages past the threshold. Fix findings before
adding more content on top.
