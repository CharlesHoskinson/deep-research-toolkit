---
name: llm-wiki-writer
description: Sixth of seven pdf-ingestion pipeline stages -- runs after knowledge-extraction and before rag-eval-harness. Reads a pdf-runs/<document_id> run directory and synthesizes its claims/entities into Open Knowledge Format wiki pages, merging into existing pages and flagging cross-document conflicts (status: conflicted) instead of silently picking a side. Use whenever claims.jsonl/entities.jsonl exist for a run and haven't been written into the knowledge base yet.
---

# LLM Wiki Writer

Pipeline position: `knowledge-extraction` → **`llm-wiki-writer`** →
`rag-eval-harness`. Full format contract:
`../../docs/contracts/pdf-ingestion-pipeline.md`.

PDF-derived wiki pages are **not a separate output format** -- they are
written directly into whatever knowledge base `.deepresearch.yml`'s
`knowledge_base.path` points at, the same one `research-knowledge-graph`
builds from web research. Read `../../docs/contracts/okf-frontmatter.md`
and `research-knowledge-graph`'s
`../research-knowledge-graph/references/wiki-operations.md` before doing
any of this -- this skill only adds two `status` values (`draft`,
`conflicted`) and an optional `source_docs` field on top of that schema, it
does not fork a parallel format.

## Inputs

From `<run_dir>` (a `pdf-runs/<document_id>/` directory, or wherever
`.deepresearch.yml`'s `knowledge_base.pdf_runs_dir` points):
- `claims.jsonl` -- one claim per line, each with `entities`, `relations`,
  and `supporting_evidence` (node_id/quote/page).
- `entities.jsonl` -- one entity per line, with `name`, `aliases`, `type`,
  `mentions`.
- `manifest.json` -- read `document_id` from here; never recompute it.

## Procedure

1. **Read** `claims.jsonl` and `entities.jsonl` from `<run_dir>`.

2. **Cluster entities into page-worthy concepts.** Not every entity needs
   its own page -- group tightly related entities (e.g. a protocol and its
   core mechanism) into one page the way existing knowledge-base pages do
   one concept per file. Skip entities that are only ever incidental
   mentions inside another entity's claims.

3. **For each page-worthy concept, query first** -- exactly like
   `research-knowledge-graph`'s `query` operation: search the knowledge
   base by title, `aliases`/`tags`, and body text for existing coverage
   before writing anything. Grep is enough; there's no separate index to
   consult.

   - **Existing page found** → merge, don't duplicate:
     - Read the page, add a new section (or extend an existing one)
       covering what this document's claims add, citing evidence loosely
       in prose (verbatim quote-checking against `provenance.jsonl` is
       `rag-eval-harness`'s job downstream, not this skill's).
     - In frontmatter: append this run's `document_id` to `source_docs`
       (create the list if absent, dedupe if already present), and bump
       `timestamp` to now. Leave the existing web `source` field alone --
       a page can have both `source` (web) and `source_docs` (PDF runs).
     - If the page was `status: seed` or `stale`, upgrade it to `draft`
       (real content now exists, but only from this one document, so it
       isn't "researched"/cross-checked yet) -- unless a conflict applies
       (see below), in which case use `conflicted` instead.
     - After editing, run `scripts/scaffold_wiki_page.py <run_dir>
       <knowledge_path> --record-updated` to log the touched path into
       `wiki_pages_written.json` -- this script call is required even
       though the script didn't write the content itself, since that file
       is `rag-eval-harness`'s audit trail of every page this run touched.

   - **No existing page** → scaffold a new one with
     `scripts/scaffold_wiki_page.py <run_dir> <knowledge_path> --type ...
     --title ... --status draft --source-docs <document_id> [--tags ...]
     [--description ...] [--body ...]`. New pages from this skill always
     start at `status: draft`, never `seed` (seed is for placeholders with
     no content) or `researched` (that implies cross-checked, multi-source
     confidence this skill hasn't earned yet). Write real body content --
     don't leave a stub; synthesize prose from the claims, and cross-link
     related concepts with real relative links (`[OWS](/standards/ows.md)`
     etc.) exactly as the existing graph does. If this introduces a new
     top-level concept, add it to the knowledge base's `index.md` under the
     right section, per `wiki-operations.md`'s `ingest` step 5.

   Under `provider: local`, `scripts/write_wiki_page.py` drafts a
   citation-gated body with the `wiki_write` model instead of you writing
   the prose by hand; review the draft page like any other before treating
   it as done -- it still needs the same query-first, merge-not-duplicate,
   and conflict checks above.

4. **Conflicts.** If two claims -- from different `document_id`s, whether
   both landing in this run or one already on an existing page -- directly
   disagree (e.g. contradictory throughput numbers, contradictory safety
   assumptions), do not silently prefer one:
   - Keep both claims' content in the page.
   - Set `status: conflicted` in frontmatter.
   - Add an explicit `## Conflicting claims` section describing the
     disagreement in prose and citing both `document_id`s (and, if
     available, the claim text/quote each comes from) so a reader
     immediately sees there's an open contradiction rather than a single
     confident-sounding number.
   - This applies even if only one of the two conflicting claims came from
     a PDF run and the other was already-`researched` web content -- the
     conflict is between claims, not between provenance types.

5. **Record every touched page.** By the end of the run,
   `<run_dir>/wiki_pages_written.json` (maintained automatically by
   `scaffold_wiki_page.py`, in both create and `--record-updated` modes)
   must list every knowledge-base-relative path this run created or edited
   -- nothing touched should be left out, and nothing untouched should be
   in it. `scaffold_wiki_page.py` also updates `manifest.json`'s
   `stages.llm-wiki-writer` key (`completed_at`, `pages_written`) on every
   call, so a partial run still leaves an inspectable manifest, and
   multiple pages written across a run accumulate in the same
   `pages_written` list rather than the last call clobbering earlier ones.

6. **Lint before finishing.** Run
   `research-knowledge-graph`'s `scripts/lint_graph.py` from the project
   root. Fix anything it flags (broken links, orphans, missing frontmatter)
   before considering the run done -- a wiki-writer run that breaks the
   graph is worse than one that writes nothing.

## Script

```
python scripts/scaffold_wiki_page.py <run_dir> <knowledge_path> \
    --type Concept --title "Hydra Head Settlement" --status draft \
    --source-docs hydra-settlement-test-fixture-a1b2c3d4 \
    --tags protocol,settlement,hydra --body "..."

python scripts/scaffold_wiki_page.py <run_dir> <knowledge_path> --record-updated
```

`<knowledge_path>` is relative to the configured knowledge base (resolved
via `.deepresearch.yml`'s `knowledge_base.path`, see
`deep_research_toolkit.config.load_config`) unless you pass an absolute
path.

Refuses to overwrite an existing file without `--force`, same as
`research-knowledge-graph`'s `scaffold_page.py` -- if a page already exists,
edit it by hand and log the edit with `--record-updated` instead of forcing
a fresh scaffold over real content.
