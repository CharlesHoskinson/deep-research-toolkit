# Wiki Operations: ingest / query / lint

Adapted from Karpathy's LLM-wiki pattern
(https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) for
this project's knowledge base (path configured in `.deepresearch.yml`,
`knowledge_base.path`).

## ingest(topic_or_urls)

1. **Query first.** Before fetching anything new, search the knowledge
   base for existing coverage (grep titles/tags/resource fields). If it's
   already covered and fresh (see staleness rule below), skip straight to
   using it.
2. **Fetch.** Use `scripts/fetch.py` (wraps Scrapling) to retrieve raw
   content — pick the right mode:
   - plain page → `--mode http` (default)
   - anti-bot/JS-heavy → `--mode stealth`
3. **Record the source.** Append a row to
   `<knowledge_base>/sources/index.md` with a new `src-XXXX` id, the
   resource URL, fetch date, and a short note.
4. **Extract and write.** Pull out the entities/claims that matter, then
   use `scripts/scaffold_page.py` to create a *new* concept doc, or open
   and edit an *existing* one if the concept already has a page — merge
   new information into the existing body and bump `timestamp`, don't just
   append a duplicate section. Cross-link to related concepts.
5. **Update the index.** If a new top-level concept file was created, add
   it to `<knowledge_base>/index.md` under the right section.

## query(question)

1. Search the knowledge base (titles, tags, body text, `resource` fields)
   before doing any live scraping.
2. Traverse markdown links from the most relevant page(s) to assemble full
   context — the graph's edges are exactly these links.
3. If the answer isn't there or is marked `status: seed`/`stale`, fall back
   to `ingest()` on the gap, then answer from the freshly written page.
4. If the query surfaces a genuinely new synthesis worth keeping (not just
   a one-off answer), file it back as a new or updated page.

## lint()

Run `scripts/lint_graph.py` periodically (after a batch of ingests, or
before starting new research) to check:

- **Orphans** — pages with no incoming markdown link from any other page
  (except intentional index pages).
- **Schema** — missing required frontmatter fields, malformed YAML,
  invalid `status` values.
- **Staleness** — `timestamp` older than the configured threshold
  (default 180 days) still marked `status: researched`.
- **Broken links** — relative markdown links pointing at files that don't
  exist.

Fix what lint finds before moving on to new ingests — a graph with known
rot compounds errors, not knowledge.
