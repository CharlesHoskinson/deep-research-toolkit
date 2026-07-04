# Tool contracts

Exact input arguments and JSON output shapes for the 8 retrieval-planner
tools. Each CLI subcommand of `scripts/query.py` is a thin shim over the
same-named method on `deep_research_toolkit.compiler.tools.Index`; the
JSON printed to stdout is exactly the method's return value
(`json.dumps(out, indent=2, ensure_ascii=False)`). None of these tools
makes an LLM call.

All commands require a built index (`<index_dir>/knowledge.duckdb`); if it
is missing, `Index.open` raises `FileNotFoundError` telling you to run the
knowledge-compiler skill's `compile.py` first.

## search-wiki

Hybrid (keyword + vector) search over wiki pages.

```
python scripts/query.py search-wiki "<query>" [--k N]
```

- `query` (positional, required): search text.
- `--k` (int, default 8): maximum number of results.

Output: a JSON **list** of page hits, best-first:

```json
[
  {
    "path": "hydra.md",
    "title": "Hydra",
    "type": "Concept",
    "status": "ok",
    "snippet": "first 200 characters of the page body"
  }
]
```

Empty list when nothing matches.

## read-page

Fetch one wiki page in full by its kb-relative path.

```
python scripts/query.py read-page <kb-relative-path>
```

- `path` (positional, required): page path as returned by `search-wiki`
  (e.g. `hydra.md`).

Output: a JSON **object**:

```json
{
  "path": "hydra.md",
  "body": "full markdown body",
  "frontmatter": {"type": "Concept", "title": "Hydra", "timestamp": "t"}
}
```

If the page does not exist: `{"path": "<path>", "error": "not found"}`.

## search-claims

Hybrid search over extracted claims, each returned with its evidence rows.

```
python scripts/query.py search-claims "<query>" [--k N] [--producer pdf|web]
```

- `query` (positional, required): search text.
- `--k` (int, default 8): maximum number of results.
- `--producer` (`pdf` | `web`, default none): filter applied *after*
  retrieval, so fewer than `k` results may come back.

Output: a JSON **list** of claims:

```json
[
  {
    "claim_id": "c-123",
    "producer": "pdf",
    "source_id": "run-abc",
    "claim": "the claim text",
    "claim_type": "factual",
    "confidence": 0.9,
    "evidence": [
      {
        "producer": "pdf",
        "source_id": "run-abc",
        "locator": "p3.b2",
        "page": 3,
        "url": null,
        "quote": "verbatim quote from the source"
      }
    ]
  }
]
```

## get-entity

Look up one entity by id or by case-insensitive exact name.

```
python scripts/query.py get-entity <name-or-id>
```

- `name_or_id` (positional, required): entity id, or entity name
  (matched with `lower(name) = lower(?)`).

Output: a JSON **object**:

```json
{
  "entity_id": "e-1",
  "name": "Hydra",
  "type": "Protocol",
  "aliases": ["Hydra Head"],
  "mentions": ["kb/hydra.md#s1"],
  "relations": [
    {"relation_id": "r-1", "subject": "e-1", "predicate": "part_of", "object": "e-2"}
  ]
}
```

- `mentions`: list of locator strings from `entity_mentions`.
- `relations`: every relation where the entity is subject *or* object.

If not found: `{"query": "<name_or_id>", "error": "entity not found"}`.

## neighbors

Graph walk over the relations table from a starting entity. Edges are
treated as undirected; `depth` bounds the number of hops.

```
python scripts/query.py neighbors <entity-id> [--depth D]
```

- `entity` (positional, required): starting entity id (use the exact
  `entity_id`, not a display name).
- `--depth` (int, default 1): maximum hop count.

Output: a JSON **list** of reachable nodes (the start node excluded),
each with its minimum hop distance, ordered by depth then node id:

```json
[
  {"node": "e-2", "depth": 1},
  {"node": "e-7", "depth": 2}
]
```

## get-sources

Provenance for a wiki page or a claim. Pass exactly one of the two flags.

```
python scripts/query.py get-sources (--page P | --claim C)
```

- `--page` (string): kb-relative page path.
- `--claim` (string): claim id.

Output with `--page` (values pulled from the page frontmatter; each may be
null if the frontmatter lacks the key):

```json
{
  "page": "hydra.md",
  "source": "...",
  "source_docs": ["..."],
  "resource": "..."
}
```

Output with `--claim` (distinct evidence sources):

```json
{
  "claim": "c-123",
  "sources": [
    {"producer": "web", "source_id": "run-xyz", "url": "https://..."}
  ]
}
```

If neither flag is given: `{"error": "pass page= or claim="}`.

## find-contradictions

Mechanical contradiction *candidates*. Takes no arguments.

```
python scripts/query.py find-contradictions
```

Output: a JSON **list** mixing two candidate kinds:

```json
[
  {
    "kind": "relation",
    "subject": "e-1",
    "predicate": "founded_in",
    "objects": ["2015", "2017"],
    "relation_ids": ["r-3", "r-9"],
    "source_ids": ["run-a", "run-b"]
  },
  {"kind": "conflicted_page", "path": "hydra.md"}
]
```

- `kind: "relation"`: the same subject+predicate maps to more than one
  distinct object.
- `kind: "conflicted_page"`: a wiki page with `status: conflicted`.

These are candidates, not verdicts — confirming them is the agent's job
(see SKILL.md).

## compose-dossier

Assemble claims plus citations into an evidence dossier, gated by the
verbatim-quote invariant.

```
python scripts/query.py compose-dossier "<query>" [--claims c1,c2] [--k N]
```

- `query` (positional, optional): if `--claims` is not given, claims are
  selected by running `search-claims` with this query.
- `--claims` (comma-separated claim ids, default none): explicit claim
  set; when given, `query` is ignored for selection.
- `--k` (int, default 12): how many claims the query selection retrieves.

Output: a JSON **object** with two lists:

```json
{
  "included": [
    {
      "claim_id": "c-123",
      "claim": "the claim text",
      "claim_type": "factual",
      "confidence": 0.9,
      "evidence": [
        {
          "claim_id": "c-123",
          "producer": "pdf",
          "source_id": "run-abc",
          "locator": "p3.b2",
          "page": 3,
          "url": null,
          "quote": "verbatim quote"
        }
      ]
    }
  ],
  "rejected": [
    {
      "claim_id": "c-456",
      "claim": "...",
      "claim_type": "...",
      "confidence": 0.5,
      "evidence": ["..."],
      "reason": "non-verbatim quote(s): [...]"
    },
    {"claim_id": "c-999", "reason": "claim_id not found in index"}
  ]
}
```

A claim is `included` only if it has at least one evidence row and
**every** quote is a verbatim substring of its source text (PDF page text
from `provenance.jsonl`, or the web run's `source.md`). Otherwise it lands
in `rejected` with a `reason` of `"no evidence"`,
`"non-verbatim quote(s): [...]"`, or `"claim_id not found in index"`
(unknown ids carry only `claim_id` and `reason`).
