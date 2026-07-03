# Chunking strategy

This stage turns a flat list of provenance units (`provenance.jsonl`) into a
flat list of retrieval nodes (`chunks.jsonl`). It runs in two passes:
grouping, then an optional split expansion. The logic lives in
`deep_research_toolkit.pdf.chunk`; `scripts/chunk_nodes.py` is a thin CLI
shim over it.

## Pass 1: grouping (`group_units`)

Walk `provenance.jsonl` in order, maintaining one "in-progress section"
accumulator:

- **`heading` unit** → flush whatever section was in progress (append it to
  the output list), then start a new section:
  - `title` = the heading's own text.
  - `section_path` = the heading unit's own `section_path` (its ancestor
    chain, as already computed by `pdf-layout-provenance`) **plus** its own
    text. This mirrors how `provenance.jsonl` represents section_path:
    a heading's `section_path` field is its *parent* chain, not including
    itself, while a paragraph's `section_path` includes the heading it sits
    directly under. So `heading.section_path + [heading.text]` reconstructs
    the full path for the section this heading introduces.
  - `member_units` seeded with just the heading unit.
- **`table` unit** → flush the in-progress section, then emit a standalone
  `table` node immediately (`title: "Table"`, `section_path` copied
  directly from the unit's own `section_path`, since a table isn't itself
  part of the heading chain).
- **`figure` / `picture` unit** → same as `table`, but `type: "figure"`,
  `title: "Figure"`.
- **Anything else** (`paragraph`, `list_item`, `caption`, ...) → append to
  the in-progress section's `member_units`. If no section is in progress
  yet (body text before any heading, or right after a table/figure closed
  one), start an implicit section using the unit's own `section_path`.

At the end, flush whatever section is still in progress.

### Text assembly and the caption exception

A node's `text` is `"\n\n".join(...)` over its `member_units`' own `text`,
**except** units of type `caption` are skipped in that join. A caption
contributes its `unit_id` and `page` to the node's `source` (so a document's
provenance trail doesn't silently drop it), but its text is not spliced into
the surrounding prose: a caption describes a table/figure, and mixing
"Figure 1: Head Lifecycle (placeholder)" into the middle of a paragraph
about open questions would read as a non-sequitur and pollute the chunk's
embedding. If a real `figure`/`picture` unit is present, its caption belongs
conceptually to that figure node instead — this stage doesn't currently
attach captions to figure nodes' text either (same reasoning), leaving that
join to be revisited if `knowledge-extraction` wants captions surfaced
verbatim on the figure node.

### `content_hash`

Computed once, from the final `text` field, via
`deep_research_toolkit.common.hashing.content_hash` (`"sha256:" +
sha256(text).hexdigest()[:16]`). This is literally what `provenance.jsonl`'s
own `content_hash` does for a single unit's text, so a single-unit node's
hash equals that unit's own `content_hash` (verified against
`tests/fixtures/reference-run-hydra-settlement/provenance.jsonl` /
`chunks.jsonl` for every 1-unit node, e.g. the `2. Architecture` and `table`
nodes). For multi-unit nodes, this hashes the exact same string that ends up
in `text` — one join, one source of truth, no separate hidden format for
the hash input.

## Pass 2: splitting (`maybe_split`)

The contract allows (but doesn't require) splitting an over-long section:
"a section that's still too long for one node splits at paragraph
boundaries and gets `parent` set to the section node." This implementation
does that:

- **Threshold:** 1500 characters of assembled section text (`SPLIT_THRESHOLD`
  in `deep_research_toolkit/pdf/chunk.py`). This is a rough proxy for "too
  long to embed and retrieve as one coherent unit" — comfortably inside
  typical embedding context windows, but big enough that ordinary short
  sections (like every section in the `reference-run-hydra-settlement`
  fixture) never trigger it. Tune it in one place if real documents show
  it's off.
- **Trigger condition:** text length over the threshold **and** more than
  one contributing non-heading, non-caption unit. A single giant paragraph
  isn't split further (this stage splits at unit boundaries, which are
  already paragraph-level from Docling — it doesn't re-segment prose by
  sentence).
- **What happens:** the section node is kept as-is (full concatenated text,
  still a valid citable/retrievable overview node), and each contributing
  unit *also* becomes its own `paragraph`-type node with:
  - the same `title`/`section_path` as the parent section,
  - `text` equal to just that one unit's text,
  - `source.unit_ids` = just that unit,
  - `links.parent` = the section node's `node_id`.
  These child nodes are inserted immediately after the section node in the
  overall sequence, so `chunks.jsonl` stays one flat, linearly-ordered file
  — no separate tree structure to walk — while retrieval can still surface
  the smaller, more precise paragraph node when that's the better match, and
  fall back to the parent section for broader context.

No section in `tests/fixtures/reference-run-hydra-settlement/` is long
enough to trigger this path (the longest is well under 1500 characters), so
the reference `chunks.jsonl` doesn't exercise it directly — but the logic
is covered by a dedicated unit test with a synthetic over-long section (see
`tests/unit/`).

## Pass 3: id assignment and linking (`build_nodes`)

Runs once, after both passes above have produced the final flat sequence:

- `node_id` = `<document_id>:n<NNN>`, zero-padded to at least 3 digits (or
  wider if there are 1000+ nodes, so ids stay sortable as strings).
- `links.previous`/`links.next` = the adjacent node's `node_id` in this
  final sequence (`null` at the very start/end). This includes split
  children — the chain is over the flattened list, not per-section.
- `links.parent` = `null` for every node except split children, which point
  at their section's `node_id`.
- `source.page_start`/`page_end` = min/max `page` across all contributing
  units (including captions, since they still occurred on a real page).

## What this deliberately does not do

- No token-count-based re-splitting of an individual long paragraph — see
  "Trigger condition" above.
- No cross-page merging heuristics beyond what `pdf-layout-provenance`
  already encodes in `section_path`/`page` per unit.
- No `summary`/`keywords`/`entities` — left as `""`/`[]`/`[]` for
  `knowledge-extraction` to fill in with an LLM call, per
  `docs/contracts/pdf-ingestion-pipeline.md`.

## The real, verified example

`tests/fixtures/reference-run-hydra-settlement/provenance.jsonl` (17 units)
→ `chunks.jsonl` (9 nodes: 8 sections + 1 table) is the real, verified
reference for this stage, produced by chaining all seven pipeline skills
together against `tests/fixtures/hydra-settlement-test-fixture.pdf`. Note in
particular that the table node's `section_path` is `["3. Throughput
Comparison"]` — the table sits under that heading, not a later one. (An
earlier, hand-derived example fixture had this wrong — pinned to `["5. Open
Questions"]` — before a bug fix in the provenance stage; if you see that
path referenced anywhere else, it's the stale, buggy example, not this real
reference run.)
