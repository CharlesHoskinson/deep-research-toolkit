# Web claim extraction

How to turn a web research run (`research-runs/<source_id>/`) into
`claims.jsonl`, `entities.jsonl`, and `relations.jsonl`. This is the web
counterpart of `knowledge-extraction`'s Part 2 (see
`../../knowledge-extraction/SKILL.md`): the same judgment calls, the same
four rules, but evidence anchors to `source.md` and chunk locators instead
of PDF pages and node ids. The knowledge compiler indexes both producers
into one table, so getting the shapes right here is what lets a dossier
mix PDF- and web-sourced evidence for the same entity.

## Inputs and outputs

Read `chunks.jsonl` in the run directory — one JSON object per heading
section of the fetched page, each with a `node_id` of the form
`<source_id>:cNN` and the section's `text`. Then write three JSONL files
directly into the same run directory, one JSON object per line:

```json
// claims.jsonl
{"schema_version": "1.0", "claim_id": "c_0001", "document_id": "<source_id>", "claim": "...", "claim_type": "architectural | empirical | definitional | comparative", "confidence": "high | medium | low", "supporting_evidence": [{"locator": "<source_id>:cNN", "quote": "...", "url": "<source_url>"}], "entities": ["ows"], "relations": [{"subject": "ows", "predicate": "defines", "object": "wallet interoperability interface"}]}

// entities.jsonl
{"schema_version": "1.0", "entity_id": "open-wallet-standard", "name": "Open Wallet Standard", "aliases": ["OWS"], "type": "standard", "mentions": ["<source_id>:c01", "<source_id>:c03"]}

// relations.jsonl
{"schema_version": "1.0", "relation_id": "r_0001", "subject": "open-wallet-standard", "predicate": "defines", "object": "wallet interoperability interface", "supporting_claim": "c_0001", "document_id": "<source_id>"}
```

The one structural difference from the PDF shape is `supporting_evidence`:

```json
{"locator": "<source_id>:cNN", "quote": "...", "url": "<source_url>"}
```

- `locator` — the `node_id` of the chunk (from `chunks.jsonl`) the quote
  comes from. There is no `page` key; web runs have no pages.
- `quote` — a verbatim substring of `source.md` (rule 2 below).
- `url` — the run's `source_url`, copied from `manifest.json`. Every
  evidence entry carries it so a claim remains attributable even when read
  out of the index, away from the run directory.

`document_id` is the run's `source_id` (the run directory name, also in
`manifest.json`). `entities.jsonl` `mentions` use the same `<source_id>:cNN`
locators.

## The four rules

### 1. One assertion per claim — no compound sentences

If a sentence asserts two things and both halves matter, split it into two
claims or pick the single load-bearing assertion. A claim should be
checkable as true or false on its own.

### 2. Every quote is a verbatim substring of `source.md`

The evidence rule, checked mechanically downstream — but note the web
twist: the quote must be an exact substring of the run's **`source.md`
file**, not of a PDF page. The dossier composer reads `source.md` whole
and does a substring check against it, so copy the exact span — do not
clean up wording, expand an abbreviation, fix punctuation, or normalize
whitespace. Chunking preserves section text but strips surrounding blank
lines and drops heading `#` markers, so when in doubt select your span
from a single paragraph and confirm it appears character-for-character in
`source.md`. If you catch yourself editing the quote to read better,
shorten the span instead of paraphrasing.

### 3. Merge entity mentions that clearly refer to the same thing

A page will use a short name and a full name interchangeably ("OWS" and
"Open Wallet Standard"). These become **one** row in `entities.jsonl`,
with the canonical/most-formal form as `name` and the others in
`aliases` — never two `entity_id`s for the same concept. Collect every
`mentions` locator across the whole page before writing the row, rather
than one row per mention. If the entity already appeared in a PDF run or
another web run, reuse the same `entity_id` so the compiler's graph joins
the mentions instead of splitting them.

### 4. Don't force it

Web pages are noisier than papers: marketing copy, hedged roadmap talk,
navigation cruft that survived cleaning. If a sentence is vague, hedged,
or you can't find a chunk whose text actually backs it, leave it out
rather than writing a low-confidence claim to hit a quota. Only emit a
`relations.jsonl` row when a claim in `claims.jsonl` actually asserts that
subject-predicate-object triple. A handful of well-evidenced claims beats
twenty shaky ones.
