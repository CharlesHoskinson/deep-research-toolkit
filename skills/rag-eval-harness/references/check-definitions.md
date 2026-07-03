# Check definitions

Each of these is a mechanical, read-only check run by `scripts/run_eval.py`.
None call an LLM — the point of this stage is to be cheap enough to run on
every document, every time, and catch the kind of silent corruption that
never raises an exception anywhere upstream.

## headings_recovered

Every line starting with `#` in `canonical.md` is compared against the
union of every `section_path` entry across all of `chunks.jsonl`. This
catches the chunking stage silently dropping a section — the most common
failure mode when a document's structure is irregular enough that a
heading-detection heuristic misses one (an unusually formatted subheading,
a heading that got misclassified as a plain paragraph by Docling, etc).
If a heading exists in the canonical text but never shows up as a
`section_path`, the reader has no way to navigate to that part of the
document through the graph — the content might still be *somewhere* in a
chunk's `text`, but it's unreachable by section, which defeats the point of
structure-aware chunking in the first place.

## tables_present

Counts `unit_type: "table"` rows in `provenance.jsonl` and compares against
the number of `.csv` files in `tables/`. This catches the table-extraction
script silently failing on one table while succeeding on others — e.g. a
table with an unusual layout that Docling's table-structure model doesn't
resolve cleanly, or an off-by-one in a loop over detected tables. A
mismatch here means some tabular data that a reader can see in the source
PDF simply isn't queryable anywhere in the knowledge graph.

## page_citations_valid

Every `supporting_evidence[].page` referenced from `claims.jsonl` must be a
page number that actually appears in `provenance.jsonl`. This is a cheap
sanity check that catches a claim citing a page that doesn't exist in the
document at all — typically a copy-paste error during claim extraction (the
model reusing a page number from a different claim) or an off-by-one from
a 0-indexed vs 1-indexed page count mismatch between stages. Cheap to check,
and a claim citing a nonexistent page is an immediate signal that something
about that claim's provenance chain is broken.

## evidence_quotes_verbatim

The most important check. Every `supporting_evidence[].quote` in
`claims.jsonl` must be an exact (verbatim) substring of the `text` field of
at least one `provenance.jsonl` unit on the cited page. This is what keeps
a claim *audit-able* rather than merely *plausible-sounding*: an LLM
writing "supporting evidence" is prone to paraphrasing the source into
something that reads more like a clean quote than what the document
actually says, which is functionally indistinguishable from hallucination
once it's sitting in a knowledge graph next to genuinely sourced claims. A
failure here does not necessarily mean the claim is false — it means the
citation backing it cannot currently be trusted without a human going back
to the source page, which is exactly the state this stage exists to catch
before it propagates into a wiki page.

## figures_accounted_for

Every entry in `figures/captions.jsonl` must be either `extracted: true`
with a real image file present in `figures/`, or have the `extracted` field
present at all (explicitly `false` counts as accounted for, per the
contract's "flag rather than silently drop" rule) — an entry with no
`extracted` field at all fails this check, since that's exactly the
"silently missing" state the contract's design is trying to avoid. This
catches figure extraction failing partway (caption detected, image never
saved) without a matching explicit flag explaining why the image is absent.

## no_ocr_garbage

Aggregates all `provenance.jsonl` unit text per page and computes the ratio
of characters outside printable ASCII + common whitespace
(`string.printable`). If more than 5% of a page's aggregate text falls
outside that set, the check fails for that page. This is a coarse proxy for
OCR/encoding corruption — the kind of failure where a scanned page or a
PDF with a broken font encoding produces text that *looks* like it
extracted successfully (no exception, non-empty string) but is actually
mojibake or garbled glyphs. Nothing upstream would necessarily notice this;
it would just quietly poison every claim, chunk, and citation derived from
that page.
