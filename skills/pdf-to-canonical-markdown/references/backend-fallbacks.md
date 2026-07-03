# Backend fallbacks (documented, not implemented)

Docling is the only backend this stage actually runs. Per the pipeline
contract's routing table (see `pdf-ingest-router`'s
`references/routing-table.md`), these are the fallbacks the original
research identified for routes Docling handles poorly — reach for them
only when a real document forces the issue, not preemptively.

## Marker — math/scientific PDFs

**When:** `classification.json` route is `scientific-math` and Docling's
markdown shows mangled or dropped LaTeX/formula regions (common with dense
equations, multi-line derivations, or unusual math fonts).

**Why it's better here:** Marker was built specifically around scientific
PDF layout and has stronger formula-to-LaTeX recovery than Docling's
general-purpose layout model. Docling still runs first per the routing
table (`scientific-math → Docling then Marker if formulas look mangled`) —
Marker is a second pass, not a replacement, so a math-light "scientific"
paper doesn't pay Marker's extra cost.

**Install note:** `pip install marker-pdf` pulls its own model weights
(separate from Docling's) and is heavier to set up; do this only when a
document actually needs it.

## MinerU — complex academic/scanned documents

**When:** dense multi-column academic layouts, or scanned documents where
Docling's default OCR path struggles with layout reconstruction (reading
order across columns, footnotes interleaved with body text).

**Why it's better here:** MinerU's layout analysis is tuned for exactly
this class of document (its training data leans heavily academic/scanned)
and it tends to preserve reading order and cross-references more reliably
than a general-purpose OCR pipeline in these specific cases.

**Install note:** MinerU has a larger dependency footprint (its own PDF
parsing + layout + OCR stack) — treat it as a targeted swap-in for
documents where Docling's OCR route (see `classification.json`'s `scanned`
route) visibly fails, not a default.

## Unstructured — typed-ETL-element needs

**When:** a downstream consumer needs strongly-typed structural elements
(e.g. `NarrativeText`, `Title`, `ListItem`, `Table` as distinct Python
classes) for an ETL-style pipeline, rather than a markdown/JSON document
tree.

**Why it's better here:** Unstructured's `partition_pdf` returns a flat
list of typed elements, which is a more natural fit than Docling's nested
document tree when the consumer wants to filter/transform by element type
directly (e.g. "give me every `Table` element across a batch of PDFs")
rather than reconstruct structure from a tree.

**Install note:** `pip install unstructured[pdf]`. Only worth adding if a
future stage genuinely needs element-typed output that `docling_raw.json`
doesn't already provide via its `texts`/`tables`/`pictures` arrays and
`label` field — check that first, since Docling's raw JSON already carries
most of the same typing (`section_header`, `text`, `table`, `picture`,
`caption`, `list_item`).
