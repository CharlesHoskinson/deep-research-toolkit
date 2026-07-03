# Routing table and decision logic

`classify_pdf.py` computes five signals (see SKILL.md) and picks exactly one
`route` by checking conditions **in priority order** — the first one that
matches wins. This doc explains why that order and those thresholds, so a
human (or a later Claude session) can tune them without re-deriving the
reasoning from scratch.

## Priority order

1. **`form`** — `has_acroform_fields` is true.
   AcroForm fields are a structural fact about the PDF (there are literally
   fillable field objects in it), not a fuzzy inference, so this checks
   first and overrides every other signal. A form with a lot of prose text
   is still a form.

2. **`scanned`** — `image_only_page_ratio >= 0.6`.
   If most pages have no extractable text layer at all, nothing downstream
   (chunking, table extraction, quote-verification) works without OCR
   first, regardless of how much math or tabular structure the images
   contain. 0.6 rather than a stricter "any page" or looser "majority"
   threshold: a handful of scanned appendix pages in an otherwise digital
   document shouldn't flip the whole document to `scanned`.

3. **`scientific-math`** — `detected_math_density == "high"`.
   Checked after `scanned` (no point routing to a math-aware backend if the
   text isn't extractable in the first place) but before `financial-legal`
   and `slide-like`, since a dense-math paper can also contain tables or be
   sparse-per-page and math handling should take priority in that case.

4. **`financial-legal`** — `table_like_page_ratio >= 0.6`.
   Financial statements, schedules, and contracts tend to be *majority*
   tabular across the whole document, not just contain one incidental
   table. The threshold is deliberately high (not e.g. 0.3) because a
   single table on an otherwise prose-heavy document (a whitepaper with one
   summary table, say) should stay `digital-text` — that's exactly the
   shape of `tests/fixtures/hydra-settlement-test-fixture.pdf` (1 table / 2
   pages = 0.5 ratio, below this threshold, correctly falls through to
   `digital-text`).
   Caveat: this ratio is noisy on very short documents (a 2-page doc with
   one table page is already at 0.5) — for documents under ~5 pages, treat
   a `financial-legal` classification as provisional and eyeball it.

5. **`slide-like`** — `avg_extractable_chars_per_page <= 150` and not
   already caught by `scanned`.
   Slide decks have short bullet-point text per page rather than flowing
   prose, but (unlike scanned pages) still have a real text layer. The
   `image_only_page_ratio < 0.6` guard keeps this from double-firing on
   documents already routed to `scanned`.

6. **`digital-text`** — default. Plain extractable text, no strong signal
   for any of the above. This is the common case and should be the most
   frequent route in practice.

## Route → backend

| route | ideal backend | why |
|---|---|---|
| `digital-text` | Docling | clean text extraction, no special handling needed |
| `financial-legal` | Docling | Docling's table structure recognition handles statements/schedules well |
| `form` | Docling | Docling reads AcroForm fields directly |
| `scientific-math` | Docling, then Marker if formulas look mangled | Docling's math/formula recognition is decent but not perfect; Marker is a documented fallback, not wired up |
| `scanned` | Docling's OCR mode | needed whenever there's no text layer to read |
| `slide-like` | render pages + describe figures | slides are visual-first; text extraction alone loses the point |

## What's actually implemented in this toolkit

Only plain Docling conversion is wired up in `pdf-to-canonical-markdown`.
`recommended_backend` in `classification.json` is therefore always
`"docling"`, regardless of route — but `notes` records the ideal backend
from the table above whenever it isn't plain Docling, so nothing is lost,
it's just deferred until a real scanned/math-heavy/slide document forces
adding a second backend (as `docs/contracts/pdf-ingestion-pipeline.md`
specifies).

## Math symbol detection

`detected_math_density` counts occurrences of a fixed set of math/LaTeX-ish
symbols and Greek letters (∑ ∫ √ ≤ ≥ ≠ ± ∞ ∂ Δ ∇ × ÷ → ↔ ∈ ∉ ⊂ ⊆ ∪ ∩ ∀ ∃ π ∝
≈ ≡, plus `\frac`, `\sum`, `\int`, `\alpha`, etc., plus the Greek alphabet)
per page, averaged across the document, then bucketed:

- `low`: average < 1.0 hits/page
- `medium`: 1.0 <= average < 5.0 hits/page
- `high`: average >= 5.0 hits/page

This is a rough proxy, not a real math-OCR pass — it will under-count math
that's rendered as images (which `scanned` should catch instead) and can
over-count documents that just use Greek letters as variable names in prose
(e.g. finance papers using α/β for alpha/beta). Treat `medium` classifications
as a hint worth a human glance, not gospel.

## Try it

```
python skills/pdf-ingest-router/scripts/classify_pdf.py \
    tests/fixtures/hydra-settlement-test-fixture.pdf --runs-dir /tmp/pdf-runs
```

Should print a run directory ending in
`hydra-settlement-test-fixture-4edb3c3c/` and classify the fixture as
`digital-text` (see `tests/fixtures/reference-run-hydra-settlement/classification.json`
for the exact reference output).
