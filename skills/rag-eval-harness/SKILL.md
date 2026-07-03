---
name: rag-eval-harness
description: Use as the last stage of the PDF ingestion pipeline (see docs/contracts/pdf-ingestion-pipeline.md), run after canonical-markdown-to-llm-nodes / knowledge-extraction have produced a run directory under the configured pdf-runs location, to catch silent corruption before trusting a document's chunks, claims, tables, or figures. Also use to check the health of a partial run (only some stages completed).
---

# rag-eval-harness

Seventh and final stage of the PDF ingestion pipeline. Everything upstream
(routing, conversion, provenance extraction, chunking, knowledge extraction)
can fail quietly: a heading that never made it into `section_path`, a claim
whose "quote" is actually a paraphrase, a figure that got silently dropped.
None of those show up as an exception — the pipeline finishes, the files
exist, and the corruption is only visible if someone reads every file by
hand. This skill is that read instead, automated and run every time.

## Quick start

```
python scripts/run_eval.py <run_dir>
```

Reads whatever files exist in `<run_dir>` (see the contract doc's "Per-
document working directory" tree) and writes `eval_report.json` +
`eval_report.md` back into that same directory, plus updates
`manifest.json`'s `stages.rag-eval-harness` entry.

Safe to run against a partial pipeline: any check whose upstream file(s)
don't exist yet is **skipped**, not failed — it shows up in the report as
passed with a `"skipped - <reason>"` detail, so re-running after a later
stage completes will pick up the real check. This means `pass_rate` only
means "score out of the checks that could run" — always read the `detail`
field, not just the number, before trusting a run.

## What it checks

Six mechanical checks, each explained in full (with the failure mode it
catches) in `references/check-definitions.md`:

- `headings_recovered` — every heading in `canonical.md` made it into some
  `section_path` in `chunks.jsonl`.
- `tables_present` — table count in `provenance.jsonl` matches CSV count in
  `tables/`.
- `page_citations_valid` — every claim's cited page actually exists.
- `evidence_quotes_verbatim` — every claim's supporting quote is a verbatim
  substring of the provenance text on its cited page (the important one —
  this is what keeps claims audit-able instead of merely plausible).
- `figures_accounted_for` — every figure caption is either extracted with a
  real file, or explicitly flagged as not extracted — never silently
  dropped.
- `no_ocr_garbage` — no page's aggregate text is more than 5%
  non-printable/mojibake characters.

## What it deliberately does NOT check (optional manual step)

The original research on RAG evaluation also recommends **Q/A retrieval
probes**: write N questions a reader should be able to answer from this
document, run them through your retrieval + prompt path, and check that the
chunk which actually answers each question is the one that comes back. This
is a genuinely good signal — it catches chunking/retrieval problems that
none of the mechanical checks above can see (e.g. a section that got split
mid-thought, or two nodes with near-duplicate text that confuse
similarity search).

It is **not** part of `run_eval.py` and does not affect `pass_rate`,
because it requires an LLM call per probe — that's a cost and latency
tradeoff this stage's automated pass should never carry silently. Treat it
as an optional manual step for high-stakes documents:

1. Read `chunks.jsonl` and write 5-10 questions whose answer clearly lives
   in one specific node.
2. For each question, run your actual retrieval path and check the
   returned node's `node_id` matches the one you expected.
3. Record misses by hand — a systematic miss pattern (e.g. always the
   wrong sibling section) is a chunking problem, not a fluke, and is worth
   fixing in `canonical-markdown-to-llm-nodes` rather than working around
   at query time.

## Output

`eval_report.json` matches the contract doc's schema exactly:
`schema_version`, `document_id`, `generated_at`, `checks: [{name, passed,
detail}]`, `pass_rate`. `eval_report.md` is the same information as a short
markdown table, for a human skimming the run directory.
