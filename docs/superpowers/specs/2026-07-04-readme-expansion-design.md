# README Expansion — Design Spec

Date: 2026-07-04
Status: Approved (brainstorming)

## Summary

Expand `README.md` from its current ~640-line form (which already describes all
ten skills with a few paragraphs each) into a comprehensive, self-contained
narrative — a mini-book of roughly 2,000–2,600 lines. A reader should be able to
understand the whole system, run it, choose models for it, and trust its output
without leaving the README, while still being pointed to `docs/` for the
authoritative schemas and decision records. The expansion adds a conceptual
spine, a worked end-to-end example, a retrieval reference, a local-model guide, a
trust/guarantees section, verification guidance, an FAQ and glossary, and a full
suite of six ASCII diagrams. Every new or rewritten section is run through the
humanizer skill before commit.

## Goals

- One document that stands alone as the primary guide to the toolkit.
- Every one of the ten skills described in several paragraphs: what it does, how
  it's invoked, the design rationale, its on-disk outputs, and what it
  deliberately does not do.
- New conceptual, reference, and guide layers around the existing per-skill
  material.
- Six ASCII diagrams (one redrawn, five new), each rendering correctly in a
  fixed-width block and readable on a narrow terminal.
- Prose that reads as human-written (humanizer pass), in the repo's existing
  dense-but-plain voice — 3–5 paragraphs per substantive section, no AI tells.
- Factual accuracy: numbers, tool contracts, config keys, and model findings
  match the code and `docs/` as they stand at this commit.

## Non-goals

- No code, config, or skill changes. This is documentation only.
- No duplication of the full authoritative schemas from
  `docs/contracts/*` — the README summarizes and links to them.
- No new marketing/benchmarks; the tone stays explanatory, not promotional.

## Source material (what the README draws on, all in-repo)

- `docs/decisions/0001-architecture.md`, `0002-knowledge-compiler.md` — the
  decisions and their rationale.
- `docs/contracts/pdf-ingestion-pipeline.md`, `okf-frontmatter.md`,
  `knowledge-compiler.md`, `schema-versions.md` — the on-disk contracts, the 8
  tool contracts, the role-routed stack, the serving notes.
- `docs/pipeline-findings.md` — the five-agent synthesis: the three-layer idea,
  the gate, the model-routing lessons, what the pipeline can write, the closed
  gaps.
- The ten `skills/*/SKILL.md` files — the per-skill behavior and the
  "deliberately does not do" notes.
- The worked example numbers come from the real Proof-of-Stake / snail run
  described in the session and the config at `snail-research/.deepresearch.yml`
  (5 sources → ~105 chunks → 103 claims / 61 entities / 66 relations → a
  compiled index → a grounded thesis). The README states these as an
  illustrative real run.

## Outline (target structure)

The README is rebuilt to this section order. Existing prose that is still
accurate is kept and folded in; sections marked NEW are written from scratch;
sections marked EXPAND grow the current material.

0. **Front matter** — title, badges, one-paragraph tagline, and a table of
   contents (anchor links) — needed at this length.
1. **Why this exists** — EXPAND to 3–4 paragraphs: the compounding-research
   problem and the one-time-write-with-evidence answer.
2. **Core ideas — the mental model** — NEW: the three layers (durable corpus /
   disposable index / judgment layer that only proposes), the verbatim-quote
   gate as the load-bearing invariant, `evidence_ref` (two producers, one
   shape), OKF, files-as-corpus vs index-as-cache. → Diagram 2.
3. **How it fits together** — EXPAND + redraw: the phase pipeline (acquire →
   chunk → extract → compile → retrieve → synthesize), marking deterministic vs
   judgment stages. → Diagram 1 (redrawn) and Diagram 6.
4. **Quick start** — EXPAND: install tiers, `drt init`, a first web run, a first
   PDF run, a first query; keep the by-hand command sequence.
5. **A worked example, end to end** — NEW: the real Proof-of-Stake / snail run
   narrated with actual numbers, from fetch to a grounded, cited thesis.
6. **The skills** — EXPAND all ten + the local backend, grouped: web research
   (research-knowledge-graph); the seven-stage PDF pipeline (pdf-ingest-router,
   pdf-to-canonical-markdown, pdf-layout-provenance,
   canonical-markdown-to-llm-nodes, knowledge-extraction, llm-wiki-writer,
   rag-eval-harness); the compiler layer (knowledge-compiler, retrieval-planner).
7. **The retrieval tools** — NEW reference: the 8 tools (a table plus prose),
   RRF fusion, `compose_dossier` + the gate + `--format md`,
   `find_contradictions`. → Diagram 4.
8. **Running local models — the role-routed stack** — NEW guide: `agent` vs
   `local`, `llm.roles` and the per-phase model table, the Ornith + qwen3.5
   non-thinking findings, serving setup (the Ollama chat-template fix,
   `max_tokens`), embeddings (Ollama vs sentence-transformers), the
   `validate-local-llm` harness. → Diagram 5.
9. **What the pipeline guarantees** — NEW: the gate as defense-in-depth (one
   shared check, three checkpoints), "a weak model can only under-produce, never
   corrupt," and the honest limits (fidelity-not-truth; source errors pass
   through; verbatim ≠ well-formed).
10. **Configuration** — EXPAND: the full `.deepresearch.yml` walkthrough (paths,
    features, `llm.roles`, embeddings).
11. **Verification & testing** — NEW (lighter): the fast/heavy test tiers, the
    reference fixtures, how to check a run (`rag-eval-harness`,
    `validate-local-llm`).
12. **Status, roadmap & what's deferred** — EXPAND.
13. **FAQ + glossary** — NEW: "how is this different from RAG?", "do I need a
    GPU?", "which model?", "is my data sent anywhere?"; a glossary (OKF,
    evidence_ref, dossier, RRF, provenance, run directory, producer, chunk/node).
14. **Contributing** — EXPAND: dev setup, test tiers, contracts to update, the
    sync guards.
15. **License** — one line.

## The six diagrams

All are ASCII in fenced blocks, ≤ ~76 columns wide so they don't overflow a
narrow terminal, box-drawing done with `+ - | v` (matching the current style, no
Unicode box characters).

1. **Master pipeline** (redraw of the current one) — the phase flow from
   `.deepresearch.yml` through the producers, the compiler, and retrieval to an
   evidence dossier; cleaner spacing and clearer stage labels than today's.
2. **Three-layer architecture** — durable corpus (files) | derived index
   (DuckDB + LanceDB) | judgment layer (agent / local model), with arrows
   showing that judgment only writes into the corpus and the index is rebuilt
   from it.
3. **`evidence_ref` normalization** — a PDF claim `{node_id, page}` and a web
   claim `{locator, url}` converging into one `EvidenceRef {producer, source_id,
   locator, page?, url?}` at index time, on-disk files untouched.
4. **Retrieval / RRF fusion** — a query fanning into DuckDB FTS and LanceDB
   vector search, the two ranked lists fused by reciprocal rank fusion, feeding
   the eight tools and `compose_dossier`'s verbatim gate.
5. **Role-routed model stack** — the phases (extract / wiki_write /
   conflict_adjudicate / synthesize / embed) routed to their models (fast
   instruct / reasoning / embedding), with the "single-model fallback" note.
6. **End-to-end data lifecycle** — one source's journey: source.md → chunks →
   claims+entities+relations (gated) → index rows → dossier → deliverable.

Each diagram is checked to render aligned in a fenced ```` ``` ```` block.

## Voice, accuracy, and the humanizer pass

- Match the README's existing register: explanatory, specific, opinionated where
  earned, plain. 3–5 paragraphs per substantive section. Prefer concrete detail
  over abstraction; state trade-offs honestly (the repo already does this —
  e.g., "designed, not yet built," "an honest placeholder").
- Punctuation follows the repo's convention: `--` in README prose (not em
  dashes), straight quotes.
- After drafting each new or rewritten prose section, run the humanizer skill
  over it and apply its fixes before committing that section. The humanizer is
  applied to prose only, not to code blocks, tables, or diagrams.
- Accuracy gate: every factual claim (test counts, the 8 tool names/args, config
  keys, model findings, the worked-example numbers, "10 skills", version 0.2.0)
  is cross-checked against the code and `docs/` before commit. Where a number
  would drift (e.g., exact test count), phrase it robustly rather than pinning a
  brittle figure.

## Build approach (for the plan)

The README is built section by section into the single `README.md` file, in
outline order, each section a small, reviewable unit ending with: the section
drafted, its diagrams rendered-checked, its prose humanized, and a commit. A
final pass verifies the table-of-contents anchors resolve, all six diagrams
render aligned, no section is a placeholder, and the cross-links to `docs/`
resolve. No tests are affected (documentation only), but the plan includes a
markdown-lint/link sanity check and a read-through against the accuracy gate.

## Risks / trade-offs accepted

- A long README can drift from the code. Mitigation: the accuracy gate at build
  time, robust phrasing for volatile numbers, and links to `docs/` as the
  authoritative source for schemas.
- Duplication between the README's per-skill sections and the `SKILL.md` files
  is acceptable and intended — the README is the narrative front door; the
  SKILL.md files are the agent-facing operational instructions.
