# Comprehensive README — Design

Date: 2026-07-03

## Goal

Replace the current stub README with a comprehensive one that gives a
reader (a potential user, contributor, or someone evaluating whether to
adopt this suite) a complete picture: why it exists, how the pipeline
works end to end, what each of the 8 shipped skills actually does, how to
configure and run it, and what's real today versus planned.

## Structure

1. **Title + pitch + badges** — one line stating what this is, license and
   CI badges.
2. **Why this exists** — the problem (LLMs re-deriving knowledge from raw
   sources on every query instead of compounding it), short, no marketing
   language.
3. **Architecture diagram** — a comprehensive ASCII diagram showing the
   full pipeline: source → OKF wiki → claims/entities/relations →
   (planned) hybrid index → retrieval tools, including the run-directory
   structure (`pdf-runs/`, `research-runs/`) and where each of the 8
   skills sits in the flow.
4. **Quick start** — install, `drt init`, first real command.
5. **Skills, in depth** — one subsection per skill (8 total), each 3-5
   paragraphs: what it does, how it works internally, inputs/outputs,
   why it's a separate skill rather than folded into a neighbor, and a
   concrete usage example. Sourced from each skill's real `SKILL.md` and
   the underlying `src/deep_research_toolkit/*.py` implementation — not
   generic filler.
6. **Configuration** — `.deepresearch.yml` fields, with a real example.
7. **Status and roadmap** — Phase 1 (shipped, tested, pass_rate 1.0 on a
   real fixture) versus Phase 2 (knowledge compiler — designed in
   `docs/decisions/0001-architecture.md`, not yet built). Explicit about
   what's real versus planned so the README doesn't oversell.
8. **Contributing and license** — short, points to `CONTRIBUTING.md` and
   `LICENSE`.

## Humanization pass

After drafting, install and run the real `blader/humanizer` tool
(`npx skills add blader/humanizer`) against the draft to strip AI-writing
tells (significance inflation, formulaic transitions, em-dash overuse,
chatbot openers, vague attributions) — using its documented default
calibration, not a personal voice sample.

## Out of scope

No new badges beyond license/CI (no coverage/PyPI badges — package isn't
published yet). No screenshots/GIFs (this is a CLI/skill suite, not a
GUI app). No separate architecture doc duplication — the README's diagram
is a summary; `docs/decisions/0001-architecture.md` and
`docs/contracts/pdf-ingestion-pipeline.md` remain the detailed reference.
