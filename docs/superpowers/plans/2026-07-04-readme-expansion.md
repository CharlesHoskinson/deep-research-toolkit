# README Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `README.md` into a comprehensive, self-contained ~2,000–2,600-line guide — concepts, all ten skills, a worked end-to-end example, a retrieval reference, a local-model guide, trust guarantees, config, FAQ/glossary, and six ASCII diagrams — with a humanizer pass on all new prose.

**Architecture:** Documentation only; no code, config, or skill changes. `README.md` is rebuilt section by section in outline order (per `docs/superpowers/specs/2026-07-04-readme-expansion-design.md`). Each section is drafted, its diagram(s) render-checked, its prose humanized, its facts cross-checked against the code/`docs/`, then committed. A final pass verifies the table-of-contents anchors, diagram alignment, and cross-links.

**Tech Stack:** Markdown, GitHub-flavored. ASCII diagrams in fenced blocks. The `humanizer` skill for prose. `git` for commits.

## Global Constraints

- **Documentation only.** Do NOT change any file under `src/`, `skills/`,
  `tests/`, `scripts/`, `.github/`, or `pyproject.toml`. Only `README.md` (and,
  if truly needed, an image-free asset) changes.
- **Commit style:** imperative sentence, NO `feat:`/`fix:` prefix, and NEVER a
  `Co-Authored-By` or AI-attribution trailer. Sole author is Charles Hoskinson;
  git identity is already configured for this repo.
- **Punctuation:** use `--` (double hyphen) in README prose, not em dashes.
  Straight quotes only, never curly quotes.
- **Diagrams:** ASCII only, fenced in triple-backtick blocks, box-drawing with
  `+ - | v` (no Unicode box characters), and **≤ 76 columns wide** so they don't
  overflow a narrow terminal. Every diagram must render left-aligned and
  vertically consistent — check it in a fixed-width view before committing.
- **Humanizer:** after drafting each new or rewritten *prose* section, run the
  `humanizer` skill over that section's prose and apply its fixes before
  committing. Do not run it over code blocks, tables, or diagrams.
- **Accuracy gate (bind every task):** every factual claim must match the repo
  at this commit. Authoritative sources: `docs/contracts/knowledge-compiler.md`
  (the 8 tool contracts, RRF k=60, role-routed stack, serving notes),
  `docs/contracts/pdf-ingestion-pipeline.md` (PDF stage outputs),
  `docs/contracts/okf-frontmatter.md` (OKF), `docs/decisions/0001-architecture.md`
  and `0002-knowledge-compiler.md`, `docs/pipeline-findings.md`, the ten
  `skills/*/SKILL.md`, `src/deep_research_toolkit/config.py` (`ROLE_DEFAULTS`,
  config keys), and `src/deep_research_toolkit/compiler/tools.py` (tool
  signatures). Suite version is **0.2.0**; there are **ten skills**. For volatile
  numbers (e.g. exact test count) use robust phrasing ("a fast unit-test suite
  that runs on every push"), never a pinned figure.
- **The worked-example numbers** (Task 5) are the real Proof-of-Stake / snail
  run: **5 web sources -> ~105 chunks -> 103 claims / 61 entities / 66 relations
  -> a compiled DuckDB+LanceDB index -> a grounded thesis**. Present them as an
  illustrative real run.
- **Preserve accurate existing prose.** The current README already describes all
  ten skills correctly; fold that prose in and expand it rather than discarding
  it. Read the current `README.md` before starting.

---

## File map

- Modify (throughout): `README.md` — the single deliverable, rebuilt section by
  section in outline order.
- Read-only references (never edited): the `docs/` and `skills/` files listed in
  the accuracy gate above.

There are no test files (documentation). "Verification" in each task means:
render-check any diagram, run the humanizer on the prose, and cross-check the
stated facts against the named source file(s).

---

## Task 1: Front matter, table of contents, and "Why this exists"

**Files:** Modify `README.md` (top of file).

- [ ] **Step 1: Read the current README and preserve the header.** Keep the
  existing `# deep-research-toolkit` title and the two badges (MIT license, CI).
  Keep the existing tagline paragraph if still accurate; tighten to one
  paragraph.

- [ ] **Step 2: Add a table of contents** immediately after the tagline, with
  anchor links to every top-level section that will exist (list them from the
  outline: Why this exists, Core ideas, How it fits together, Quick start, A
  worked example, The skills, The retrieval tools, Running local models, What
  the pipeline guarantees, Configuration, Verification and testing, Status and
  roadmap, FAQ and glossary, Contributing, License). Use GitHub anchor
  conventions (lowercase, spaces -> hyphens).

- [ ] **Step 3: Write "Why this exists"** (EXPAND to 3-4 paragraphs). Cover:
  (a) ask an LLM the same research question twice and it redoes all the work --
  re-fetch, re-read, re-derive; nothing sticks; (b) that's fine for a one-off,
  a bad foundation for anything that compounds (a project, an internal KB, a
  standards comparison revisited for months); (c) this toolkit writes every
  fetch, PDF, and claim down once, in a durable format, with the evidence
  attached, so the tenth question is answered from what's already there instead
  of re-scraping; (d) the wiki isn't the knowledge -- it's the source corpus;
  the claims-with-citations built from it are what an agent reasons over. Draw
  from the current README's "Why this exists" and `docs/pipeline-findings.md`.

- [ ] **Step 4: Humanizer pass** on the new prose (Steps 1, 3). Apply fixes.

- [ ] **Step 5: Accuracy check** -- the TOC anchors match the section titles
  you'll create; the tagline says "ten skills" and "web retrieval, PDF
  ingestion, and knowledge compilation."

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "Rebuild README front matter: tagline, table of contents, why-this-exists"
```

---

## Task 2: Core ideas -- the mental model (+ Diagram 2)

**Files:** Modify `README.md` (add `## Core ideas` after "Why this exists").

**Interfaces:** Introduces the vocabulary later sections reuse verbatim -- "the
three layers," "the verbatim-quote gate," "evidence_ref," "OKF," "producer,"
"run directory." Use these exact terms consistently for the rest of the README.

- [ ] **Step 1: Write the section intro** -- one paragraph: most of the system
  follows from a few ideas; understanding these makes every skill obvious.

- [ ] **Step 2: Write "Three layers"** (2-3 paragraphs). The durable corpus
  (markdown wiki + per-run JSONL, git-tracked, the audit trail); the derived
  index (DuckDB + LanceDB, git-ignored, always a full rebuild, so it can never
  silently disagree with the files); the judgment layer (agent or local model)
  trusted only to *propose* claims, never to certify them. Source:
  `docs/pipeline-findings.md` "What the pipeline is," `docs/decisions/0001`.

- [ ] **Step 3: Insert Diagram 2 (three-layer architecture).** Use exactly this
  ASCII (verify alignment):

```
+---------------------------------------------------------------+
|  JUDGMENT LAYER     agent (in-session)  or  local model       |
|  proposes claims; never certifies them                        |
+-------------------------------+-------------------------------+
                                |  writes (only if gated)
                                v
+---------------------------------------------------------------+
|  DURABLE CORPUS   (git-tracked -- the audit trail)            |
|  knowledge_base/*.md   pdf-runs/<id>/   research-runs/<id>/    |
|  claims + entities + relations, each with a verbatim quote    |
+-------------------------------+-------------------------------+
                                |  compiled (full rebuild)
                                v
+---------------------------------------------------------------+
|  DERIVED INDEX    (git-ignored -- a rebuildable cache)        |
|  DuckDB (full-text + graph)   +   LanceDB (vectors)           |
+-------------------------------+-------------------------------+
                                |  queried by
                                v
                8 cheap, deterministic retrieval tools
```

- [ ] **Step 4: Write "The verbatim-quote gate"** (2 paragraphs). The one hard
  invariant: a claim is admitted only if its supporting quote is an exact
  substring of the source it cites. The check is deliberately dumb, runs the
  same way regardless of which model produced the claim, and is enforced at
  three points (extraction, dossier composition, the eval harness) from one
  shared `common.verbatim` function. Consequence: a weak or hallucinating model
  can only under-produce, never fabricate a citation that looks real. Source:
  `src/deep_research_toolkit/common/verbatim.py`, `docs/pipeline-findings.md`.

- [ ] **Step 5: Write "evidence_ref: two producers, one shape"** (1-2
  paragraphs), then insert Diagram 3 (evidence_ref normalization):

```
   PDF claim evidence              WEB claim evidence
   { node_id, quote, page }        { locator, quote, url }
            \                            /
             \   normalize_evidence()   /   at index time only;
              \      (the compiler)    /    on-disk files untouched
               v                      v
         EvidenceRef { producer, source_id, locator,
                       quote, page?, url? }
                          |
                          v
              one uniform shape every retrieval tool reads
```

- [ ] **Step 6: Write "OKF and run directories"** (1 paragraph). OKF = Open
  Knowledge Format: markdown + YAML frontmatter, one concept per file,
  cross-linked with relative links that are the graph's edges. A "run directory"
  (`pdf-runs/<id>/` or `research-runs/<id>/`) is the per-document working set,
  git-tracked as the auditable record. A "producer" is web or PDF. Source:
  `docs/contracts/okf-frontmatter.md`, `docs/contracts/pdf-ingestion-pipeline.md`.

- [ ] **Step 7: Humanizer pass** on all prose in this section (not the diagrams).

- [ ] **Step 8: Accuracy check** -- `EvidenceRef` fields match
  `compiler/schema.py`; the three gate checkpoints match `common/verbatim.py`
  usage; DuckDB/LanceDB roles correct.

- [ ] **Step 9: Commit**

```bash
git add README.md
git commit -m "Add README core-ideas section: three layers, the verbatim gate, evidence_ref, OKF"
```

---

## Task 3: How it fits together (redraw Diagram 1)

**Files:** Modify `README.md` (rework the existing `## How it fits together`).

- [ ] **Step 1: Write the intro** (2 paragraphs): the toolkit is a pipeline of
  phases -- acquire, chunk, extract, compile, retrieve, synthesize -- and the
  key property is that the LLM only touches three of them (extract, wiki-write,
  synthesize); everything else is deterministic Python or a local embedding
  model. Note each PDF gets its own `<pdf_runs_dir>/<document_id>/` and it's
  git-tracked on purpose (the auditable record, not scratch).

- [ ] **Step 2: Replace the current diagram with Diagram 1 (master pipeline).**
  Use exactly this ASCII (verify alignment, ≤76 cols):

```
                       .deepresearch.yml
             (topic, paths, features, llm.roles)
                              |
          +-------------------+-------------------+
          | WEB                               PDF |
          v                                       v
  research-knowledge-graph              pdf-ingest-router
  fetch + chunk (Scrapling)             classify + route
          |                    pdf-to-canonical-markdown (Docling)
          |                    pdf-layout-provenance
          |                    canonical-markdown-to-llm-nodes
          v                                       v
   research-runs/<id>/                     pdf-runs/<id>/
          |            chunks.jsonl (structure-aware)
          +-------------------+-------------------+
                              v
              knowledge-extraction        [JUDGMENT: LLM]
        claims + entities + relations, verbatim-gated
                              v
              llm-wiki-writer             [JUDGMENT: LLM]
           merge into knowledge_base/ (OKF pages)
                              v
              knowledge-compiler          [deterministic]
        DuckDB (FTS + graph)  +  LanceDB (vectors)
                              v
              retrieval-planner           [deterministic]
      8 tools, RRF fusion, compose_dossier (verbatim gate)
                              v
                     evidence dossier
             claims + citations, ready to answer from
```

- [ ] **Step 3: Write the walk-through** (2-3 paragraphs) naming each phase and
  what it hands to the next, and stating which are deterministic vs judgment.

- [ ] **Step 4: Humanizer pass** on the prose.

- [ ] **Step 5: Accuracy check** -- stage names/order match
  `docs/contracts/pdf-ingestion-pipeline.md` and the skills list.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "Rebuild README how-it-fits-together with a cleaner master pipeline diagram"
```

---

## Task 4: Quick start (expand)

**Files:** Modify `README.md` (rework `## Quick start`).

- [ ] **Step 1: Write the install + tiers paragraph.** `pip install
  "deep-research-toolkit[pdf]"` (or `[web]`, `[compiler]`, `[full]`), then
  `drt init`. Explain the four extras and what each pulls in (from
  `pyproject.toml`: web=scrapling; pdf=docling/pypdf/pdfplumber;
  compiler=duckdb/lancedb/sentence-transformers/openai). Mention `scrapling
  install` for the web tier's Playwright browsers, and `docs/environment.md` for
  Docling/Playwright first-run downloads.

- [ ] **Step 2: Explain `drt init`** (1-2 paragraphs): asks the project's topic
  and where the knowledge base lives, writes `.deepresearch.yml`, scaffolds the
  KB, and copies skills into `.claude/skills/` and `.agents/skills/` for Claude
  Code and Codex. Keep the current README's explanation, expanded.

- [ ] **Step 3: Keep and expand the by-hand PDF command sequence** already in
  the README (classify -> convert -> provenance -> chunk -> run_eval), noting
  where the LLM step (knowledge-extraction) fits.

- [ ] **Step 4: Add a "first web research + query" mini-sequence** in prose:
  ask an agent to "research X for the knowledge base," or by hand fetch a
  source, `start_research_run.py`, extract, `compile.py`, then `query.py
  search-claims "..."`. Cross-reference Task 5 for the full worked example.

- [ ] **Step 5: Humanizer pass** on the prose.

- [ ] **Step 6: Accuracy check** -- extras and their packages match
  `pyproject.toml`; script names match `skills/*/scripts/`.

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "Expand README quick start: install tiers, drt init, first web/PDF/query runs"
```

---

## Task 5: A worked example, end to end (+ Diagram 6)

**Files:** Modify `README.md` (add `## A worked example, end to end`).

- [ ] **Step 1: Write the framing** (1 paragraph): to make the pipeline concrete,
  here is a real run that turned five Wikipedia sources on proof-of-stake into a
  queryable, cited knowledge base and a grounded thesis, using a fully local
  model stack.

- [ ] **Step 2: Narrate the run** (4-5 paragraphs), each phase with its real
  numbers: (a) fetch 5 sources -> `research-runs/<id>/` with `source.md` +
  `chunks.jsonl`, ~105 chunks total; (b) local-model extraction produced **103
  claims, 61 entities, 66 relations**, with the verbatim gate dropping the
  non-verbatim ones; (c) `compile.py` built the DuckDB+LanceDB index using a
  Qwen embedding model; (d) `query.py search-claims` and `compose-dossier`
  returned grounded, verbatim-quoted results; (e) a reasoning model synthesized
  a thesis where every checked number traced back to an exact source quote.
  State that the extraction ran in minutes on a fast instruct model, and that
  the same run on a reasoning model was far slower -- motivating the role-routed
  stack (Task 8/Task 10).

- [ ] **Step 3: Insert Diagram 6 (end-to-end data lifecycle):**

```
  a source (a URL or a PDF)
        |  fetch / convert
        v
  source.md  /  canonical.md
        |  chunk (structure-aware: one node per heading, table, figure)
        v
  chunks.jsonl
        |  extract   [JUDGMENT + verbatim gate]
        v
  claims.jsonl   entities.jsonl   relations.jsonl
        |  compile (+ embed)
        v
  DuckDB + LanceDB index
        |  query -> compose_dossier
        v
  an evidence dossier -> a grounded, cited deliverable
```

- [ ] **Step 4: Humanizer pass** on the prose.

- [ ] **Step 5: Accuracy check** -- numbers match the Global Constraints
  worked-example figures; phase names match the skills.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "Add README worked example: a real proof-of-stake research run end to end"
```

---

## Task 6: The skills -- overview + web research

**Files:** Modify `README.md` (start `## The skills`).

- [ ] **Step 1: Write the "why many small skills" overview** (2 paragraphs).
  Ten skills; seven form the PDF pipeline, one handles web research, two form the
  compiler layer. Each is deliberately small so only the running stage's
  instructions load into context (progressive disclosure), keeping any
  conversation's context proportional to the work. Source: current README "The
  skills" intro, `docs/decisions/0001`.

- [ ] **Step 2: Write `### research-knowledge-graph`** (4-5 paragraphs): the
  web-research half, used directly and repeatedly. Wraps Scrapling (http default,
  stealth on a bot-challenge/403), stores findings as OKF pages, and runs
  Karpathy's LLM-wiki loop as `ingest` / `query` / `lint`. The `status` field
  (`seed`/`researched`/`stale`/`draft`/`conflicted`). Reads `.deepresearch.yml`
  for scope, never guessing. The new claim-extraction step: `start_research_run.py`
  scaffolds `research-runs/<source_id>/` mirroring a PDF run so the compiler
  indexes web and PDF claims uniformly. Source: `skills/research-knowledge-graph/
  SKILL.md`, `skills/research-knowledge-graph/references/*`.

- [ ] **Step 3: Humanizer pass** on the prose.

- [ ] **Step 4: Accuracy check** against the SKILL.md.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "Add README skills overview and the research-knowledge-graph deep dive"
```

---

## Task 7: The skills -- the seven-stage PDF pipeline

**Files:** Modify `README.md` (continue `## The skills`).

Write a `###` subsection for each of the seven PDF stages, 3-4 paragraphs each,
covering what it does, how it's invoked, the design rationale, its on-disk
output, and what it deliberately does not do. Preserve and expand the current
README's already-accurate prose for these. Source: each
`skills/<name>/SKILL.md` and `docs/contracts/pdf-ingestion-pipeline.md`.

- [ ] **Step 1:** `### pdf-ingest-router` -- five real signals (pypdf/pdfplumber)
  into six routes; computes the stable `document_id`; only Docling wired up but
  records the *ideal* backend so the gap is visible in `classification.json`.
- [ ] **Step 2:** `### pdf-to-canonical-markdown` -- Docling to `canonical.md` +
  `docling_raw.json`; the single-retry-on-transient-network fix; Marker/MinerU/
  Unstructured documented-but-not-wired.
- [ ] **Step 3:** `### pdf-layout-provenance` -- recovers page/section_path/bbox
  per structural unit into `provenance.jsonl`; the numbering-depth heuristic for
  `section_path`; the honest `confidence: 1.0` placeholder (no OCR yet).
- [ ] **Step 4:** `### canonical-markdown-to-llm-nodes` -- structure-aware
  chunking into `chunks.jsonl` (one node per heading section / table / figure),
  the long-section split, and the intentionally-empty `summary`/`keywords`/
  `entities` fields.
- [ ] **Step 5:** `### knowledge-extraction` -- deterministic tables/figures plus
  LLM-written claims/entities/relations; the evidence rule (verbatim quotes);
  the "short list of solid claims beats a long shaky one" note.
- [ ] **Step 6:** `### llm-wiki-writer` -- merge claims into OKF pages, the
  `draft`/`conflicted` statuses, `wiki_pages_written.json`, and the lint at the
  end.
- [ ] **Step 7:** `### rag-eval-harness` -- the six mechanical checks (now
  including the chunk-based verbatim check), skip-don't-fail on partial runs, and
  the deliberately-excluded Q/A retrieval probes.
- [ ] **Step 8: Humanizer pass** over all seven subsections' prose.
- [ ] **Step 9: Accuracy check** each subsection against its SKILL.md and the
  contract doc (note that `rag-eval-harness`'s verbatim check is chunk-based per
  the recent gate unification).
- [ ] **Step 10: Commit**

```bash
git add README.md
git commit -m "Add README deep dives for the seven-stage PDF pipeline"
```

---

## Task 8: The skills -- the compiler layer + the local backend

**Files:** Modify `README.md` (finish `## The skills`).

- [ ] **Step 1:** `### knowledge-compiler` (3-4 paragraphs) -- compiles
  `knowledge_base/` + `pdf-runs/` + `research-runs/` into DuckDB (FTS + graph) +
  LanceDB (vectors); full rebuild per run; git-ignored `index_dir`; the
  `evidence_ref` normalization at index time. Source: `skills/knowledge-compiler/
  SKILL.md`, `docs/contracts/knowledge-compiler.md`.
- [ ] **Step 2:** `### retrieval-planner` (3-4 paragraphs) -- eight deterministic
  LLM-free tools; RRF fusion; `compose_dossier`'s verbatim hard gate;
  `find_contradictions` as mechanical candidates only. Point forward to the full
  reference in "The retrieval tools" (Task 9).
- [ ] **Step 3:** `### The optional local LLM backend` (3-4 paragraphs) -- the
  `agent` default (the in-session agent is the extraction step) vs opt-in
  `local` (an OpenAI-compatible endpoint); programmatic `extract_claims.py` with
  the auto-drop gate. Point forward to "Running local models" (Task 10).
- [ ] **Step 4: Humanizer pass** on the prose.
- [ ] **Step 5: Accuracy check** against the SKILL.md files and
  `docs/contracts/knowledge-compiler.md`.
- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "Add README deep dives for knowledge-compiler, retrieval-planner, and the local backend"
```

---

## Task 9: The retrieval tools reference (+ Diagram 4)

**Files:** Modify `README.md` (add `## The retrieval tools`).

- [ ] **Step 1: Write the intro** (2 paragraphs): the eight tools are cheap and
  deterministic (no LLM at query time), so an agent can call them freely and
  compose them; judgment happens at compile/extraction time, not here. Source:
  `docs/contracts/knowledge-compiler.md`, `src/deep_research_toolkit/compiler/
  tools.py`.

- [ ] **Step 2: Insert Diagram 4 (retrieval / RRF fusion):**

```
                        query string
                             |
             +---------------+---------------+
             v                               v
      DuckDB FTS / BM25              LanceDB vector search
      (lexical, keyword)             (semantic, embeddings)
             |                               |
             v                               v
        ranked list A                   ranked list B
              \                             /
               v   reciprocal rank fusion  v   (k = 60)
                      fused ranking
                             |
      +---------+-----------+-----------+------------------+
      v         v           v           v                  v
  search_    search_    get_entity   neighbors    compose_dossier
   wiki      claims                              (verbatim gate:
                                                  included / rejected)
```

- [ ] **Step 3: Add the tool table** -- the eight tools with a one-line contract
  each (name, args, what it returns), copied faithfully from `tools.py` /
  `docs/contracts/knowledge-compiler.md`: `search_wiki(query, k)`,
  `read_page(path)`, `search_claims(query, k, producer)`,
  `get_entity(name_or_id)`, `neighbors(entity, depth)`,
  `get_sources(page|claim)`, `find_contradictions()`,
  `compose_dossier(query|claim_ids, k)`.

- [ ] **Step 4: Write `compose_dossier` + the gate** (2 paragraphs): returns
  `included` / `rejected`; a claim reaches `included` only if every quote is a
  verbatim substring of its chunk; `--format md` renders a self-citing dossier
  (claims + quotes + sources inline). And `find_contradictions` (1 paragraph):
  mechanical `(subject, predicate)`-with-conflicting-objects candidates plus
  `conflicted` pages; confirmation is an agent step.

- [ ] **Step 5: Humanizer pass** on the prose.

- [ ] **Step 6: Accuracy check** -- the eight signatures match `tools.py`; RRF
  k=60; `--format md` exists in `skills/retrieval-planner/scripts/query.py`.

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "Add README retrieval-tools reference with the RRF fusion diagram"
```

---

## Task 10: Running local models -- the role-routed stack (+ Diagram 5)

**Files:** Modify `README.md` (add `## Running local models`).

- [ ] **Step 1: Write the intro** (2 paragraphs): the pipeline touches an LLM in
  only a few phases, and they want different models -- extraction is high-volume
  and wants a fast instruct model; synthesis is one-shot and wants a reasoning
  model. `llm.roles` routes each phase; `get_backend(config, role=...)` resolves
  it; a single-model config still works. Source: `config.py` `ROLE_DEFAULTS`,
  `docs/contracts/knowledge-compiler.md` "Role-routed model stack."

- [ ] **Step 2: Insert Diagram 5 (role-routed model stack):**

```
  PHASE                     ROLE (llm.roles)      MODEL KIND
  ----------------------    ------------------    -----------------
  extract  (high volume)    extract               fast instruct
  wiki-write                wiki_write            mid/large instruct
  contradiction review      conflict_adjudicate   reasoning
  synthesize / thesis       synthesize            reasoning
  code-agent work           code_agent            agentic-coding
  ----------------------    ------------------    -----------------
  embeddings                llm.embedding_model   Ollama or sentence-
                                                  transformers

  get_backend(config, role="extract") -> the model for that phase
  any role left unset      -> falls back to the flat llm.local model
```

- [ ] **Step 3: Write the role table** in prose/table form matching
  `ROLE_DEFAULTS` (extract: non-thinking, json, small budget; synthesize:
  thinking, large budget; etc.), and a short YAML `llm.roles` example (copy the
  shape from `docs/contracts/knowledge-compiler.md`).

- [ ] **Step 4: Write "the non-thinking finding"** (2 paragraphs) -- the
  operational lesson: a model's non-thinking switch is a promise about the model,
  not the serving stack; both a reasoning model (Ornith-1.0-9B) and a hybrid
  model (qwen3.5:9b) ignored `think:false`/`/no_think` under the tested Ollama
  build and reasoned to the token ceiling with no output, so the `extract` role
  needs a true instruct model (e.g. qwen2.5:7b-instruct). Source:
  `docs/contracts/knowledge-compiler.md` "Serving a local reasoning model,"
  `docs/pipeline-findings.md`.

- [ ] **Step 5: Write "serving + embeddings + validation"** (2 paragraphs) --
  the Ollama chat-template fix (stock GGUFs may ship no template; build one with
  a ChatML `TEMPLATE`), the generous `max_tokens`, the temperature caution;
  embeddings route by name shape (`qwen3-embedding:4b` -> endpoint, else
  sentence-transformers, LanceDB infers the dim); `scripts/validate-local-llm.py`
  measures recovery against the reference fixture.

- [ ] **Step 6: Humanizer pass** on the prose.

- [ ] **Step 7: Accuracy check** -- roles/defaults match `ROLE_DEFAULTS`; the
  findings match the contract doc; embedding routing matches `compiler/embed.py`.

- [ ] **Step 8: Commit**

```bash
git add README.md
git commit -m "Add README local-models guide: the role-routed stack and the serving findings"
```

---

## Task 11: What the pipeline guarantees

**Files:** Modify `README.md` (add `## What the pipeline guarantees`).

- [ ] **Step 1: Write "the gate, in depth"** (2 paragraphs): the verbatim check
  is re-derived from disk independently at extraction, dossier composition, and
  the eval harness, all through the one `common.verbatim` function, so a bug or
  a hand-edit at any one stage can't launder an unverified claim into "included."
  A quote that spans two structural units within a chunk is treated consistently
  by all three (the gates were unified so they agree).

- [ ] **Step 2: Write "what it does and doesn't protect"** (2-3 paragraphs):
  it guarantees a quote is present verbatim in the cited chunk (so a weak model
  can only under-produce, never fabricate). It does NOT guarantee the quote
  supports the claim, that the source is *true* (a source's own errors pass
  through as grounded claims), or that a verbatim quote is a clean, un-truncated
  sentence. State these plainly. Source: `docs/pipeline-findings.md` "honest
  limits," and the "gaps closed" update.

- [ ] **Step 3: Humanizer pass** on the prose.

- [ ] **Step 4: Accuracy check** against `docs/pipeline-findings.md` and
  `common/verbatim.py`.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "Add README section on what the pipeline guarantees (and what it does not)"
```

---

## Task 12: Configuration (expand)

**Files:** Modify `README.md` (rework `## Configuration`).

- [ ] **Step 1: Keep and expand the `.deepresearch.yml` walkthrough.** Show the
  full annotated YAML: `version`, `knowledge_base` (path, pdf_runs_dir,
  research_runs_dir, index_dir), `topic` (name, scope_hint, tags), `features`
  (web_research, pdf_ingestion, knowledge_compiler), `llm` (provider,
  embedding_model, `roles`, `local`). Copy the shape faithfully from
  `src/deep_research_toolkit/cli.py`'s `DEFAULT_YAML_TEMPLATE` and `config.py`.

- [ ] **Step 2: Write a paragraph on discovery** -- found by walking up from cwd
  like `.git`; nothing hardcodes a topic or path; `drt init` writes a starter
  and sets `features.*` from the chosen tier. Note the `schema_version` fields
  and point to `docs/contracts/schema-versions.md`.

- [ ] **Step 3: Write a short paragraph on `llm.roles`** cross-linking Task 10.

- [ ] **Step 4: Humanizer pass** on the prose.

- [ ] **Step 5: Accuracy check** -- every key matches `config.py` / the init
  template; the `llm.roles` keys match `ROLE_DEFAULTS`.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "Expand README configuration: full .deepresearch.yml walkthrough incl. llm.roles"
```

---

## Task 13: Verification & testing + Status and roadmap

**Files:** Modify `README.md` (add `## Verification and testing`, rework
`## Status and roadmap`).

- [ ] **Step 1: Write "Verification and testing"** (2-3 paragraphs): the fast
  tier runs on every push (unit + light integration, no heavy deps -- torch-free
  via an injected fake embedder); the heavy tier (weekly / manual) runs real
  Docling and real embeddings; the reference fixtures under `tests/fixtures/`;
  how to check a run in practice (`rag-eval-harness` on a run dir;
  `validate-local-llm.py` for a local model). Use robust phrasing for the test
  count (no pinned number). Source: `.github/workflows/ci.yml`,
  `docs/pipeline-findings.md`.

- [ ] **Step 2: Rework "Status and roadmap"** (EXPAND): built-and-tested (both
  producer stacks, the compiler layer, the CLI, the role-routed local backend,
  the dual plugin manifests); designed-but-deferred (incremental compilation, a
  reranker stage, GraphRAG-style community detection, an MCP query server, a
  multi-package split) -- point to `docs/decisions/0002` for the reasoning.
  Source: current README, `docs/decisions/0002`, `CHANGELOG.md`.

- [ ] **Step 3: Humanizer pass** on the prose.

- [ ] **Step 4: Accuracy check** against CI, the ADRs, and the CHANGELOG.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "Add README verification-and-testing section; expand status and roadmap"
```

---

## Task 14: FAQ, glossary, contributing, license

**Files:** Modify `README.md` (add `## FAQ`, `## Glossary`, rework
`## Contributing`, add `## License`).

- [ ] **Step 1: Write the FAQ** (6-8 Q/A, 1-3 sentences each): how is this
  different from a generic RAG pipeline? (the verbatim gate + files-as-corpus +
  two-producers-one-graph); do I need a GPU? (only for the optional local model
  tier; the agent default and the deterministic tools need none); which local
  model should I use? (a fast instruct model for extraction, a reasoning model
  for synthesis -- see the role-routed stack); is my data sent anywhere? (no --
  the corpus, index, and embeddings are local; only if you point a provider at a
  remote endpoint); can I use it for PDFs and the web together? (yes -- one
  graph, normalized at index time); does it work in Codex as well as Claude
  Code? (yes -- one shared `skills/` tree, dual manifests). Source:
  `docs/pipeline-findings.md`, the ADRs.

- [ ] **Step 2: Write the glossary** -- short definitions for: OKF, evidence_ref,
  dossier, RRF, provenance, run directory, producer, chunk/node, claim, the
  verbatim gate, role (llm.roles), the compiler. Draw definitions from the
  contract docs.

- [ ] **Step 3: Rework Contributing** (EXPAND): dev setup (`pip install -e
  ".[dev,pdf]" ...`), the fast/heavy test tiers, what to re-run when you change
  an on-disk format (`sync-skill-templates.py`, the manifest/template sync
  guards), and where the contracts live. Source: `CONTRIBUTING.md`,
  `scripts/*`, the CI.

- [ ] **Step 4: Add a one-line License section** -- MIT, per `LICENSE`.

- [ ] **Step 5: Humanizer pass** on the FAQ and glossary and contributing prose.

- [ ] **Step 6: Accuracy check** -- FAQ answers match the design; glossary terms
  match the contracts; contributing commands match `CONTRIBUTING.md`/CI.

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "Add README FAQ and glossary; expand contributing; add license section"
```

---

## Task 15: Final pass -- TOC, diagrams, accuracy, links

**Files:** Modify `README.md` (whole-file review).

- [ ] **Step 1: Verify the table of contents.** Every TOC anchor resolves to a
  real heading; every top-level section appears in the TOC; order matches.

- [ ] **Step 2: Render-check all six diagrams.** Open each fenced block and
  confirm it renders left-aligned, columns ≤ 76, no wrapped lines, arrows and
  boxes vertically consistent. Fix any misalignment.

- [ ] **Step 3: Full accuracy read-through** against the Global Constraints
  accuracy gate: "ten skills," version 0.2.0, the eight tool names/args, the
  role names, the worked-example numbers (103 / 61 / 66, 5 sources, ~105
  chunks), config keys, and no pinned brittle test count. Fix any drift.

- [ ] **Step 4: Link check.** Every `docs/...` and `skills/...` cross-link points
  at a file that exists; run:

```bash
python - <<'PY'
import re, pathlib
txt = pathlib.Path("README.md").read_text(encoding="utf-8")
missing = []
for m in re.findall(r"\]\((docs/[^)]+|skills/[^)]+|[A-Z]+\.md)\)", txt):
    if not pathlib.Path(m).exists():
        missing.append(m)
print("missing links:", missing or "none")
PY
```
Expected: `missing links: none`.

- [ ] **Step 5: Placeholder scan.** Grep the README for `TODO`, `TBD`,
  `FIXME`, `XXX`, `...` used as a placeholder; there should be none (ellipses
  inside example JSON/quotes are fine).

- [ ] **Step 6: Final humanizer sweep.** Re-read the whole README for AI tells
  the per-section passes missed (repeated openings, rule-of-three runs,
  em dashes -- should be `--`); fix inline.

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "Final README pass: verify TOC anchors, diagram alignment, links, and accuracy"
```

---

## Self-review notes (for the executor)

- **Spec coverage:** every spec section maps to a task -- front matter/why (T1),
  core ideas + D2/D3 (T2), how-it-fits + D1 (T3), quick start (T4), worked
  example + D6 (T5), the ten skills (T6-T8), retrieval reference + D4 (T9),
  local models + D5 (T10), guarantees (T11), configuration (T12),
  verification + roadmap (T13), FAQ/glossary/contributing/license (T14), final
  pass (T15). All six diagrams are placed (D1 T3, D2 T2, D3 T2, D4 T9, D5 T10,
  D6 T5).
- **No placeholders:** the six diagrams are given as literal ASCII; prose
  sections are given as concrete content briefs with exact facts and named
  sources (writing the full final prose in the plan would duplicate the README).
  The humanizer + accuracy steps are concrete, per section.
- **Consistency:** the vocabulary introduced in T2 ("three layers," "verbatim
  gate," "evidence_ref," "producer," "run directory") is reused by name in every
  later task; the eight tool names in T9 match T8's forward reference; the role
  names in T10 match T12's config keys and T5's worked example.
- **Documentation only:** no task touches `src/`, `skills/`, `tests/`, CI, or
  `pyproject.toml`; the only executable step is a read-only link-check script.
