# deep-research-toolkit

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![CI](https://github.com/CharlesHoskinson/deep-research-toolkit/actions/workflows/ci.yml/badge.svg)

A deep-research skill suite for **Claude Code** and **Codex**, at version
0.2.0: ten skills for web retrieval, PDF ingestion, and knowledge
compilation into an evidence-backed knowledge base an agent can query,
cite, and keep building on instead of starting over on every question.

**Contents**

- [Why this exists](#why-this-exists)
- [Core ideas](#core-ideas)
- [How it fits together](#how-it-fits-together)
- [Quick start](#quick-start)
- [A worked example](#a-worked-example-end-to-end)
- [The skills](#the-skills)
- [The retrieval tools](#the-retrieval-tools)
- [Running local models](#running-local-models)
- [What the pipeline guarantees](#what-the-pipeline-guarantees)
- [Configuration](#configuration)
- [Verification and testing](#verification-and-testing)
- [Status and roadmap](#status-and-roadmap)
- [FAQ and glossary](#faq-and-glossary)
- [Contributing](#contributing)
- [License](#license)

## Why this exists

Ask an LLM the same research question twice, a week apart, and it does the
same work twice: fetch the same pages, re-read the same PDF, re-derive the
same conclusions from scratch. Nothing it learned the first time sticks
around. That's fine for a one-off question. It's a bad foundation for
anything that compounds: a research project, an internal knowledge base, a
standards comparison that gets revisited for months.

This toolkit fixes that specific problem. Every fetch, every PDF, every
claim gets written down once, in a durable format, with the evidence
attached, so the tenth question about a topic is answered by reading
what's already there instead of re-scraping the internet. Durable means
plain files: markdown wiki pages and per-run JSONL records, kept in git,
treated as the audit trail. The search index built from them (DuckDB for
text and graph queries, LanceDB for vectors) is disposable and always
rebuilt from scratch, so there is no state in which the index and the
files can silently disagree.

The wiki isn't the knowledge, though. It's the source corpus. What an
agent actually reasons over is the layer built on top of it: claims, each
one tied to a verbatim quote from a named source. A claim is admitted only
if its quote is an exact substring of the source it cites, and that check
is mechanical -- a dumb substring comparison, not a model grading its own
work. So the question "where did this come from?" always has an answer you
can verify by looking, rather than a model's say-so.

All three layers of this are built and tested today: a web-research skill,
a seven-stage PDF ingestion pipeline, and a knowledge compiler that turns
everything the first two produced into a queryable hybrid index with a
small set of retrieval tools on top. The two producers write into the same
knowledge base, using the same format, so it doesn't matter whether a fact
came from a webpage or a whitepaper: it ends up in one place, checkable
the same way, and the compiler indexes both sides into one graph.

## Core ideas

Most of this toolkit follows from a small number of ideas. Once you have
them, each of the ten skills described later reads as a consequence rather
than a separate thing to learn -- of course the chunker doesn't guess at
summaries, of course a weak local model can't corrupt the corpus. This
section defines the vocabulary the
rest of the README reuses -- the three layers, the verbatim-quote gate,
evidence_ref, OKF, producer, run directory -- so that when a later section
says "the gate rejects it," you already know which gate and why it exists.

### The three layers

The first layer is the durable corpus: markdown wiki pages plus per-run
JSONL files, all kept in git. This is the audit trail. Every claim, entity,
and relation the system knows lives here as a plain file, each claim
carrying a verbatim quote from a named source, and "how do we know this?"
is answered by opening the file and reading, not by querying anything. If
every other layer were deleted, the corpus alone would still be a complete
and inspectable record.

The second layer is the derived index: DuckDB for full-text and graph
queries, LanceDB for vectors. It's git-ignored, and it's rebuilt from
scratch on every compile -- never patched incrementally. That sounds
wasteful until you notice what it buys: there is no state in which the
index and the files can silently disagree. The index is either current or
stale, and stale is fixed by recompiling, which takes seconds at the
per-project scale this toolkit serves. Nothing ever has to patch the index
to match a changed file, so a whole class of cache-invalidation bugs
can't exist here.

The third layer is the judgment layer: an LLM, whether the in-session
agent reading a SKILL.md or a local model behind an OpenAI-compatible
endpoint. It's trusted to do exactly one kind of work -- propose. It
proposes claims, proposes entity merges, proposes wiki prose. It is never
trusted to certify its own output, because certification is a mechanical
check that no model gets to perform on itself. That division is what the
next idea enforces.

```
+---------------------------------------------------------------+
|  JUDGMENT LAYER     agent (in-session)  or  local model       |
|  proposes claims; never certifies them                        |
+-------------------------------+-------------------------------+
                                |  writes (only if gated)
                                v
+---------------------------------------------------------------+
|  DURABLE CORPUS   (git-tracked -- the audit trail)            |
|  knowledge_base/*.md   pdf-runs/<id>/   research-runs/<id>/   |
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

### The verbatim-quote gate

The one hard invariant: a claim is admitted only if its supporting quote is
an exact substring of the source text it cites -- character for character,
contiguous, no normalization. The check is deliberately dumb. It's a Python
`in` test, it runs identically no matter which model produced the claim,
and it's enforced at three points -- claim extraction, dossier composition,
and the eval harness -- all calling the same `verbatim_ok` function in
`common/verbatim.py`. One shared function matters more than it might seem:
because every stage resolves the source text the same way (re-read from the
run directory's `chunks.jsonl` on disk, independent of the compiled index),
a claim admitted at extraction can never be rejected later because two
stages disagreed about what "the source" was.

The consequence is what makes the whole design safe to operate. A weak or
hallucinating model can only under-produce -- fewer of its proposed claims
survive the gate -- but it cannot fabricate a citation that looks real,
because a paraphrased or invented quote fails the substring test no matter
how plausible it reads. That's why the optional local-model backend exists
at all: swapping a frontier model for a 9B one changes how much gets
extracted, not whether what's extracted can be trusted.

### evidence_ref: two producers, one shape

Claims arrive from two producers, and their evidence is genuinely
different in shape. A PDF claim cites a `node_id`, a quote, and a page
number; a web claim cites a chunk `locator`, a quote, and a URL. Rather
than forcing both producers to write some invented common format to disk,
the compiler absorbs the asymmetry in one function, `normalize_evidence()`,
at index time only. On-disk files keep their native shapes untouched, and
everything downstream reads one uniform record: the `EvidenceRef`, with
fields `producer`, `source_id`, `locator`, `quote`, and optional `page`
and `url`. Every retrieval tool, and every dossier the toolkit composes,
handles evidence through that one shape and never needs to know or care
which producer a claim came from.

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

### OKF and run directories

Three more terms round out the vocabulary. OKF is the Open Knowledge
Format: every knowledge-base page is markdown with YAML frontmatter, one
concept per file, cross-linked with ordinary relative markdown links --
and those links are the graph's edges, checked by lint, not decoration. A
run directory (`pdf-runs/<id>/` or `research-runs/<id>/`) is the per-source
working set: everything one document's ingestion produced, from raw
conversion output through chunks to extracted claims, git-tracked on
purpose as the auditable record of how a conclusion was reached. And a
producer is simply which side wrote a run -- `pdf` or `web` -- the only
distinction `EvidenceRef` needs to preserve after normalization.

## How it fits together

Every skill in this toolkit is a phase of one pipeline: acquire, chunk,
extract, compile, retrieve, synthesize. What's worth noticing is how
little of it
an LLM touches. Extraction, wiki-writing, and the final answer composed
from a dossier are judgment work, marked [JUDGMENT: LLM] in the diagram.
Everything else -- fetching, Docling conversion, provenance recovery,
chunking, index compilation, all eight retrieval tools -- is deterministic
Python, plus one local embedding model at compile time. Rerun a
deterministic stage and you get the same files. Swap the model behind a
judgment stage and the verbatim-quote gate still decides what gets in.

Each source gets a run directory: `<pdf_runs_dir>/<document_id>/` for a
PDF, `research-runs/<source_id>/` for a web source worth mining. These are
git-tracked on purpose. A run directory holds everything the pipeline
produced for one document -- the raw conversion output, the chunks, the
claims with their citations, the eval report -- and it's the auditable
record of how a conclusion was reached, not scratch space you'd
`.gitignore`. The three layers map onto the diagram directly: run
directories and `knowledge_base/` are the durable corpus, the two marked
stages are the judgment layer, and the compiler's output is the derived
index, the one thing kept out of git because any checkout can rebuild it.

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

Sources come in on two sides. On the web side, `research-knowledge-graph`
fetches with Scrapling and chunks the fetched page by heading section. The
PDF side takes four stages to reach the same point: `pdf-ingest-router`
classifies the file and creates its run directory,
`pdf-to-canonical-markdown` runs the one Docling conversion,
`pdf-layout-provenance` recovers what markdown throws away (page, section
path, bounding box), and `canonical-markdown-to-llm-nodes` chunks along
that recovered structure. Both sides hand the same thing to the next
phase: a run directory with a structure-aware `chunks.jsonl`. From here
on, the only trace of which side a source came from is its producer field.

The two judgment stages sit in the middle. `knowledge-extraction` has an
LLM read the chunks and write claims, entities, and relations, and every
claim's quote passes the verbatim-quote gate before it counts.
`llm-wiki-writer` merges those claims into `knowledge_base/` as OKF pages,
updating an existing page where one exists and marking pages `conflicted`
where two sources disagree. (`rag-eval-harness` then re-checks the whole
run directory mechanically; it isn't drawn above because it gates a run
rather than handing anything downstream.)

The last two stages are deterministic again. `knowledge-compiler` rebuilds
the index from scratch -- DuckDB for full-text and graph queries, LanceDB
for vectors -- and normalizes each producer's evidence into one
`evidence_ref` shape as it goes. `retrieval-planner` puts eight tools over
that index, fusing keyword and vector rankings with RRF. The last of them,
`compose_dossier`, re-applies the verbatim-quote gate against the run
directory's source text and emits the evidence dossier. What an agent does
with that dossier -- synthesize an answer -- is the third and final place
an LLM appears.

## Quick start

### Install a tier, then `drt init`

```bash
pip install "deep-research-toolkit[pdf]"   # or [web], [compiler], [full]
drt init                                    # scaffold .deepresearch.yml + a knowledge base
```

The base package is deliberately tiny (PyYAML and the `drt` CLI, nothing
else), and the four extras map onto the three pipeline layers, so you
install only the machinery you'll actually run:

- **`web`** pulls in [Scrapling](https://github.com/d4vinci/Scrapling),
  the retrieval library behind `research-knowledge-graph`. Its stealth
  mode drives a real browser, so after installing run `scrapling install`
  once to download Playwright's browser binaries -- that's a separate step
  `pip` doesn't do for you.
- **`pdf`** pulls in Docling (the conversion engine), plus `pypdf` and
  `pdfplumber` (the classification signals `pdf-ingest-router` computes).
  Docling fetches its OCR and layout models on first use, not at install
  time.
- **`compiler`** pulls in DuckDB, LanceDB, sentence-transformers (the
  local embedding model), and the `openai` client, which exists solely so
  the optional local-LLM backend can talk to any OpenAI-compatible
  endpoint. No hosted API key is required for anything in this tier.
- **`full`** is the union of the other three.

Both Docling and Playwright download sizable assets the first time they
run, well after `pip install` has reported success. `docs/environment.md`
covers what gets fetched, where it lands, and how to pre-warm it -- worth
five minutes before you start, rather than discovering a model download
mid-pipeline on a flaky connection.

`drt init` asks no questions -- everything comes in as flags. `--tier`
takes `web`, `pdf`, `compiler`, or `full` (the default) and sets the
`features.*` flags in `.deepresearch.yml` to match, so pass the tier you
actually installed. `--topic-name` and `--scope-hint` fill in the topic
block, and `--knowledge-base` relocates the knowledge base from its
default `knowledge_base/`. A realistic first run looks like:

```bash
drt init --tier pdf --topic-name "Perovskite stability" \
  --scope-hint "Degradation mechanisms and encapsulation, not manufacturing economics"
```

A bare `drt init` still works, but it writes `topic.name: "(unnamed
project)"` with a placeholder scope hint and every `features.*` flag set
to true. Either way, the generated `.deepresearch.yml` is meant to be
opened and edited afterward. That file matters more than it looks: every
skill reads its topic and scope from there instead of hardcoding one, so
"research X for the knowledge base" means your X, not some generic
default.

Beyond the config file, `drt init` scaffolds the knowledge base directory
and copies each skill (its `SKILL.md` plus scripts) into both
`.claude/skills/` and `.agents/skills/` -- the first is where Claude Code
looks for skills, the second is where Codex does. After that, open the project in either
agent and ask it to research something, or point it at a PDF; the relevant
skill's `SKILL.md` takes it from there. The by-hand sequences below are
for when you want to see each stage individually, or script them.

### First PDF, by hand

The seven-stage PDF pipeline can be driven one script at a time. The first
four stages are deterministic Python:

```bash
python .claude/skills/pdf-ingest-router/scripts/classify_pdf.py your-file.pdf
# prints a run directory, e.g. pdf-runs/your-file-a1b2c3d4/
python .claude/skills/pdf-to-canonical-markdown/scripts/convert.py pdf-runs/your-file-a1b2c3d4
python .claude/skills/pdf-layout-provenance/scripts/extract_provenance.py pdf-runs/your-file-a1b2c3d4
python .claude/skills/canonical-markdown-to-llm-nodes/scripts/chunk_nodes.py pdf-runs/your-file-a1b2c3d4
```

At this point the run directory holds `chunks.jsonl`, and the pipeline
needs judgment: an LLM has to read those chunks and write `claims.jsonl`,
`entities.jsonl`, and `relations.jsonl`. That's the `knowledge-extraction`
stage, done inside an agent session (its `SKILL.md` carries the rules,
chiefly that every claim's quote must be verbatim from the cited page), or
programmatically via the local-LLM backend if you've configured one. Then
the eval harness re-checks the whole run mechanically:

```bash
python .claude/skills/rag-eval-harness/scripts/run_eval.py pdf-runs/your-file-a1b2c3d4
```

A passing eval means the run survived six mechanical checks -- every quote
really is a substring of its cited page, every table made it to disk, no
heading got lost in chunking, and so on. "Nothing crashed" is a much lower
bar, and it's not the one this stage sets.

### First web source, and the first query

The web side is meant to be agent-driven: open the project in Claude Code
or Codex and ask it to "research X for the knowledge base." The agent
reads your topic from `.deepresearch.yml`, checks what the knowledge base
already has, fetches only what's missing, and writes or updates wiki
pages.

By hand, the same flow is four scripts and one judgment step. Fetch a
page, then scaffold a run directory from it:

```bash
python .claude/skills/research-knowledge-graph/scripts/fetch.py <url> --out source.md
python .claude/skills/research-knowledge-graph/scripts/start_research_run.py <url> --content-file source.md
```

That creates `research-runs/<source_id>/` with the content chunked by
heading section -- the web-side mirror of a PDF run. Extraction is the
same judgment step as above (an agent, or the local backend), writing
verbatim-gated claims into the run directory. Then compile everything into
the index and ask it something:

```bash
python .claude/skills/knowledge-compiler/scripts/compile.py
python .claude/skills/retrieval-planner/scripts/query.py search-claims "your first question"
```

`compile.py` rebuilds the DuckDB + LanceDB index from whatever the corpus
holds, PDF runs and web runs alike, and `query.py` puts all eight
retrieval tools on the command line (`search-claims` is one; `search-wiki`,
`get-entity`, and `compose-dossier` are covered under
[retrieval-planner](#retrieval-planner)). For this whole loop run
end-to-end on a real document, with real output at each stage, see
[A worked example](#a-worked-example-end-to-end).

## A worked example, end to end

The quick start gives you the commands; this section shows what they
actually produce. What follows is one real run: five Wikipedia articles
on proof-of-stake, fetched, extracted, compiled into a queryable
knowledge base, and then queried to write a grounded thesis -- all of it
on a fully local model stack, with no hosted API anywhere in the loop.
The numbers below come from that single run. Read them for a sense of
scale, not as a benchmark.

The run started with `research-knowledge-graph` fetching the five
articles. Each one became a `research-runs/<id>/` directory holding the
page as `source.md` and its structure-aware chunking as `chunks.jsonl`,
one node per heading section. Five sources came to roughly 105 chunks,
and the chunk is the unit everything downstream speaks in: claims cite
chunks, chunks cite sources.

Extraction is the judgment step. `knowledge-extraction` read those
chunks on a local instruct model and wrote `claims.jsonl`,
`entities.jsonl`, and `relations.jsonl` into each run directory. What
survived the verbatim gate: 103 claims, 61 entities, and 66 relations.
"Survived" is the right word -- any claim whose quote was not an exact
substring of its cited source got dropped, which is exactly where a
local model's habit of paraphrasing gets caught. The pass took minutes
on a fast instruct model; rerunning the same extraction on a reasoning
model was far slower. That gap is the practical case for routing each
pipeline role to a model suited to it, which [Running local
models](#running-local-models) covers.

Compiling was one command. `compile.py`, from the `knowledge-compiler`
skill, rebuilt the DuckDB + LanceDB index from the five runs, embedding
claims and chunks with a local Qwen embedding model so lexical and
vector search work against the same corpus. The index is always rebuilt
from the files on disk, so this step is safe to rerun whenever the
corpus changes.

Then the questions. `query.py search-claims`, from `retrieval-planner`,
answered ad-hoc queries with claims that each carried their verbatim
quote and source attribution, and `compose-dossier` gathered everything
relevant to a broader question into a single evidence dossier -- the
package a writing model actually consumes.

Last, synthesis. A reasoning model took the dossier and wrote a thesis
on proof-of-stake. Because every claim in the dossier carries an exact
quote, the draft was checkable line by line: every number that got
spot-checked traced back to a verbatim quote in a named source. The
model can still be wrong, but "where did this come from?" now has a
mechanical answer instead of a plausible one.

Compressed to one picture, here is the lifecycle that run walked
through:

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

## The skills

Ten skills ship today. Seven form the PDF ingestion pipeline, one handles
web research, and two form the knowledge-compiler layer that indexes and
queries what the other eight produce. Every one of them is deliberately
small. A single monolithic "research skill" would drag its whole
instruction set into context on every use, whether the task at hand needed
the PDF-parsing details or not; splitting the work into ten focused skills
means the agent loads only the instructions for the stage actually
running. This is the progressive-disclosure model both Claude Code and
Codex use for skill discovery, and it keeps a conversation's context usage
proportional to the work being done rather than to the size of the
toolkit.

The skills can stay small because they don't carry state or topic
knowledge. Every hand-off between stages is a file with a documented
schema -- a manifest, a JSONL record, an OKF page -- so each skill only has
to explain its own stage, never the whole pipeline. Deterministic work
(fetching, chunking, hashing, linting, index building) lives in scripts
the skills call, while the two genuinely judgment-based stages, claim
extraction and wiki synthesis, are the agent itself reasoning over those
files with the skill supplying the rules -- the split
`docs/decisions/0001-architecture.md` commits to. And nothing
topic-specific is baked in anywhere: per-project scope lives in
`.deepresearch.yml`, which is what lets one shared `skills/` tree serve
both platforms and any research topic without editing a skill.

### research-knowledge-graph

This is the web-research half of the toolkit, and the one meant to be used
directly and repeatedly, not once per document. It wraps
[Scrapling](https://github.com/d4vinci/Scrapling) for retrieval through
its `fetch.py` script: plain HTTP by default, with a stealth mode reserved
for when a plain fetch gets blocked by an anti-bot challenge or a flat 403
-- stealth gets past things an ordinary web-fetch tool cannot. Findings
are stored as Open Knowledge Format pages: markdown files with YAML
frontmatter, one concept per file, cross-linked with ordinary relative
markdown links. Those links aren't decoration; they are the graph's edges,
and the lint step treats them as load-bearing.

The maintenance discipline comes from Andrej Karpathy's LLM-wiki pattern,
and it shows up as three named operations: `ingest`, `query`, and `lint`.
`ingest` doesn't append a new file for every fetch. It searches the
knowledge base first, fetches only what's missing, records the source as a
`src-XXXX` row in the knowledge base's sources index, and then -- if the
concept already has a page -- merges the new findings into that page and
bumps its timestamp instead of creating a near-duplicate. `query` searches
the existing graph and walks its links before doing any live scraping at
all, on the theory that re-deriving an answer you've already written down
is wasted work. `lint` is the health check: `lint_graph.py` walks every
page looking for orphans nothing links to, broken relative links, missing
or malformed frontmatter, and `researched` pages whose timestamp has
passed a staleness threshold (180 days by default).

Every page's frontmatter carries a `status` field with five possible
values: `seed` for a placeholder that exists but hasn't been researched
yet, `researched` once it has real content, `stale` once lint flags it as
overdue for a refresh, and two written by the PDF pipeline's
`llm-wiki-writer` stage (covered below) -- `draft` for a page synthesized
from a single ingested source and not yet cross-checked, and `conflicted`
for a page where two sources directly disagree. The status field is what
lets `query` decide whether an existing page can answer a question as-is,
or whether a `seed` or `stale` marking means a fresh `ingest` has to
happen first.

None of this is tied to a topic. The skill's first instruction is to read
`.deepresearch.yml` (found by walking up from the current directory, the
same way git finds `.git`): `topic.scope_hint` says what this project's
research is actually about, and `knowledge_base.path` says where the pages
live. If the file doesn't exist, the skill tells the user to run `drt
init` rather than inventing a scope or a directory to write into. So
asking an agent to "research X for the knowledge base" means: read the
config, search what's already there, fetch only the gaps, write or update
a page.

Web research doesn't stop at wiki prose. It also produces claims -- the
same evidence-backed records the PDF pipeline has emitted from the start.
When
a fetched source is substantial enough to mine, `start_research_run.py`
scaffolds a `research-runs/<source_id>/` directory that mirrors a PDF
run: the fetched content saved verbatim as `source.md`, one chunk per
heading section in `chunks.jsonl`, and a `manifest.json` marking the
producer as `web`. The agent then writes `claims.jsonl`,
`entities.jsonl`, and `relations.jsonl` into that same directory itself --
the skill leaves extraction unscripted on purpose, since deciding what
counts as an atomic, well-evidenced claim is a judgment call -- under the
rule that every supporting quote must be a verbatim substring of
`source.md`. That symmetry is the point. The knowledge compiler indexes
web runs and PDF runs into one table, so a claim about an entity from a
webpage and a claim about the same entity from a whitepaper land in the
same queryable graph, checked by the same evidence gate. Only the evidence
shape differs (a web claim cites a chunk locator and a URL, a PDF claim a
node id and a page number), and the compiler normalizes that difference
away at index time without touching either producer's files on disk.

### pdf-ingest-router

The first of the seven PDF stages, and the only one that ever looks at
signals *about* a PDF rather than its actual content. Before any real
parsing happens, this skill answers one question: what kind of document is
this, and which of the toolkit's backends should handle it? It computes a
stable `document_id` (a slug of the filename plus a hash of the file's
bytes, so the same PDF always gets the same identity even if you run this
twice) and creates that document's working directory. Everything after
this stage reads and writes into `<pdf_runs_dir>/<document_id>/` and never
touches the original PDF path again.

The classification itself comes from five real signals, computed with
`pypdf` and `pdfplumber` rather than guessed: average extractable
characters per page (is there even a text layer?), whether the PDF has
AcroForm fields (it's a fillable form regardless of anything else), what
fraction of pages have essentially no extractable text (probably scanned),
what fraction have detectable tables, and a rough density of math and
Greek-letter symbols per page. Those five signals resolve, in a fixed
priority order, into one of six routes: `digital-text`, `scanned`,
`scientific-math`, `form`, `financial-legal`, or `slide-like`, each with a
backend that would be ideal for it in principle.

That last qualifier matters: right now, only one backend is actually wired
up. Every route's `recommended_backend` comes back as `docling`, because
that's the only conversion path this toolkit implements today. The
classification still records what the *ideal* backend would be (Docling's
OCR mode for scanned documents, Marker as a fallback when Docling mangles
formulas in a math-heavy paper), so the gap is visible in the data rather
than silently papered over. When a real scanned or equation-dense document
eventually needs a second backend, the routing decision that justifies
adding one is already sitting in `classification.json`, not something that
has to be reverse-engineered later.

Running it is one command, `python scripts/classify_pdf.py <pdf_path>`,
and it's the only stage in the pipeline that takes a PDF path at all. The
runs directory resolves from `.deepresearch.yml`'s
`knowledge_base.pdf_runs_dir` unless `--runs-dir` overrides it, with a
plain `pdf-runs/` fallback for zero-config exploration, and the script
prints the run directory it created -- the one argument every later stage
takes. Everything the stage decides lands in `classification.json`, and a
`manifest.json` gets started alongside it with the `document_id`, the
source path, and a content hash: the file every later stage reads
`document_id` back out of, appending its own entry rather than ever
overwriting what came before. One thing that never lands in the run
directory is the PDF itself -- the manifest records its path and hash, not
its bytes.

### pdf-to-canonical-markdown

The actual PDF parsing happens here, and it happens once: this stage turns
the source file into a canonical markdown rendering plus Docling's own
structured JSON export, and every stage downstream works from those two
outputs instead of re-opening the PDF. `canonical.md` is what a human would
want to read; `docling_raw.json` is what the next few stages actually
parse, kept in Docling's own shape rather than reprocessed into something
this toolkit invented, since downstream code already knows how to walk
Docling's `texts`/`tables`/`pictures`/`pages` structure directly.

Invocation is where the pipeline's one CLI convention starts:
`python scripts/convert.py <run_dir>`, where the run directory is the path
`classify_pdf.py` printed. The source PDF's location is never passed on
the command line -- the script reads `source_file` back out of
`manifest.json`, converts it with Docling (table structure recognition
explicitly enabled), writes the two outputs, and appends its own `stages`
entry to the manifest recording the parser name and version. That append
goes through `common.manifest.update_stage`, which merges into the
existing manifest rather than replacing it; a stage clobbering another
stage's manifest entry was a real bug in the original prototype, and the
fix is now pinned by a regression test.

One piece of this stage is worth calling out specifically because it came
from a real failure, not a hypothetical one: even a plain digital-text PDF
makes Docling try to fetch OCR and layout models on first use, and on a
flaky connection that download can fail mid-stream. The fix isn't to
disable OCR to dodge the flakiness (this toolkit's scanned-document route
depends on OCR working later). It's a single retry around the whole
conversion call when a transient network error shows up. That retry
survived being carried over into this generalized version of the toolkit
on purpose; it's exactly the kind of unglamorous fix that's easy to
"simplify away" during a rewrite, and only notice it's missing the next
time a download blips.

Only Docling is implemented. Marker, MinerU, and Unstructured are
documented as fallback options for when a real scanned or math-heavy
document shows up and Docling's output looks wrong (mangled formulas,
blank pages where there should be OCR'd text), but none of them are
installed or called from this stage today. Building that out is deferred
until an actual document forces the question, not built speculatively
ahead of one.

### pdf-layout-provenance

Markdown is good for reading and bad for citing. The moment
`pdf-to-canonical-markdown` renders a PDF into `canonical.md`, it throws
away exactly the information a citation needs: which page a sentence came
from, which section it sits under, where on the page it physically is.
This stage exists to recover that before anything downstream makes a
factual claim it can't back up. It walks Docling's structured export in
true document order, not headings first and tables afterward -- processing
text and tables as separate passes is enough to attach a table to the wrong
section when it sits between two paragraphs -- and emits one record per
structural unit: heading,
paragraph, table, figure, caption, or list item, each carrying its page
number, its section path, its bounding box, and a content hash.

The command is `python scripts/extract_provenance.py <run_dir>`, and it
reads exactly two files: `docling_raw.json` for the structure and
`manifest.json` for the document's identity. The document-order walk
follows Docling's `body.children` list, resolving each `$ref` pointer to
its `texts` or `tables` item, mapping Docling's `label` onto one of the
six `unit_type` values, and maintaining a stack of active headings that
becomes each unit's `section_path`. One JSON object per unit goes out to
`provenance.jsonl`, each line carrying its `schema_version`, and the
manifest gains a `unit_count`.

Getting `section_path` right turned out to need a small heuristic, because
Docling's own heading-level field isn't always trustworthy. In this
toolkit's own test fixture, "2. Architecture" and "2.1 Head Lifecycle" both
come back from Docling at the same nominal level, even though the second
one is obviously a subsection of the first. This stage disambiguates by
counting the dot-separated numbers in a heading's own numbering instead:
"2." reads as depth one, "2.1" as depth two. It only falls back to
Docling's raw level field for headings that have no numbering to parse at
all, like a document's title. It's a heuristic, not a guarantee, and it
says so.

Every downstream stage that cites a page number or checks a quote against
source text is really checking against this stage's output, `provenance.jsonl`.
It's the layer that makes the difference between "the model said so" and
"here's the exact page and bounding box this came from." That's also
why `confidence` is honestly recorded as `1.0` across the board right now:
this pipeline doesn't run OCR yet, so there's no real per-element
confidence score to report, and a fabricated one would be worse than an
honest placeholder. Table cell structure is the other deliberate
exclusion: tables get flattened here into a pipe-separated text rendering,
just enough to hash and locate them, because the real per-cell CSV
extraction is `knowledge-extraction`'s job two stages later.

### canonical-markdown-to-llm-nodes

Chunking is where a lot of retrieval pipelines quietly go wrong, usually by
tokenizing the document and cutting every N tokens without regard for what
those tokens actually contain. That approach breaks two things this
toolkit specifically depends on: it splits a heading from the paragraph it
introduces (or a sentence in half across a table boundary), which makes for
worse embeddings and worse retrieval; and it can produce a chunk whose
citation spans two different provenance units, so a claim built from it
can't be checked cleanly against one page. This stage chunks along the
document's own structure instead: one node per heading section, one node
per table, one node per figure, because `pdf-layout-provenance` already
recovered exactly that structure in the previous stage. Nothing here is
guessing where the meaningful boundaries are.

`python scripts/chunk_nodes.py <run_dir>` is the whole invocation; the
logic lives in `deep_research_toolkit.pdf.chunk.chunk_nodes`, so the same
function is importable and unit-testable without going through a
subprocess. The grouping rules are short. A heading opens a new `section`
node, and the paragraphs and list items after it fold in until the next
heading, table, or figure; tables and figures always get nodes of their
own. Captions get particular treatment: a caption contributes its
`unit_id` and page to whichever node it falls under, but its text is not
spliced into the surrounding narrative, because a figure caption dropped
mid-paragraph reads as a non-sequitur. Every node carries
`links.previous` and `links.next`, chaining the whole file in document
order.

Long sections get a second pass. When a section's combined text runs past
roughly 1,500 characters and was built from more than one paragraph, the
section node stays as a coarse, still-citable overview, and each
contributing paragraph *also* becomes its own smaller node, linked back to
the section as its parent. That keeps `chunks.jsonl` as one flat,
sequentially-linked file, easy for `rag-eval-harness` to walk end to end,
while still giving retrieval smaller, more precise nodes to work with
when a section runs long. This was built as real functionality from the
start, not left as a "future work" note, because a pipeline that only
handles short documents cleanly isn't actually done.

What this stage deliberately does not do is fill in `summary`, `keywords`,
or `entities` on the nodes it writes. Those fields go out as empty
placeholders on purpose. Deciding what a chunk is really about, and which
entities it mentions, takes actual judgment about meaning, which is a job
for the next stage and the LLM reading its output, not a job for a chunking
algorithm to fake with a keyword-frequency count.

### knowledge-extraction

This is the stage where the pipeline stops being purely mechanical, and it
says so plainly rather than pretending otherwise. Half of its six outputs
come from scripts: tables pulled out of Docling's cell data into real CSV
files, figures pulled into PNGs, and a caption record that never silently
drops a figure just because Docling didn't capture usable pixel data for
it. The other three (`claims.jsonl`, `entities.jsonl`, `relations.jsonl`)
are written directly by an LLM reading the previous stage's chunked
nodes, because deciding whether a sentence expresses one real, checkable
claim, or whether "Hydra" and "Hydra Head" refer to the same thing, isn't
something a regex should be trusted to decide.

The deterministic half is two scripts, `extract_tables.py` and
`extract_figures.py`, each taking the run directory and each reading
`docling_raw.json` rather than the original PDF. The table script places
every cell's text at its recorded row and column offset, leaving
spanned-over cells blank instead of repeating a value, and writes one
`tables/table_NN.csv` per table. The figure script tries to materialize
`figures/figure_NN.png` from Docling's image data but always writes a row
to `figures/captions.jsonl` regardless: a figure with a caption but no
extractable pixels is recorded with `extracted: false` and a note, never
dropped, because the eval harness downstream counts figure references,
not just successful PNGs. Both scripts are idempotent, and each merges
only its own counts into the manifest, so re-running one never wipes out
the claim, entity, and relation counts the LLM half records.

The rule that does the most work here is the evidence rule: every claim's
supporting quote has to be copy-pasted verbatim from the cited node's own
text, never paraphrased, never cleaned up for readability, always with the
page it appears on. That constraint is checked mechanically downstream by
`rag-eval-harness`, and it's the single property that keeps this toolkit's
claims audit-able rather than merely plausible-sounding. A claim with a
quote that doesn't actually appear in its cited source is worse than no
claim at all, because it looks verified when it isn't. The other rules that shape a good extraction pass
are almost as load-bearing: one checkable assertion per claim rather than a
compound sentence trying to do two jobs, entity mentions merged under
their most formal name with variants kept as aliases instead of spawning
duplicate rows, and a hard rule against forcing a claim or relation the
source text doesn't actually support just to hit some notional quota.

It's normal, and expected, for a good extraction pass to produce a short
list: the toolkit's own reference example pulls five well-evidenced claims
out of a two-page document, not twenty shaky ones. A short list of claims
that all survive the evidence check is worth more than a long one that
doesn't.

### llm-wiki-writer

The second-to-last stage is where a document's claims stop being specific
to that one PDF's run directory and become part of the shared knowledge
base, the same one `research-knowledge-graph` writes to from the web. It
reads `claims.jsonl` and `entities.jsonl`, groups entities into
page-worthy concepts (not every entity needs its own file), and for each
one, searches the existing knowledge base before writing anything. If a
page for that concept already exists, this stage merges the new claims
into it and bumps its timestamp rather than creating a near-duplicate file.
That's exactly the discipline `research-knowledge-graph` already applies to web
research, applied here to PDF-derived content instead.

The writing itself is the agent's judgment, but the bookkeeping runs
through one script. New pages get scaffolded with
`scripts/scaffold_wiki_page.py <run_dir> <knowledge_path> --type ...
--title ... --status draft --source-docs <document_id>`, and the script
refuses to overwrite an existing file without `--force` -- if a page
already exists, the agent edits it by hand and then calls the script with
`--record-updated` so the touched path still gets logged. The
`<knowledge_path>` is relative to whatever `.deepresearch.yml`'s
`knowledge_base.path` points at; there is no separate PDF-wiki directory,
and this stage adds nothing to the page format beyond the two status
values below and an optional `source_docs` frontmatter field listing
which PDF runs a page draws on.

New pages always start at `status: draft` (a value this pipeline adds on
top of the existing `seed`/`researched`/`stale` set), because a page
synthesized from a single document hasn't been cross-checked against
anything else yet, and calling it `researched` would overstate how solid
it is. The other status value this stage introduces, `conflicted`, handles
the case where two documents flatly disagree: contradictory throughput
numbers, contradictory safety assumptions. Rather than quietly picking
whichever claim sounds more confident, this stage keeps both, marks the
page `conflicted`, and writes an explicit section describing the
disagreement and which document each side comes from. An unresolved
contradiction that's visible is more useful than a resolved-looking answer
that's actually just a coin flip.

Every page this stage touches, whether newly created or merged into, gets
logged to `wiki_pages_written.json` in the run directory: the audit trail
`rag-eval-harness` reads to know exactly which pages this run is
responsible for. The stage finishes by running
`research-knowledge-graph`'s `lint_graph.py` over the whole knowledge
base. A wiki-writer run that leaves the graph broken (an orphaned
page, a dangling link) is treated as worse than one that wrote nothing at
all.

### rag-eval-harness

The last stage exists because everything before it can fail silently. A
heading that never made it into a chunk's section path, a claim whose
"quote" turns out to be a paraphrase, a figure that got dropped without a
trace. None of those raise an exception. The pipeline finishes, the files
all exist, and the corruption is only visible if someone reads every
output by hand. This stage automates that read, every time, so a document
doesn't get trusted just because the earlier stages didn't crash.

Six mechanical checks run against whatever's in a run directory:
`headings_recovered` (every heading in the canonical markdown made it into
some chunk's section path), `tables_present` (the table count in the
provenance data matches the CSV count on disk), `page_citations_valid`
(every claim's cited page actually exists), `evidence_quotes_verbatim`
(every claim's supporting quote appears character for character in the
chunk it cites), `figures_accounted_for` (every figure reference was
either extracted or explicitly flagged as not extracted), and
`no_ocr_garbage` (no page's text crosses a small non-printable/mojibake
threshold). Each one is designed to fail loudly on the specific corruption
it exists to catch, rather than passing by default. The verbatim check
deserves a note: it is chunk-based, and it runs through the one shared
gate in `common/verbatim.py` that claim extraction and dossier composition
also apply. One definition of "the source text," enforced everywhere,
means a quote admitted at extraction time can't be rejected here because
two stages quietly disagreed about what to check it against.

`python scripts/run_eval.py <run_dir>` runs everything and writes two
reports back into the run directory: `eval_report.json`, the
machine-readable version with a `pass_rate`, and `eval_report.md`, the
same checks as a table a human can skim. It's safe to point at a
half-finished pipeline. Any check whose input files don't exist yet is
marked skipped, not failed, so a run that has only reached the chunking
stage reports honestly on what could be checked instead of drowning in
spurious failures. The flip side is that `pass_rate` only ever means
"score out of the checks that could run" -- the per-check `detail` field
is the thing to read, not just the number.

One thing this stage deliberately leaves out of its automated score:
question-and-answer retrieval probes, where you write questions a reader
should be able to answer from the document and confirm retrieval actually
surfaces the right chunk. That's a genuinely useful check. It catches
problems none of the six mechanical checks can see, like a section that
got split mid-thought, but it needs an LLM call per question, and that's
a cost this stage's automated pass should never carry without being asked.
It's documented as an optional manual step for documents where it's worth
the extra spend, not baked into `pass_rate`.

### knowledge-compiler

Everything above produces files: wiki pages, chunked nodes, claims with
citations. Files are the right durable format (auditable, diffable,
git-tracked), but they're a poor query surface. Answering "what do we know
about X, and what's the evidence?" by grepping a knowledge base gets
slower and lossier as the corpus grows, and it misses semantically related
material that doesn't share keywords. This skill compiles the whole corpus
(`knowledge_base/`, `pdf-runs/`, `research-runs/`) into a hybrid index:
DuckDB for full-text search over pages and claims plus the graph tables
(wiki links, entities, relations, evidence), and LanceDB for vector search
over wiki pages and claims, embedded locally with sentence-transformers
(`all-MiniLM-L6-v2` by default, configurable). Search queries run both
engines and fuse the two rankings with reciprocal rank fusion, so a hit
only one side finds still surfaces.

Compilation is a full rebuild every run, on purpose. At the scale this
toolkit actually serves (a per-project knowledge base on one machine), a
rebuild costs seconds, and it means there is no cache-invalidation state
to reason about: the index is either current or stale, and stale is fixed
by recompiling. For the same reason the index lives in a git-ignored
directory (`.deepresearch/index/` by default) rather than being committed.
The run directories and the wiki are the record; the index is derived from
them and any checkout can regenerate it.

The one piece of judgment-free normalization the compiler does is the
`evidence_ref`: PDF claims cite `node_id` + `page`, web claims cite a
chunk locator + URL, and the compiler maps both into a single
producer-agnostic evidence shape as it indexes. Neither producer's files
change on disk; the asymmetry is absorbed in one function at index time,
which is what lets every retrieval tool below treat evidence uniformly
regardless of where it came from. The full index schema and normalization
mapping are in `docs/contracts/knowledge-compiler.md`, and the build-time
design decisions in `docs/decisions/0002-knowledge-compiler.md`.

### retrieval-planner

The query half of the compiler layer: eight small tools over the compiled
index, exposed as one CLI (`scripts/query.py`) that prints JSON. Six are
plain lookups: `search_wiki` and `search_claims` (hybrid keyword+vector
search), `read_page` (fetch one wiki page whole), `get_entity` (an entity
with its aliases, mentions, and relations), `neighbors` (a bounded graph
walk over the relation table), and `get_sources` (provenance for a page or
a claim). None of the eight makes an LLM call, which is a design rule
inherited from ADR 0001, not an implementation accident: tools that are
cheap and deterministic can be called freely and composed by an agent
mid-conversation, while a tool hiding a model call inside would have
unpredictable cost and non-reproducible output.

The seventh tool, `find_contradictions`, is deliberately dumber than it
sounds. It reports mechanical candidates only: relation triples where one
subject and predicate map to more than one distinct object, plus any wiki
page already marked `conflicted`. Deciding whether "founded in 2015" vs
"founded in 2017" is a real contradiction or whether "supports X" and
"supports Y" are just both true takes judgment, and judgment is the
agent's job, done in one batched pass over all candidates rather than a
hidden model call per candidate.

The eighth, `compose_dossier`, is where everything upstream pays off. It
assembles a set of claims (picked by query or by explicit ids) with their
full citations into an evidence dossier, and it applies the toolkit's
verbatim rule as a hard gate: a claim reaches `included` only if every
supporting quote is an exact substring of its source (the cited PDF page's
text, or the web run's `source.md`). Anything else lands in `rejected`
with an explicit reason, never silently mixed in with the verified
material. The result is the artifact the whole pipeline exists to produce:
a set of claims an agent can answer from, where every line traces back to
a quote that actually appears, character for character, in a real source.

### The optional local LLM backend

Everywhere this toolkit needs judgment (deciding what counts as an atomic
claim, merging entity mentions, synthesizing a wiki page), the default
worker is the in-session agent itself, reading the relevant SKILL.md. That
default is unchanged, and it's still the recommendation: frontier-model
judgment is exactly what those steps need. What's new is an opt-in
alternative for the extraction step: set `llm.provider: local` in
`.deepresearch.yml` and point it at any OpenAI-compatible endpoint (Ollama
on `:11434/v1`, vLLM on `:8000/v1`) serving a local model such as
`Ornith-1.0-9B`, and `extract_claims.py` will run claim extraction
programmatically over a run's chunks: useful for batch-processing many
sources without burning agent context on each one.

The reason this is safe to offer at all is the verbatim gate. Programmatic
extraction runs every proposed claim through the same exact-substring
check `compose_dossier` uses, and auto-drops any claim whose evidence
quotes aren't verbatim in the source, before anything is written to
`claims.jsonl`. A smaller local model can therefore only under-produce
(fewer claims survive), never corrupt the corpus with paraphrases that
look like citations. `scripts/validate-local-llm.py` measures exactly
that: how much of the reference extraction a given local model recovers,
and how many of its proposals the gate had to drop.

## Configuration

Everything project-specific lives in one file, `.deepresearch.yml`, at your
project's root, discovered by walking up from wherever a command is run,
the same way `.git` is found. Nothing in any skill hardcodes a topic, a
directory name, or a research scope; all of that comes from here.

```yaml
version: 1

knowledge_base:
  path: knowledge_base/
  pdf_runs_dir: pdf-runs/
  research_runs_dir: research-runs/
  index_dir: .deepresearch/index/

topic:
  name: "Your project's research topic"
  scope_hint: >
    A short description of what's in scope, so a skill reading this
    knows what it's actually researching instead of guessing.
  tags: []

features:
  web_research: true
  pdf_ingestion: true
  knowledge_compiler: false

llm:
  provider: anthropic          # "anthropic"/"agent": the in-session agent
  model: claude-sonnet-4-5     # does extraction; "local": an OpenAI-compatible
  api_key_env: ANTHROPIC_API_KEY   # endpoint does it (see llm.local below)
  embedding_model: all-MiniLM-L6-v2
  local:                       # only read when provider: local
    base_url: http://localhost:11434/v1
    model: Ornith-1.0-9B
    api_key_env: OPENAI_API_KEY

scrapling:
  default_mode: http
  rate_limit_seconds: 1.0
```

`drt init` writes a starter version of this file and asks what tier you
need (`web`, `pdf`, `compiler`, or `full`), which sets the `features.*`
flags accordingly. See `docs/contracts/pdf-ingestion-pipeline.md`,
`docs/contracts/okf-frontmatter.md`, and
`docs/contracts/knowledge-compiler.md` for the full schema every artifact
in this toolkit follows, including the `schema_version` fields that make
future changes to these formats detectable rather than silent.

## Status and roadmap

**Built and tested:** all three layers above — web research, the
seven-stage PDF pipeline, and the knowledge-compiler layer (the
`knowledge-compiler` and `retrieval-planner` skills, suite version
0.2.0) — including the `drt` CLI, the dual Claude Code/Codex plugin
manifests, a fast unit-test suite that runs on every push, and two heavy
integration tests: one that
runs the entire PDF pipeline through real Docling conversion against a
test fixture and checks for a perfect score on every eval check, and one
that compiles a real corpus into the DuckDB + LanceDB index with the real
embedding model and exercises every retrieval tool against it. All of it
has been verified against a real installed package, not just a
development checkout, so what's described above is what actually runs.

**Deferred on purpose:** incremental index compilation (the compiler does
a full rebuild each run, which is seconds at the scale this toolkit
serves), an MCP query server over the finished knowledge base,
GraphRAG-style community detection, and a learned reranker — each judged
premature at per-project, single-machine scale rather than forgotten. The
reasoning for the original architecture is in
`docs/decisions/0001-architecture.md`; the decisions made while building
the compiler layer (the injectable test embedder, the index-time
evidence normalization, the opt-in local LLM backend) are in
`docs/decisions/0002-knowledge-compiler.md`.

## Contributing

See `CONTRIBUTING.md` for the development setup, test tiers, and what to
update when you change an on-disk format. `LICENSE` covers the terms:
MIT, so build on it freely.
