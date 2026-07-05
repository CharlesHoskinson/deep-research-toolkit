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
- [A worked example, end to end](#a-worked-example-end-to-end)
- [The skills](#the-skills)
- [The retrieval tools](#the-retrieval-tools)
- [Running local models](#running-local-models)
- [What the pipeline guarantees](#what-the-pipeline-guarantees)
- [Configuration](#configuration)
- [Verification and testing](#verification-and-testing)
- [Status and roadmap](#status-and-roadmap)
- [FAQ](#faq)
- [Glossary](#glossary)
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
section defines the vocabulary the rest of the README reuses -- the three
layers, the verbatim-quote gate, evidence_ref, OKF, producer, run
directory -- so that when a later section says "the gate rejects it," you
already know which gate and why it exists.

### The three layers

The first layer is the durable corpus: markdown wiki pages plus per-run
JSONL files, all kept in git. This is the audit trail. Every claim, entity,
and relation the system knows lives here as a plain file, each claim
carrying a verbatim quote from a named source, and "how do we know this?"
is answered by opening the file and reading, not by querying anything. If
every other layer were deleted, the corpus alone would still be a complete
and inspectable record.

The second layer is the derived index: DuckDB for full-text and graph
queries, LanceDB for vectors. It's meant to be git-ignored, and it's
rebuilt from scratch on every compile -- never patched incrementally.
That sounds wasteful until you notice what it buys: there is no state in
which the index and the files can silently disagree. The index is either
current or stale, and stale is fixed by recompiling, which takes seconds
at the per-project scale this toolkit serves. Nothing ever has to patch
the index to match a changed file, so a whole class of cache-invalidation
bugs can't exist here.

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
|  DERIVED INDEX    (keep git-ignored -- a rebuildable cache)   |
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
how plausible it reads. That's why a local-model backend can be the
default at all: swapping a frontier model for a 7B one changes how much
gets extracted, not whether what's extracted can be trusted.

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
little of it an LLM touches. Extraction, wiki-writing, and the final
answer composed from a dossier are judgment work, marked [JUDGMENT: LLM]
in the diagram. Everything else -- fetching, Docling conversion,
provenance recovery, chunking, index compilation, all eight retrieval
tools -- is deterministic Python, plus one local embedding model at
compile time. Rerun a deterministic stage and you get the same files.
Swap the model behind a judgment stage and the verbatim-quote gate still
decides what gets in.

Each source gets a run directory: `<pdf_runs_dir>/<document_id>/` for a
PDF, `research-runs/<source_id>/` for a web source worth mining. These are
git-tracked on purpose. A run directory holds everything the pipeline
produced for one document -- the raw conversion output, the chunks, the
claims with their citations, the eval report -- and it's the auditable
record of how a conclusion was reached, not scratch space you'd
`.gitignore`. The three layers map onto the diagram directly: run
directories and `knowledge_base/` are the durable corpus, the two marked
stages are the judgment layer, and the compiler's output is the derived
index, the one thing to keep out of git because any checkout can
rebuild it.

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
  fallback local embedding path), and the `openai` client, which exists
  solely so the local-LLM backend can talk to any OpenAI-compatible
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
opened and edited afterward. That file matters more than it looks: the
skills read their topic and scope from there instead of hardcoding one,
so "research X for the knowledge base" means your X, not some generic
default.

Beyond the config file, `drt init` scaffolds the knowledge base directory
and copies each skill (its `SKILL.md` plus scripts) into both
`.claude/skills/` and `.agents/skills/` -- the first is where Claude Code
looks for skills, the second is where Codex does. After that, open the
project in either agent and ask it to research something, or point it at
a PDF; the relevant skill's `SKILL.md` takes it from there. The by-hand
sequences below are for when you want to see each stage individually, or
script them.

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
stage, run programmatically through the local-LLM backend under the
default `provider: local`, or done by hand inside an agent session if
you've set `llm.provider: agent` (either way its `SKILL.md` carries the
rules, chiefly that every claim's quote must be verbatim from the cited
page). Then the eval harness re-checks the whole run mechanically:

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
same judgment step as above (the local backend by default, or an agent
under `provider: agent`), writing
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
extraction and wiki synthesis, are an LLM (the configured local model,
or the in-session agent under `provider: agent`) reasoning over those
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
When a fetched source is substantial enough to mine,
`start_research_run.py` scaffolds a `research-runs/<source_id>/`
directory that mirrors a PDF run: the fetched content saved verbatim as
`source.md`, one chunk per heading section in `chunks.jsonl`, and a
`manifest.json` marking the producer as `web`. Extraction then writes
`claims.jsonl`, `entities.jsonl`, and `relations.jsonl` into that same
directory -- the skill's `extract_claims.py` on the default local
provider, or the agent by hand under `provider: agent` -- under the rule
that every supporting quote must be a verbatim substring of `source.md`. That symmetry is the point. The
knowledge compiler indexes web runs and PDF runs into one table, so a
claim about an entity from a webpage and a claim about the same entity
from a whitepaper land in the same queryable graph, checked by the same
evidence gate. Only the evidence shape differs (a web claim cites a
chunk locator and a URL, a PDF claim a node id and a page number), and
the compiler normalizes that difference away at index time without
touching either producer's files on disk.

### pdf-ingest-router

The first of the seven PDF stages, and the only one that ever looks at
signals *about* a PDF rather than its actual content. Before any real
parsing happens, this skill answers one question: what kind of document is
this, and which of the toolkit's backends should handle it? It computes a
stable `document_id` (a slug of the filename plus a hash of the file's
bytes, so the same PDF always gets the same identity even if you run this
twice) and creates that document's working directory. Everything after
this stage reads and writes into `<pdf_runs_dir>/<document_id>/`; no
later stage is ever passed the PDF path again -- the one that needs it
reads `source_file` back out of the manifest.

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
structural unit: heading, paragraph, table, figure, caption, or list
item, each carrying its page number, its section path, its bounding box,
and a content hash.

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

Every downstream stage that cites a page number or checks a quote
against source text is really checking against this stage's output,
`provenance.jsonl`. It's the layer that makes the difference between
"the model said so" and "here's the exact page and bounding box this
came from." That's also why `confidence` is honestly recorded as `1.0`
across the board right now: this pipeline doesn't run OCR yet, so
there's no real per-element confidence score to report, and a
fabricated one would be worse than an honest placeholder. Table cell
structure is the other deliberate exclusion: tables get flattened here
into a pipe-separated text rendering,
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
claim at all, because it looks verified when it isn't. The other rules
that shape a good extraction pass are almost as load-bearing: one
checkable assertion per claim rather than a compound sentence trying to
do two jobs, entity mentions merged under their most formal name with
variants kept as aliases instead of spawning duplicate rows, and a hard
rule against forcing a claim or relation the source text doesn't
actually support just to hit some notional quota.

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
into it and bumps its timestamp rather than creating a near-duplicate
file. That's exactly the discipline `research-knowledge-graph` already
applies to web research, applied here to PDF-derived content instead.

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
logged to `wiki_pages_written.json` in the run directory: the record of
which pages this run created or updated. The stage finishes by running
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
over wiki pages and claims, embedded locally (`qwen3-embedding:8b` by
default, served through Ollama; a plain sentence-transformers name works
too). The two engines cover each
other's blind spots: FTS finds the exact term you typed, vectors find the
paragraph that says the same thing in different words.

Building the index is one command, the skill's
`python scripts/compile.py [--index-dir DIR]`, after a one-time
`pip install "deep-research-toolkit[compiler]"` for the DuckDB, LanceDB,
and sentence-transformers dependencies. The embedding model is a
one-time cost too: an `ollama pull` for the default Ollama embedder, or
an automatic first-run download for a sentence-transformers one (the
same offline-after cost as Docling's models). Every run is fully local,
and it prints row counts on
success so you can sanity-check that the corpus you expected actually got
indexed. The build also refuses two footguns before touching anything:
it won't compile into the project root or into the knowledge base itself,
and it won't delete a non-empty directory unless it contains a
`knowledge.duckdb` from a previous build.

Compilation is a full rebuild every run, on purpose. At the scale this
toolkit actually serves (a per-project knowledge base on one machine), a
rebuild costs seconds, and it means there is no cache-invalidation state
to reason about: the index is either current or stale, and stale is fixed
by recompiling. For the same reason you'll want the index directory
(`.deepresearch/index/` by default) git-ignored rather than committed.
The run directories and the wiki are the record; the index is derived from
them and any checkout can regenerate it.

The one piece of judgment-free normalization the compiler does is the
`evidence_ref`: PDF claims cite `node_id` + `page`, web claims cite a
chunk locator + URL, and the compiler maps both into a single
producer-agnostic evidence shape as it indexes. Neither producer's files
change on disk; the asymmetry is absorbed in one function at index time,
which is what lets every retrieval tool below treat evidence uniformly
regardless of where it came from. That's also the boundary of what
compilation does: it maps fields, it never interprets them. There is no
LLM call anywhere in a compile, no summarizing, no deduplication
judgment -- if two runs extracted overlapping claims, both are indexed,
and deciding what they add up to is left to the agent querying them.
The full index schema and normalization
mapping are in `docs/contracts/knowledge-compiler.md`, and the build-time
design decisions in `docs/decisions/0002-knowledge-compiler.md`.

### retrieval-planner

The query half of the compiler layer: eight small tools over the compiled
index, exposed as one CLI (`scripts/query.py`) that prints JSON. Six are
plain lookups: `search_wiki` and `search_claims` (each runs both the
lexical and the vector engine and fuses the two rankings with reciprocal
rank fusion, the standard k=60, so a hit only one side finds still
surfaces), `read_page` (fetch one wiki page whole), `get_entity` (an entity
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
supporting quote is an exact substring of the chunk it cites, re-read
from the run directory's `chunks.jsonl`. Anything else lands in `rejected`
with an explicit reason, never silently mixed in with the verified
material. The result is the artifact the whole pipeline exists to produce:
a set of claims an agent can answer from, where every line traces back to
a quote that actually appears, character for character, in a real source.

Two failure modes are handled without drama. If there's no index at all,
opening it fails immediately with a message pointing at the
knowledge-compiler skill, rather than returning empty results that look
like an empty corpus. If the vector tables can't be opened but the DuckDB
side can, search degrades to lexical-only with a logged warning instead
of dying -- worse recall, but every result still real. The exact
arguments and output shape of each tool are specified field by field in
`skills/retrieval-planner/references/tool-contracts.md`, and
[The retrieval tools](#the-retrieval-tools) below walks through all
eight with examples.

### The local LLM backend

Everywhere this toolkit needs judgment (deciding what counts as an atomic
claim, merging entity mentions, synthesizing a wiki page), the work goes
to an LLM, and the default worker is a local model stack. That is what
`llm.provider: local` means, and it is what `drt init` writes: an
OpenAI-compatible endpoint (Ollama on `:11434/v1` by default; vLLM on
`:8000/v1` works the same) serving `qwen2.5:7b-instruct` as the flat
fallback model, a per-role Qwen model for each pipeline phase under
`llm.roles`, and `qwen3-embedding:8b` for embeddings. It expects a
running endpoint with those models pulled; [Running local
models](#running-local-models) covers the operational details.

Under this provider, extraction runs programmatically: `python
scripts/extract_claims.py <run_dir>` (the script ships in both
`knowledge-extraction` for PDF runs and `research-knowledge-graph` for
web runs) reads a run's `chunks.jsonl`, hands the model bounded batches
of chunks, and writes `claims.jsonl`, `entities.jsonl`, and
`relations.jsonl` back into the run directory. The prompt is a task
brief, not a schema dump: it states the goal, the typed output contract,
and the verbatim-quote invariant as a checkable precondition, then lets
the model plan its own pass. A batch whose output can't be parsed
(usually a model running out of tokens mid-reasoning) is retried as
smaller halves, and what still fails is surfaced as a `parse_failures`
count rather than recorded as "no claims here". The payoff is
throughput: forty sources overnight without spending agent context on
each one.

The opt-out is `llm.provider: agent` in `.deepresearch.yml` (the config
also accepts `anthropic` as a synonym): no local models at all, with the
in-session agent itself doing the LLM work, reading the relevant
SKILL.md. It asks nothing of your machine -- no server to run, no models
to pull, no API key to set -- and the backend enforces the division
honestly rather than papering over it: ask it to complete a prompt and
it raises an error on purpose, with a message explaining that under this
provider a script doing its own extraction is a usage mistake, not a
missing dependency.

The reason a local model is safe to trust here is the verbatim gate. Before
anything is written, every proposed claim runs through the same
exact-substring check (`common/verbatim.py`) that `compose_dossier` and
the eval harness apply: each evidence quote must appear character for
character in the chunk the model was shown, or the claim is auto-dropped
into a `dropped` list instead of `claims.jsonl`. A smaller local model
can therefore only under-produce -- fewer claims survive -- never corrupt
the corpus with paraphrases dressed up as citations.
`scripts/validate-local-llm.py` measures exactly that trade: how much of
a reference extraction a given model recovers, and how many of its
proposals the gate had to drop.

What this backend deliberately is not: a switch that moves the whole
toolkit onto programmatic model calls. Claim extraction is the main
programmatic caller, because it is the one high-volume step whose output
is mechanically checkable; wiki synthesis, conflict adjudication, and
research planning remain agent work even under `provider: local`. Getting
a local model to perform well at extraction has its own operational
details -- the chat template a reasoning model needs, generous token
budgets, the `llm.roles` map that routes each phase to a model suited to
it -- all covered in [Running local models](#running-local-models) below.

## The retrieval tools

Everything above this point spends effort: fetching, converting,
extracting, compiling. Retrieval spends almost none. The eight tools in
`src/deep_research_toolkit/compiler/tools.py` are plain Python over the
compiled DuckDB and LanceDB tables, and none of them makes an LLM call
at query time. That is a design rule (ADR 0001), not a current
limitation: a tool with a model call inside has unpredictable cost and
output that varies between runs, while these behave like database
queries. Ask twice, get the same answer twice.

Cheap and deterministic is also what makes the tools composable. An
agent answering a question does not need to budget its calls: it can
search claims, read the pages that surfaced, pull an entity, walk its
neighbors, then compose a dossier from the claim ids it collected, all
in one conversation. The judgment that makes those answers worth
trusting happened upstream -- extraction decided what counts as a claim
and pinned each one to a verbatim quote, the wiki writer merged sources
and flagged conflicts, the compiler froze the result into tables. Query
time only reads.

When a tool takes a query string, retrieval is hybrid. The query runs
through both engines, and the two rankings are fused with reciprocal
rank fusion (`compiler/search.py`, the standard k=60), so a hit that
only one engine finds still competes for a spot in the final list:

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

The eight tools, as implemented in `tools.py`. The retrieval-planner CLI
(`skills/retrieval-planner/scripts/query.py`) exposes each one as a
kebab-case subcommand that prints JSON.

| Tool | Arguments | Returns |
|------|-----------|---------|
| `search_wiki` | `query, k=8` | Fused-ranked wiki pages: path, title, type, status, and a 200-char snippet each |
| `read_page` | `path` | One full wiki page: body plus parsed frontmatter |
| `search_claims` | `query, k=8, producer=None` | Fused-ranked claims with their evidence rows; `producer` filters to `pdf` or `web` |
| `get_entity` | `name_or_id` | The entity's id, name, type, aliases, mentions, and every relation it appears in |
| `neighbors` | `entity, depth=1` | Graph neighbors, from a depth-bounded walk over the relation table |
| `get_sources` | `page=` or `claim=` | Provenance: a page's source frontmatter, or the distinct sources behind a claim |
| `find_contradictions` | none | Contradiction candidates: conflicting relation triples plus `conflicted` pages |
| `compose_dossier` | `query=None, claim_ids=None, k=12` | An evidence dossier split into `included` and `rejected` claim lists |

`compose_dossier` is the tool the rest of the pipeline exists to feed.
Given a query it takes the top `k` claim hits (default 12); given
explicit `claim_ids` it skips search entirely. For each claim it gathers
the evidence rows and applies the verbatim gate: every supporting quote
must appear character for character in the chunk it cites, where the
chunk text is re-read from the run directory's `chunks.jsonl` and
matched by the evidence's locator -- the same shared check in
`common/verbatim.py` that extraction and the eval harness apply. A claim
with no evidence, or with even one quote that fails the substring test,
lands in `rejected` with an explicit reason; only claims whose every
quote survives reach `included`. Note what the gate checks against: the
chunk the extractor was actually shown, not a rendered PDF page or a
whole `source.md`. Quote and source text are resolved the same way at
every stage, so a claim admitted at extraction time is never silently
rejected here because two stages disagreed on what "the source" means.

The JSON shape is for agents; `--format md` on the CLI's
`compose-dossier` subcommand is for deliverables. It renders `included`
as a self-citing markdown dossier: each claim numbered, its verbatim
quotes as blockquotes with source and locator inline, and a closing note
counting anything omitted as not verbatim-verifiable. The document
carries its own audit trail, so a reader can check any line against the
cited source without ever opening a run directory.

`find_contradictions` returns candidates, not verdicts. Mechanically, it
reports two things: relation rows where a single `(subject, predicate)`
pair maps to more than one distinct object, and wiki pages whose status
is `conflicted`. Plenty of candidates are innocent -- a person can hold
two titles, a project can have two funders -- and deciding which ones
are real contradictions takes judgment. That confirmation is an agent
step, run as one batched pass over the whole candidate list (the
retrieval-planner SKILL.md drives it), never a model call hidden inside
the tool.

## Running local models

The pipeline calls an LLM in only a few places, and those places want
different models. Claim extraction runs once per chunk-batch per source
-- hundreds of calls over a real corpus -- and wants a fast instruct
model that emits JSON and stops. Synthesis is a single judgment call at
the end of a run and benefits from a model that reasons before it
answers. Contradiction review sits with synthesis; wiki writing sits
closer to extraction. Serving all of them with one model is
systematically wrong for at least one of them.

So under `provider: local` the config routes each phase to its own
model. `llm.roles` maps a role name to a model, sampling settings, a
thinking mode, and a response format, and
`get_backend(config, role="extract")` resolves the backend for that
phase. Each role ships with its own default model (the Qwen stack
below), and the fallback is back-compat friendly: a role that doesn't
name a model uses the flat `llm.local.model` if your project set that
key explicitly, and the role's shipped Qwen default otherwise. A
single-model setup keeps working unchanged.

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
  role without a model     -> flat llm.local.model if set explicitly,
                              else the role's shipped Qwen default
```

The per-role defaults (`ROLE_DEFAULTS` in `config.py`) encode what each
phase needs, model included -- this table is the default stack:

| Role | Model | Thinking | `temperature` | `max_tokens` | `response_format` |
|------|-------|----------|---------------|--------------|-------------------|
| `extract` | `qwen2.5:7b-instruct` | off | 0.0 | 3000 | `json` |
| `wiki_write` | `qwen3.6:35b-a3b` | off | 0.2 | 4096 | -- |
| `conflict_adjudicate` | `qwen3.6:27b` | on | 0.2 | 8192 | -- |
| `synthesize` | `qwen3.6:27b` | on | 0.4 | 12000 | -- |
| `code_agent` | `Ornith-1.0-9B` | on | 0.6 | 16000 | -- |

`top_p` (0.95) and `top_k` (20) come from `llm.local` unless a role
overrides them, as do `base_url` and `api_key_env`. `drt init` writes
this stack out explicitly (the full block is in
[Configuration](#configuration)), and any field of any role can be
overridden under `llm.roles.<role>` in `.deepresearch.yml`. To run
everything on one model instead, set `llm.local.model` and skip the
`roles` block:

```yaml
llm:
  provider: local
  local:
    base_url: http://localhost:11434/v1
    model: qwen2.5:7b-instruct   # every role runs on this one model
```

One finding from testing is worth stating plainly: a model's
"non-thinking" switch is a promise about the model, not about your
serving stack. Two model families that advertise one -- the reasoning
model Ornith-1.0-9B and the hybrid model `qwen3.5:9b` -- both ignored
every attempt to disable thinking under the Ollama builds we tested,
whether via the `think: false` request parameter or a `/no_think`
prompt directive. Each reasoned until it hit the token ceiling and
returned nothing.

That failure lands hardest on `extract`, the one role that runs at
volume and wants an immediate JSON answer. The fix was not to wait for
the switch to start working; it was to stop depending on a switch. A
true instruct model such as `qwen2.5:7b-instruct` has no reasoning path
to suppress, so there is nothing for the serving layer to get wrong.
"Does this model support non-thinking mode" turns out to be the wrong
question. The right one is "does this serving stack, at this version,
honor that switch" -- and the only way to answer it is to test the exact
(model, server, version) triple you intend to run.

Getting a reasoning model to work at all involves two more serving
details. First, the model needs its real chat template: the stock
`Ornith-1.0-9B-GGUF` on Hugging Face ships without one, so Ollama falls
back to a raw passthrough that silences the model's `<think>` reasoning
and can degenerate into repetition loops. Build a local model from it
with a proper ChatML `TEMPLATE` (`ollama create <name> -f Modelfile`)
and point `llm.local.model` at that name. Second, budget tokens
generously -- `max_tokens` defaults to 16000 because a reasoning model
spends thousands of tokens thinking before it answers, and a low cap
truncates it mid-thought -- and keep the documented sampling
(`temperature` 0.6, `top_p` 0.95, `top_k` 20): lowering the temperature
for "determinism" backfires into repetition on this model family.

Embeddings are a separate axis, set by `llm.embedding_model` and routed
by the shape of the name: an Ollama tag -- anything with a `:`, like
the default `qwen3-embedding:8b` -- embeds through the same local
endpoint, while a plain sentence-transformers name (`all-MiniLM-L6-v2`,
for example) runs through sentence-transformers. LanceDB infers the vector
dimension from the model's output, so swapping embedding models is a
one-line config change and a recompile, with no schema migration. To
judge whether a candidate generative model is good enough,
`scripts/validate-local-llm.py` runs it against the reference
extraction fixture and reports how much of the reference it recovers
and how many of its proposals the verbatim gate dropped. It is a manual
harness, not part of CI.

## What the pipeline guarantees

### The gate, in depth

One function decides whether a quote counts as evidence: `verbatim_ok`
in `common/verbatim.py`, an exact-substring check with no
normalization. The quote must appear in the cited chunk's text
character for character, contiguously, or the claim is dropped. That
check runs at three points in the pipeline -- when the extractor
proposes a claim (`llm/extract.py`), when the dossier is composed
(`compiler/dossier.py`), and in the eval harness (`pdf/eval.py`) -- and
all three import the same function. Each stage also re-derives the text
it checks against from disk, reading the chunk out of the run
directory's `chunks.jsonl` rather than trusting the compiled index or
whatever an earlier stage recorded.

That redundancy is deliberate. Because every checkpoint resolves the
source text independently, a bug in one stage -- or a hand-edit to an
intermediate file -- cannot launder an unverified claim into the final
output; the next gate re-checks the quote from scratch. The three
checkpoints used to be three separately written checks, and they had
drifted on one detail: what "the source text" means for a quote that
spans two structural units within a chunk. A claim could pass
extraction and then be silently rejected downstream, purely because the
gates disagreed by construction. They were unified into the one
`common.verbatim` function with one definition of the chunk text, and a
regression test pins the spanning case, so a quote that passes one gate
passes all three.

### What it does and doesn't protect against

The guarantee is narrow, and worth stating exactly: every claim in the
record carries at least one quote that appears verbatim in the chunk it
cites. That is a hard bound on fabrication. Put a weak local model
through the pipeline and its failure mode is fewer claims -- proposals
with bad quotes die at the gate -- rather than invented facts sitting
in the record. A weak model can only under-produce; it cannot corrupt
what gets admitted.

The gate does not judge meaning. A model can attach a perfectly real
quote to a conclusion the quote does not actually support, and the
substring check will pass, because relevance is a judgment call and the
gate is mechanical on purpose. Nor does the gate make a source right:
it guarantees fidelity to the source, not truth, so a source's own
errors pass straight through as grounded claims, exact quote and all. A
verbatim quote is also not necessarily a clean one -- a fragment cut
off mid-sentence is still an exact substring.

So read "verified" as "traceable." The pipeline makes "where did this
come from?" mechanically answerable, down to an exact quote in an
archived source. Whether that source is correct, and whether the quote
means what the claim says it means, still takes a reader.

## Configuration

Everything project-specific lives in one file, `.deepresearch.yml`, at
your project's root. Here it is in full, annotated. The shape below is
exactly what `drt init` writes:

```yaml
# .deepresearch.yml -- deep-research-toolkit project configuration
version: 1                           # format version of this config file itself

knowledge_base:
  path: knowledge_base               # the compiled wiki (OKF markdown) lives here
  pdf_runs_dir: pdf-runs             # one run directory per ingested PDF
  research_runs_dir: research-runs   # one run directory per web-research source
  index_dir: .deepresearch/index     # DuckDB + LanceDB index; rebuilt on demand, so keep it git-ignored

topic:
  name: "Perovskite stability"       # what "research X for the knowledge base" resolves X to
  scope_hint: >                      # how the skills know what is in scope
    Degradation mechanisms and encapsulation,
    not manufacturing economics.
  tags: []                           # free-form labels; empty is fine

features:                            # written by `drt init` from --tier; edit freely later
  web_research: false                # research-knowledge-graph (--tier web or full)
  pdf_ingestion: true                # the seven-stage PDF pipeline (--tier pdf or full)
  knowledge_compiler: false          # index + retrieval layer (--tier compiler or full)

llm:
  # Local, role-routed Qwen stack served by Ollama is the default. It needs a
  # running Ollama endpoint (see llm.local.base_url) with the models below
  # pulled. To run without local models -- letting the in-session agent do the
  # extraction by hand instead -- set: provider: agent
  provider: local                    # local | agent ("anthropic" is a synonym for agent)
  embedding_model: qwen3-embedding:8b  # Ollama tag; a plain sentence-transformers name works too
  local:                             # only read when provider: local
    base_url: http://localhost:11434/v1  # any OpenAI-compatible endpoint (Ollama, vLLM, ...)
    model: qwen2.5:7b-instruct   # flat fallback (role=None, and any role below without its own model)
    api_key_env: OPENAI_API_KEY      # local servers usually ignore the key; the var can stay unset
    temperature: 0.6                 # flat defaults; roles override these per phase
    top_p: 0.95
    top_k: 20
    max_tokens: 16000
  # Per-phase models. extract stays a true instruct model (qwen2.5:7b-instruct),
  # NOT qwen3.5:9b -- under the Ollama builds tested it ignored non-thinking mode
  # and produced nothing on extraction.
  roles:
    extract:
      model: qwen2.5:7b-instruct
    wiki_write:
      model: qwen3.6:35b-a3b
    conflict_adjudicate:
      model: qwen3.6:27b
    synthesize:
      model: qwen3.6:27b
    code_agent:
      model: Ornith-1.0-9B

scrapling:                           # web retrieval behavior (web tier)
  default_mode: http                 # http | stealth; stealth drives a real browser
  rate_limit_seconds: 1.0            # minimum delay between fetches
```

Discovery mirrors `.git`: the skill scripts call `load_config()`, which
walks upward from the current directory until it finds a
`.deepresearch.yml`, so commands work from any subdirectory of the
project. Nothing in any skill hardcodes a topic, a directory name, or a
research scope; if it is project-specific, it comes from this file. `drt
init` writes the starter non-interactively, entirely from flags --
`--tier` sets the `features.*` block (`web`, `pdf`, `compiler`, or
`full`), and `--topic-name`, `--scope-hint`, and `--knowledge-base` fill
in the rest. The `version: 1` at the top covers only this config file.
The artifacts the pipeline writes -- manifests, chunks, claim files, OKF
frontmatter -- each carry their own `schema_version` field, and
`docs/contracts/schema-versions.md` is the registry mapping suite
versions to the schema versions they produce and accept.

The `llm.roles` block deserves the extra explanation it gets in [Running
local models](#running-local-models): different pipeline phases want
different models, so under `provider: local` each of the five roles --
`extract`, `wiki_write`, `conflict_adjudicate`, `synthesize`,
`code_agent` -- can point at its own model with its own sampling
settings, thinking mode, and response format. The defaults live in
`ROLE_DEFAULTS` in `config.py`, and the fallback splits by field.
`model` follows a back-compat rule: a role that doesn't name one uses
the flat `llm.local.model` if your project set that key explicitly, and
the role's shipped Qwen default otherwise. Leave `base_url`,
`api_key_env`, `top_p`, or `top_k` out of a role and it inherits the
flat `llm.local` value; leave out `thinking`, `temperature`,
`max_tokens`, or `response_format` and the role gets its own
`ROLE_DEFAULTS` value instead. Omit `temperature` under `extract`, say,
and you get 0.0 (the extract default), not whatever `llm.local` sets. A
single-model setup still needs no `roles` block at all: set
`llm.local.model` and every role runs on that one model, with per-phase
reasoning and sampling defaults that suit each phase.

## Verification and testing

CI runs in two tiers, split by dependency weight rather than by
importance. The fast tier runs on every push to main and every pull
request, on both Ubuntu and Windows. It installs the `dev` and `pdf`
extras plus DuckDB and LanceDB directly, but skips the `compiler` extra,
so sentence-transformers never lands in the environment: the compiler
tests run against real DuckDB and LanceDB, with an injected fake
embedder (deterministic, hash-derived vectors) standing in for
sentence-transformers, so the whole compile-and-query path is exercised
end to end without downloading or loading an embedding model. The same
job checks that the two plugin manifests and the skill templates are in
sync, runs the fast suite (dozens of unit and light integration tests)
with coverage, and lints with ruff; an advisory mypy job and a pip-audit
job run alongside it. This tier is the gate every change has to pass.

The heavy tier is the one that touches real models, and it never runs on
an ordinary push: its triggers are a weekly cron (Monday mornings, UTC)
and manual dispatch. It installs the full `[compiler]` extra and runs
the tests marked `heavy`: one drives the entire PDF pipeline through a
real Docling conversion and requires a perfect score on every eval
check, the other compiles a real corpus with the real `all-MiniLM-L6-v2`
embedding model and exercises every retrieval tool against the result.
It reports rather than blocks, because its first run downloads models
and can fail for reasons that have nothing to do with the code. Both
tiers measure against the reference material in `tests/fixtures/`: a
small generated PDF (with the script that generates it), a complete
reference run directory for that PDF holding every stage's output from
classification through the eval reports, a matching web-run directory,
and a small reference knowledge base.

The same checks work outside CI. To verify a run you just produced,
point the eval harness at its run directory
(`python .claude/skills/rag-eval-harness/scripts/run_eval.py <run_dir>`)
and read the six mechanical checks it reports -- verbatim quotes, valid
page citations, recovered headings, accounted-for tables and figures --
rather than settling for "nothing crashed." To judge whether a local
model is up to extraction duty, `scripts/validate-local-llm.py` runs it
against the reference extraction fixture and reports how much of the
reference it recovers and how many of its proposals the verbatim gate
had to drop. That one is a manual harness for when you are choosing a
model, not part of CI.

## Status and roadmap

**Built and tested:** both producer stacks and everything downstream of
them. On the producer side, the web-research skill
(`research-knowledge-graph`, including its claim-extraction step) and
the seven-stage PDF pipeline. On the consumer side, the compiler layer:
the `knowledge-compiler` and `retrieval-planner` skills, at suite
version 0.2.0. Around all of it, the `drt` CLI (`init`, `upgrade`,
`doctor`, `migrate`), the role-routed local backend (`llm.roles`, so
each pipeline phase runs on a model suited to it), and the dual plugin
manifests that serve Claude Code and Codex from one shared `skills/`
tree. The fast suite covers this on every push to main and every pull
request, and the two heavy
integration tests exercise exactly what the fast suite fakes: real
Docling conversion of the fixture PDF with a perfect eval score
required, and a real-embedding compile with every retrieval tool run
against the result. All of it has been verified against a real installed
package, not just a development checkout, so what's described above is
what actually runs.

**Designed, not yet built:** five items, each deferred deliberately,
with the reasoning recorded in
`docs/decisions/0002-knowledge-compiler.md` (and, for the older ones, in
`docs/decisions/0001-architecture.md`):

- **Incremental compilation.** The compiler rebuilds the index from
  scratch on every run. At per-project scale a rebuild is seconds, and
  it removes cache-invalidation bugs as a category; re-embedding only
  changed pages waits until a real corpus makes rebuild time hurt.
- **A learned reranker stage.** Retrieval today fuses lexical and vector
  rankings with RRF, which is cheap and deterministic. A reranker sits
  on ADR 0001's deferred list, not in the code.
- **GraphRAG-style community detection.** Flagged in ADR 0001 as
  premature at per-project, single-machine scale, and nothing since has
  changed that judgment.
- **An MCP query server.** The retrieval tools are CLI-first on purpose;
  ADR 0001 calls a read-only MCP layer over the finished knowledge base
  a reasonable later addition, once a corpus outgrows file-scan search.
- **A multi-package split.** One package with extras (`web`, `pdf`,
  `compiler`, `full`) ships today; independent per-tier PyPI packages
  are explicitly deferred past v1.

Each of these was judged premature at the scale this toolkit serves,
rather than forgotten. If one of them starts to matter for your corpus,
the two ADRs are where the reasoning -- and the sketch of what would be
built -- lives.

## FAQ

**How is this different from a generic RAG pipeline?**
Mostly in what it refuses to trust. A claim only enters the corpus if
its supporting quote survives the verbatim gate -- an exact substring
check against the source -- so a model can under-produce but can't
fabricate a citation. The corpus itself is plain files in git; the
DuckDB + LanceDB index is just a rebuildable cache over them. And the
two producers (PDF and web) feed one graph, their different evidence
shapes normalized into a single `evidence_ref` at index time.

**Do I need a GPU?**
For the shipped defaults, realistically yes. The default is the local
Qwen stack served by Ollama -- roughly 7B to 35B models for the
generative roles, plus `qwen3-embedding:8b` for embeddings -- and models
that size want a GPU (see [Running local
models](#running-local-models)). To run without local models, set
`llm.provider: agent`, which has the in-session agent do the extraction
and synthesis by hand, and point `llm.embedding_model` at a small
sentence-transformers model such as `all-MiniLM-L6-v2`; then every stage
is deterministic Python plus a CPU-friendly embedder at compile time.

**Which local model should I use?**
No single model is right for every phase, which is what `llm.roles` is
for. Use a genuinely non-reasoning instruct model (something like
`qwen2.5:7b-instruct`) for `extract`, since that role runs at volume and
wants immediate JSON, and a reasoning model for `synthesize` and
`conflict_adjudicate`. `scripts/validate-local-llm.py` measures a
candidate against the reference extraction fixture before you commit to
it.

**Is my data sent anywhere?**
Not by default. The default provider is local: the corpus, the index,
the embeddings, and every model call stay on your machine, and no tier
requires a hosted API key. Fetching a web source talks to that website,
of course. Your corpus leaves the machine only if you point
`llm.local.base_url` at a remote endpoint yourself, or if you run under
`llm.provider: agent`, where the in-session agent's own hosted API sees
whatever it reads.

**Can I mix PDFs and web sources in one knowledge base?**
Yes -- that's the normal case. Both producers write run directories with
the same claims/entities/relations schemas, the compiler indexes both
and normalizes their different evidence shapes at index time, and every
retrieval tool works over the result without caring which side a claim
came from.

**Does it work in Codex as well as Claude Code?**
Yes. There is one shared `skills/` tree serving both platforms through
dual plugin manifests (`.claude-plugin/` and `.codex-plugin/`), and
`drt init` copies the skills into both `.claude/skills/` and
`.agents/skills/`. CI checks the two manifests never drift.

**What happens when a model hallucinates a quote?**
The claim is dropped. A paraphrased or invented quote fails the
substring test no matter how plausible it reads, so it never reaches
the corpus -- under the local backend it lands in an explicit `dropped`
list, and `compose_dossier` re-checks every quote later and reports
rejects with reasons rather than passing them through.

**So everything in the corpus is true?**
No. The gate guarantees fidelity to the source, not truth -- a source's
own errors pass straight through as grounded claims, and a quote being
present doesn't prove it supports the claim built around it. What you
get is that "where did this come from?" always resolves to a named
source and an exact quote instead of a model's say-so.

## Glossary

- **OKF (Open Knowledge Format)** -- the convention every knowledge-base
  page follows: markdown with YAML frontmatter (required `type`,
  `title`, `timestamp`; optional fields like `status` and `aliases`),
  one concept per file, cross-linked with relative markdown links that
  are the graph's edges. Defined in `docs/contracts/okf-frontmatter.md`.
- **evidence_ref** -- the single producer-agnostic evidence shape
  (`producer`, `source_id`, `locator`, `quote`, optional `page` and
  `url`) the compiler normalizes each claim's evidence into at index
  time. On-disk run files keep their native shapes; everything
  downstream of the index sees only this one.
- **dossier** -- the output of `compose_dossier`: the claims selected to
  answer a query, split into `included` and `rejected`, where inclusion
  requires at least one evidence row and every quote passing the
  verbatim gate against its source text.
- **RRF (Reciprocal Rank Fusion)** -- the rule hybrid search uses to
  merge the lexical (BM25) and vector rankings: each result scores the
  sum of `1 / (k + rank)` over the lists it appears in, with k = 60.
  Rank-based, so the two incomparable score scales need no calibration.
- **provenance** -- the layout facts markdown conversion throws away,
  recovered into `provenance.jsonl`: one line per structural unit with
  its page, section path, bounding box, and content hash. Page
  citations are validated against this file.
- **run directory** -- the per-source working set,
  `pdf-runs/<document_id>/` or `research-runs/<source_id>/`: manifest,
  chunks, claims, entities, relations, and eval output for one
  document, git-tracked as the auditable record of how a conclusion was
  reached.
- **producer** -- which side wrote a run, `pdf` or `web`; the only
  distinction `evidence_ref` preserves after normalization.
- **chunk / node** -- one retrieval unit in `chunks.jsonl`, cut along
  document structure (one node per heading section, tables and figures
  as their own nodes), never a fixed token-count split. Each carries a
  `node_id`, a type, a section path, and its source pages and units.
- **claim** -- one factual assertion in `claims.jsonl`, with a type, a
  confidence, and supporting evidence whose quote must be verbatim from
  the cited source.
- **the verbatim gate** -- the exact-substring check (`verbatim_ok` in
  `common/verbatim.py`) a claim's quote must pass against its cited
  source text; enforced at extraction, dossier composition, and eval,
  with no normalization and no fuzzy matching.
- **role (`llm.roles`)** -- a named pipeline phase (`extract`,
  `wiki_write`, `conflict_adjudicate`, `synthesize`, `code_agent`) that
  the local provider routes to its own model, sampling, thinking mode,
  and response format. A role without its own model uses the flat
  `llm.local.model` if the project set one explicitly, and its shipped
  Qwen default otherwise.
- **the compiler** -- the `knowledge-compiler` stage: a full,
  from-scratch rebuild of the disposable DuckDB + LanceDB index from
  the corpus, normalizing evidence as it goes. If the index and the
  files disagree, the files win -- you recompile.

## Contributing

Development happens against an editable install. Clone the repo, make a
virtualenv, and install with the dev extras:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows; source .venv/bin/activate on Linux/macOS
pip install -e ".[dev,full]"
```

CI's fast tier installs a leaner set --
`pip install -e ".[dev,pdf]" "duckdb>=1.0" "lancedb>=0.15"` -- on
purpose, keeping the `compiler` extra (and with it
sentence-transformers) out of the environment; use that line if you
want to reproduce CI exactly. Either way, `docs/environment.md` covers
the Python version floor and the assets Docling and Playwright download
on first use, well after `pip install` has reported success.

Tests come in two tiers, split by dependency weight. The fast suite,
`pytest -m "not heavy"`, runs on every push to main and every pull
request on both Ubuntu and Windows, and it has to pass before you open a
PR. Tests marked `heavy` make real Docling and embedding-model calls
against `tests/fixtures/`; they run on a weekly cron and manual
dispatch, not on your PR, so run `pytest -m heavy` locally only when
you've touched the PDF core or the embedding path.

The on-disk formats are the load-bearing part of this repo, and their
contracts live in `docs/contracts/` -- manifest keys, JSONL row shapes,
OKF frontmatter, the schema-version registry. If you change any of
them, bump the relevant `schema_version` in
`docs/contracts/schema-versions.md` and add a `CHANGELOG.md` entry
under "Schema changes."

Two sync guards run in CI, and both have a local command:

- `python scripts/check-manifests-in-sync.py` verifies the two plugin
  manifests (`.claude-plugin/plugin.json` and
  `.codex-plugin/plugin.json`) agree on every shared field. Run it if
  you touched either manifest.
- `python scripts/check-skill-templates-in-sync.py` verifies
  `src/deep_research_toolkit/skill_templates/` is a byte-for-byte copy
  of the canonical `skills/` tree -- that copy is what ships inside the
  wheel, compared by content hash so a stale tree can't slip through.
  If you changed anything under `skills/`, regenerate it with
  `python scripts/sync-skill-templates.py`; it's a one-way copy, so
  never edit `skill_templates/` directly.

`ruff check src/ skills/` lints, and CI runs mypy as an advisory job.
`CONTRIBUTING.md` is the short authoritative version of all of the
above.

## License

MIT -- see [`LICENSE`](LICENSE).
