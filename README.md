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
- [A worked example](#a-worked-example)
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

```
+---------------------------------------------------------+
|                    .deepresearch.yml                    |
|             (topic, kb path, feature flags)             |
+---------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------+
| research-knowledge-graph  (web)                         |
| Scrapling fetch -- http or stealth                      |
+---------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------+
| pdf-ingest-router                                       |
| classify (digital-text/scanned/form/...) + route        |
+---------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------+
| pdf-to-canonical-markdown                               |
| Docling conversion -> canonical.md + raw JSON           |
+---------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------+
| pdf-layout-provenance                                   |
| page / section_path / bbox per structural unit          |
+---------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------+
| canonical-markdown-to-llm-nodes                         |
| structure-aware chunking -> chunks.jsonl                |
+---------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------+
| knowledge-extraction                                    |
| tables/figures (code) + claims/entities (LLM)           |
+---------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------+
| llm-wiki-writer                                         |
| merge into the knowledge base, flag conflicts           |
+---------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------+
| knowledge_base/  (Open Knowledge Format)                |
| markdown + YAML frontmatter, cross-linked,              |
| one file per concept, git-tracked                       |
+---------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------+
| rag-eval-harness                                        |
| headings recovered? citations verbatim?                 |
| figures accounted for?  -> eval_report.json             |
+---------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------+
| knowledge compiler                                      |
| DuckDB (full-text + graph) + LanceDB (vectors)          |
+---------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------+
| retrieval-planner tools (8)                             |
| search_wiki / read_page / search_claims / get_entity    |
| neighbors / get_sources / find_contradictions /         |
| compose_dossier                                         |
+---------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------+
| evidence dossier                                        |
| claims + citations, ready to answer from                |
+---------------------------------------------------------+
```

Every PDF gets its own working directory,
`<pdf_runs_dir>/<document_id>/`, holding everything the seven stages
produced: the raw Docling export, the chunked nodes, the claims with their
page citations, the eval report. Web sources substantial enough to mine
for claims get the same treatment in `research-runs/<source_id>/`: the
fetched content, its chunks, and the claims extracted from it. Both kinds
of run directory are git-tracked on purpose: they're meant to be the
auditable record of how a conclusion was reached, not scratch space you'd
`.gitignore`. The compiled index is the one exception — it's a derived
cache, rebuilt from the run directories on demand, and stays out of git.

## Quick start

```bash
pip install "deep-research-toolkit[pdf]"   # or [web], [compiler], [full]
drt init                                    # scaffold .deepresearch.yml + a knowledge base
```

`drt init` asks what this project's research is about and writes that
into `.deepresearch.yml`, alongside where the knowledge base should live.
It also copies the skill files into `.claude/skills/` and `.agents/skills/`
so both Claude Code and Codex can find them in this project. From there,
open the project in either one and ask it to research something, or point
it at a PDF. The relevant skill's `SKILL.md` takes it from there.

To ingest a PDF by hand instead of through an agent session:

```bash
python .claude/skills/pdf-ingest-router/scripts/classify_pdf.py your-file.pdf
# prints a run directory, e.g. pdf-runs/your-file-a1b2c3d4/
python .claude/skills/pdf-to-canonical-markdown/scripts/convert.py pdf-runs/your-file-a1b2c3d4
python .claude/skills/pdf-layout-provenance/scripts/extract_provenance.py pdf-runs/your-file-a1b2c3d4
python .claude/skills/canonical-markdown-to-llm-nodes/scripts/chunk_nodes.py pdf-runs/your-file-a1b2c3d4
# claims.jsonl / entities.jsonl / relations.jsonl need an LLM reading chunks.jsonl --
# that's what knowledge-extraction's SKILL.md is for, inside an actual agent session
python .claude/skills/rag-eval-harness/scripts/run_eval.py pdf-runs/your-file-a1b2c3d4
```

See `docs/environment.md` before you start on Docling/Playwright's
first-run downloads: they're separate from `pip install` and worth
knowing about ahead of time rather than mid-pipeline.

## The skills

Ten skills ship today. Seven of them form the PDF ingestion pipeline, one
handles web research, and two form the knowledge-compiler layer that
indexes and queries what the other eight produce. Every one of them is
deliberately small. That's not an accident. A single monolithic "research
skill" would load its entire instruction set into context on every use,
whether you needed the PDF-parsing details or not. Splitting the work into
ten focused skills means Claude or Codex only pulls in the instructions
for the stage actually running, which keeps any single conversation's
context usage proportional to what it's actually doing.

### research-knowledge-graph

This is the web-research half of the toolkit, and the one meant to be used
directly and repeatedly, not just once per document. It wraps
[Scrapling](https://github.com/d4vinci/Scrapling) for retrieval (plain
HTTP by default, a stealth mode when a site returns a bot-detection
challenge or a flat 403) and stores everything it finds as Open Knowledge
Format pages: markdown files with YAML frontmatter, one concept per file,
cross-linked to each other with ordinary relative markdown links. Those
links aren't decoration; they're the graph's edges, and the skill's lint
step treats them as load-bearing.

The operational discipline here comes from Andrej Karpathy's LLM-wiki
pattern, and it shows up as three named operations: `ingest`, `query`, and
`lint`. `ingest` doesn't just append a new file for every fetch. It
checks the knowledge base first, and if a concept already has a page, it
merges new findings into that existing page and bumps its timestamp rather
than creating a near-duplicate. `query` searches the existing graph and
walks its links before doing any live scraping at all, on the theory that
re-deriving an answer you've already written down is wasted work. `lint`
is the health check: it walks every page looking for orphans (pages
nothing links to), broken links, missing or malformed frontmatter, and
entries marked `researched` that have gone stale past a configurable
threshold.

Every page's frontmatter carries a `status`: `seed` for a placeholder that
exists but hasn't been researched yet, `researched` once it has real
content, `stale` once lint flags it as overdue for a refresh, plus two more
values the PDF pipeline adds (`draft` and `conflicted`, covered under
`llm-wiki-writer` below). That status field is what lets `query` decide
whether an existing page is good enough to answer from, or whether it needs
a fresh `ingest` first.

In practice, using this skill looks like asking an agent to "research X for
the knowledge base." It reads `.deepresearch.yml` to find out what this
project's research scope actually is (never guessing from its own generic
description), searches what's already there, fetches only what's missing,
and writes or updates a page. Nothing about the skill itself is tied to any
particular topic; the topic lives entirely in that one config file.

Web research also does what PDF ingestion has done from the start: turn a
substantial source into evidence-backed claims, not just wiki prose. When
a fetched page is worth mining, `start_research_run.py` scaffolds a
`research-runs/<source_id>/` directory that deliberately mirrors a PDF
run: the fetched content saved verbatim as `source.md`, one chunk per
heading section in `chunks.jsonl`, and a `manifest.json` marking the
producer as `web`. The agent then extracts `claims.jsonl`,
`entities.jsonl`, and `relations.jsonl` into that same directory, under
the same rules the PDF pipeline enforces, chiefly that every supporting
quote must be a verbatim substring of `source.md`. That symmetry is the
point: the knowledge compiler indexes web runs and PDF runs into one
table, so a claim about some entity from a webpage and a claim about the
same entity from a whitepaper end up in the same queryable graph, checked
by the same evidence gate. Only the evidence shape differs (a web claim
cites a chunk locator and a URL; a PDF claim cites a node id and a page
number), and the compiler normalizes that difference away at index time
without touching either producer's files on disk.

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

Everything this stage decides gets written to `classification.json`, and a
`manifest.json` gets started alongside it: the file every later stage
reads `document_id` back out of, appending its own entry rather than ever
overwriting what came before.

### pdf-to-canonical-markdown

The actual PDF parsing happens here, and it happens once: this stage turns
the source file into a canonical markdown rendering plus Docling's own
structured JSON export, and every stage downstream works from those two
outputs instead of re-opening the PDF. `canonical.md` is what a human would
want to read; `docling_raw.json` is what the next few stages actually
parse, kept in Docling's own shape rather than reprocessed into something
this toolkit invented, since downstream code already knows how to walk
Docling's `texts`/`tables`/`pictures`/`pages` structure directly.

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
true document order, not headings first and tables afterward (that ordering
matters: it's enough to attach a table to the wrong section if it happens to
sit between two paragraphs), and emits one record per structural unit: heading,
paragraph, table, figure, caption, or list item, each carrying its page
number, its section path, its bounding box, and a content hash.

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
honest placeholder.

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
says so plainly rather than pretending otherwise. Two of its six outputs
come from scripts: pulling tables out of Docling's cell data into real CSV
files, and pulling figures into PNGs alongside a caption record that never
silently drops a figure just because Docling didn't capture usable pixel
data for it. The other three outputs (`claims.jsonl`, `entities.jsonl`,
`relations.jsonl`) are written directly by an LLM reading the previous
stage's chunked nodes, because deciding whether a sentence expresses one
real, checkable claim, or whether "Hydra" and "Hydra Head" refer to the
same thing, isn't something a regex should be trusted to decide.

The rule that does the most work here is the evidence rule: every claim's
supporting quote has to be copy-pasted verbatim from the source text on the
page it cites, never paraphrased, never cleaned up for readability. That
constraint is checked mechanically downstream by `rag-eval-harness`, and
it's the single property that keeps this toolkit's claims audit-able rather
than merely plausible-sounding. A claim with a quote that doesn't actually
appear on the cited page is worse than no claim at all, because it looks
verified when it isn't. The other rules that shape a good extraction pass
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
responsible for. The stage finishes by running the knowledge base's
lint check. A wiki-writer run that leaves the graph broken (an orphaned
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

Six mechanical checks run against whatever's in a run directory: whether
every heading recovered from the canonical markdown made it into some
chunk's section path, whether the table count in the provenance data
matches the CSV count on disk, whether every claim's cited page actually
exists, whether every claim's quote is genuinely verbatim on that page,
whether every figure reference was either extracted or explicitly flagged
as not extracted, and whether any page's text looks like OCR garbage. Each
one is designed to fail loudly on the specific corruption it exists to
catch, rather than passing by default. The harness is also safe to run
against a half-finished pipeline: any check whose input files don't exist
yet is marked skipped, not failed, so `pass_rate` always reflects only the
checks that had something real to check.

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
