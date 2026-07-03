# deep-research-toolkit

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![CI](https://github.com/CharlesHoskinson/deep-research-toolkit/actions/workflows/ci.yml/badge.svg)

A deep-research skill suite for **Claude Code** and **Codex**: web
retrieval, PDF ingestion, and knowledge compilation into an evidence-backed
knowledge base an agent can query, cite, and keep building on, instead of
starting over on every question.

## Why this exists

Ask an LLM the same research question twice, a week apart, and it does the
same work twice: fetch the same pages, re-read the same PDF, re-derive the
same conclusions from scratch. Nothing it learned the first time sticks
around. That's fine for a one-off question. It's a bad foundation for
anything that compounds: a research project, an internal knowledge base, a
standards comparison that gets revisited for months.

This toolkit exists to fix that specific problem: every fetch, every PDF,
every claim gets written down once, in a durable format, with the evidence
attached, so the tenth question about a topic is answered by reading what's
already there instead of re-scraping the internet. The wiki isn't the
knowledge. It's the source corpus. The claims and citations built from it
are what an agent actually reasons over.

Two pieces of this are built and tested today: a web-research skill and a
seven-stage PDF ingestion pipeline. Both write into the same knowledge base,
using the same format, so it doesn't matter whether a fact came from a
webpage or a whitepaper: it ends up in one place, checkable the same way.

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

   - - - - - - -  designed, not yet built  - - - - - - -   
                            |
                            v
+---------------------------------------------------------+
| knowledge compiler                                      |
| DuckDB (full-text + graph) + LanceDB (vectors)          |
+---------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------+
| retrieval-planner tools                                 |
| search_wiki / search_claims / get_entity / ...          |
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
page citations, the eval report. It's git-tracked on purpose: it's meant to
be the auditable record of how a conclusion was reached, not scratch space
you'd `.gitignore`.

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

Eight skills ship today. Seven of them form the PDF ingestion pipeline;
the eighth handles web research. Every one of them is deliberately small.
That's not an accident. A single monolithic "research skill" would load
its entire instruction set into context on every use, whether you needed
the PDF-parsing details or not. Splitting the work into eight focused
skills means Claude or Codex only pulls in the instructions for the stage
actually running, which keeps any single conversation's context usage
proportional to what it's actually doing.

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
  provider: anthropic
  model: claude-sonnet-4-5
  api_key_env: ANTHROPIC_API_KEY

scrapling:
  default_mode: http
  rate_limit_seconds: 1.0
```

`drt init` writes a starter version of this file and asks what tier you
need (`web`, `pdf`, `compiler`, or `full`), which sets the `features.*`
flags accordingly. See `docs/contracts/pdf-ingestion-pipeline.md` and
`docs/contracts/okf-frontmatter.md` for the full schema every artifact in
this toolkit follows, including the `schema_version` fields that make
future changes to these formats detectable rather than silent.

## Status and roadmap

**Built and tested:** both skill stacks above (web research and the
seven-stage PDF pipeline), including the `drt` CLI, the dual Claude
Code/Codex plugin manifests, 36 fast unit tests, and a heavy integration
test that runs the entire PDF pipeline through real Docling conversion
against a test fixture and checks for a perfect score on every eval
check. All of it has been verified against a real installed package, not
just a development checkout, so what's described above is what actually
runs.

**Designed, not yet built:** the knowledge-compiler layer sketched in the
lower half of the diagram above: a hybrid DuckDB and LanceDB index over
everything the two pipelines have produced, and a small set of retrieval
tools (`search_wiki`, `search_claims`, `get_entity`, `neighbors`, and a
few others) that would let an agent query the accumulated knowledge base
directly instead of grepping files by hand. The full reasoning behind that
design (including what was deliberately left out, like GraphRAG-style
community detection and a hosted query server, both judged premature at
the scale this toolkit actually runs at) is in
`docs/decisions/0001-architecture.md`.

## Contributing

See `CONTRIBUTING.md` for the development setup, test tiers, and what to
update when you change an on-disk format. `LICENSE` covers the terms:
MIT, so build on it freely.
