# What This Pipeline Taught Us — and What It Lets Us Write

A synthesis of a five-agent examination of the deep-research-toolkit: its
architecture, the correctness of its pipeline, the lessons from running it on
local models, what it can actually produce, and the state of its tests. Each
agent read the code and the real end-to-end snail run independently; the
findings below are where they converged, and where they found problems worth
fixing.

## The conclusion in one sentence

The value of this pipeline is not that a model reads sources and writes an
answer — models already do that — it is that every factual claim the system
records is mechanically tied to a verbatim quote from a named source, so the
question "where did this come from?" always has a checkable answer instead of
a model's say-so; that one property is what turns a research assistant into a
research *instrument*.

## What the pipeline is

Underneath the ten skills there is a single, disciplined idea: separate three
things that most retrieval-and-generation systems blur together. The first is
a durable evidence corpus — markdown wiki pages and per-run JSONL files, kept
in git, treated as the audit trail. The second is a disposable derived index —
DuckDB for full-text and graph queries, LanceDB for vectors — that is
git-ignored and always rebuilt from scratch, so there is no state in which the
index and the files can silently disagree. The third is the judgment layer: an
LLM, whether the in-session agent or a local model, which is trusted only to
*propose* claims and never to *certify* them. Certification is a mechanical,
non-LLM substring check.

That last move is the whole system. A claim is admitted only if its supporting
quote is an exact substring of the source it cites. The check is deliberately
dumb, it runs the same way no matter which model produced the claim, and it is
re-derived from disk at three separate points in the pipeline. Everything else
— the producer-agnostic evidence normalization, the eight retrieval tools, the
role-routed model stack, the ability to run weak local models safely — is
downstream of that one guarantee holding.

## What we learned building and running it

### Design: isolate the asymmetries, don't paper over them

The cleanest parts of the codebase are the places where a genuine difference
between two things was confined to one small function rather than smeared
across the system. PDF claims cite a node id and a page; web claims cite a
chunk locator and a URL — genuinely different because the sources are
different — and the two shapes are unified into one internal form only inside
the compiler, at index time, leaving both producers' files untouched on disk.
The same discipline shows up in the PDF router recording an *ideal* backend it
hasn't built yet: the gap between intended and implemented lives visibly in the
data instead of being hidden. The lesson an outside engineer should take is
that generalizing a bespoke tool is mostly an exercise in finding the
asymmetries and naming them, not in forcing false uniformity.

### Model routing: route by phase, and don't trust the switch

The strongest operational lesson came from getting local models to actually
work, and it inverts the usual instinct. The instinct is to pick the single
best model and use it everywhere. The pipeline needs the opposite: extraction
is high-volume, well-specified work that wants a fast model that emits JSON and
stops, while synthesis is a one-shot judgment call that benefits from a model
that reasons. Trying to serve both with one model is systematically wrong for
one of them. So the config routes each phase to its own model, mode, and token
budget, and falls back to a single model when you only have one — sophistication
that is additive, not required.

The sharper, more surprising lesson is that a model's capability claims are
promises about the model, not about your inference stack. Two different model
families that advertise a "non-thinking" mode — a reasoning model and a hybrid
model — both ignored every request to disable thinking under the Ollama builds
we tested, reasoned until they hit the token ceiling, and returned nothing.
The fix was not to wait for the switch to work; it was to stop depending on a
switch and use a true instruct model, which has no reasoning path to suppress.
The right question is never "does this model support non-thinking mode," it is
"does *this serving stack, at this version,* honor that switch" — and the only
way to answer it is to test the exact (model, server, version) triple.

Two smaller lessons rode along. Reasoning models fill whatever token budget you
give them, so when output is truncated the lever is how much work you hand them
per call, not just the output cap — bounding the input into batches is what
actually fixed it. And embeddings are a separate axis entirely: routing the
embedding model by the shape of its name, decoupled from the generative stack,
let us swap a small sentence-transformer for a much stronger Qwen embedding
model with a one-line config change and no schema migration, because the vector
store infers dimension from the data.

### Correctness: defense in depth, done right — and where it drifts

The examination confirmed the load-bearing invariant is real and well-defended:
the verbatim check is re-derived independently from disk at extraction time, at
dossier-composition time, and in the PDF eval harness, so a bug or a hand-edit
at any single stage cannot launder an unverified claim into a trusted one. That
is genuinely good engineering, and it is why a weak or off-label model can only
under-produce, never corrupt the record.

But the same choice carries the seed of a real problem, and the correctness
pass found it: three independently written implementations of "the same" check
have quietly drifted apart in exactly the place that matters — the shape of the
text they check against. Extraction accepts a quote that spans two structural
units because it checks against the chunk the model was shown; the dossier gate
rebuilds page text with a different separator; the eval harness checks each unit
separately and never concatenates at all. The consequence is that a claim can be
admitted by extraction and then silently rejected by the other two gates, not
because the model was wrong but because the three gates disagree by construction.
The lesson is precise: when you enforce an invariant in more than one place, pin
down the shape of the haystack as part of the invariant, and share one function
rather than re-authoring it three times.

The examination also surfaced that the hard gate on claim quotes coexists with
no enforced integrity anywhere else in the schema. A relation can reference a
supporting claim that was dropped for a bad quote, leaving a dangling pointer.
Entity identity across independently processed sources rests entirely on the
model choosing the same slug for the same real-world thing, with no cross-source
merge step. Neither is catastrophic, but both are easy for a reader of the code
to assume are guaranteed when they are not — and "everything downstream of a
passed gate is safe" is exactly the false inference the architecture invites.

### Testing: test the mechanical, refuse to fake the judgment

The most honest thing about the test suite is what it declines to test. The
extraction and synthesis *judgment* is not mocked; the default agent backend
raises rather than pretending to make a judgment call, because a faked judgment
test proves nothing. What is tested is everything mechanical — the verbatim
gate, rank fusion, the graph walk, batching, id resolution, role routing — as
plain deterministic functions with real assertions. That split is the right
model for an LLM-in-the-loop system.

The gaps are equally clear and worth naming. The entire PDF-processing core is
exercised only by a weekly, non-blocking, model-dependent integration test, so
a regression in it is caught by nothing on a normal push — even though the
fixtures needed to test it as fast replay tests are already checked in. The CLI,
the first thing a new user runs, has no tests at all. Static typing and coverage
tooling are installed but never invoked. And a sync guard that is supposed to
keep two directory trees identical compares them by size and timestamp rather
than content, which can pass on a stale tree. None of these is fatal; all are
the ordinary debt of a repo moving fast, and all are cheap to close.

## What we can write as a result of this pipeline

The pipeline produces a two-layer artifact: a mechanically verifiable evidence
base — claims, entities, and relations, each claim bound to at least one
verbatim quote with a source locator — and, on top of it, any number of
documents synthesized from that base. The snail run is a small but real proof.
Five Wikipedia sources became 103 claims, 61 entities, and 66 relations; a local
reasoning model then wrote a coherent thesis on the nature of snails across human
history. Checked against the evidence, every specific number and named fact in
that thesis — a giant snail's 39.3-centimetre length, Pliny recording snails as
Roman elite fare, the French New Year accounting for seventy percent of annual
escargot sales — traces back to an exact quote in an archived source. The
examiner who verified this did not find a single invented fact.

What that buys over asking a model to "write about snails" is a hard boundary
against fabrication rather than a hope of accuracy. It makes a specific class of
documents writable with a property they normally lack: a cited literature or
standards survey where every sentence has a checkable source; an evidence
dossier for a decision, where each input fact is traceable rather than recalled;
a contradiction report that mechanically surfaces where sources disagree; a
living knowledge base that grows as sources are added and stays queryable; and
an auditable claim ledger that is itself the fact-check artifact. The
distinguishing feature across all of them is the same: "where did this come
from?" resolves to a source and an exact quote, not to a model's memory.

The honest limits belong next to the promise. The gate guarantees fidelity to
the source, not truth — a source's own errors pass straight through as grounded
claims. It guarantees a quote is present, not that it supports the claim built
around it, and not that the quote is a clean, un-truncated sentence. And, as
built today, the polished deliverable and its audit trail are two separate
files: the thesis reads as ordinary prose with no inline citations, and a reader
has to know to go back to the run directory to verify it. The traceability lives
in the pipeline, not yet in the document.

## Gaps worth closing next

The examination turned up a short list of concrete, actionable improvements,
roughly in priority order:

1. **Unify the three verbatim gates.** Make extraction, dossier, and eval share
   one check with one definition of the source text, so a claim admitted at one
   stage cannot be silently rejected at another. This is the one finding that
   touches the system's core guarantee.
2. **Drop or repair dangling references.** When a claim is gate-dropped, drop or
   flag the relations that cite it, rather than writing a pointer to a claim that
   will never exist.
3. **Make deliverables self-citing.** Have the synthesis step emit inline
   citations from the same claims it draws on, so the audit trail travels with
   the document instead of living in a separate directory.
4. **Add a cross-source entity-merge step at compile time**, so entity identity
   does not rest on the model independently choosing the same slug across sources.
5. **Close the fast-CI gap on the PDF core** with fixture-replay tests using the
   intermediate artifacts already in the repo, and add tests for the CLI.
6. **Retry or narrow batches on parse failure**, so a dense source does not lose
   whole sections to a truncated call that is merely counted, never recovered.

None of these changes the thesis of the system. They tighten the places where
its per-part correctness does not yet compose into end-to-end correctness — and
finding them is exactly what running the pipeline on a real, messy corpus was
for.

## Update: gaps closed

Every gap above has since been fixed, each with tests:

1. **The three verbatim gates are now one.** Extraction, dossier composition,
   and the eval harness all call a single `common.verbatim` check against the
   exact chunk a claim cites; a regression test asserts the three agree on a
   quote that spans two provenance units — the case that used to be admitted by
   one gate and silently rejected by the others.
2. **Dangling references are dropped.** A relation whose supporting claim was
   gate-dropped is no longer written, so nothing points at a claim id that isn't
   in `claims.jsonl`.
3. **Deliverables can self-cite.** `compose_dossier` renders to markdown
   (`compose-dossier --format md`) with each claim's verbatim quote and source
   inline, so the audit trail travels with the document.
4. **Entities merge across sources at compile time.** The compiler collapses a
   shared `entity_id` into one row — union of aliases, a deterministic canonical
   name, the most common type — instead of letting `get_entity` pick a source
   arbitrarily.
5. **The fast tier now covers the PDF core and the CLI.** Fixture-replay unit
   tests exercise `pdf/provenance`, `pdf/chunk`, `pdf/eval`, and `pdf/extract`
   on every push (no Docling needed), and the `drt` CLI has real tests; the
   heavy PDF test no longer flakes on a fixture-metadata hash drift.
6. **Failed batches are retried, not lost.** A batch whose output can't be
   parsed is split into smaller halves and retried before it counts as a parse
   failure, and abbreviated chunk ids resolve only when unambiguous.

Also closed along the way: the skill-template sync guard now compares by content
hash rather than size/mtime, CI reports coverage and runs an advisory mypy pass,
and the id-resolution and bare-output edge cases surfaced by the snail run are
guarded. The suite went from a fast tier of ~90 tests to ~160, and both heavy
integration tests (real Docling, real embeddings) pass.
