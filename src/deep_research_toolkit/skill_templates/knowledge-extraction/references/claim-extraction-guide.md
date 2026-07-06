# Claim / Entity / Relation Extraction Guide

This is the part of `knowledge-extraction` that needs the agent's judgment,
not a script (see `SKILL.md` — same split `research-knowledge-graph` makes
between `scripts/fetch.py` and the human-judgment merge step described in
its `references/wiki-operations.md`). This guide works through
`tests/fixtures/hydra-settlement-test-fixture.pdf`'s own content — read
alongside `tests/fixtures/reference-run-hydra-settlement/claims.jsonl`,
`entities.jsonl`, and `relations.jsonl`, the real verified output for that
fixture (produced by chaining all 7 skills end-to-end, not a hand-derived
approximation).

## What a good claim looks like

A claim is **one checkable assertion**, traceable to an exact quote on an
exact page. Take `c_0001` from the example:

```json
{"schema_version": "1.0", "claim_id": "c_0001", "claim": "Hydra can be used as a synchronous settlement layer over Cardano-style eUTXO state.", "claim_type": "architectural", "confidence": "high", "supporting_evidence": [{"node_id": "hydra-settlement-test-fixture-4edb3c3c:n005", "quote": "Hydra can be used as a synchronous settlement layer over Cardano-style eUTXO state", "page": 1}], "entities": ["hydra", "cardano", "eutxo"], "relations": [{"subject": "hydra", "predicate": "serves_as", "object": "synchronous settlement layer"}]}
```

The source paragraph (node `n005`) actually reads:

> Because every state transition inside the Head requires unanimous
> signatures from all participants, Hydra can be used as a synchronous
> settlement layer over Cardano-style eUTXO state: once all parties sign a
> snapshot, that state is final among them even before it touches the main
> chain.

Notice what the claim *didn't* do: it didn't try to also assert "state is
final once all parties sign a snapshot" in the same claim, even though
that's in the same sentence and clearly related. That's a second,
definitional-flavored assertion about what finality means here — if it's
worth keeping, it's its own claim with its own evidence span, not folded
into `c_0001`. One claim, one assertion.

The `quote` (`"Hydra can be used as a synchronous settlement layer over
Cardano-style eUTXO state"`) is an exact substring of the paragraph above —
copy-pasted, not retyped. Note it deliberately stops before the colon
rather than including the rest of the sentence, because the colon marks
where a second idea starts.

## Bad claim patterns to avoid

**Compound sentence crammed into one claim.** The throughput table backs
`c_0003`:

```json
{"schema_version": "1.0", "claim": "Hydra Heads in the small (3-participant) configuration reach 1,000 TPS with sub-1s finality, versus 250 TPS and ~20s finality for the baseline L1.", "claim_type": "empirical", "confidence": "medium", "supporting_evidence": [{"node_id": "...:n009", "quote": "Hydra Head (small),3,\"1,000\",<1s", "page": 1}]}
```

This one *is* a compound sentence, and that's a deliberate exception worth
understanding: it's a single comparison drawn from two rows of one table,
and the two halves (small-Head numbers, baseline numbers) are meaningless
without each other — "1,000 TPS" alone doesn't say what it's being compared
to. When the "compound" is really one comparative assertion that would lose
its meaning split apart, keep it together and mark it `comparative`-shaped
reasoning even under an `empirical` type. What you should *not* do is merge
two genuinely independent assertions this way just to save a claim slot —
e.g. don't combine "Hydra reaches 1,000 TPS" with "Hydra Head safety relies
on unanimity" into one claim just because both mention Hydra. If in doubt,
ask: does removing half of this sentence leave the other half still true
and checkable on its own? If yes, split it.

Also notice `confidence: "medium"` here, not `"high"` — the underlying
evidence is a CSV-rendered table row (`quote` is the raw
`"Hydra Head (small),3,\"1,000\",<1s"` cell text, not prose), so the claim
is paraphrasing structured data into a sentence. That paraphrase step is
exactly why this one gets `medium` instead of `high`: the *claim* wording
is the agent's own, even though the *quote* is verbatim.

**Paraphrasing the quote.** Never write a quote like `"Hydra achieves high
throughput"` when the source says `"1,000 TPS"` — that's not a substring
match and will fail the mechanical evidence check. If the wording you want
to cite isn't contiguous in the source text, either find a shorter span
that is contiguous, or accept a `medium`/`low` confidence claim that leans
on the *nearest* exact quote rather than inventing one.

**Asserting more than the text supports.** `c_0005` is deliberately hedged:

```json
{"schema_version": "1.0", "claim": "It is an open question how Hydra Heads compose with delegated-signing standards like the Open Wallet Standard (OWS) when an autonomous agent holds one of the Head's signing keys.", "claim_type": "comparative", "confidence": "medium"}
```

The source text (node `n008`) literally frames this as an open question,
not a resolved architectural fact — the claim's wording ("It is an open
question...") mirrors that framing instead of overclaiming that Hydra
*does* compose with OWS. Don't upgrade a hedge into a confident claim.

## Entities: merge, don't multiply

The fixture uses "Hydra" (short form, e.g. in the intro) and "Hydra Head"
(the more formal protocol name used throughout the architecture section)
for the same protocol. The example collapses both into **one** row:

```json
{"schema_version": "1.0", "entity_id": "hydra-head", "name": "Hydra Head", "aliases": ["Hydra", "Head"], "type": "protocol", "mentions": ["...:n002", "...:n004", "...:n005", "...:n007", "...:n009"]}
```

`name` is the more formal/canonical form; `aliases` covers every other
surface form the document actually uses ("Hydra", and bare "Head" where
context makes it unambiguous); `mentions` is the *union* of every node_id
where any of those surface forms appears — collected once, across the whole
document, rather than emitting a separate entity row per mention or per
alias. Before writing an entity row, scan `chunks.jsonl` for every place the
concept comes up under any name, then write one row.

Contrast with `eutxo`, which gets one alias (`"extended UTXO"`) because
that's a genuine expansion of the same abbreviation, not a different
concept — same test: do all these surface forms point at one thing a
reader would consider identical? If yes, merge. If a document uses the same
word for two genuinely different things (rare, but watch for it — e.g. a
paper using "Head" for both the Hydra protocol concept and a literal
document section heading), don't merge those.

## Relations: only what claims actually assert

Every row in `relations.jsonl` should trace back to a `relations` entry
already present in some `claims.jsonl` row's own `relations` list, via
`supporting_claim`:

```json
{"schema_version": "1.0", "relation_id": "r_0001", "subject": "hydra-head", "predicate": "serves_as", "object": "synchronous settlement layer", "supporting_claim": "c_0001", "document_id": "..."}
```

Don't invent a relation the claims don't state just because it seems
plausible or would make the graph more connected — e.g. don't add
`hydra-head --composes_with--> ows` as a confident relation when `c_0005`
only supports `open_question_relates_to`. If the text doesn't commit to a
relationship, the relation predicate should reflect that uncertainty
(`open_question_relates_to`, not `composes_with`) rather than smoothing it
into something more definite than the source.

## Quick checklist before writing the files

- [ ] Every claim is one assertion (or one comparison that's genuinely
      inseparable — see the table example above), not a compound sentence.
- [ ] Every `supporting_evidence[].quote` is pasted verbatim from the node's
      `text`, with the matching `page`.
- [ ] Every entity mentioned under multiple surface forms is one row with
      `aliases`, not several rows.
- [ ] Every relation traces back to a claim that actually asserts it,
      via `supporting_claim`.
- [ ] Nothing was added just to hit a target count — a short, well-evidenced
      set beats a padded one.
