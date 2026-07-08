"""Recipe-B final SFT dataset assembly (`scripts/assemble-sft.py`'s backing
module): takes the three already-gate-verified pools this repo has on disk --
the 612-chunk local self-distillation set (`sft-dataset-recipe-b/`), the
90-chunk frontier bait set (`datasets/frontier-bait/`), and a small
self-authored general-instruction set -- and turns them into one final
`train.jsonl`/`val.jsonl`/`manifest.json` under `sft-final/`.

This module does NOT call any model or gate anything itself: every claim it
touches already passed `tunekit.dataset.gate_claim` at generation time. Its
only job is CURATION (fixing a real over-production problem in the local
set), MERGING (bait + general into the same conversation-record shape), and
RE-SPLITTING (a fresh, stratified, deterministic train/val cut over the
combined pool).

## Why curation is a heuristic cap, not the ideal per-claim support filter

Design doc §5.2 frames "keep claims supported by >=2 of N (temperature-varied)
samples" as the correct precision/bait knob. That requires, per claim, how
many of a chunk's N samples independently produced it. Investigated here:
`tunekit.dataset.escalating_k_sample` accumulates ALL of a round's
gate-passing claims from all k completions into one flat list (`accepted`)
with no per-claim sample-count kept; `tunekit.dataset.build_sft_dataset`'s
`dedup_claims` step (which runs ACROSS THE WHOLE DATASET, not per chunk) then
keeps the first occurrence of each `claim_key` and silently drops the rest --
again with no count of how many occurrences were merged. `escalation-log.json`
(this repo's `sft-dataset-recipe-b/escalation-log.json`) only records
ROUND-LEVEL aggregates (`{"k", "completions", "parsed", "accepted"}` per
chunk-round) -- never which specific claims recurred across completions. The
one artifact that WOULD carry per-claim provenance
(`<out_dir>/accepted.partial.jsonl`, written by `build_sft_dataset`'s
checkpoint path) does not exist on disk for this build (confirmed: no
`accepted.partial.jsonl`/`progress.json` anywhere in the repo) -- it is not a
committed artifact of a finished build. So a true support>=2 filter cannot be
reconstructed after the fact for this dataset.

Fallback (documented deviation, `curate_claims` below): per chunk, (1) collapse
near-duplicate paraphrases via token-Jaccard over normalized claim text,
keeping the longer-supporting-quote representative of each cluster (a proxy
for "value" -- a longer verbatim span is less likely to be a trivial/templated
claim); (2) cap the survivors at `DEFAULT_PER_RECORD_CAP` per chunk, still
ranked by quote length. This is coarser than a true support filter (it cannot
distinguish "said once, said confidently" from "said once, said by chance"),
but empirically lands the local set's atomicity in the target 2-8/record band
(see `scripts/assemble-sft.py`'s report) without discarding whole chunks.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from ..common.hashing import file_hash
from .dataset import dataset_hash, dedup_claims, to_conversation_record

#: Token-Jaccard similarity (over normalized claim text) at/above which two
#: claims in the SAME chunk are treated as paraphrase duplicates and
#: collapsed to one (the longer-quote survivor). Chosen empirically (see the
#: module docstring / assemble-sft.py report): 0.3 is the lowest threshold
#: that does real dedup work on this dataset's actual paraphrase patterns
#: without the per-chunk claim cap alone doing all the work (0.4+ collapses
#: to "median == cap" -- see the report for the sweep).
DEFAULT_JACCARD_THRESHOLD = 0.3

#: Per-chunk cap on curated claims (design doc §5.2's "atomicity ~1.33"
#: guardrail reframed as a hard per-record ceiling -- see module docstring).
DEFAULT_PER_RECORD_CAP = 8

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Near-duplicate collapse + per-record cap (the curation heuristic)
# ---------------------------------------------------------------------------

def normalized_tokens(text: str) -> set[str]:
    """Lowercased, punctuation-stripped, whitespace-collapsed token set --
    deliberately coarser than `llm.selfconsistency.claim_key`'s normalization
    (which only collapses exact near-identical strings): this is used to
    catch PARAPHRASES (different wording, same fact) that survived the
    original build's exact-normalized-text dedup."""
    folded = _PUNCT_RE.sub(" ", (text or "").lower())
    return {tok for tok in _WS_RE.split(folded.strip()) if tok}


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity; two empty sets are defined as identical (1.0) so
    an empty-claim-text edge case collapses rather than always surviving."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def claim_quote_len(claim: dict) -> int:
    """Sum of supporting-quote character lengths across a claim's evidence
    (almost always exactly one entry in this dataset) -- the "value" proxy
    curation ranks by: a longer verbatim span is less likely to be a
    trivial/templated restatement."""
    return sum(len(ev.get("quote") or "") for ev in (claim.get("supporting_evidence") or []))


def curate_claims(claims: list[dict], *, jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
                  cap: int = DEFAULT_PER_RECORD_CAP) -> list[dict]:
    """Curates one chunk's already gate-passed claim list: rank by
    `claim_quote_len` (descending), greedily drop any claim whose normalized
    token set is >= `jaccard_threshold` similar to an already-kept claim
    (paraphrase collapse), stop once `cap` survivors are kept. Order of the
    output is rank order (longest-quote first), not input order."""
    ranked = sorted(claims, key=claim_quote_len, reverse=True)
    kept: list[dict] = []
    kept_tokens: list[set[str]] = []
    for claim in ranked:
        tokens = normalized_tokens(claim.get("claim", ""))
        if any(jaccard(tokens, kt) >= jaccard_threshold for kt in kept_tokens):
            continue
        kept.append(claim)
        kept_tokens.append(tokens)
        if len(kept) >= cap:
            break
    return kept


# ---------------------------------------------------------------------------
# Record I/O + local-set curation
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


_OUTPUT_TAG_RE = re.compile(r"^<output>(.*)</output>$", re.DOTALL)


def _parse_assistant_body(content: str) -> tuple[dict, bool]:
    """Returns (parsed_json_obj, was_thinking_wrapped). Handles both the
    bare-JSON (`thinking=False`, this dataset's actual shape) and
    `<output>...</output>`-wrapped (`thinking=True`) assistant contracts --
    see `tunekit.dataset.to_conversation_record`."""
    m = _OUTPUT_TAG_RE.match(content.strip())
    if m:
        return json.loads(m.group(1)), True
    return json.loads(content), False


def _render_assistant_body(obj: dict, wrapped: bool) -> str:
    body = json.dumps(obj, ensure_ascii=False)
    return f"<output>{body}</output>" if wrapped else body


def record_claim_count(record: dict) -> int:
    """Number of claims in a conversation record's assistant turn (0 if the
    record isn't extraction-shaped, e.g. a general-instruction record)."""
    try:
        obj, _ = _parse_assistant_body(record["messages"][-1]["content"])
    except (KeyError, IndexError, ValueError):
        return 0
    return len(obj.get("claims") or [])


def atomicity_stats(records: list[dict]) -> dict:
    """Claims-per-record summary stats (mean/median/min/max/n) over an
    extraction-shaped record list -- used to report atomicity before/after
    curation."""
    counts = sorted(record_claim_count(r) for r in records)
    n = len(counts)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    mid = n // 2
    median = counts[mid] if n % 2 else (counts[mid - 1] + counts[mid]) / 2
    return {
        "n": n,
        "mean": sum(counts) / n,
        "median": median,
        "min": counts[0],
        "max": counts[-1],
    }


def curate_local_record(record: dict, *, jaccard_threshold: float = DEFAULT_JACCARD_THRESHOLD,
                        cap: int = DEFAULT_PER_RECORD_CAP) -> dict:
    """Returns a NEW record (input untouched) with its assistant turn's
    `claims` replaced by `curate_claims`'s survivors; `system`/`user` turns
    (the rendered extraction prompt, independent of which claims survive) and
    `entities`/`relations` (always `[]` in this dataset) are carried over
    unchanged. Adds `component: "local"`."""
    messages = record["messages"]
    obj, wrapped = _parse_assistant_body(messages[-1]["content"])
    curated = curate_claims(obj.get("claims") or [], jaccard_threshold=jaccard_threshold, cap=cap)
    new_obj = {"claims": curated, "entities": obj.get("entities", []), "relations": obj.get("relations", [])}
    new_messages = [*messages[:-1], {"role": "assistant", "content": _render_assistant_body(new_obj, wrapped)}]
    return {"messages": new_messages, "locator": record.get("locator"), "component": "local"}


# ---------------------------------------------------------------------------
# Frontier bait merge
# ---------------------------------------------------------------------------

def load_corpus_chunk_texts(training_corpus_dir: str | Path) -> dict[str, dict]:
    """locator -> chunk dict (`{"locator", "text", ...}`), read from every
    `<doc>/chunks.jsonl` under `training_corpus_dir` -- the same per-doc-dir
    shape `scripts/build-sft-dataset.py`'s `load_training_chunks` reads,
    reimplemented here (not imported -- that script's hyphenated filename
    isn't import-friendly) to look up a bait chunk's actual text for prompt
    rendering."""
    training_corpus_dir = Path(training_corpus_dir)
    chunks: dict[str, dict] = {}
    for doc_dir in sorted(p for p in training_corpus_dir.iterdir() if p.is_dir()):
        for row in load_jsonl(doc_dir / "chunks.jsonl"):
            locator = row.get("locator") or row.get("node_id")
            if locator:
                chunks[locator] = row
    return chunks


def merge_bait_candidate_file(candidate_path: str | Path, chunk_by_locator: dict[str, dict],
                              producer: str = "web") -> dict:
    """One frontier-bait candidate file (`{"bait_locator", "candidates": [{
    "candidate_id", "claims": [...]}, ...]}`, k=3 authored candidates per
    chunk, already gate-verified at 1.0 pass rate per
    `datasets/frontier-bait/manifest.json`) -> ONE conversation record,
    union+deduped across its k candidates via the SAME
    `tunekit.dataset.dedup_claims` (claim-text+locator key) the local build's
    cross-dataset dedup uses, then rendered via `to_conversation_record` --
    the identical function the local records were built with, so the output
    shape matches byte-for-byte (system/user rendered by
    `llm.extract.build_extraction_prompt`, assistant a bare
    `{"claims", "entities", "relations"}` object). Adds `component: "bait"`."""
    rec = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    locator = rec["bait_locator"]
    chunk = chunk_by_locator[locator]
    all_claims = [claim for candidate in rec["candidates"] for claim in candidate["claims"]]
    merged = dedup_claims(all_claims)
    conv = to_conversation_record(chunk, merged, producer=producer)
    return {**conv, "component": "bait"}


def merge_all_bait(candidates_dir: str | Path, chunk_by_locator: dict[str, dict],
                   producer: str = "web") -> list[dict]:
    candidates_dir = Path(candidates_dir)
    return [
        merge_bait_candidate_file(p, chunk_by_locator, producer=producer)
        for p in sorted(candidates_dir.glob("*.json"))
    ]


# ---------------------------------------------------------------------------
# General instruction data (design doc §6.1: "mix 10-20% general instruction
# data against forgetting"). Self-authored here (documented in the manifest's
# `general_source` field) -- not sourced from any external dataset. 25 short,
# generic, non-extraction topics (deliberately disjoint from every training-
# corpus/eval-corpus subject) x 4 task types (summarize / rewrite / Q&A /
# format-conversion) = 100 records.
# ---------------------------------------------------------------------------

GENERAL_TOPICS: list[dict] = [
    dict(
        topic="photosynthesis",
        passage=("Plants make their own food using sunlight, water, and carbon dioxide. "
                 "Chlorophyll in their leaves absorbs light energy and uses it to convert "
                 "water and carbon dioxide into glucose, releasing oxygen as a byproduct. "
                 "This process happens mostly in daylight, in the leaf's chloroplasts."),
        summary="Plants use chlorophyll to convert sunlight, water, and carbon dioxide into glucose, releasing oxygen.",
        casual_rewrite=("So basically, plants are little solar-powered food factories: their leaves "
                        "soak up sunlight and use it to turn water and CO2 into sugar, and oxygen just "
                        "comes out as a bonus."),
        qa_question="What byproduct is released during photosynthesis?",
        qa_answer="Oxygen.",
        bullets=["Uses sunlight, water, and carbon dioxide as inputs",
                 "Chlorophyll in leaves absorbs the light energy",
                 "Produces glucose for the plant",
                 "Releases oxygen as a byproduct",
                 "Occurs mainly in the chloroplasts during daylight"],
    ),
    dict(
        topic="pour-over coffee",
        passage=("A pour-over starts with medium-fine ground coffee in a paper filter. "
                 "Pour a small amount of hot water over the grounds first and let it sit for "
                 "thirty seconds so the coffee releases trapped gas -- this is called the "
                 "bloom. Then pour the rest of the water slowly in circles until you reach "
                 "the desired amount."),
        summary="Pour-over coffee involves blooming the grounds with a little hot water, then slowly pouring the rest in circles.",
        casual_rewrite=("Making pour-over is easy once you know the trick: wet the grounds a "
                        "little first and give it 30 seconds to puff up and breathe, then just "
                        "keep pouring hot water in slow circles till you've got enough coffee."),
        qa_question="Why do you let the coffee sit for thirty seconds after the first pour?",
        qa_answer="To let the coffee bloom -- release trapped gas built up in the grounds.",
        bullets=["Use medium-fine ground coffee in a paper filter",
                 "Pour a small amount of hot water first",
                 "Let it bloom for about thirty seconds",
                 "Pour the remaining water slowly in circles",
                 "Stop once you reach the desired amount"],
    ),
    dict(
        topic="bicycle tire change",
        passage=("To change a flat bicycle tire, first release the brake and remove the "
                 "wheel from the frame. Use tire levers to pry one side of the tire off the "
                 "rim, then pull out the old inner tube. Check the tire's inside for sharp "
                 "debris before inserting a new tube, partially inflated, and working the "
                 "tire back onto the rim."),
        summary="Changing a bike tire means removing the wheel, prying off the tire, checking for debris, and fitting a new tube.",
        casual_rewrite=("Got a flat? Pop the wheel off, pry the tire off the rim with a couple "
                        "of levers, yank out the old tube, and run your fingers around inside "
                        "the tire to make sure nothing sharp is still in there before you put "
                        "the new tube in and pop the tire back on."),
        qa_question="Why should you check the inside of the tire before fitting a new tube?",
        qa_answer="To make sure no sharp debris is left that would puncture the new tube.",
        bullets=["Release the brake and remove the wheel",
                 "Pry one side of the tire off the rim with tire levers",
                 "Remove the old inner tube",
                 "Check the tire's interior for sharp debris",
                 "Insert and inflate the new tube, then reseat the tire"],
    ),
    dict(
        topic="printing press history",
        passage=("Johannes Gutenberg introduced movable-type printing to Europe around 1440. "
                 "Before this, books were copied by hand, which was slow and expensive. "
                 "Gutenberg's press used reusable metal letters that could be arranged into "
                 "any text, dramatically speeding up book production and lowering costs, "
                 "which helped literacy spread across Europe."),
        summary="Gutenberg's movable-type press, introduced around 1440, made books far cheaper and faster to produce than hand-copying.",
        casual_rewrite=("Before Gutenberg came along around 1440, every single book had to be "
                        "copied out by hand -- painfully slow. His big idea was reusable metal "
                        "letters you could rearrange for any text, which made printing books way "
                        "faster and cheaper."),
        qa_question="Around what year did Gutenberg introduce movable-type printing to Europe?",
        qa_answer="Around 1440.",
        bullets=["Introduced by Johannes Gutenberg around 1440",
                 "Replaced slow, expensive hand-copying of books",
                 "Used reusable metal letters arranged into any text",
                 "Sped up production and lowered costs",
                 "Helped literacy spread across Europe"],
    ),
    dict(
        topic="compound interest",
        passage=("Compound interest is interest calculated on both the original amount saved "
                 "or borrowed and on the interest already added to it. Because each period's "
                 "interest is added to the principal before the next period's interest is "
                 "calculated, the balance grows faster over time than with simple interest, "
                 "especially over long periods."),
        summary="Compound interest grows a balance faster than simple interest because interest is calculated on interest already earned.",
        casual_rewrite=("Compound interest is basically interest on your interest -- every time "
                        "it's added to your balance, the next round of interest is calculated on "
                        "the bigger total, so your money snowballs faster the longer you leave it."),
        qa_question="Why does a balance grow faster under compound interest than simple interest?",
        qa_answer="Because interest is calculated on the principal plus previously added interest, not just the original principal.",
        bullets=["Calculated on principal plus previously added interest",
                 "Each period's interest is folded into the balance",
                 "Grows faster than simple interest over time",
                 "Effect is strongest over long time horizons",
                 "Applies to both savings and debt"],
    ),
    dict(
        topic="job interview tips",
        passage=("Before a job interview, research the company's products, mission, and "
                 "recent news so you can speak knowledgeably about why you want to work "
                 "there. Prepare specific examples from past experience that show relevant "
                 "skills. Arrive a few minutes early, and prepare a couple of thoughtful "
                 "questions to ask the interviewer at the end."),
        summary="Good interview prep means researching the company, preparing concrete examples, arriving early, and having questions ready.",
        casual_rewrite=("Before you walk into an interview, look up what the company actually "
                        "does and what's been in the news about them, have a few real stories "
                        "ready that show off your skills, show up a little early, and think of a "
                        "couple of good questions to ask at the end."),
        qa_question="What should a candidate prepare to show relevant skills during an interview?",
        qa_answer="Specific examples from past experience.",
        bullets=["Research the company's products, mission, and recent news",
                 "Prepare specific examples from past experience",
                 "Arrive a few minutes early",
                 "Prepare thoughtful questions for the interviewer",
                 "Be ready to explain why you want the role"],
    ),
    dict(
        topic="minor cut first aid",
        passage=("For a minor cut, first wash your hands, then rinse the wound with clean "
                 "water to remove dirt. Apply gentle pressure with a clean cloth to stop any "
                 "bleeding. Once bleeding stops, apply an antibiotic ointment and cover the "
                 "cut with a bandage, changing it daily or whenever it gets wet or dirty."),
        summary="Treat a minor cut by cleaning it, applying pressure to stop bleeding, then covering it with ointment and a bandage.",
        casual_rewrite=("If you nick yourself, wash your hands first, rinse the cut with clean "
                        "water, press a clean cloth on it till the bleeding stops, then dab on "
                        "some antibiotic ointment and slap a bandage over it -- just swap the "
                        "bandage daily or whenever it gets wet."),
        qa_question="What should you do immediately after rinsing a minor cut?",
        qa_answer="Apply gentle pressure with a clean cloth to stop any bleeding.",
        bullets=["Wash your hands before treating the cut",
                 "Rinse the wound with clean water",
                 "Apply gentle pressure to stop bleeding",
                 "Apply antibiotic ointment once bleeding stops",
                 "Cover with a bandage and change it daily"],
    ),
    dict(
        topic="rainbow formation",
        passage=("Rainbows form when sunlight enters a raindrop, slows down, and bends. "
                 "The light reflects off the inside back of the drop and bends again as it "
                 "exits, splitting into its component colors because each wavelength bends by "
                 "a slightly different amount. This is why a rainbow always appears opposite "
                 "the sun."),
        summary="Rainbows appear when sunlight bends entering and exiting raindrops, splitting into colors by wavelength.",
        casual_rewrite=("A rainbow happens because sunlight hits a raindrop, bends going in, "
                        "bounces off the back, and bends again coming out -- and since every "
                        "color bends by a slightly different amount, you get the whole spectrum "
                        "spread out, always on the opposite side of the sky from the sun."),
        qa_question="Why does sunlight split into different colors inside a raindrop?",
        qa_answer="Because each wavelength of light bends by a slightly different amount.",
        bullets=["Sunlight enters and slows down inside the raindrop",
                 "Light reflects off the inside back of the drop",
                 "Light bends again on exiting the drop",
                 "Each wavelength bends a different amount, separating colors",
                 "The rainbow appears opposite the sun"],
    ),
    dict(
        topic="packing a carry-on",
        passage=("To pack a carry-on efficiently, roll clothes instead of folding them to "
                 "save space and reduce wrinkles. Put heavier items like shoes near the wheels "
                 "or bottom of the bag. Keep liquids in a clear, sealed bag under the airline "
                 "limit, and pack anything you need during the flight, like medication or "
                 "chargers, in an easy-to-reach pocket."),
        summary="Efficient carry-on packing means rolling clothes, placing heavy items low, and keeping liquids and essentials accessible.",
        casual_rewrite=("Roll your clothes instead of folding -- saves space and cuts down on "
                        "wrinkles. Put the heavy stuff like shoes near the bottom, keep your "
                        "liquids in a clear bag under the limit, and stash anything you'll need "
                        "mid-flight somewhere you can grab it fast."),
        qa_question="Where should heavier items like shoes be placed in a carry-on?",
        qa_answer="Near the wheels or bottom of the bag.",
        bullets=["Roll clothes instead of folding to save space",
                 "Place heavier items near the bottom of the bag",
                 "Keep liquids in a clear, sealed bag under the limit",
                 "Pack in-flight essentials in an easy-to-reach pocket",
                 "Aim to reduce wrinkles and maximize space"],
    ),
    dict(
        topic="weather vs climate",
        passage=("Weather describes short-term atmospheric conditions like today's "
                 "temperature, rain, or wind. Climate describes the average weather pattern "
                 "of a region over a long period, typically decades. A single hot day doesn't "
                 "change a region's climate classification, but a shift in average conditions "
                 "over many years does."),
        summary="Weather is short-term atmospheric conditions, while climate is the long-term average pattern over decades.",
        casual_rewrite=("Weather is just what's happening outside right now -- rain, wind, "
                        "temperature today. Climate is the bigger picture: the average pattern "
                        "over decades. One freak hot day doesn't change a place's climate; a "
                        "shift over many years does."),
        qa_question="What is the main difference between weather and climate?",
        qa_answer="Weather is short-term conditions; climate is the long-term average pattern over decades.",
        bullets=["Weather = short-term atmospheric conditions",
                 "Climate = long-term average weather pattern",
                 "Climate is typically measured over decades",
                 "A single day's weather doesn't change climate",
                 "Climate reflects sustained shifts over many years"],
    ),
    dict(
        topic="composting basics",
        passage=("A backyard compost pile needs a mix of 'greens' like vegetable scraps and "
                 "grass clippings, and 'browns' like dry leaves and cardboard. Turning the "
                 "pile regularly adds oxygen, which speeds up decomposition. Keeping the pile "
                 "about as moist as a wrung-out sponge helps the microbes that break down the "
                 "material stay active."),
        summary="Good compost mixes nitrogen-rich greens with carbon-rich browns, gets turned regularly, and stays evenly moist.",
        casual_rewrite=("Composting is just mixing your veggie scraps and grass clippings "
                        "('greens') with dry leaves and cardboard ('browns'), giving it a turn "
                        "now and then so it gets air, and keeping it about as damp as a "
                        "wrung-out sponge."),
        qa_question="What does turning a compost pile regularly accomplish?",
        qa_answer="It adds oxygen, which speeds up decomposition.",
        bullets=["Mix 'greens' (vegetable scraps, grass clippings)",
                 "Mix 'browns' (dry leaves, cardboard)",
                 "Turn the pile regularly to add oxygen",
                 "Keep moisture about like a wrung-out sponge",
                 "Active microbes break the material down"],
    ),
    dict(
        topic="thank-you notes",
        passage=("A good thank-you note mentions the specific gift or gesture, explains why "
                 "it mattered to you, and is sent within a couple of weeks. Keeping it short "
                 "and personal matters more than fancy wording. A handwritten note is a nice "
                 "touch, but a sincere email is far better than no note at all."),
        summary="A good thank-you note names the specific gift, explains why it mattered, and is sent promptly.",
        casual_rewrite=("Keep your thank-you note simple: mention what they actually gave you, "
                        "say why it meant something, and send it within a couple weeks. It "
                        "doesn't need fancy words -- handwritten is nice, but a genuine email "
                        "beats no note at all."),
        qa_question="Within roughly what timeframe should a thank-you note be sent?",
        qa_answer="Within a couple of weeks.",
        bullets=["Mention the specific gift or gesture",
                 "Explain why it mattered to you",
                 "Send it within about two weeks",
                 "Keep it short and personal",
                 "A sincere email beats no note at all"],
    ),
    dict(
        topic="water cycle",
        passage=("The water cycle moves water between the earth's surface and the "
                 "atmosphere. The sun evaporates water from oceans and lakes into vapor, "
                 "which rises and cools until it condenses into clouds. Eventually the water "
                 "falls back as precipitation -- rain or snow -- and collects in rivers, "
                 "lakes, and groundwater before the cycle repeats."),
        summary="The water cycle moves water through evaporation, condensation into clouds, and precipitation back to the surface.",
        casual_rewrite=("The water cycle is just water going round and round: the sun heats up "
                        "oceans and lakes so water evaporates into vapor, that vapor cools and "
                        "turns into clouds, and eventually it falls back down as rain or snow "
                        "before starting all over again."),
        qa_question="What causes water vapor to condense into clouds?",
        qa_answer="The rising vapor cools as it goes up, causing it to condense.",
        bullets=["Sun evaporates water from oceans and lakes",
                 "Vapor rises and cools",
                 "Cooled vapor condenses into clouds",
                 "Water falls back as precipitation (rain or snow)",
                 "Collects in rivers, lakes, and groundwater, then repeats"],
    ),
    dict(
        topic="sleep hygiene",
        passage=("Good sleep hygiene includes going to bed and waking up at consistent times, "
                 "even on weekends. Avoiding screens and caffeine in the hour or two before "
                 "bed helps the body wind down. A cool, dark, quiet room also makes it easier "
                 "to fall and stay asleep."),
        summary="Good sleep hygiene means a consistent schedule, avoiding screens and caffeine before bed, and a cool, dark, quiet room.",
        casual_rewrite=("Want better sleep? Go to bed and wake up around the same time every "
                        "day, even weekends, skip the screens and coffee for an hour or two "
                        "before bed, and keep your room cool, dark, and quiet."),
        qa_question="What kind of room environment helps with falling and staying asleep?",
        qa_answer="A cool, dark, and quiet room.",
        bullets=["Keep a consistent sleep and wake schedule",
                 "Avoid screens before bed",
                 "Avoid caffeine in the hours before bed",
                 "Keep the bedroom cool, dark, and quiet",
                 "Consistency matters even on weekends"],
    ),
    dict(
        topic="balanced meal plate",
        passage=("A simple way to build a balanced meal is to fill half the plate with "
                 "vegetables and fruit, a quarter with whole grains like brown rice or whole "
                 "wheat pasta, and a quarter with a protein source like fish, beans, or "
                 "chicken. Adding a small amount of healthy fat, like olive oil or nuts, "
                 "rounds out the meal."),
        summary="A balanced plate is roughly half vegetables and fruit, a quarter whole grains, and a quarter protein, plus a little healthy fat.",
        casual_rewrite=("Easiest way to think about a balanced meal: half your plate is veggies "
                        "and fruit, a quarter is whole grains like brown rice, a quarter is "
                        "protein like fish or beans, and toss in a bit of healthy fat like olive "
                        "oil or nuts."),
        qa_question="What fraction of the plate should whole grains make up?",
        qa_answer="About a quarter of the plate.",
        bullets=["Half the plate: vegetables and fruit",
                 "A quarter: whole grains (brown rice, whole wheat pasta)",
                 "A quarter: a protein source (fish, beans, chicken)",
                 "Add a small amount of healthy fat",
                 "Simple ratio-based approach, not exact calorie counting"],
    ),
    dict(
        topic="library classification",
        passage=("The Dewey Decimal System organizes library books into ten broad classes, "
                 "numbered 000 to 999, by subject -- for example 500 covers science and 800 "
                 "covers literature. Each book gets a more specific number within its class, "
                 "so books on the same narrow topic sit together on the shelf, making browsing "
                 "by subject straightforward."),
        summary="The Dewey Decimal System groups library books into ten numbered subject classes so related books shelve together.",
        casual_rewrite=("Libraries that use Dewey Decimal sort books into ten big number-coded "
                        "subjects -- like 500 for science, 800 for literature -- and then give "
                        "each book an even more specific number so anything on the same topic "
                        "ends up sitting on the same shelf."),
        qa_question="What subject area does the 500 class cover in the Dewey Decimal System?",
        qa_answer="Science.",
        bullets=["Ten broad numbered classes, 000-999",
                 "Classes grouped by subject",
                 "500 covers science, 800 covers literature",
                 "More specific numbers narrow down within a class",
                 "Keeps books on the same topic shelved together"],
    ),
    dict(
        topic="car oil changes",
        passage=("A typical oil change involves draining the old oil from the engine through "
                 "a drain plug, replacing the oil filter, and refilling with the manufacturer- "
                 "recommended type and amount of fresh oil. Most vehicles need this roughly "
                 "every 5,000 to 7,500 miles, though the exact interval depends on the oil "
                 "type and the car's manual."),
        summary="An oil change drains old oil, replaces the filter, and refills with fresh oil, typically every 5,000-7,500 miles.",
        casual_rewrite=("Basic oil change: drain the old oil out through the plug, swap in a "
                        "new filter, and pour in fresh oil of the type your car wants. Most cars "
                        "want this done every 5,000 to 7,500 miles or so, depending on the oil "
                        "and what the manual says."),
        qa_question="Roughly how often do most vehicles need an oil change?",
        qa_answer="Roughly every 5,000 to 7,500 miles.",
        bullets=["Drain the old oil through the drain plug",
                 "Replace the oil filter",
                 "Refill with manufacturer-recommended oil",
                 "Typical interval: 5,000-7,500 miles",
                 "Exact interval depends on oil type and car manual"],
    ),
    dict(
        topic="starting a vegetable garden",
        passage=("To start a vegetable garden, pick a spot that gets at least six hours of "
                 "sun a day and has well-draining soil. Start with a few easy crops like "
                 "tomatoes, lettuce, or beans. Water consistently, and space plants according "
                 "to the seed packet so they have room to grow without crowding each other."),
        summary="Starting a vegetable garden means choosing a sunny, well-drained spot, easy crops, and consistent spacing and watering.",
        casual_rewrite=("If you're starting a veggie garden, find a spot that gets six-plus "
                        "hours of sun and drains well, start easy with stuff like tomatoes, "
                        "lettuce, or beans, water it consistently, and give the plants the "
                        "spacing the seed packet says so they're not crowding each other out."),
        qa_question="How many hours of sun should a vegetable garden spot get at minimum?",
        qa_answer="At least six hours a day.",
        bullets=["Choose a spot with at least six hours of sun",
                 "Make sure the soil drains well",
                 "Start with easy crops like tomatoes, lettuce, or beans",
                 "Water consistently",
                 "Space plants per the seed packet's instructions"],
    ),
    dict(
        topic="time zones",
        passage=("Time zones exist because the earth rotates, so different places face the "
                 "sun at different times. Rather than every town keeping its own local solar "
                 "time, the world is divided into roughly 24 standard zones, each about 15 "
                 "degrees of longitude wide, so nearby regions share a common clock time."),
        summary="Time zones divide the world into roughly 24 standard bands so nearby regions share a common clock despite the earth's rotation.",
        casual_rewrite=("Time zones are a thing because the earth spins, so the sun hits "
                        "different spots at different times. Instead of every town having its "
                        "own weird local time, we split the globe into about 24 zones, each "
                        "roughly 15 degrees of longitude, so nearby places can share a clock."),
        qa_question="Roughly how many standard time zones does the world use?",
        qa_answer="Roughly 24.",
        bullets=["Caused by the earth's rotation",
                 "World divided into roughly 24 standard zones",
                 "Each zone is about 15 degrees of longitude wide",
                 "Lets nearby regions share a common clock time",
                 "Avoids every town keeping its own local solar time"],
    ),
    dict(
        topic="Pomodoro technique",
        passage=("The Pomodoro technique breaks work into focused 25-minute intervals called "
                 "pomodoros, each followed by a short 5-minute break. After four pomodoros, "
                 "you take a longer break of 15 to 30 minutes. The idea is that short, timed "
                 "bursts of focus followed by real breaks reduce burnout and improve "
                 "concentration."),
        summary="The Pomodoro technique alternates 25-minute focused work intervals with short breaks, plus a longer break every four rounds.",
        casual_rewrite=("Pomodoro is simple: work in focused 25-minute chunks, take a 5-minute "
                        "breather after each one, and after four of those, give yourself a "
                        "longer 15-to-30-minute break. Short bursts plus real breaks supposedly "
                        "keep you sharper and less burned out."),
        qa_question="How long is the break after four completed pomodoros?",
        qa_answer="15 to 30 minutes.",
        bullets=["Work in focused 25-minute intervals (pomodoros)",
                 "Take a short 5-minute break after each interval",
                 "Take a longer 15-30 minute break after four pomodoros",
                 "Aims to reduce burnout",
                 "Aims to improve concentration"],
    ),
    dict(
        topic="glacier formation",
        passage=("Glaciers form where snowfall over many years exceeds melting, so layers of "
                 "snow build up and compress under their own weight into dense ice. Over time, "
                 "the sheer weight of accumulated ice causes the glacier to slowly flow "
                 "downhill under gravity, carving valleys and moving rock and debris as it "
                 "goes."),
        summary="Glaciers form when accumulated snow compresses into ice over many years and slowly flows downhill under its own weight.",
        casual_rewrite=("Glaciers happen when more snow falls each year than melts, so it piles "
                        "up and gets squeezed into thick ice over time. Eventually all that "
                        "weight makes the ice slowly creep downhill, carving out valleys and "
                        "dragging rock along with it."),
        qa_question="What causes a glacier to slowly flow downhill?",
        qa_answer="The sheer weight of the accumulated ice, pulled by gravity.",
        bullets=["Forms where snowfall exceeds melting over years",
                 "Layers of snow compress into dense ice",
                 "Accumulated weight causes slow downhill flow",
                 "Driven by gravity",
                 "Carves valleys and moves rock and debris"],
    ),
    dict(
        topic="training a puppy to sit",
        passage=("To teach a puppy to sit, hold a treat close to its nose and slowly move "
                 "your hand upward, which naturally makes the puppy's head follow and its "
                 "bottom lower to the ground. The moment it sits, say 'sit' and give the "
                 "treat. Repeating this consistently in short sessions helps the puppy "
                 "associate the word with the action."),
        summary="You teach a puppy to sit by luring its head up with a treat until it naturally sits, then rewarding it while saying the command.",
        casual_rewrite=("Teaching sit is easy: hold a treat right by the puppy's nose and lift "
                        "your hand up slowly -- its head follows and its butt hits the ground. "
                        "Right when that happens, say 'sit' and give the treat. Do that a bunch "
                        "in short sessions and it clicks."),
        qa_question="What should you say the moment the puppy sits down?",
        qa_answer="The word 'sit', right before giving the treat.",
        bullets=["Hold a treat close to the puppy's nose",
                 "Move your hand upward to lure the head back",
                 "The puppy's bottom naturally lowers to the ground",
                 "Say 'sit' the moment it sits, then reward",
                 "Repeat consistently in short sessions"],
    ),
    dict(
        topic="email etiquette",
        passage=("Professional emails should have a clear subject line describing the "
                 "content, a brief greeting, and a message that gets to the point quickly. "
                 "Proofreading before sending catches typos and unclear phrasing. It is also "
                 "considerate to reply within a day or two, even if just to say you need more "
                 "time to give a full answer."),
        summary="Good email etiquette means a clear subject line, a concise message, proofreading, and prompt replies.",
        casual_rewrite=("Good email habits: use a subject line that actually says what it's "
                        "about, keep the message short and to the point, proofread before you "
                        "hit send, and try to reply within a day or two even if it's just 'give "
                        "me more time on this'."),
        qa_question="What should a professional email's subject line do?",
        qa_answer="Clearly describe the content of the email.",
        bullets=["Use a clear, descriptive subject line",
                 "Start with a brief greeting",
                 "Get to the point quickly",
                 "Proofread before sending",
                 "Reply within a day or two when possible"],
    ),
    dict(
        topic="marathon distance history",
        passage=("The modern marathon distance of 26.2 miles was standardized at the 1908 "
                 "London Olympics, when the course was lengthened so it could start at "
                 "Windsor Castle and finish in front of the royal viewing box at the stadium. "
                 "Earlier Olympic marathons had used varying distances closer to 25 miles."),
        summary="The 26.2-mile marathon distance was fixed at the 1908 London Olympics to fit a course from Windsor Castle to the royal box.",
        casual_rewrite=("Ever wonder why a marathon is exactly 26.2 miles? That number got "
                        "locked in at the 1908 London Olympics, when they stretched the course "
                        "so it could start at Windsor Castle and end right in front of the royal "
                        "box. Before that, Olympic marathons were shorter and varied."),
        qa_question="At which Olympic Games was the 26.2-mile marathon distance standardized?",
        qa_answer="The 1908 London Olympics.",
        bullets=["Standard distance is 26.2 miles",
                 "Fixed at the 1908 London Olympics",
                 "Course lengthened to start at Windsor Castle",
                 "Finished in front of the royal viewing box",
                 "Earlier Olympic marathons used shorter, varying distances"],
    ),
    dict(
        topic="folding a fitted sheet",
        passage=("To fold a fitted sheet neatly, tuck one corner inside another on each end "
                 "so you end up holding two folded corners in each hand, then lay it flat and "
                 "fold it like a rectangle. Smoothing out the elastic edges as you go keeps "
                 "the final fold from looking lumpy."),
        summary="Folding a fitted sheet neatly means tucking corners into each other in pairs, then folding the result like a flat rectangle.",
        casual_rewrite=("The trick to folding a fitted sheet: tuck one corner inside its "
                        "neighbor on each end so you're holding two folded corners per hand, "
                        "then just lay the whole thing flat and fold it up like a regular "
                        "rectangle, smoothing the elastic edges as you go."),
        qa_question="What should you do with the elastic edges while folding a fitted sheet?",
        qa_answer="Smooth them out as you fold, so the result doesn't look lumpy.",
        bullets=["Tuck one corner inside another on each end",
                 "End up holding two folded corners in each hand",
                 "Lay the sheet flat",
                 "Fold it like a rectangle",
                 "Smooth elastic edges to avoid lumps"],
    ),
]

assert len(GENERAL_TOPICS) == 25, f"expected 25 general topics, got {len(GENERAL_TOPICS)}"

_GENERAL_SYSTEM_PROMPT = "You are a helpful, concise assistant."

_TASK_BUILDERS = (
    lambda t: ("Summarize the following passage in one sentence.\n\n" + t["passage"], t["summary"]),
    lambda t: ("Rewrite the following passage in a more casual, conversational tone.\n\n" + t["passage"],
              t["casual_rewrite"]),
    lambda t: ("Passage: " + t["passage"] + "\n\nQuestion: " + t["qa_question"], t["qa_answer"]),
    lambda t: ("Convert the following passage into a bulleted list of its key points.\n\n" + t["passage"],
              "\n".join(f"- {b}" for b in t["bullets"])),
)


def synthesize_general_instructions(n: int = 100, seed: int = 42) -> list[dict]:
    """Self-authored general-instruction pool (see module docstring): 25
    topics x 4 task types (summarize / casual-rewrite / Q&A /
    prose-to-bullets), each rendered as one conversation record with a
    plain assistant-persona system prompt (NOT the extraction prompt) so
    the model sees a genuinely different register/task in the mix. `n`
    (default 100, the full pool) truncates the deterministically-ordered
    pool if smaller; `seed` reorders the pool deterministically (does not
    change its content -- every pair is fixed, hand-authored text) so a
    caller asking for `n < 100` still gets a representative, reproducible
    subset rather than always the first `n` topics' pairs."""
    pool: list[dict] = []
    for topic in GENERAL_TOPICS:
        for build in _TASK_BUILDERS:
            user, assistant = build(topic)
            pool.append({
                "messages": [
                    {"role": "system", "content": _GENERAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                "locator": None,
                "component": "general",
            })
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    n = min(n, len(pool))
    return [pool[i] for i in order[:n]]


# ---------------------------------------------------------------------------
# Stratified re-split over the combined pool
# ---------------------------------------------------------------------------

def split_train_val(records: list[dict], val_fraction: float = 0.10, seed: int = 42) -> tuple[list, list]:
    """Deterministic shuffle-then-split, identical algorithm to
    `tunekit.dataset.split_train_val` (duplicated rather than imported so
    this module has no import-order dependency quirks; kept byte-identical
    on purpose -- see that function's docstring for the exact contract)."""
    n = len(records)
    if n == 0:
        return [], []
    order = list(range(n))
    random.Random(seed).shuffle(order)
    n_val = (max(1, round(n * val_fraction)) if val_fraction > 0 else 0)
    n_val = min(n_val, n)
    val_idx = set(order[:n_val])
    train = [records[i] for i in range(n) if i not in val_idx]
    val = [records[i] for i in range(n) if i in val_idx]
    return train, val


def stratified_split(records_by_component: dict[str, list[dict]], *, val_fraction: float = 0.10,
                     seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Splits EACH component independently (same `val_fraction`/`seed`), so
    every component (local/bait/general) is held out proportionally in val
    -- not just the pool as a whole -- then concatenates and
    deterministically reshuffles train and val separately so the final
    files aren't grouped by component."""
    train: list[dict] = []
    val: list[dict] = []
    for name in sorted(records_by_component):
        t, v = split_train_val(records_by_component[name], val_fraction=val_fraction, seed=seed)
        train.extend(t)
        val.extend(v)
    random.Random(seed).shuffle(train)
    random.Random(seed).shuffle(val)
    return train, val


# ---------------------------------------------------------------------------
# Hashing + manifest
# ---------------------------------------------------------------------------

def component_hash(records: list[dict]) -> str:
    """Canonical-JSON sha256 over one component's full record list (order as
    constructed, i.e. BEFORE the final stratified split/shuffle) -- reuses
    `tunekit.dataset.dataset_hash`'s exact canonicalization so this is
    consistent with the rest of the codebase's hash convention."""
    return dataset_hash(records)


def source_corpus_hash(training_corpus_dir: str | Path) -> str:
    """sha256 over every `(locator, text)` pair in `training_corpus_dir`
    (the full 36-doc/702-chunk committed training corpus, local+bait
    combined), sorted by locator so the hash is independent of file-read
    order -- this is the `source_corpus_hash` manifest field the design doc
    (§6.1) calls for."""
    chunks = load_corpus_chunk_texts(training_corpus_dir)
    joined = "".join(f"{loc}\x00{chunks[loc].get('text', '')}\n" for loc in sorted(chunks))
    return "sha256:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()


def final_dataset_hash(train_jsonl_path: str | Path) -> str:
    """sha256 of the final `train.jsonl` FILE BYTES (not a re-canonicalized
    in-memory form) -- reproducible with a plain `sha256sum train.jsonl`."""
    return "sha256:" + file_hash(train_jsonl_path)


def build_generator_digests(local_records: list[dict], escalation_log: list[dict],
                            bait_n_claims: int, general_n_records: int) -> dict:
    """Post-curation generator-digest breakdown: attributes each SURVIVING
    local claim to the teacher model that generated its chunk (looked up by
    locator in `escalation_log`, one entry per chunk -- a chunk is always
    routed to exactly one teacher, see `tunekit.dataset.route_for_chunk`),
    plus the frontier bait teacher's claim count and the general set's
    record count (no claim concept applies there)."""
    teacher_by_locator = {row["locator"]: row.get("teacher_model") for row in escalation_log
                          if "teacher_model" in row}
    digests: dict[str, int] = {}
    for rec in local_records:
        model = teacher_by_locator.get(rec.get("locator"), "unknown")
        digests[model] = digests.get(model, 0) + record_claim_count(rec)
    digests["frontier-agent-opus"] = bait_n_claims
    digests["self-authored (assembling agent)"] = general_n_records
    return digests


CURATION_METHOD = (
    "Support-count filter (design doc §5.2, keep claims supported by >=2 of N "
    "samples) was investigated and found NOT recoverable from this build's "
    "artifacts: tunekit.dataset.escalating_k_sample pools all of a round's "
    "gate-passing claims from every completion into one flat accepted list "
    "with no per-claim sample-count; build_sft_dataset's cross-dataset "
    "dedup_claims then keeps only the first occurrence of each claim_key and "
    "drops the rest without recording how many occurrences were merged; "
    "escalation-log.json records only round-level aggregates ({k, "
    "completions, parsed, accepted} per chunk), never per-claim provenance; "
    "and no accepted.partial.jsonl/progress.json checkpoint (which WOULD "
    "carry per-claim, per-completion records) survives on disk for this "
    "build. Fallback heuristic applied instead, per chunk: (1) rank "
    "gate-passing claims by total supporting-quote character length "
    "descending (longer verbatim span ~ higher-value, less trivial claim); "
    "(2) greedily drop any claim whose normalized-token-set Jaccard "
    "similarity to an already-kept claim is >= 0.3 (paraphrase collapse); "
    "(3) cap the survivors at 8 per chunk. This is coarser than a true "
    "support filter (cannot distinguish 'said once, said with confidence' "
    "from 'said once, by chance') but is documented here as the deviation "
    "and was validated empirically to land atomicity in a sane 2-8/record "
    "band (see manifest's atomicity_after)."
)
