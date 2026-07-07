#!/usr/bin/env python3
"""Mechanical gate for the stratified eval corpus (tests/fixtures/eval-corpus/).

The corpus is the permanent measuring stick for the extraction/wiki/synthesize/
adjudicate roles (see docs/superpowers/plans/2026-07-06-eval-suite-implementation.md,
Task 6). This script enforces its contract so a broken or drifted corpus fails
loudly instead of silently degrading eval numbers:

  - every doc directory has manifest.json, chunks.jsonl, reference-claims.jsonl
  - chunk rows carry a "<doc>#c<NNN>" locator and are 80-450 words
  - total chunk count and per-slice quotas (from corpus-index.json) are met
  - every reference claim quote is a verbatim substring of its own chunk
  - each chunk is cited by 2-6 reference claims
  - bait chunks (near-copies of a same-doc source sentence) are recorded in
    "bait_sources" and actually contain a near-copy (>=80% word overlap, not
    identical) of a sentence in their source chunk
  - contradiction_pairs are {"a", "b", "verdict", "note"} objects; verdicts are
    "contradiction" or "not_contradiction"; locators exist; "contradiction"
    pairs span different documents; enough pairs carry verdict "contradiction"
  - corpus_version (sha256 over sorted chunk texts) matches what's on disk

Usage:
  python scripts/check-eval-corpus.py [corpus_dir]      # validate (exit 1 on problems)
  python scripts/check-eval-corpus.py [corpus_dir] --stamp   # recompute + write corpus_version
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_research_toolkit.common.verbatim import verbatim_ok  # noqa: E402

DEFAULT_CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "eval-corpus"

KNOWN_SLICES = {"prose", "dense-facts", "table", "list", "unicode", "math", "long", "bait"}

VALID_VERDICTS = {"contradiction", "not_contradiction"}

DEFAULT_QUOTAS = {
    "prose": 60, "dense-facts": 30, "table": 15, "list": 15,
    "unicode": 12, "math": 12, "long": 15, "bait": 15,
}
DEFAULT_TOTAL_RANGE = (180, 220)
DEFAULT_WORD_RANGE = (80, 450)
DEFAULT_CLAIMS_PER_CHUNK_RANGE = (2, 6)
DEFAULT_MIN_CONTRADICTION_PAIRS = 8
DEFAULT_BAIT_OVERLAP_THRESHOLD = 0.8

_LOCATOR_TAIL_RE = re.compile(r"^c\d{3}$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _doc_of(locator: str) -> str:
    return locator.split("#", 1)[0]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(f"{path}:{line_no}: {e.msg}", e.doc, e.pos) from e
    return rows


def _sentences(text: str) -> list[str]:
    flat = " ".join(text.split())
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(flat) if s.strip()]


def _word_set(sentence: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(sentence)}


def _bait_near_copy_exists(bait_text: str, source_text: str, threshold: float) -> bool:
    """True when some sentence in bait_text is a near-copy (word-overlap >=
    threshold) of some sentence in source_text, but the two sentences are not
    character-for-character identical."""
    for b in _sentences(bait_text):
        b_words = _word_set(b)
        if not b_words:
            continue
        for s in _sentences(source_text):
            if b == s:
                continue  # verbatim-identical is not a "near"-copy
            s_words = _word_set(s)
            if not s_words:
                continue
            union = b_words | s_words
            overlap = len(b_words & s_words) / len(union) if union else 0.0
            if overlap >= threshold:
                return True
    return False


def compute_corpus_version(chunk_texts: dict[str, str]) -> str:
    """sha256 over the sorted (by locator) chunk texts -- a single stamp that
    changes if any chunk's text changes or a chunk is added/removed."""
    joined = "\n".join(text for _, text in sorted(chunk_texts.items()))
    return f"sha256:{hashlib.sha256(joined.encode('utf-8')).hexdigest()}"


def _walk_docs(corpus_dir: Path, word_range: tuple[int, int]) -> dict:
    """Read every doc dir's manifest/chunks/reference-claims, collecting
    structural errors plus the data later checks need (chunk texts, per-chunk
    reference-claim citation counts)."""
    errors: list[str] = []
    chunk_texts: dict[str, str] = {}
    claim_citations: dict[str, int] = {}

    doc_dirs = sorted(p for p in corpus_dir.iterdir() if p.is_dir())
    if not doc_dirs:
        errors.append(f"no document directories found under {corpus_dir}")

    for doc_dir in doc_dirs:
        doc_id = doc_dir.name
        manifest_path = doc_dir / "manifest.json"
        chunks_path = doc_dir / "chunks.jsonl"
        claims_path = doc_dir / "reference-claims.jsonl"
        missing = [name for name, p in (
            ("manifest.json", manifest_path),
            ("chunks.jsonl", chunks_path),
            ("reference-claims.jsonl", claims_path),
        ) if not p.is_file()]
        if missing:
            errors.append(f"{doc_id}: missing required file(s): {', '.join(missing)}")
            continue

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errors.append(f"{doc_id}: manifest.json is not valid JSON: {e}")
            manifest = {}
        if manifest.get("document_id") != doc_id:
            errors.append(f"{doc_id}: manifest document_id {manifest.get('document_id')!r} "
                          f"does not match directory name")

        try:
            chunk_rows = _read_jsonl(chunks_path)
        except json.JSONDecodeError as e:
            errors.append(f"{doc_id}: chunks.jsonl has invalid JSON: {e}")
            chunk_rows = []

        doc_chunk_texts: dict[str, str] = {}
        for i, row in enumerate(chunk_rows):
            locator = row.get("locator")
            text = row.get("text")
            if not locator or not isinstance(locator, str):
                errors.append(f"{doc_id}: chunks.jsonl row {i} has no string 'locator'")
                continue
            if not isinstance(text, str) or not text:
                errors.append(f"{doc_id}: chunk {locator!r} has no string 'text'")
                continue
            if _doc_of(locator) != doc_id or "#" not in locator or \
                    not _LOCATOR_TAIL_RE.match(locator.split("#", 1)[1]):
                errors.append(f"{doc_id}: chunk locator {locator!r} does not match "
                              f"'<doc>#c<NNN>' format")
            if locator in doc_chunk_texts:
                errors.append(f"{doc_id}: duplicate chunk locator {locator!r}")
            doc_chunk_texts[locator] = text
            wc = len(text.split())
            if not (word_range[0] <= wc <= word_range[1]):
                errors.append(f"{doc_id}: chunk {locator!r} has {wc} words, outside "
                              f"[{word_range[0]}, {word_range[1]}]")

        for locator in doc_chunk_texts:
            if locator in chunk_texts:
                errors.append(f"duplicate chunk locator across the corpus: {locator!r}")
        chunk_texts.update(doc_chunk_texts)

        try:
            claim_rows = _read_jsonl(claims_path)
        except json.JSONDecodeError as e:
            errors.append(f"{doc_id}: reference-claims.jsonl has invalid JSON: {e}")
            claim_rows = []

        for row in claim_rows:
            claim_id = row.get("claim_id", "<unknown>")
            evidence = row.get("supporting_evidence") or []
            if not evidence:
                errors.append(f"{doc_id}: claim {claim_id!r} has no supporting_evidence")
                continue
            locs_in_claim: set[str] = set()
            for ev in evidence:
                loc = ev.get("locator")
                quote = ev.get("quote") or ""
                if loc not in doc_chunk_texts:
                    errors.append(f"{doc_id}: claim {claim_id!r} cites unknown locator {loc!r}")
                    continue
                if not verbatim_ok(quote, doc_chunk_texts[loc]):
                    errors.append(f"{doc_id}: claim {claim_id!r} quote is not verbatim in "
                                  f"{loc!r}: {quote!r}")
                locs_in_claim.add(loc)
            for loc in locs_in_claim:
                claim_citations[loc] = claim_citations.get(loc, 0) + 1

    return {"errors": errors, "chunk_texts": chunk_texts, "claim_citations": claim_citations}


def validate(
    corpus_dir,
    *,
    quotas: dict[str, int] | None = None,
    total_range: tuple[int, int] | None = None,
    word_range: tuple[int, int] | None = None,
    claims_per_chunk_range: tuple[int, int] | None = None,
    min_contradiction_pairs: int | None = None,
    bait_overlap_threshold: float | None = None,
) -> list[str]:
    """Validate a corpus directory against the eval-corpus contract. Returns a
    list of human-readable problem strings (empty when the corpus is clean).

    The scale-dependent thresholds (quotas, total_range, min_contradiction_pairs)
    are parameters with production defaults so tests can drive this against a
    tiny mini-corpus without tripping the full-scale minimums."""
    corpus_dir = Path(corpus_dir)
    quotas = dict(DEFAULT_QUOTAS) if quotas is None else dict(quotas)
    total_range = DEFAULT_TOTAL_RANGE if total_range is None else total_range
    word_range = DEFAULT_WORD_RANGE if word_range is None else word_range
    claims_range = DEFAULT_CLAIMS_PER_CHUNK_RANGE if claims_per_chunk_range is None else claims_per_chunk_range
    min_pairs = DEFAULT_MIN_CONTRADICTION_PAIRS if min_contradiction_pairs is None else min_contradiction_pairs
    overlap_threshold = DEFAULT_BAIT_OVERLAP_THRESHOLD if bait_overlap_threshold is None else bait_overlap_threshold

    index_path = corpus_dir / "corpus-index.json"
    if not index_path.is_file():
        return [f"missing corpus-index.json at {index_path}"]
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"corpus-index.json is not valid JSON: {e}"]

    walked = _walk_docs(corpus_dir, word_range)
    errors: list[str] = list(walked["errors"])
    chunk_texts: dict[str, str] = walked["chunk_texts"]
    claim_citations: dict[str, int] = walked["claim_citations"]

    total = len(chunk_texts)
    if not (total_range[0] <= total <= total_range[1]):
        errors.append(f"total chunk count {total} outside [{total_range[0]}, {total_range[1]}]")

    for locator in chunk_texts:
        n = claim_citations.get(locator, 0)
        if not (claims_range[0] <= n <= claims_range[1]):
            errors.append(f"chunk {locator!r} is cited by {n} reference claim(s), outside "
                          f"[{claims_range[0]}, {claims_range[1]}]")

    index_chunks = index.get("chunks") or {}
    index_locators = set(index_chunks)
    disk_locators = set(chunk_texts)
    for loc in sorted(index_locators - disk_locators):
        errors.append(f"corpus-index.json references locator {loc!r} with no matching chunk on disk")
    for loc in sorted(disk_locators - index_locators):
        errors.append(f"chunk {loc!r} is not tagged in corpus-index.json 'chunks'")

    slice_counts = {s: 0 for s in KNOWN_SLICES}
    for loc, meta in index_chunks.items():
        for s in (meta or {}).get("slices") or []:
            if s not in KNOWN_SLICES:
                errors.append(f"chunk {loc!r} has unknown slice tag {s!r}")
            else:
                slice_counts[s] += 1
    for slice_name, minimum in quotas.items():
        got = slice_counts.get(slice_name, 0)
        if got < minimum:
            errors.append(f"slice {slice_name!r} has {got} chunk(s), needs >= {minimum}")

    bait_sources = index.get("bait_sources") or {}
    bait_locators = {loc for loc, meta in index_chunks.items() if "bait" in ((meta or {}).get("slices") or [])}
    for loc in sorted(bait_locators):
        if loc not in bait_sources:
            errors.append(f"bait chunk {loc!r} has no entry in corpus-index.json 'bait_sources'")
            continue
        source_loc = bait_sources[loc]
        if source_loc not in chunk_texts:
            errors.append(f"bait_sources[{loc!r}] references unknown locator {source_loc!r}")
            continue
        if _doc_of(loc) != _doc_of(source_loc):
            errors.append(f"bait_sources[{loc!r}] = {source_loc!r} is not in the same document")
            continue
        if not _bait_near_copy_exists(chunk_texts[loc], chunk_texts[source_loc], overlap_threshold):
            errors.append(f"bait chunk {loc!r} has no near-copy sentence (>={overlap_threshold:.0%} "
                          f"word overlap, not verbatim-identical) of a sentence in source {source_loc!r}")

    pairs = index.get("contradiction_pairs") or []
    genuine = 0
    for pair in pairs:
        if not isinstance(pair, dict) or not all(k in pair for k in ("a", "b", "verdict")):
            errors.append(f"malformed contradiction pair (needs 'a', 'b', 'verdict'): {pair!r}")
            continue
        a, b, verdict = pair["a"], pair["b"], pair["verdict"]
        if verdict not in VALID_VERDICTS:
            errors.append(f"contradiction pair ({a!r}, {b!r}) has invalid verdict {verdict!r} "
                          f"(must be one of {sorted(VALID_VERDICTS)})")
            continue
        a_ok = a in chunk_texts
        b_ok = b in chunk_texts
        if not a_ok:
            errors.append(f"contradiction pair references unknown locator {a!r}")
        if not b_ok:
            errors.append(f"contradiction pair references unknown locator {b!r}")
        if verdict == "contradiction":
            genuine += 1
            if a_ok and b_ok and _doc_of(a) == _doc_of(b):
                errors.append(f"contradiction pair ({a!r}, {b!r}) is within the same document, "
                              f"needs to span different documents")
    if genuine < min_pairs:
        errors.append(f"only {genuine} pair(s) with verdict 'contradiction', needs >= {min_pairs}")

    stated_version = index.get("corpus_version")
    if not stated_version:
        errors.append("corpus-index.json missing 'corpus_version'")
    else:
        recomputed = compute_corpus_version(chunk_texts)
        if stated_version != recomputed:
            errors.append(f"corpus_version mismatch: stated {stated_version!r}, recomputed "
                          f"{recomputed!r} (run --stamp to refresh)")

    return errors


def _stamp(corpus_dir: Path) -> str:
    index_path = corpus_dir / "corpus-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    walked = _walk_docs(corpus_dir, DEFAULT_WORD_RANGE)
    version = compute_corpus_version(walked["chunk_texts"])
    index["corpus_version"] = version
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus_dir", nargs="?", default=str(DEFAULT_CORPUS_DIR),
                        help="Corpus directory (default: tests/fixtures/eval-corpus)")
    parser.add_argument("--stamp", action="store_true",
                        help="Recompute corpus_version from the chunks on disk and write it "
                             "into corpus-index.json, instead of validating")
    args = parser.parse_args(argv)
    corpus_dir = Path(args.corpus_dir)

    if args.stamp:
        version = _stamp(corpus_dir)
        print(f"stamped corpus_version={version} into {corpus_dir / 'corpus-index.json'}")
        return 0

    errors = validate(corpus_dir)
    if errors:
        print(f"{len(errors)} problem(s) found in {corpus_dir}:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"OK: {corpus_dir} passes all checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
