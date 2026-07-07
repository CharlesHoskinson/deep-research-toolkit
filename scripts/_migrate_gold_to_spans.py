#!/usr/bin/env python3
"""One-off migration: move the eval-corpus gold reference claims from the old
{quote} evidence shape to the span shape {start_char, end_char, quote}.

For every tests/fixtures/eval-corpus/<doc>/reference-claims.jsonl, loads the
doc's chunks.jsonl and, for each evidence row, locates the stored quote in the
cited chunk: start_char = chunk_text.find(quote), end_char = start_char +
len(quote). The quote itself is kept (it is now the derived slice, which the
recall metric still matches on).

Fails loudly (raises) if any quote is not found verbatim in its chunk or cites
an unknown locator -- that would mean pre-existing non-verbatim gold, which
must be fixed by hand, never papered over with a bad span.

Usage:
  python scripts/_migrate_gold_to_spans.py [corpus_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "eval-corpus"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def migrate_doc(doc_dir: Path) -> tuple[int, int]:
    """Rewrite one doc's reference-claims.jsonl with span evidence. Returns
    (claims, evidence_rows) migrated. Raises on any quote not found."""
    chunks = {row["locator"]: row["text"] for row in _read_jsonl(doc_dir / "chunks.jsonl")}
    claims_path = doc_dir / "reference-claims.jsonl"
    claims = _read_jsonl(claims_path)

    n_evs = 0
    out_lines = []
    for claim in claims:
        claim_id = claim.get("claim_id", "<unknown>")
        new_evs = []
        for ev in claim.get("supporting_evidence") or []:
            loc = ev.get("locator")
            quote = ev.get("quote")
            if loc not in chunks:
                raise ValueError(f"{doc_dir.name}: claim {claim_id!r} cites unknown "
                                 f"locator {loc!r}")
            if not quote:
                raise ValueError(f"{doc_dir.name}: claim {claim_id!r} has an evidence "
                                 f"row with no quote")
            start = chunks[loc].find(quote)
            if start < 0:
                raise ValueError(f"{doc_dir.name}: claim {claim_id!r} quote is NOT a "
                                 f"verbatim slice of {loc!r} -- pre-existing "
                                 f"non-verbatim gold, fix by hand: {quote!r}")
            new_ev = {"locator": loc, "start_char": start,
                      "end_char": start + len(quote), "quote": quote}
            for k, v in ev.items():
                if k not in new_ev:
                    new_ev[k] = v
            new_evs.append(new_ev)
            n_evs += 1
        claim["supporting_evidence"] = new_evs
        out_lines.append(json.dumps(claim, ensure_ascii=False))

    claims_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return len(claims), n_evs


def main() -> int:
    corpus_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CORPUS_DIR
    doc_dirs = sorted(p for p in corpus_dir.iterdir() if p.is_dir())
    if not doc_dirs:
        raise SystemExit(f"no document directories under {corpus_dir}")
    total_claims = total_evs = 0
    for doc_dir in doc_dirs:
        n_claims, n_evs = migrate_doc(doc_dir)
        total_claims += n_claims
        total_evs += n_evs
        print(f"{doc_dir.name}: {n_claims} claim(s), {n_evs} evidence row(s) -> spans")
    print(f"done: {len(doc_dirs)} doc(s), {total_claims} claim(s), "
          f"{total_evs} evidence row(s) migrated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
