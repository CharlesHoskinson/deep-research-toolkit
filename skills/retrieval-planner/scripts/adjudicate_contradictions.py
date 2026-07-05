#!/usr/bin/env python3
"""Adjudicate contradiction candidates with the local conflict_adjudicate model.

Pipe candidates in from the deterministic tool, write verdicts JSONL out:

  python scripts/query.py find-contradictions > candidates.json
  python scripts/adjudicate_contradictions.py candidates.json --out adjudications.jsonl

Verdicts are proposals with rationales -- review them before acting on the
corpus. Under provider: agent this exits with guidance; run the SKILL.md
batched adjudication pass instead."""
import argparse
import json
import sys
from pathlib import Path

from deep_research_toolkit.config import load_config
from deep_research_toolkit.llm.adjudicate import adjudicate_candidates
from deep_research_toolkit.llm.backend import LLMBackendNotConfigured, get_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidates", help="JSON file from `query.py find-contradictions` ('-' for stdin)")
    parser.add_argument("--out", default="adjudications.jsonl")
    args = parser.parse_args()

    raw = sys.stdin.read() if args.candidates == "-" else Path(args.candidates).read_text(encoding="utf-8")
    candidates = json.loads(raw)
    if not candidates:
        print("no candidates -- nothing to adjudicate")
        return

    try:
        backend = get_backend(load_config(), role="conflict_adjudicate")
        result = adjudicate_candidates(candidates, backend)
    except LLMBackendNotConfigured as e:
        sys.exit(str(e))

    with open(args.out, "w", encoding="utf-8") as f:
        for v in result["verdicts"]:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    print(f"{len(result['verdicts'])} verdict(s) -> {args.out}; "
          f"{len(result['invalid'])} invalid row(s); {result['parse_failures']} parse failure(s)")
    if result["invalid"]:
        print(json.dumps(result["invalid"], indent=2))


if __name__ == "__main__":
    main()
