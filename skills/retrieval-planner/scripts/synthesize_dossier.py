#!/usr/bin/env python3
"""Synthesize a cited thesis over a composed dossier with the local model.

  python scripts/query.py compose-dossier --claims c1,c2 --format json > dossier.json
  python scripts/synthesize_dossier.py dossier.json --question "..." [--out thesis.md]

Output: the model's thesis (citation-gated) followed by the standard
self-citing dossier appendix. Under provider: agent this exits with
guidance; synthesize in-session instead."""
import argparse
import json
import sys
from pathlib import Path

from deep_research_toolkit.compiler.dossier import render_dossier_markdown
from deep_research_toolkit.config import load_config
from deep_research_toolkit.llm.backend import LLMBackendNotConfigured, get_backend
from deep_research_toolkit.llm.synthesize import synthesize_thesis
from deep_research_toolkit.llm.wiki import CitationError


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dossier", help="JSON file from `query.py compose-dossier --format json` ('-' for stdin)")
    parser.add_argument("--question", required=True)
    parser.add_argument("--out", help="write markdown here instead of stdout")
    args = parser.parse_args()

    raw = sys.stdin.read() if args.dossier == "-" else Path(args.dossier).read_text(encoding="utf-8")
    try:
        dossier = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"invalid JSON in {args.dossier}: {e}")
    try:
        backend = get_backend(load_config(), role="synthesize")
        out = synthesize_thesis(args.question, dossier, backend)
    except (LLMBackendNotConfigured, CitationError, ValueError) as e:
        sys.exit(str(e))

    doc = (f"# Synthesis: {args.question}\n\n{out['thesis']}\n\n---\n\n"
           + render_dossier_markdown(dossier))
    if args.out:
        Path(args.out).write_text(doc, encoding="utf-8")
        print(f"wrote {args.out} (coverage {out['citations']['coverage']:.2f})")
    else:
        print(doc)


if __name__ == "__main__":
    main()
