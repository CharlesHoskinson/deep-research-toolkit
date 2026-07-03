#!/usr/bin/env python3
"""Thin CLI shim: walk docling_raw.json and write provenance.jsonl via
deep_research_toolkit.pdf.provenance.
"""
import argparse
import sys
from pathlib import Path

from deep_research_toolkit.pdf.provenance import extract_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="pdf-runs/<document_id>/ produced by earlier pipeline stages")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    try:
        unit_count = extract_provenance(run_dir)
    except FileNotFoundError as e:
        sys.exit(str(e))

    print(f"wrote {unit_count} provenance units to {run_dir / 'provenance.jsonl'}")


if __name__ == "__main__":
    main()
