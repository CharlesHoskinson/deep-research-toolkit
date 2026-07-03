#!/usr/bin/env python3
"""Thin CLI shim: extract figures + captions from docling_raw.json via
deep_research_toolkit.pdf.extract.extract_figures."""
import argparse
import sys

from deep_research_toolkit.pdf.extract import extract_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="<pdf_runs_dir>/<document_id>/ directory")
    args = parser.parse_args()

    try:
        count = extract_figures(args.run_dir)
    except FileNotFoundError as e:
        sys.exit(str(e))

    print(f"{count} figure(s) recorded in {args.run_dir}/figures/captions.jsonl")


if __name__ == "__main__":
    main()
