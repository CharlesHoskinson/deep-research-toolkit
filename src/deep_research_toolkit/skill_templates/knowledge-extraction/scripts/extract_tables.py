#!/usr/bin/env python3
"""Thin CLI shim: extract tables from docling_raw.json via
deep_research_toolkit.pdf.extract.extract_tables."""
import argparse
import sys

from deep_research_toolkit.pdf.extract import extract_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="<pdf_runs_dir>/<document_id>/ directory")
    args = parser.parse_args()

    try:
        count = extract_tables(args.run_dir)
    except FileNotFoundError as e:
        sys.exit(str(e))

    print(f"{count} table(s) extracted to {args.run_dir}/tables")


if __name__ == "__main__":
    main()
