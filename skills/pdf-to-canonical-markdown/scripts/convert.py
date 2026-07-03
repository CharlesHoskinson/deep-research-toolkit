#!/usr/bin/env python3
"""Thin CLI shim: convert a run's source PDF via deep_research_toolkit.pdf.convert.

python scripts/convert.py <run_dir>
"""
import argparse
import sys

from deep_research_toolkit.pdf.convert import DoclingNotInstalled, convert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="pdf-runs/<document_id> directory")
    args = parser.parse_args()

    try:
        convert(args.run_dir)
    except (FileNotFoundError, ValueError, DoclingNotInstalled) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
