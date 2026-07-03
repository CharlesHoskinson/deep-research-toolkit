#!/usr/bin/env python3
"""Thin CLI shim: health-check a run directory via deep_research_toolkit.pdf.eval.

python scripts/run_eval.py <run_dir>
"""
import argparse
import sys
from pathlib import Path

from deep_research_toolkit.pdf.eval import run_eval


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", help="pdf-runs/<document_id> directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        sys.exit(f"no such run directory: {run_dir}")

    run_eval(run_dir)


if __name__ == "__main__":
    main()
