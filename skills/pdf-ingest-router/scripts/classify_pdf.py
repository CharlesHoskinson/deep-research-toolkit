#!/usr/bin/env python3
"""Thin CLI shim: classify a PDF via deep_research_toolkit.pdf.router."""
import argparse
import sys

from deep_research_toolkit.config import load_config, resolve_path
from deep_research_toolkit.pdf.router import classify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", help="path to the source PDF")
    parser.add_argument("--runs-dir", default=None,
                         help="Overrides .deepresearch.yml's knowledge_base.pdf_runs_dir")
    args = parser.parse_args()

    cfg = load_config()
    runs_dir = resolve_path(args.runs_dir, cfg.pdf_runs_path, "pdf-runs")

    try:
        run_dir = classify(args.pdf_path, runs_dir)
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(str(e))

    print(run_dir)


if __name__ == "__main__":
    main()
