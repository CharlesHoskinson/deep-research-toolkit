#!/usr/bin/env python3
"""Thin CLI shim: scaffold a web research run for claim extraction.

python scripts/start_research_run.py <url> --content-file PATH [--research-runs-dir DIR]
"""
import argparse
import sys
from pathlib import Path

from deep_research_toolkit.config import load_config, resolve_path
from deep_research_toolkit.web.research_run import start_research_run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("--content-file", required=True, help="Path to the fetched, cleaned markdown/text")
    parser.add_argument("--research-runs-dir", default=None)
    args = parser.parse_args()

    config = load_config()
    runs_dir = resolve_path(args.research_runs_dir, config.research_runs_path, "research-runs")
    content = Path(args.content_file).read_text(encoding="utf-8")
    run_dir = start_research_run(args.url, content, runs_dir)
    print(f"research run: {run_dir}")


if __name__ == "__main__":
    sys.exit(main())
