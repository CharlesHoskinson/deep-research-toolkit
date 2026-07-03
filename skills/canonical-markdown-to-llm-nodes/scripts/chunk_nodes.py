#!/usr/bin/env python3
"""Thin CLI shim: turn provenance.jsonl into chunks.jsonl via
deep_research_toolkit.pdf.chunk.

Usage:
    python scripts/chunk_nodes.py <run_dir>
"""
import argparse
import sys

from deep_research_toolkit.pdf.chunk import chunk_nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="<pdf_runs_dir>/<document_id> directory")
    args = parser.parse_args()

    try:
        node_count = chunk_nodes(args.run_dir)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")

    print(f"wrote {args.run_dir}/chunks.jsonl ({node_count} nodes)")
    print(f"updated {args.run_dir}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
