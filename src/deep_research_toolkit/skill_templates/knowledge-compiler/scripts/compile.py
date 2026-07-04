#!/usr/bin/env python3
"""Thin CLI shim: build the knowledge index via deep_research_toolkit.compiler.build.

python scripts/compile.py [--index-dir DIR]

Set DRT_FAKE_EMBEDDER=1 to use the deterministic test embedder (no model
download) -- for CI and smoke tests only, never for a real corpus.
"""
import argparse
import os
import sys

from deep_research_toolkit.compiler.build import compile_index
from deep_research_toolkit.compiler.embed import FakeEmbedder
from deep_research_toolkit.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-dir", default=None, help="Override knowledge_base.index_dir")
    args = parser.parse_args()

    config = load_config()
    if args.index_dir:
        config.index_dir = resolve_path(args.index_dir, config.index_dir, ".deepresearch/index")

    embedder = FakeEmbedder() if os.environ.get("DRT_FAKE_EMBEDDER") == "1" else None
    stats = compile_index(config, embedder=embedder)
    print("compiled index:", stats)
    print(f"index at: {config.index_dir}")


if __name__ == "__main__":
    sys.exit(main())
