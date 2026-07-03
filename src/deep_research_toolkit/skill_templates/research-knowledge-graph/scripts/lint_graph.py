#!/usr/bin/env python3
"""Thin CLI shim: health-check the knowledge base via deep_research_toolkit.common.lint."""
import argparse
import sys

from deep_research_toolkit.common.lint import lint_knowledge_base
from deep_research_toolkit.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-base-dir", default=None,
                         help="Overrides .deepresearch.yml's knowledge_base.path")
    parser.add_argument("--stale-days", type=int, default=180)
    args = parser.parse_args()

    cfg = load_config()
    knowledge_base_dir = resolve_path(args.knowledge_base_dir, cfg.knowledge_base_path, "knowledge_base")

    try:
        problems = lint_knowledge_base(knowledge_base_dir, stale_days=args.stale_days)
    except FileNotFoundError as e:
        sys.exit(str(e))

    file_count = sum(1 for _ in knowledge_base_dir.rglob("*.md"))
    if not problems:
        print(f"clean: {file_count} files checked, no problems found")
        return 0

    print(f"{len(problems)} problem(s) across {file_count} files:")
    for p in problems:
        print(f"  {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
