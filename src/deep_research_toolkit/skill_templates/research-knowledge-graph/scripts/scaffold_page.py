#!/usr/bin/env python3
"""Thin CLI shim: create a new OKF page via deep_research_toolkit.common.scaffold."""
import argparse
import sys
from pathlib import Path

from deep_research_toolkit.common.scaffold import PageAlreadyExists, scaffold_page
from deep_research_toolkit.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Where to create the doc, relative to the knowledge base "
                                      "(e.g. standards/ows.md), or an absolute/explicit path.")
    parser.add_argument("--type", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--description")
    parser.add_argument("--resource")
    parser.add_argument("--aliases", help="comma-separated")
    parser.add_argument("--tags", help="comma-separated")
    parser.add_argument("--source", help="id of a row in <knowledge_base>/sources/index.md")
    parser.add_argument("--status", choices=["seed", "researched", "stale", "draft", "conflicted"], default="seed")
    parser.add_argument("--body", default="", help="markdown body; defaults to a stub heading")
    parser.add_argument("--force", action="store_true", help="overwrite if the file already exists")
    args = parser.parse_args()

    given = Path(args.path)
    if given.is_absolute():
        path = given
    else:
        cfg = load_config()
        path = cfg.knowledge_base_path / given

    try:
        scaffold_page(
            path,
            type=args.type,
            title=args.title,
            description=args.description,
            resource=args.resource,
            aliases=[a.strip() for a in args.aliases.split(",")] if args.aliases else None,
            tags=[t.strip() for t in args.tags.split(",")] if args.tags else None,
            source=args.source,
            status=args.status,
            body=args.body,
            force=args.force,
        )
    except PageAlreadyExists as e:
        sys.exit(str(e))

    print(f"wrote {path}")


if __name__ == "__main__":
    main()
