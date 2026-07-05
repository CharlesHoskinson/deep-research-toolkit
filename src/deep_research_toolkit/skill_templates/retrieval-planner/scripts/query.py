#!/usr/bin/env python3
"""Thin CLI shim over deep_research_toolkit.compiler.tools.Index. Prints JSON.

Subcommands: search-wiki, read-page, search-claims, get-entity, neighbors,
get-sources, find-contradictions, compose-dossier.

Set DRT_FAKE_EMBEDDER=1 to use the deterministic test embedder (CI/smoke only).
"""
import argparse
import json
import os
import sys

from deep_research_toolkit.compiler.embed import FakeEmbedder
from deep_research_toolkit.compiler.tools import Index
from deep_research_toolkit.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("find-contradictions")
    p = sub.add_parser("search-wiki")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=8)
    p = sub.add_parser("read-page")
    p.add_argument("path")
    p = sub.add_parser("search-claims")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--producer", choices=["pdf", "web"], default=None)
    p = sub.add_parser("get-entity")
    p.add_argument("name_or_id")
    p = sub.add_parser("neighbors")
    p.add_argument("entity")
    p.add_argument("--depth", type=int, default=1)
    p = sub.add_parser("get-sources")
    p.add_argument("--page", default=None)
    p.add_argument("--claim", default=None)
    p = sub.add_parser("compose-dossier")
    p.add_argument("query", nargs="?", default=None)
    p.add_argument("--claims", default=None, help="comma-separated claim_ids")
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--format", choices=["json", "md"], default="json",
                   help="md renders a self-citing markdown dossier (claims + verbatim quotes + sources)")
    args = parser.parse_args()

    embedder = FakeEmbedder() if os.environ.get("DRT_FAKE_EMBEDDER") == "1" else None
    idx = Index.open(load_config(), embedder=embedder)
    try:
        if args.cmd == "search-wiki":
            out = idx.search_wiki(args.query, k=args.k)
        elif args.cmd == "read-page":
            out = idx.read_page(args.path)
        elif args.cmd == "search-claims":
            out = idx.search_claims(args.query, k=args.k, producer=args.producer)
        elif args.cmd == "get-entity":
            out = idx.get_entity(args.name_or_id)
        elif args.cmd == "neighbors":
            out = idx.neighbors(args.entity, depth=args.depth)
        elif args.cmd == "get-sources":
            out = idx.get_sources(page=args.page, claim=args.claim)
        elif args.cmd == "find-contradictions":
            out = idx.find_contradictions()
        elif args.cmd == "compose-dossier":
            claim_ids = args.claims.split(",") if args.claims else None
            out = idx.compose_dossier(query=args.query, claim_ids=claim_ids, k=args.k)
            if args.format == "md":
                from deep_research_toolkit.compiler.dossier import render_dossier_markdown
                print(render_dossier_markdown(out))
                return
        else:
            parser.error("unknown command")
        print(json.dumps(out, indent=2, ensure_ascii=False))
    finally:
        idx.close()


if __name__ == "__main__":
    sys.exit(main())
