#!/usr/bin/env python3
"""Programmatic wiki page writer (provider: local, role: wiki_write).

Reads gate-passed claims from <run_dir>/claims.jsonl (optionally filtered to
one entity), asks the local wiki_write model for a cited body, mechanically
validates the [claim:<id>] markers, then scaffolds the page and records it in
the run's audit trail. Under provider: agent this exits with guidance --
write the page yourself per SKILL.md.

  python scripts/write_wiki_page.py <run_dir> <knowledge_path> \\
      --type Concept --title "Ouroboros Praos" [--entity "Praos"] [--force]
"""
import argparse
import json
import sys
from pathlib import Path

from deep_research_toolkit.common.scaffold import PageAlreadyExists, scaffold_page
from deep_research_toolkit.config import load_config
from deep_research_toolkit.llm.backend import LLMBackendNotConfigured, get_backend
from deep_research_toolkit.llm.wiki import CitationError, write_wiki_body
from deep_research_toolkit.pdf.wiki_writer import record_wiki_page


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("knowledge_path")
    parser.add_argument("--type", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--entity", help="only claims mentioning this entity (case-insensitive substring)")
    parser.add_argument("--description")
    parser.add_argument("--resource")
    parser.add_argument("--aliases", help="comma-separated")
    parser.add_argument("--tags", help="comma-separated")
    parser.add_argument("--source-docs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    claims_path = run_dir / "claims.jsonl"
    if not claims_path.is_file():
        sys.exit(f"no claims.jsonl in {run_dir} -- run extraction first")
    claims = [json.loads(l) for l in claims_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.entity:
        needle = args.entity.lower()
        claims = [c for c in claims
                  if needle in (c.get("claim") or "").lower()
                  or any(needle in (e or "").lower() for e in (c.get("entities") or []))]
    if not claims:
        sys.exit("no claims matched -- nothing to synthesize")

    config = load_config()
    try:
        backend = get_backend(config, role="wiki_write")
        out = write_wiki_body(args.title, args.type, claims, backend)
    except (LLMBackendNotConfigured, CitationError, ValueError) as e:
        sys.exit(str(e))

    given = Path(args.knowledge_path)
    path = given if given.is_absolute() else config.knowledge_base_path / given
    try:
        scaffold_page(
            path, type=args.type, title=args.title, description=args.description,
            resource=args.resource,
            aliases=[a.strip() for a in args.aliases.split(",")] if args.aliases else None,
            tags=[t.strip() for t in args.tags.split(",")] if args.tags else None,
            source_docs=[d.strip() for d in args.source_docs.split(",")] if args.source_docs else None,
            status="draft", body=out["body"], force=args.force,
        )
    except PageAlreadyExists as e:
        sys.exit(str(e))
    record_wiki_page(args.run_dir, args.knowledge_path)
    rep = out["citations"]
    print(f"wrote {path} ({len(rep['cited'])} claim(s) cited, coverage {rep['coverage']:.2f})")


if __name__ == "__main__":
    main()
