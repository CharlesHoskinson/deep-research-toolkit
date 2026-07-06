#!/usr/bin/env python3
"""Thin CLI shim: create or record a PDF-derived Open Knowledge Format wiki page.

Page content is written by deep_research_toolkit.common.scaffold.scaffold_page
(shared with research-knowledge-graph's scaffold_page.py) -- this script adds
the --source-docs / extended --status handling the PDF pipeline needs on top
of it, plus the wiki_pages_written.json + manifest.json audit-trail
bookkeeping in deep_research_toolkit.pdf.wiki_writer.record_wiki_page.

Two modes:

1. Create (default) -- writes a brand new page's frontmatter + body.
   python scaffold_wiki_page.py <run_dir> <knowledge_path> \\
       --type Concept --title "..." --status draft \\
       --source-docs doc-id-a1b2c3d4 [--description ...] [--tags ...] [--body ...]

2. --record-updated -- the page already exists and the agent merged new
   claims into it directly (Read + Edit), following the same "edit it directly,
   merge into the body, bump timestamp" convention research-knowledge-graph
   uses. This mode does not touch the file's content at all; it only logs
   the path into wiki_pages_written.json (and manifest.json) so the audit
   trail is complete.
   python scaffold_wiki_page.py <run_dir> <knowledge_path> --record-updated
"""
import argparse
import sys
from pathlib import Path

from deep_research_toolkit.common.scaffold import PageAlreadyExists, scaffold_page
from deep_research_toolkit.config import load_config
from deep_research_toolkit.pdf.wiki_writer import record_wiki_page

STATUS_CHOICES = ["seed", "researched", "stale", "draft", "conflicted"]


def _resolve_knowledge_path(knowledge_path: str) -> Path:
    """An absolute path is used as-is; otherwise resolved against
    .deepresearch.yml's knowledge_base.path (zero-config fallback:
    "knowledge_base/"), matching research-knowledge-graph's scaffold_page.py.
    """
    given = Path(knowledge_path)
    if given.is_absolute():
        return given
    cfg = load_config()
    return cfg.knowledge_base_path / given


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="pdf-runs/<document_id> directory for this ingestion run")
    parser.add_argument(
        "knowledge_path",
        help="Where the page lives, relative to the configured knowledge base "
        "(e.g. concepts/hydra-settlement.md), or an absolute path.",
    )
    parser.add_argument("--type", help="Index | Tool | Standard | Pattern | Product | Concept | Entity")
    parser.add_argument("--title")
    parser.add_argument("--description")
    parser.add_argument("--resource")
    parser.add_argument("--aliases", help="comma-separated")
    parser.add_argument("--tags", help="comma-separated")
    parser.add_argument("--source", help="id of a row in <knowledge_base>/sources/index.md (web-scraped provenance)")
    parser.add_argument(
        "--source-docs",
        help="comma-separated document_id(s) from the pdf run directories this page was synthesized from",
    )
    parser.add_argument("--status", choices=STATUS_CHOICES, default="draft")
    parser.add_argument("--body", default="", help="markdown body; defaults to a stub heading")
    parser.add_argument("--force", action="store_true", help="overwrite if the file already exists")
    parser.add_argument(
        "--record-updated",
        action="store_true",
        help="don't write page content; just log an existing page (already merged by hand) into the audit trail",
    )
    args = parser.parse_args()

    path = _resolve_knowledge_path(args.knowledge_path)

    if args.record_updated:
        if not path.exists():
            sys.exit(f"{path} does not exist -- --record-updated is for logging an already-edited page.")
        record_wiki_page(args.run_dir, args.knowledge_path)
        print(f"recorded update to {path} in {args.run_dir}/wiki_pages_written.json")
        return

    if not args.type or not args.title:
        sys.exit("--type and --title are required unless --record-updated is passed")

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
            source_docs=[d.strip() for d in args.source_docs.split(",")] if args.source_docs else None,
            status=args.status,
            body=args.body,
            force=args.force,
        )
    except PageAlreadyExists as e:
        sys.exit(str(e))

    record_wiki_page(args.run_dir, args.knowledge_path)

    print(f"wrote {path}")


if __name__ == "__main__":
    main()
