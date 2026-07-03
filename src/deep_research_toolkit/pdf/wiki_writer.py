"""llm-wiki-writer's audit-trail bookkeeping: which knowledge-base-relative
paths a pdf-runs/<document_id>/ run touched.

This module does NOT write page content -- that's
`deep_research_toolkit.common.scaffold.scaffold_page` (shared with
research-knowledge-graph's scaffold_page.py). This module is purely the
PDF-pipeline-specific run-directory tracking layer on top of it: appending
(deduped) to <run_dir>/wiki_pages_written.json and merging a
`pages_written` list into manifest.json's `stages.llm-wiki-writer` entry,
per docs/contracts/pdf-ingestion-pipeline.md.

Ported from agentictrading's scaffold_wiki_page.py
(record_touched_page/update_manifest), which did this same bookkeeping
inline in a single script -- split out here so it's shared by both the
create-mode and --record-updated-mode code paths in
skills/llm-wiki-writer/scripts/scaffold_wiki_page.py without duplication.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..common.manifest import load_manifest, update_stage

WIKI_PAGES_WRITTEN_FILENAME = "wiki_pages_written.json"


def _knowledge_relative(knowledge_path: str | Path) -> str:
    """Normalize to forward slashes, matching the contract's manifest.json
    example (paths like "knowledge/concepts/hydra-settlement.md")."""
    return str(knowledge_path).replace(os.sep, "/")


def _load_touched(audit_path: Path) -> list[str]:
    if not audit_path.exists():
        return []
    with open(audit_path, encoding="utf-8") as f:
        try:
            touched = json.load(f)
        except json.JSONDecodeError:
            return []
    return touched if isinstance(touched, list) else []


def record_wiki_page(run_dir: str | Path, knowledge_base_relative_path: str | Path) -> list[str]:
    """Append `knowledge_base_relative_path` (deduped, order-preserving) to
    <run_dir>/wiki_pages_written.json, and merge it into manifest.json's
    `stages.llm-wiki-writer.pages_written` list.

    Safe to call multiple times per run (a run can write several pages) --
    each call reads back whatever `pages_written` the manifest already has
    for this stage and merges into it, since `common.manifest.update_stage`
    replaces the whole stage entry rather than merging fields within it;
    without this read-merge step, a second page's call would clobber the
    first page's entry.

    Returns the full deduped list of paths touched by this run so far.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    audit_path = run_dir / WIKI_PAGES_WRITTEN_FILENAME
    touched = _load_touched(audit_path)
    rel = _knowledge_relative(knowledge_base_relative_path)
    if rel not in touched:
        touched.append(rel)

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(touched, f, indent=2)
        f.write("\n")

    # Merge with whatever pages_written the manifest's llm-wiki-writer stage
    # already recorded (from an earlier call in this same run), rather than
    # relying solely on wiki_pages_written.json, so a manifest that was
    # hand-edited or out of sync still ends up correct.
    manifest = load_manifest(run_dir)
    existing_pages: list[str] = []
    if manifest is not None:
        existing_pages = manifest.get("stages", {}).get("llm-wiki-writer", {}).get("pages_written", [])
    merged = list(dict.fromkeys([*existing_pages, *touched]))  # dedupe, preserve order

    update_stage(run_dir, "llm-wiki-writer", pages_written=merged)

    return merged
