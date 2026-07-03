"""Health-check an OKF knowledge base: orphans, schema errors, broken
links, staleness. Producer-agnostic -- checks pages regardless of whether
they came from web research or PDF ingestion.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

from .frontmatter import find_links, parse_okf, resolve_link, validate_frontmatter


@dataclass
class LintProblem:
    category: str  # schema | broken-link | orphan | stale
    path: Path
    detail: str

    def __str__(self) -> str:
        return f"[{self.category}] {self.path}: {self.detail}"


def lint_knowledge_base(knowledge_base_dir: Path, stale_days: int = 180) -> list[LintProblem]:
    knowledge_base_dir = Path(knowledge_base_dir)
    if not knowledge_base_dir.is_dir():
        raise FileNotFoundError(f"no such directory: {knowledge_base_dir}")

    all_files = sorted(knowledge_base_dir.rglob("*.md"))
    docs: dict[Path, dict | None] = {}
    problems: list[LintProblem] = []
    incoming_links: dict[Path, set[Path]] = {f: set() for f in all_files}

    for path in all_files:
        text = path.read_text(encoding="utf-8")
        page = parse_okf(text, path=path)
        if page is None:
            problems.append(LintProblem("schema", path, "missing frontmatter block"))
            docs[path] = None
            continue
        docs[path] = page.frontmatter

        for field_problem in validate_frontmatter(page.frontmatter):
            problems.append(LintProblem("schema", path, field_problem))

        for target in find_links(text):
            if target.startswith(("http://", "https://")):
                continue
            resolved = resolve_link(target, path, knowledge_base_dir)
            if resolved not in all_files:
                problems.append(LintProblem("broken-link", path, f"links to missing file '{target}'"))
            else:
                incoming_links[resolved].add(path)

    for path, frontmatter in docs.items():
        if not frontmatter:
            continue
        is_index = frontmatter.get("type") == "Index" or path.name == "index.md"
        if not is_index and not incoming_links.get(path):
            problems.append(LintProblem("orphan", path, "no incoming links from other graph pages"))

        timestamp = frontmatter.get("timestamp")
        status = frontmatter.get("status")
        if timestamp and status == "researched":
            try:
                ts = timestamp if isinstance(timestamp, datetime.datetime) else datetime.datetime.fromisoformat(
                    str(timestamp).replace("Z", "+00:00")
                )
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=datetime.timezone.utc)
                now = datetime.datetime.now(ts.tzinfo)
                age_days = (now - ts).days
                if age_days > stale_days:
                    problems.append(
                        LintProblem("stale", path, f"last updated {age_days} days ago (status still 'researched')")
                    )
            except ValueError:
                problems.append(LintProblem("schema", path, f"unparseable timestamp '{timestamp}'"))

    return problems
