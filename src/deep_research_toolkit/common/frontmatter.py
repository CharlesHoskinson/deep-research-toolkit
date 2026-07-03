"""Open Knowledge Format (OKF) frontmatter: read/write markdown+YAML pages.

One parser/writer shared by every skill that touches knowledge_base/ pages
(research-knowledge-graph's ingest/query/lint, llm-wiki-writer, and later
the knowledge compiler) -- generalized from agentictrading's
scaffold_okf.py/lint_graph.py, which each reimplemented a slice of this.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

OKF_SCHEMA_VERSION = "1.0"
REQUIRED_FIELDS = ["type", "title", "timestamp"]
STATUS_VALUES = {"seed", "researched", "stale", "draft", "conflicted"}

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_LINK_RE = re.compile(r"\]\(([^)]+\.md)\)")


@dataclass
class OkfPage:
    frontmatter: dict[str, Any]
    body: str
    path: Path | None = None

    @property
    def links(self) -> list[str]:
        return find_links(self.body)


def find_links(text: str) -> list[str]:
    """Every relative-markdown-link target (`](...)` ending in `.md`) in text."""
    return _LINK_RE.findall(text)


def parse_okf(text: str, path: Path | None = None) -> OkfPage | None:
    """Parse a markdown+YAML-frontmatter document. Returns None if there's
    no frontmatter block at all (caller decides whether that's an error).
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return None
    frontmatter = yaml.safe_load(match.group(1)) or {}
    body = text[match.end():]
    return OkfPage(frontmatter=frontmatter, body=body, path=path)


def read_okf(path: Path) -> OkfPage | None:
    with open(path, encoding="utf-8") as f:
        return parse_okf(f.read(), path=path)


def render_okf(frontmatter: dict[str, Any], body: str) -> str:
    out = ["---\n"]
    out.append(yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True))
    out.append("---\n\n")
    out.append(body)
    return "".join(out)


def write_okf(path: Path, frontmatter: dict[str, Any], body: str) -> None:
    frontmatter.setdefault("okf_version", OKF_SCHEMA_VERSION)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_okf(frontmatter, body))


def validate_frontmatter(frontmatter: dict[str, Any]) -> list[str]:
    """Return a list of problems (empty if valid). Does not raise -- callers
    like lint decide what to do with problems; scaffolding callers can
    choose to raise on a non-empty list.
    """
    problems = []
    for field_name in REQUIRED_FIELDS:
        if field_name not in frontmatter:
            problems.append(f"missing required field '{field_name}'")
    status = frontmatter.get("status")
    if status is not None and status not in STATUS_VALUES:
        problems.append(f"invalid status '{status}' (must be one of {sorted(STATUS_VALUES)})")
    return problems


def resolve_link(target: str, current_file: Path, knowledge_base_dir: Path) -> Path:
    """A link starting with '/' resolves from the knowledge_base root; a
    link without a leading '/' resolves relative to the current file's
    directory. Matches the convention documented in okf-schema.md.
    """
    if target.startswith("/"):
        return (knowledge_base_dir / target.lstrip("/")).resolve()
    return (current_file.parent / target).resolve()
