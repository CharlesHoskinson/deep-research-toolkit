"""Create a new OKF page, refusing to silently overwrite an existing one.

Shared by both producers' "scaffold a page" scripts (research-knowledge-
graph's scaffold_page.py and llm-wiki-writer's scaffold_wiki_page.py) --
the only difference between them is the extra source_docs/status-enum
handling the PDF pipeline needs, which the wiki-writer's thin script layers
on top of this.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .frontmatter import write_okf


class PageAlreadyExists(FileExistsError):
    pass


def scaffold_page(
    path: Path,
    *,
    type: str,
    title: str,
    description: str | None = None,
    resource: str | None = None,
    aliases: list[str] | None = None,
    tags: list[str] | None = None,
    source: str | None = None,
    source_docs: list[str] | None = None,
    status: str = "seed",
    body: str = "",
    force: bool = False,
    now_iso: str | None = None,
) -> Path:
    path = Path(path)
    if path.exists() and not force:
        raise PageAlreadyExists(
            f"{path} already exists. This concept may already have a page — "
            "edit it instead of creating a duplicate, or pass force=True/--force to overwrite."
        )

    import datetime

    frontmatter: dict[str, Any] = {
        "type": type,
        "title": title,
        "timestamp": now_iso
        or datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    if description:
        frontmatter["description"] = description
    if resource:
        frontmatter["resource"] = resource
    if aliases:
        frontmatter["aliases"] = aliases
    if tags:
        frontmatter["tags"] = tags
    if source:
        frontmatter["source"] = source
    if source_docs:
        frontmatter["source_docs"] = source_docs
    frontmatter["status"] = status

    write_okf(path, frontmatter, body or f"# {title}\n")
    return path
