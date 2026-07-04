"""Web research runs: research-runs/<source_id>/ mirrors a PDF run so the
knowledge compiler indexes web- and PDF-sourced claims uniformly. See
docs/contracts/knowledge-compiler.md."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

from ..common.hashing import content_hash

MANIFEST_SCHEMA_VERSION = "1.0"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "source"


def web_source_id(url: str, content: str) -> str:
    parsed = urlparse(url)
    base = _slug(f"{parsed.netloc}{parsed.path}")
    return f"{base}-{content_hash(content, length=8).split(':')[1]}"


def chunk_markdown(text: str, source_id: str) -> list[dict]:
    sections: list[tuple[str, list[str]]] = []
    current_title, current_lines = None, []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            if current_title is not None or current_lines:
                sections.append((current_title or "", current_lines))
            current_title = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_title is not None or current_lines:
        sections.append((current_title or "", current_lines))
    if not sections:
        sections = [("", text.splitlines())]

    nodes = []
    for i, (title, lines) in enumerate(sections, start=1):
        body = "\n".join(lines).strip()
        node_text = (title + "\n\n" + body).strip() if title else body
        nodes.append({
            "schema_version": "1.0",
            "node_id": f"{source_id}:c{str(i).zfill(2)}",
            "source_id": source_id, "type": "section", "title": title,
            "text": node_text, "content_hash": content_hash(node_text),
        })
    return nodes


def _now_iso() -> str:
    import datetime
    return (datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z"))


def start_research_run(url: str, content: str, research_runs_dir: Path) -> Path:
    research_runs_dir = Path(research_runs_dir)
    source_id = web_source_id(url, content)
    run_dir = research_runs_dir / source_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "source.md").write_text(content, encoding="utf-8")

    nodes = chunk_markdown(content, source_id)
    with open(run_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for node in nodes:
            f.write(json.dumps(node, ensure_ascii=False) + "\n")

    (run_dir / "manifest.json").write_text(json.dumps({
        "schema_version": MANIFEST_SCHEMA_VERSION, "producer": "web", "document_id": source_id,
        "source_url": url, "content_hash": content_hash(content), "fetched_at": _now_iso(),
        "chunk_count": len(nodes),
    }, indent=2) + "\n", encoding="utf-8")
    return run_dir
