"""Synthetic-fixture tests for lint_knowledge_base -- injects known defects
and confirms exactly those are caught, with no false positives on a clean
graph. Mirrors the real defect-injection approach used to verify the
original prototype's lint_graph.py."""
import datetime

from deep_research_toolkit.common.frontmatter import write_okf
from deep_research_toolkit.common.lint import lint_knowledge_base


def _write(kb, rel_path, frontmatter, body="# Page\n"):
    write_okf(kb / rel_path, frontmatter, body)


def test_clean_graph_has_no_problems(tmp_path):
    kb = tmp_path / "kb"
    _write(kb, "index.md", {"type": "Index", "title": "Index", "timestamp": "2026-07-03T00:00:00Z"})
    _write(
        kb,
        "concepts/a.md",
        {"type": "Concept", "title": "A", "timestamp": "2026-07-03T00:00:00Z", "status": "seed"},
        "# A\n\nSee [index](/index.md).\n",
    )
    # link the page from index so it's not an orphan
    (kb / "index.md").write_text(
        (kb / "index.md").read_text(encoding="utf-8") + "\n[A](/concepts/a.md)\n", encoding="utf-8"
    )

    problems = lint_knowledge_base(kb)
    assert problems == []


def test_orphan_detected(tmp_path):
    kb = tmp_path / "kb"
    _write(kb, "concepts/orphan.md", {"type": "Concept", "title": "Orphan", "timestamp": "2026-07-03T00:00:00Z"})

    problems = lint_knowledge_base(kb)
    assert any(p.category == "orphan" for p in problems)


def test_broken_link_detected(tmp_path):
    kb = tmp_path / "kb"
    _write(kb, "index.md", {"type": "Index", "title": "Index", "timestamp": "2026-07-03T00:00:00Z"})
    (kb / "index.md").write_text(
        (kb / "index.md").read_text(encoding="utf-8") + "\n[Missing](/concepts/does-not-exist.md)\n",
        encoding="utf-8",
    )

    problems = lint_knowledge_base(kb)
    assert any(p.category == "broken-link" for p in problems)


def test_stale_researched_page_detected(tmp_path):
    kb = tmp_path / "kb"
    old_ts = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=300)).isoformat().replace("+00:00", "Z")
    _write(kb, "index.md", {"type": "Index", "title": "Index", "timestamp": "2026-07-03T00:00:00Z"})
    _write(kb, "concepts/stale.md", {"type": "Concept", "title": "Stale", "timestamp": old_ts, "status": "researched"})
    (kb / "index.md").write_text(
        (kb / "index.md").read_text(encoding="utf-8") + "\n[Stale](/concepts/stale.md)\n", encoding="utf-8"
    )

    problems = lint_knowledge_base(kb, stale_days=180)
    assert any(p.category == "stale" for p in problems)


def test_seed_status_is_never_flagged_stale(tmp_path):
    kb = tmp_path / "kb"
    old_ts = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1000)).isoformat().replace("+00:00", "Z")
    _write(kb, "index.md", {"type": "Index", "title": "Index", "timestamp": "2026-07-03T00:00:00Z"})
    _write(kb, "concepts/seed.md", {"type": "Concept", "title": "Seed", "timestamp": old_ts, "status": "seed"})
    (kb / "index.md").write_text(
        (kb / "index.md").read_text(encoding="utf-8") + "\n[Seed](/concepts/seed.md)\n", encoding="utf-8"
    )

    problems = lint_knowledge_base(kb, stale_days=180)
    assert not any(p.category == "stale" for p in problems)


def test_missing_required_field_detected(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir(parents=True)
    (kb / "bad.md").write_text("---\ntype: Concept\n---\n\n# Bad\n", encoding="utf-8")  # missing title, timestamp

    problems = lint_knowledge_base(kb)
    schema_problems = [p for p in problems if p.category == "schema"]
    assert any("title" in p.detail for p in schema_problems)
    assert any("timestamp" in p.detail for p in schema_problems)
