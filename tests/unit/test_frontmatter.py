from deep_research_toolkit.common.frontmatter import (
    find_links,
    parse_okf,
    render_okf,
    resolve_link,
    validate_frontmatter,
    write_okf,
)


def test_parse_okf_round_trip():
    text = (
        "---\n"
        "type: Concept\n"
        "title: Test\n"
        "timestamp: 2026-07-03T00:00:00Z\n"
        "status: draft\n"
        "---\n\n"
        "# Test\nBody with a [link](/concepts/other.md).\n"
    )
    page = parse_okf(text)
    assert page is not None
    assert page.frontmatter["title"] == "Test"
    assert page.links == ["/concepts/other.md"]


def test_parse_okf_no_frontmatter_returns_none():
    assert parse_okf("# Just a heading\n\nNo frontmatter here.\n") is None


def test_find_links():
    assert find_links("See [a](/x.md) and [b](y.md) and [c](https://example.com).") == ["/x.md", "y.md"]


def test_validate_frontmatter_required_fields():
    problems = validate_frontmatter({"type": "Concept"})
    assert any("title" in p for p in problems)
    assert any("timestamp" in p for p in problems)


def test_validate_frontmatter_bad_status():
    problems = validate_frontmatter({"type": "Concept", "title": "X", "timestamp": "Y", "status": "bogus"})
    assert any("status" in p for p in problems)


def test_validate_frontmatter_all_new_status_values_accepted():
    for status in ["seed", "researched", "stale", "draft", "conflicted"]:
        problems = validate_frontmatter({"type": "Concept", "title": "X", "timestamp": "Y", "status": status})
        assert problems == []


def test_write_okf_adds_okf_version(tmp_path):
    path = tmp_path / "page.md"
    write_okf(path, {"type": "Concept", "title": "X", "timestamp": "Y"}, "# X\n")
    page = parse_okf(path.read_text(encoding="utf-8"))
    assert page.frontmatter["okf_version"] == "1.0"


def test_resolve_link_leading_slash_resolves_from_kb_root(tmp_path):
    kb_root = tmp_path / "kb"
    current_file = kb_root / "concepts" / "a.md"
    result = resolve_link("/standards/b.md", current_file, kb_root)
    assert result == (kb_root / "standards" / "b.md").resolve()


def test_resolve_link_relative_resolves_from_current_dir(tmp_path):
    kb_root = tmp_path / "kb"
    current_file = kb_root / "concepts" / "a.md"
    result = resolve_link("b.md", current_file, kb_root)
    assert result == (kb_root / "concepts" / "b.md").resolve()
