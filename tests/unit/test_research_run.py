import json
from deep_research_toolkit.web.research_run import web_source_id, chunk_markdown, start_research_run


def test_web_source_id_stable_and_content_sensitive():
    a = web_source_id("https://ex.com/ows", "content one")
    b = web_source_id("https://ex.com/ows", "content one")
    c = web_source_id("https://ex.com/ows", "different content")
    assert a == b            # same url+content -> same id
    assert a != c            # content change -> different id
    assert a.startswith("ex-com-ows-")   # _slug turns "ex.com/ows" into "ex-com-ows"


def test_chunk_markdown_splits_on_headings():
    nodes = chunk_markdown("# A\nalpha text\n## B\nbeta text\n", "src-1")
    assert [n["title"] for n in nodes] == ["A", "B"]
    assert nodes[0]["node_id"] == "src-1:c01"
    assert "alpha" in nodes[0]["text"]


def test_start_research_run_writes_layout(tmp_path):
    run = start_research_run("https://ex.com/ows", "# OWS\nThe Open Wallet Standard.\n", tmp_path)
    assert (run / "source.md").read_text(encoding="utf-8").startswith("# OWS")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["producer"] == "web" and manifest["source_url"] == "https://ex.com/ows"
    assert (run / "chunks.jsonl").is_file()
