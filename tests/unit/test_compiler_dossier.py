import json
import duckdb
from types import SimpleNamespace
from deep_research_toolkit.compiler.schema import create_tables
from deep_research_toolkit.compiler.dossier import verbatim_ok, compose_dossier


def test_verbatim_ok_is_exact_substring():
    assert verbatim_ok("settles instantly", "text where it settles instantly here")
    assert not verbatim_ok("settles quickly", "text where it settles instantly here")
    assert not verbatim_ok("", "anything")


def test_compose_dossier_drops_paraphrased_claim(tmp_path):
    # the cited chunk (in chunks.jsonl) is the quote target -- same as extraction
    run = tmp_path / "research-runs" / "src-1"
    run.mkdir(parents=True)
    (run / "chunks.jsonl").write_text(json.dumps(
        {"node_id": "src-1:c0", "text": "Hydra settles instantly among participants."}) + "\n", encoding="utf-8")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")

    con = duckdb.connect(":memory:"); create_tables(con)
    con.executemany("INSERT INTO claims (claim_id, producer, source_id, claim) VALUES (?,?,?,?)",
                    [("c1", "web", "src-1", "good"), ("c2", "web", "src-1", "bad")])
    con.executemany(
        "INSERT INTO claim_evidence (claim_id, producer, source_id, locator, page, url, quote) VALUES (?,?,?,?,?,?,?)",
        [("c1", "web", "src-1", "src-1:c0", None, "u", "settles instantly"),
         ("c2", "web", "src-1", "src-1:c0", None, "u", "settles very fast")],  # paraphrase -> not verbatim
    )
    result = compose_dossier(con, cfg, ["c1", "c2"])
    assert [c["claim_id"] for c in result["included"]] == ["c1"]
    assert [c["claim_id"] for c in result["rejected"]] == ["c2"]


def test_gates_agree_on_a_quote_spanning_provenance_units(tmp_path):
    # Regression for the three-gates-disagree finding: a quote that spans two
    # provenance units within one chunk. Extraction accepts it (chunk text), and
    # the dossier gate + the pdf eval harness must now AGREE (all chunk-based),
    # not silently reject it because they re-derived a different source text.
    from deep_research_toolkit.common.verbatim import verbatim_ok as gate
    from deep_research_toolkit.pdf.eval import check_evidence_quotes_verbatim

    run = tmp_path / "pdf-runs" / "doc-x"
    run.mkdir(parents=True)
    chunk_text = "First part.\n\nSecond part."  # built from two provenance units joined with a blank line
    (run / "chunks.jsonl").write_text(json.dumps(
        {"node_id": "doc-x:n1", "text": chunk_text}) + "\n", encoding="utf-8")
    (run / "claims.jsonl").write_text(json.dumps(
        {"claim_id": "c1", "supporting_evidence": [
            {"node_id": "doc-x:n1", "quote": "First part.\n\nSecond part.", "page": 1}]}) + "\n", encoding="utf-8")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")

    con = duckdb.connect(":memory:"); create_tables(con)
    con.execute("INSERT INTO claims (claim_id, producer, source_id, claim) VALUES ('c1','pdf','doc-x','x')")
    con.execute("INSERT INTO claim_evidence (claim_id, producer, source_id, locator, page, url, quote) "
                "VALUES ('c1','pdf','doc-x','doc-x:n1',1,NULL,'First part.\n\nSecond part.')")

    assert gate("First part.\n\nSecond part.", chunk_text)                     # shared gate
    assert compose_dossier(con, cfg, ["c1"])["included"]                        # dossier gate agrees
    assert check_evidence_quotes_verbatim(run)["passed"]                        # eval harness agrees


def test_render_dossier_markdown_is_self_citing():
    from deep_research_toolkit.compiler.dossier import render_dossier_markdown
    d = {
        "included": [{"claim": "Snails are molluscs.",
                      "evidence": [{"quote": "Snails are molluscs", "url": "http://x", "locator": "s:c01", "page": None}]}],
        "rejected": [{"claim_id": "c9"}],
    }
    md = render_dossier_markdown(d)
    assert "Snails are molluscs." in md
    assert '"Snails are molluscs"' in md and "http://x" in md   # citation travels inline
    assert "1 claim(s) omitted" in md
