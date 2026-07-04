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
    # web source.md is the quote target
    run = tmp_path / "research-runs" / "src-1"
    run.mkdir(parents=True)
    (run / "source.md").write_text("Hydra settles instantly among participants.", encoding="utf-8")
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
