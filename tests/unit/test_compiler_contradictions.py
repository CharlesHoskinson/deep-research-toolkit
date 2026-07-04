import duckdb
from deep_research_toolkit.compiler.schema import create_tables
from deep_research_toolkit.compiler.contradictions import find_candidates


def test_conflicting_objects_are_flagged():
    con = duckdb.connect(":memory:"); create_tables(con)
    con.executemany(
        "INSERT INTO relations (relation_id, subject, predicate, object, source_id) VALUES (?,?,?,?,?)",
        [("r1", "hydra", "throughput", "1000 TPS", "docA"),
         ("r2", "hydra", "throughput", "500 TPS", "docB"),
         ("r3", "hydra", "phase_count", "4", "docA")],
    )
    cands = [c for c in find_candidates(con) if c["kind"] == "relation"]
    assert len(cands) == 1
    assert cands[0]["subject"] == "hydra" and set(cands[0]["objects"]) == {"1000 TPS", "500 TPS"}


def test_conflicted_status_pages_are_flagged():
    con = duckdb.connect(":memory:"); create_tables(con)
    con.execute("INSERT INTO wiki_pages (path, status) VALUES ('c/x.md', 'conflicted')")
    con.execute("INSERT INTO wiki_pages (path, status) VALUES ('c/y.md', 'draft')")
    paths = [c["path"] for c in find_candidates(con) if c["kind"] == "conflicted_page"]
    assert paths == ["c/x.md"]
