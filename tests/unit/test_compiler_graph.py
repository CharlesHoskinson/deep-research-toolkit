import duckdb
from deep_research_toolkit.compiler.schema import create_tables
from deep_research_toolkit.compiler.graph import neighbors


def _con():
    con = duckdb.connect(":memory:")
    create_tables(con)
    con.executemany(
        "INSERT INTO relations (relation_id, subject, predicate, object) VALUES (?, ?, ?, ?)",
        [("r1", "hydra", "serves_as", "settlement"),
         ("r2", "hydra", "open_question", "ows"),
         ("r3", "ows", "defined_by", "spec")],
    )
    return con


def test_neighbors_depth_1():
    got = {n["node"] for n in neighbors(_con(), "hydra", depth=1)}
    assert got == {"settlement", "ows"}


def test_neighbors_depth_2_reaches_further():
    got = {n["node"] for n in neighbors(_con(), "hydra", depth=2)}
    assert "spec" in got  # hydra -> ows -> spec


def test_neighbors_excludes_self():
    assert all(n["node"] != "hydra" for n in neighbors(_con(), "hydra", depth=3))
