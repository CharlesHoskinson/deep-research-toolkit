"""Graph walks over the relation and wiki-link tables via DuckDB recursive
CTEs. Edges are treated as undirected for reachability. UNION (not UNION ALL)
terminates on cycles."""
from __future__ import annotations


def _walk(con, edge_sql: str, start: str, depth: int) -> list[dict]:
    sql = f"""
    WITH RECURSIVE edges(a, b) AS ({edge_sql}),
    walk(node, depth) AS (
        SELECT ?, 0
        UNION
        SELECT CASE WHEN e.a = w.node THEN e.b ELSE e.a END, w.depth + 1
        FROM walk w JOIN edges e ON (e.a = w.node OR e.b = w.node)
        WHERE w.depth < ?
    )
    SELECT node, min(depth) AS depth FROM walk WHERE node <> ?
    GROUP BY node ORDER BY depth, node
    """
    return [{"node": r[0], "depth": r[1]} for r in con.execute(sql, [start, int(depth), start]).fetchall()]


def neighbors(con, entity: str, depth: int = 1) -> list[dict]:
    return _walk(con, "SELECT subject, object FROM relations WHERE subject IS NOT NULL AND object IS NOT NULL",
                 entity, depth)


def wiki_link_neighbors(con, path: str, depth: int = 1) -> list[dict]:
    return _walk(con, "SELECT from_path, to_path FROM wiki_links", path, depth)
