"""Lexical (DuckDB FTS/BM25) + vector (LanceDB) search, fused with
Reciprocal Rank Fusion. RRF constant k=60 is the standard default."""
from __future__ import annotations


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def fts_search(con, table: str, id_col: str, text_col: str, query: str, k: int) -> list[str]:
    sql = (
        f"SELECT {id_col}, fts_main_{table}.match_bm25({id_col}, ?) AS score "
        f"FROM {table} WHERE score IS NOT NULL ORDER BY score DESC LIMIT {int(k)}"
    )
    try:
        return [row[0] for row in con.execute(sql, [query]).fetchall()]
    except Exception:
        # FTS index absent (e.g. the table had zero rows at compile time) -> no lexical hits.
        return []


def vector_search(lancedb_handle, table: str, embedder, query: str, k: int) -> list[str]:
    try:
        tbl = lancedb_handle.open_table(table)
    except Exception:
        return []
    qvec = embedder.embed([query])[0]
    rows = tbl.search(qvec).limit(k).to_list()
    return [r["id"] for r in rows]


def hybrid_search(con, lancedb_handle, embedder, *, table, id_col, text_col, vec_table, query, k) -> list[str]:
    lexical = fts_search(con, table, id_col, text_col, query, k)
    vector = vector_search(lancedb_handle, vec_table, embedder, query, k) if lancedb_handle else []
    return [i for i, _ in rrf_fuse([lexical, vector])][:k]
