"""Mechanical contradiction *candidate* detection at compile/query time.
Confirming a candidate is a real contradiction is a batched LLM step driven
by the retrieval-planner SKILL.md -- never done in this module (ADR 0001
decision #3: tools stay cheap and deterministic)."""
from __future__ import annotations


def find_candidates(con) -> list[dict]:
    candidates: list[dict] = []

    rows = con.execute("""
        SELECT subject, predicate,
               list(DISTINCT object)  AS objects,
               list(relation_id)      AS relation_ids,
               list(DISTINCT source_id) AS source_ids
        FROM relations
        WHERE subject IS NOT NULL AND predicate IS NOT NULL AND object IS NOT NULL
        GROUP BY subject, predicate
        HAVING count(DISTINCT object) > 1
    """).fetchall()
    for subject, predicate, objects, relation_ids, source_ids in rows:
        candidates.append({
            "kind": "relation", "subject": subject, "predicate": predicate,
            "objects": list(objects), "relation_ids": list(relation_ids),
            "source_ids": list(source_ids),
        })

    for (path,) in con.execute(
        "SELECT path FROM wiki_pages WHERE status = 'conflicted' ORDER BY path"
    ).fetchall():
        candidates.append({"kind": "conflicted_page", "path": path})

    return candidates
