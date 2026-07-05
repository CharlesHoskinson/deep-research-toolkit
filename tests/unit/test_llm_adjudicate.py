import json

from deep_research_toolkit.llm.adjudicate import adjudicate_candidates

CANDS = [
    {"kind": "relation", "subject": "praos", "predicate": "introduced_in",
     "objects": ["2017", "2018"], "relation_ids": ["r1", "r2"], "source_ids": ["s1", "s2"]},
]


class StubBackend:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user, **kw):
        return self.reply


def _reply(verdicts):
    return "<output>" + json.dumps(verdicts) + "</output>"


def test_valid_verdict_accepted():
    reply = _reply([{"subject": "praos", "predicate": "introduced_in",
                     "verdict": "contradiction", "rationale": "years differ",
                     "relation_ids": ["r1", "r2"]}])
    out = adjudicate_candidates(CANDS, StubBackend(reply))
    assert len(out["verdicts"]) == 1
    assert out["invalid"] == []


def test_bad_enum_and_foreign_relation_ids_are_invalid():
    reply = _reply([{"subject": "praos", "predicate": "introduced_in",
                     "verdict": "maybe", "rationale": "?", "relation_ids": ["r9"]}])
    out = adjudicate_candidates(CANDS, StubBackend(reply))
    assert out["verdicts"] == []
    assert len(out["invalid"]) == 1


def test_unparseable_reply_counts_as_parse_failure():
    out = adjudicate_candidates(CANDS, StubBackend("no json at all"))
    assert out["parse_failures"] == 1
