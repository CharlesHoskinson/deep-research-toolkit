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


CANDS2 = [
    {"kind": "relation", "subject": "praos", "predicate": "introduced_in",
     "objects": ["2017", "2018"], "relation_ids": ["r1", "r2"], "source_ids": ["s1"]},
    {"kind": "relation", "subject": "ouroboros", "predicate": "extends",
     "objects": ["bft", "praos"], "relation_ids": ["r3", "r4"], "source_ids": ["s2"]},
]


def test_cross_candidate_relation_ids_are_invalid():
    reply = _reply([{"subject": "praos", "predicate": "introduced_in",
                     "verdict": "contradiction", "rationale": "x",
                     "relation_ids": ["r1", "r3"]}])
    out = adjudicate_candidates(CANDS2, StubBackend(reply))
    assert out["verdicts"] == []
    assert "not this candidate's" in out["invalid"][0]["reason"]


def test_missing_subject_predicate_is_invalid():
    reply = _reply([{"verdict": "contradiction", "rationale": "x", "relation_ids": ["r1"]}])
    out = adjudicate_candidates(CANDS, StubBackend(reply))
    assert out["verdicts"] == []
    assert "names no supplied candidate" in out["invalid"][0]["reason"]


def test_duplicate_verdicts_keep_first_flag_second():
    reply = _reply([
        {"subject": "praos", "predicate": "introduced_in", "verdict": "contradiction",
         "rationale": "a", "relation_ids": ["r1", "r2"]},
        {"subject": "praos", "predicate": "introduced_in", "verdict": "not_contradiction",
         "rationale": "b", "relation_ids": ["r1"]},
    ])
    out = adjudicate_candidates(CANDS, StubBackend(reply))
    assert len(out["verdicts"]) == 1 and out["verdicts"][0]["rationale"] == "a"
    assert "duplicate verdict" in out["invalid"][0]["reason"]


def test_object_wrapped_array_is_unwrapped():
    reply = "<output>" + json.dumps({"verdicts": [
        {"subject": "praos", "predicate": "introduced_in", "verdict": "not_contradiction",
         "rationale": "coexist", "relation_ids": ["r1"]}]}) + "</output>"
    out = adjudicate_candidates(CANDS, StubBackend(reply))
    assert len(out["verdicts"]) == 1


def test_looping_rationale_row_is_invalid_but_clean_row_is_kept():
    # One parseable reply: row 1's rationale degenerated into a >=40-word loop,
    # row 2 is clean -- row-level gating keeps the clean verdict.
    reply = _reply([
        {"subject": "praos", "predicate": "introduced_in", "verdict": "contradiction",
         "rationale": "years differ " * 25, "relation_ids": ["r1", "r2"]},
        {"subject": "ouroboros", "predicate": "extends", "verdict": "not_contradiction",
         "rationale": "extensions coexist", "relation_ids": ["r3"]},
    ])
    out = adjudicate_candidates(CANDS2, StubBackend(reply))
    assert out["parse_failures"] == 0
    assert len(out["verdicts"]) == 1
    assert out["verdicts"][0]["subject"] == "ouroboros"
    assert len(out["invalid"]) == 1
    assert "repetition" in out["invalid"][0]["reason"]


def test_batching_isolates_parse_failures():
    class PerCallBackend:
        def __init__(self, replies):
            self.replies = list(replies)

        def complete(self, system, user, **kw):
            return self.replies.pop(0)

    good = _reply([{"subject": "ouroboros", "predicate": "extends",
                    "verdict": "not_contradiction", "rationale": "x", "relation_ids": ["r3"]}])
    out = adjudicate_candidates(CANDS2, PerCallBackend(["garbage", good]), batch_size=1)
    assert out["parse_failures"] == 1
    assert len(out["verdicts"]) == 1
