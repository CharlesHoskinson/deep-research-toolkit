import json

from deep_research_toolkit.llm import extract

CHUNK = "Leaders rotate every epoch; followers verify the epoch signature."


class SeqBackend:
    """Returns a different payload per complete() call, to simulate N samples."""
    thinking = False

    def __init__(self, payloads):
        self._p = list(payloads)
        self._i = 0

    def complete(self, system, user, **kw):
        r = self._p[min(self._i, len(self._p) - 1)]
        self._i += 1
        return r


def _ev(s, e):
    return [{"locator": "d#c1", "start_char": s, "end_char": e}]


def _payload(claims):
    return json.dumps({"claims": claims, "entities": [], "relations": []})


def test_samples_union_recovers_more_claims(tmp_path):
    (tmp_path / "chunks.jsonl").write_text(json.dumps({"locator": "d#c1", "text": CHUNK}) + "\n", encoding="utf-8")
    # pass 1 finds claim A (span 0-7), pass 2 finds claim B (span 21-46)
    a = {"claim_id": "a", "claim": "Leaders rotate.", "confidence": "high", "supporting_evidence": _ev(0, 7)}
    b = {"claim_id": "b", "claim": "Followers verify.", "confidence": "high", "supporting_evidence": _ev(21, 46)}
    backend = SeqBackend([_payload([a]), _payload([b])])
    summary = extract.extract_claims_to_run(tmp_path, "web", None, backend, samples=2, min_support=1)
    assert summary["written"] == 2  # union of both passes


def _write_chunk(tmp_path):
    (tmp_path / "chunks.jsonl").write_text(json.dumps({"locator": "d#c1", "text": CHUNK}) + "\n", encoding="utf-8")


def test_min_support_drops_single_pass_claims(tmp_path):
    _write_chunk(tmp_path)
    a = {"claim_id": "a", "claim": "Leaders rotate.", "confidence": "high", "supporting_evidence": _ev(0, 7)}
    b = {"claim_id": "b", "claim": "Followers verify.", "confidence": "high", "supporting_evidence": _ev(21, 46)}
    common = {"claim_id": "c", "claim": "Epochs exist.", "confidence": "high", "supporting_evidence": _ev(21, 26)}
    backend = SeqBackend([_payload([a, common]), _payload([b, common])])
    summary = extract.extract_claims_to_run(tmp_path, "web", None, backend, samples=2, min_support=2)
    assert summary["written"] == 1          # only the claim seen in both passes
    assert summary["support_filtered"] == 2  # a and b were each seen once
    written = [json.loads(x) for x in (tmp_path / "claims.jsonl").read_text(encoding="utf-8").splitlines() if x]
    assert [c["claim_id"] for c in written] == ["p0_c"]  # first-seen (pass-0) dict, per-pass tagged


def test_entities_and_relations_come_from_pass_zero_only(tmp_path):
    _write_chunk(tmp_path)
    a = {"claim_id": "a", "claim": "Leaders rotate.", "confidence": "high", "supporting_evidence": _ev(0, 7)}
    b = {"claim_id": "b", "claim": "Followers verify.", "confidence": "high", "supporting_evidence": _ev(21, 46)}
    p0 = json.dumps({"claims": [a],
                     "entities": [{"entity_id": "leader", "name": "Leader", "aliases": [],
                                   "type": "role", "mentions": ["d#c1"]}],
                     "relations": [{"relation_id": "r1", "subject": "leader", "predicate": "rotates",
                                    "object": "epoch", "supporting_claim": "a"}]})
    p1 = json.dumps({"claims": [b],
                     "entities": [{"entity_id": "follower", "name": "Follower", "aliases": [],
                                   "type": "role", "mentions": ["d#c1"]}],
                     "relations": [{"relation_id": "r2", "subject": "follower", "predicate": "verifies",
                                    "object": "signature", "supporting_claim": "b"}]})
    backend = SeqBackend([p0, p1])
    summary = extract.extract_claims_to_run(tmp_path, "web", None, backend, samples=2, min_support=1)
    assert summary["written"] == 2
    assert summary["entities"] == 1 and summary["relations"] == 1  # pass 0 only
    ents = [json.loads(x) for x in (tmp_path / "entities.jsonl").read_text(encoding="utf-8").splitlines() if x]
    rels = [json.loads(x) for x in (tmp_path / "relations.jsonl").read_text(encoding="utf-8").splitlines() if x]
    assert [e["entity_id"] for e in ents] == ["leader"]  # entity ids are keys, not per-pass tagged
    assert [r["relation_id"] for r in rels] == ["p0_r1"]  # pass-0 relation resolves against unioned kept


def test_coverage_pass_adds_missed_claims_and_stops_early(tmp_path):
    _write_chunk(tmp_path)
    a = {"claim_id": "a", "claim": "Leaders rotate.", "confidence": "high", "supporting_evidence": _ev(0, 7)}
    b = {"claim_id": "b", "claim": "Followers verify.", "confidence": "high", "supporting_evidence": _ev(21, 46)}
    # pass 0 finds a; coverage pass 1 adds b; coverage pass 2 returns an empty
    # claims array -> loop stops early without burning the third allowed pass.
    backend = SeqBackend([_payload([a]), _payload([b]), _payload([])])
    summary = extract.extract_claims_to_run(tmp_path, "web", None, backend, coverage_passes=3)
    assert summary["written"] == 2
    assert backend._i == 3  # pass 0 + 2 coverage passes, early stop before the 3rd
    assert summary["parse_failures"] == 0  # a valid empty coverage answer is not a parse failure


def test_default_args_run_a_single_deterministic_pass(tmp_path):
    _write_chunk(tmp_path)
    a = {"claim_id": "a", "claim": "Leaders rotate.", "confidence": "high", "supporting_evidence": _ev(0, 7)}

    class Recorder(SeqBackend):
        def __init__(self, payloads):
            super().__init__(payloads)
            self.kwargs = []

        def complete(self, system, user, **kw):
            self.kwargs.append(kw)
            return super().complete(system, user, **kw)

    backend = Recorder([_payload([a])])
    summary = extract.extract_claims_to_run(tmp_path, "web", None, backend)
    assert summary["written"] == 1 and summary["samples"] == 1 and summary["support_filtered"] == 0
    assert backend.kwargs == [{}]  # exactly one call, no sampling overrides


def test_cross_pass_claim_ids_are_distinct_no_collision(tmp_path):
    # Regression: each pass is an independent call that restarts its own
    # claim_id numbering, and on a single-batch source no batch prefix is added.
    # Both passes emit "c1" for DIFFERENT claims; union_claims dedups by content,
    # not id, so both survive -> they must NOT share an id in claims.jsonl, or a
    # claim_id-keyed downstream lookup would blend evidence across the two.
    _write_chunk(tmp_path)
    x = {"claim_id": "c1", "claim": "Leaders rotate.", "confidence": "high", "supporting_evidence": _ev(0, 7)}
    y = {"claim_id": "c1", "claim": "Followers verify.", "confidence": "high", "supporting_evidence": _ev(21, 46)}
    backend = SeqBackend([_payload([x]), _payload([y])])
    summary = extract.extract_claims_to_run(tmp_path, "web", None, backend, samples=2, min_support=1)
    assert summary["written"] == 2
    written = [json.loads(v) for v in (tmp_path / "claims.jsonl").read_text(encoding="utf-8").splitlines() if v]
    ids = [c["claim_id"] for c in written]
    assert len(ids) == 2 and len(set(ids)) == 2  # distinct primary keys, no collision
