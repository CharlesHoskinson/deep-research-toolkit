import json
from pathlib import Path
from deep_research_toolkit.llm import extract

CHUNK = "Validators gossip block headers before the body arrives on the wire."

class FakeBackend:
    thinking = False
    def __init__(self, payload): self._payload = payload
    def complete(self, system, user, **kw): return self._payload

def _run(tmp_path, payload):
    (tmp_path / "chunks.jsonl").write_text(
        json.dumps({"locator": "doc#c1", "text": CHUNK}) + "\n", encoding="utf-8")
    return extract.extract_claims_to_run(tmp_path, "web", config=None,
                                         backend=FakeBackend(payload))

def test_span_claim_kept_and_quote_derived(tmp_path):
    # "Validators gossip block headers" == CHUNK[0:31]
    payload = json.dumps({"claims": [{
        "claim_id": "c1", "claim": "Validators gossip headers first.",
        "claim_type": "architectural", "confidence": "high",
        "supporting_evidence": [{"locator": "doc#c1", "start_char": 0, "end_char": 31}],
    }], "entities": [], "relations": []})
    summary = _run(tmp_path, payload)
    assert summary["written"] == 1
    row = json.loads((tmp_path / "claims.jsonl").read_text(encoding="utf-8").splitlines()[0])
    ev = row["supporting_evidence"][0]
    assert ev["quote"] == "Validators gossip block headers"  # derived from the span
    assert row["citable"] is True

def test_out_of_bounds_span_is_dropped(tmp_path):
    payload = json.dumps({"claims": [{
        "claim_id": "c2", "claim": "bogus", "confidence": "low",
        "supporting_evidence": [{"locator": "doc#c1", "start_char": 0, "end_char": 9999}],
    }], "entities": [], "relations": []})
    summary = _run(tmp_path, payload)
    assert summary["written"] == 0
    assert "c2" in summary["dropped"]
