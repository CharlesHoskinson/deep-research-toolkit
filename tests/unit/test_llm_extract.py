import json
from types import SimpleNamespace

from deep_research_toolkit.llm.extract import extract_claims_to_run, parse_claims_response


class _FakeBackend:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, user, **kw):
        return self.payload


def test_parse_claims_tolerates_surrounding_prose():
    text = 'Here are the claims:\n[{"claim_id": "c1", "claim": "x"}]\nDone.'
    assert parse_claims_response(text)[0]["claim_id"] == "c1"


def test_extract_drops_non_verbatim_quotes(tmp_path):
    run = tmp_path / "research-runs" / "src-1"
    run.mkdir(parents=True)
    (run / "source.md").write_text("Hydra settles instantly.", encoding="utf-8")
    (run / "chunks.jsonl").write_text(json.dumps(
        {"node_id": "src-1:c01", "text": "Hydra settles instantly."}) + "\n", encoding="utf-8")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")
    payload = json.dumps([
        {"claim_id": "c1", "claim": "good", "supporting_evidence": [
            {"locator": "src-1:c01", "quote": "settles instantly", "url": "u"}]},
        {"claim_id": "c2", "claim": "bad", "supporting_evidence": [
            {"locator": "src-1:c01", "quote": "settles very fast", "url": "u"}]},
    ])
    result = extract_claims_to_run(run, "web", cfg, _FakeBackend(payload))
    written = [json.loads(line) for line in (run / "claims.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert [c["claim_id"] for c in written] == ["c1"]
    assert result["dropped"] and result["written"] == 1


def test_extract_pdf_keeps_verbatim_and_drops_paraphrase(tmp_path):
    run = tmp_path / "pdf-runs" / "doc-abc"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"document_id": "doc-abc"}), encoding="utf-8")
    (run / "provenance.jsonl").write_text(json.dumps(
        {"page": 1, "text": "Hydra settles synchronously among participants."}) + "\n", encoding="utf-8")
    (run / "chunks.jsonl").write_text(json.dumps(
        {"node_id": "doc-abc:n5", "text": "Hydra settles synchronously among participants.",
         "source": {"page_start": 1}}) + "\n", encoding="utf-8")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")
    payload = json.dumps([
        {"claim_id": "c1", "claim": "good", "supporting_evidence": [
            {"node_id": "doc-abc:n5", "quote": "settles synchronously", "page": 1}]},
        {"claim_id": "c2", "claim": "bad", "supporting_evidence": [
            {"node_id": "doc-abc:n5", "quote": "settles very fast", "page": 1}]},
    ])
    result = extract_claims_to_run(run, "pdf", cfg, _FakeBackend(payload))
    written = [json.loads(line) for line in (run / "claims.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert [c["claim_id"] for c in written] == ["c1"]
    assert result["dropped"] == ["c2"] and result["written"] == 1
