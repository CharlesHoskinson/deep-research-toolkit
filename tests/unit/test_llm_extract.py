import json
from types import SimpleNamespace

from deep_research_toolkit.llm.extract import (
    extract_claims_to_run,
    parse_claims_response,
    parse_extraction_response,
)


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


def test_extract_pdf_gate_uses_chunk_text_not_provenance(tmp_path):
    # The chunk the model is shown joins two source units with a blank line ("\n\n"); the
    # re-derived provenance page text joins the same units with a single "\n". A quote that
    # spans that boundary is verbatim in the chunk the model actually quoted from, but NOT in
    # the provenance page text. The extract gate must accept it -- it is aligned to chunk
    # text (what the prompt showed the model), not to a re-derived provenance view.
    run = tmp_path / "pdf-runs" / "doc-x"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"document_id": "doc-x"}), encoding="utf-8")
    (run / "provenance.jsonl").write_text(
        json.dumps({"page": 1, "text": "First part."}) + "\n"
        + json.dumps({"page": 1, "text": "Second part."}) + "\n", encoding="utf-8")
    (run / "chunks.jsonl").write_text(json.dumps(
        {"node_id": "doc-x:n1", "text": "First part.\n\nSecond part.", "source": {"page_start": 1}}) + "\n",
        encoding="utf-8")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")
    payload = json.dumps([
        {"claim_id": "c1", "claim": "spans boundary", "supporting_evidence": [
            {"node_id": "doc-x:n1", "quote": "First part.\n\nSecond part.", "page": 1}]},
    ])
    result = extract_claims_to_run(run, "pdf", cfg, _FakeBackend(payload))
    assert result["written"] == 1 and result["dropped"] == []


def test_extract_resolves_abbreviated_chunk_ids(tmp_path):
    # A reasoning model often shortens a long chunk id (emits "n5" for "doc-abc:n5").
    # The gate must still resolve it, keep the verbatim claim, and rewrite the
    # evidence id to the canonical chunk id for downstream tools.
    run = tmp_path / "pdf-runs" / "doc-abc"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"document_id": "doc-abc"}), encoding="utf-8")
    (run / "provenance.jsonl").write_text(json.dumps(
        {"page": 1, "text": "Hydra settles synchronously among participants."}) + "\n", encoding="utf-8")
    (run / "chunks.jsonl").write_text(json.dumps(
        {"node_id": "doc-abc:n5", "text": "Hydra settles synchronously among participants.",
         "source": {"page_start": 1}}) + "\n", encoding="utf-8")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")
    payload = "<output>" + json.dumps([
        {"claim_id": "c1", "claim": "good", "supporting_evidence": [
            {"node_id": "n5", "quote": "settles synchronously", "page": 1}]},
    ]) + "</output>"
    result = extract_claims_to_run(run, "pdf", cfg, _FakeBackend(payload))
    written = [json.loads(x) for x in (run / "claims.jsonl").read_text(encoding="utf-8").splitlines() if x]
    assert result["written"] == 1
    assert written[0]["supporting_evidence"][0]["node_id"] == "doc-abc:n5"  # rewritten to canonical


def test_parse_extraction_reads_output_block_and_ignores_reasoning():
    text = ('<think>I will plan, draft, verify each quote, then emit.</think>\n'
            'Here is the result:\n<output>\n'
            '{"claims":[{"claim_id":"c1"}],"entities":[{"entity_id":"e1"}],"relations":[]}\n'
            '</output>\nDone.')
    p = parse_extraction_response(text)
    assert [c["claim_id"] for c in p["claims"]] == ["c1"]
    assert [e["entity_id"] for e in p["entities"]] == ["e1"]
    assert p["relations"] == []


def test_extract_full_contract_writes_all_three_files(tmp_path):
    run = tmp_path / "research-runs" / "src-9"
    run.mkdir(parents=True)
    (run / "source.md").write_text("Hydra settles instantly. Cardano uses eUTXO.", encoding="utf-8")
    (run / "chunks.jsonl").write_text(json.dumps(
        {"node_id": "src-9:c01", "text": "Hydra settles instantly. Cardano uses eUTXO."}) + "\n", encoding="utf-8")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")
    payload = "<think>plan then verify</think>\n<output>\n" + json.dumps({
        "claims": [{"claim_id": "c1", "claim": "Hydra settles instantly.",
                    "supporting_evidence": [{"locator": "src-9:c01", "quote": "Hydra settles instantly", "url": None}]}],
        "entities": [{"entity_id": "hydra", "name": "Hydra", "aliases": [], "type": "protocol",
                      "mentions": ["src-9:c01"]}],
        "relations": [{"relation_id": "r1", "subject": "cardano", "predicate": "uses", "object": "eutxo",
                       "supporting_claim": "c1"}],
    }) + "\n</output>"
    result = extract_claims_to_run(run, "web", cfg, _FakeBackend(payload))
    assert result["written"] == 1 and result["entities"] == 1 and result["relations"] == 1
    claims = [json.loads(x) for x in (run / "claims.jsonl").read_text(encoding="utf-8").splitlines() if x]
    ents = [json.loads(x) for x in (run / "entities.jsonl").read_text(encoding="utf-8").splitlines() if x]
    rels = [json.loads(x) for x in (run / "relations.jsonl").read_text(encoding="utf-8").splitlines() if x]
    assert claims[0]["document_id"] == "src-9" and claims[0]["schema_version"] == "1.0"
    assert ents[0]["entity_id"] == "hydra" and ents[0]["schema_version"] == "1.0"
    assert rels[0]["predicate"] == "uses" and rels[0]["document_id"] == "src-9"
