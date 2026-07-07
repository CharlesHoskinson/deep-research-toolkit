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
        # chunk[6:23] == "settles instantly"
        {"claim_id": "c1", "claim": "good", "supporting_evidence": [
            {"locator": "src-1:c01", "start_char": 6, "end_char": 23, "url": "u"}]},
        # span slice != the echoed quote (near-quote) -> gate failure
        {"claim_id": "c2", "claim": "bad", "supporting_evidence": [
            {"locator": "src-1:c01", "start_char": 6, "end_char": 23,
             "quote": "settles very fast", "url": "u"}]},
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
        # chunk[6:27] == "settles synchronously"
        {"claim_id": "c1", "claim": "good", "supporting_evidence": [
            {"node_id": "doc-abc:n5", "start_char": 6, "end_char": 27, "page": 1}]},
        # echoed quote is a paraphrase of the span it points at -> gate failure
        {"claim_id": "c2", "claim": "bad", "supporting_evidence": [
            {"node_id": "doc-abc:n5", "start_char": 6, "end_char": 27,
             "quote": "settles very fast", "page": 1}]},
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
        # span 0:25 covers "First part.\n\nSecond part." in the CHUNK text; the
        # echoed quote must be checked against that same chunk text, not provenance.
        {"claim_id": "c1", "claim": "spans boundary", "supporting_evidence": [
            {"node_id": "doc-x:n1", "start_char": 0, "end_char": 25,
             "quote": "First part.\n\nSecond part.", "page": 1}]},
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
        # chunk[6:27] == "settles synchronously"
        {"claim_id": "c1", "claim": "good", "supporting_evidence": [
            {"node_id": "n5", "start_char": 6, "end_char": 27, "page": 1}]},
    ]) + "</output>"
    result = extract_claims_to_run(run, "pdf", cfg, _FakeBackend(payload))
    written = [json.loads(x) for x in (run / "claims.jsonl").read_text(encoding="utf-8").splitlines() if x]
    assert result["written"] == 1
    assert written[0]["supporting_evidence"][0]["node_id"] == "doc-abc:n5"  # rewritten to canonical


def test_extract_resolves_abbreviated_entity_mentions(tmp_path):
    # The model abbreviates chunk ids in entity mentions too ("c01" for
    # "src-7:c01"); those must resolve to canonical chunk ids so entity_mentions
    # joins back to chunks. Unresolvable mentions are dropped, not left dangling.
    run = tmp_path / "research-runs" / "src-7"
    run.mkdir(parents=True)
    (run / "source.md").write_text("Cardano uses proof of stake.", encoding="utf-8")
    (run / "chunks.jsonl").write_text(json.dumps(
        {"node_id": "src-7:c01", "text": "Cardano uses proof of stake."}) + "\n", encoding="utf-8")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")
    payload = "<output>" + json.dumps({
        "claims": [{"claim_id": "c1", "claim": "Cardano uses proof of stake.",
                    "supporting_evidence": [  # chunk[0:27] == "Cardano uses proof of stake"
                        {"locator": "c01", "start_char": 0, "end_char": 27, "url": None}]}],
        "entities": [{"entity_id": "cardano", "name": "Cardano", "aliases": [], "type": "blockchain",
                      "mentions": ["c01"]},
                     {"entity_id": "ghost", "name": "Ghost", "aliases": [], "type": "x", "mentions": ["c99"]}],
        "relations": [],
    }) + "</output>"
    result = extract_claims_to_run(run, "web", cfg, _FakeBackend(payload))
    ents = {e["entity_id"]: e for e in
            (json.loads(x) for x in (run / "entities.jsonl").read_text(encoding="utf-8").splitlines() if x)}
    assert ents["cardano"]["mentions"] == ["src-7:c01"]   # abbreviated -> canonical
    assert ents["ghost"]["mentions"] == []                # unresolvable -> dropped
    assert result["written"] == 1                         # claim's abbreviated locator also resolved


def test_extract_drops_relations_referencing_dropped_claims(tmp_path):
    run = tmp_path / "research-runs" / "src-r"
    run.mkdir(parents=True)
    (run / "source.md").write_text("Alpha is real.", encoding="utf-8")
    (run / "chunks.jsonl").write_text(json.dumps({"node_id": "src-r:c01", "text": "Alpha is real."}) + "\n",
                                      encoding="utf-8")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")
    payload = json.dumps({
        "claims": [
            {"claim_id": "good", "claim": "Alpha is real.",  # chunk[0:13] == "Alpha is real"
             "supporting_evidence": [{"locator": "src-r:c01", "start_char": 0, "end_char": 13, "url": None}]},
            {"claim_id": "bad", "claim": "x",  # span slice != echoed quote -> dropped
             "supporting_evidence": [{"locator": "src-r:c01", "start_char": 0, "end_char": 13,
                                      "quote": "paraphrase not present", "url": None}]},
        ],
        "entities": [],
        "relations": [
            {"relation_id": "r1", "subject": "a", "predicate": "p", "object": "b", "supporting_claim": "good"},
            {"relation_id": "r2", "subject": "a", "predicate": "p", "object": "c", "supporting_claim": "bad"},
            {"relation_id": "r3", "subject": "a", "predicate": "p", "object": "d"},  # no supporting_claim
        ],
    })
    result = extract_claims_to_run(run, "web", cfg, _FakeBackend(payload))
    ids = {json.loads(x)["relation_id"] for x in (run / "relations.jsonl").read_text(encoding="utf-8").splitlines() if x}
    assert ids == {"r1", "r3"}  # r2 pointed at the dropped claim
    assert result["relations"] == 2


def test_extract_retries_failed_batch_by_splitting(tmp_path):
    run = tmp_path / "research-runs" / "src-rt"
    run.mkdir(parents=True)
    with open(run / "chunks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"node_id": "src-rt:c01", "text": "Alpha fact."}) + "\n")
        f.write(json.dumps({"node_id": "src-rt:c02", "text": "Beta fact."}) + "\n")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")

    class _Splitty:  # full 2-chunk batch fails to parse; each 1-chunk half succeeds
        def complete(self, system, user, **kw):
            has1, has2 = "src-rt:c01" in user, "src-rt:c02" in user
            if has1 and has2:
                return "garbage, no json object at all"
            cid, q = ("src-rt:c01", "Alpha fact") if has1 else ("src-rt:c02", "Beta fact")
            return "<output>" + json.dumps({
                "claims": [{"claim_id": "c1", "claim": q,
                            "supporting_evidence": [{"locator": cid, "start_char": 0,
                                                     "end_char": len(q), "url": None}]}],
                "entities": [], "relations": []}) + "</output>"

    result = extract_claims_to_run(run, "web", cfg, _Splitty(), batch_size=2)
    assert result["written"] == 2 and result["parse_failures"] == 0  # the split recovered both


def test_extract_retry_batches_get_failure_note_and_temperature(tmp_path):
    # A full 2-chunk batch fails to parse; each halved 1-chunk retry batch must
    # carry the parse-failure note appended to its user prompt and a temperature
    # override, per the retry-mutation contract (identical-prompt retries
    # reproduce identical failures).
    run = tmp_path / "research-runs" / "src-rn"
    run.mkdir(parents=True)
    with open(run / "chunks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"node_id": "src-rn:c01", "text": "Alpha fact."}) + "\n")
        f.write(json.dumps({"node_id": "src-rn:c02", "text": "Beta fact."}) + "\n")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")

    class _FailOnce:
        def __init__(self):
            self.calls = []

        def complete(self, system, user, **kw):
            self.calls.append((user, kw))
            if len(self.calls) == 1:
                return "garbage, no json object at all"
            cid, q = ("src-rn:c01", "Alpha fact") if "src-rn:c01" in user else ("src-rn:c02", "Beta fact")
            return "<output>" + json.dumps({
                "claims": [{"claim_id": "c1", "claim": q,
                            "supporting_evidence": [{"locator": cid, "start_char": 0,
                                                     "end_char": len(q), "url": None}]}],
                "entities": [], "relations": []}) + "</output>"

    backend = _FailOnce()
    result = extract_claims_to_run(run, "web", cfg, backend, batch_size=2)
    assert result["written"] == 2 and result["parse_failures"] == 0
    assert len(backend.calls) == 3
    first_user, first_kw = backend.calls[0]
    assert "NOTE: a previous attempt" not in first_user
    assert first_kw.get("temperature") is None
    for retry_user, retry_kw in backend.calls[1:]:
        assert "NOTE: a previous attempt on these chunks failed to parse as the required JSON. " \
               "Emit ONLY the contract JSON." in retry_user
        assert retry_kw.get("temperature") == 0.25


def test_extract_ambiguous_abbreviated_id_is_not_resolved(tmp_path):
    run = tmp_path / "research-runs" / "src-am"
    run.mkdir(parents=True)
    with open(run / "chunks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"node_id": "src-am:axn5", "text": "Alpha fact here."}) + "\n")
        f.write(json.dumps({"node_id": "src-am:byn5", "text": "Beta fact here."}) + "\n")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")
    # "n5" is a bare suffix of BOTH chunk ids -> ambiguous -> must not resolve -> claim dropped
    payload = "<output>" + json.dumps({
        "claims": [{"claim_id": "c1", "claim": "x",
                    "supporting_evidence": [{"locator": "n5", "start_char": 0, "end_char": 15, "url": None}]}],
        "entities": [], "relations": []}) + "</output>"
    result = extract_claims_to_run(run, "web", cfg, _FakeBackend(payload), batch_size=6)
    assert result["written"] == 0 and result["dropped"] == ["c1"]


def test_extract_tolerates_bare_string_claims(tmp_path):
    # Some instruct models emit claims as bare strings ({"claims": ["text"]})
    # instead of objects. Those have no evidence and must be skipped, not crash.
    run = tmp_path / "research-runs" / "src-s"
    run.mkdir(parents=True)
    (run / "source.md").write_text("Snails are molluscs.", encoding="utf-8")
    (run / "chunks.jsonl").write_text(json.dumps({"node_id": "src-s:c01", "text": "Snails are molluscs."}) + "\n",
                                      encoding="utf-8")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")
    payload = json.dumps({
        "claims": ["Snails are molluscs.",  # bare string -> skipped, no crash
                   {"claim_id": "c1", "claim": "Snails are molluscs.",
                    "supporting_evidence": [  # chunk[0:19] == "Snails are molluscs"
                        {"locator": "src-s:c01", "start_char": 0, "end_char": 19, "url": None}]}],
        "entities": ["not a dict", {"entity_id": "snail", "name": "Snail", "mentions": ["src-s:c01"]}],
        "relations": ["also not a dict"],
    })
    result = extract_claims_to_run(run, "web", cfg, _FakeBackend(payload))
    assert result["written"] == 1 and result["entities"] == 1  # only the well-formed items survive


def test_extract_tolerates_bare_string_evidence_rows(tmp_path):
    # Measured live (gemma4:26b, 2026-07-07): a claim object whose
    # supporting_evidence list holds a bare STRING instead of an evidence
    # object. A malformed evidence row can't pass the verbatim gate, so the
    # claim is dropped -- never a crash.
    run = tmp_path / "research-runs" / "src-e"
    run.mkdir(parents=True)
    (run / "source.md").write_text("Snails are molluscs.", encoding="utf-8")
    (run / "chunks.jsonl").write_text(json.dumps({"node_id": "src-e:c01", "text": "Snails are molluscs."}) + "\n",
                                      encoding="utf-8")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")
    payload = json.dumps({
        "claims": [
            {"claim_id": "c1", "claim": "bad evidence row",
             "supporting_evidence": ["Snails are molluscs"]},  # string, not object
            {"claim_id": "c2", "claim": "good",  # chunk[0:19] == "Snails are molluscs"
             "supporting_evidence": [{"locator": "src-e:c01", "start_char": 0, "end_char": 19, "url": None}]},
        ],
        "entities": [], "relations": [],
    })
    result = extract_claims_to_run(run, "web", cfg, _FakeBackend(payload))
    written = [json.loads(line) for line in (run / "claims.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert [c["claim_id"] for c in written] == ["c2"]
    assert result["dropped"] == ["c1"] and result["written"] == 1


def test_extract_batches_sources_and_merges_entities(tmp_path):
    # A source larger than batch_size is extracted in bounded batches; claim ids
    # are prefixed for uniqueness and the same entity seen in two batches merges.
    run = tmp_path / "research-runs" / "src-b"
    run.mkdir(parents=True)
    (run / "source.md").write_text("Alpha fact here. Beta fact here.", encoding="utf-8")
    with open(run / "chunks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"node_id": "src-b:c01", "text": "Alpha fact here."}) + "\n")
        f.write(json.dumps({"node_id": "src-b:c02", "text": "Beta fact here."}) + "\n")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")

    class _PerBatch:
        def complete(self, system, user, **kw):
            cid, quote = ("src-b:c01", "Alpha fact here") if "src-b:c01" in user else ("src-b:c02", "Beta fact here")
            return "<output>" + json.dumps({
                "claims": [{"claim_id": "c1", "claim": quote,
                            "supporting_evidence": [{"locator": cid, "start_char": 0,
                                                     "end_char": len(quote), "url": None}]}],
                "entities": [{"entity_id": "e", "name": "E", "aliases": [], "type": "x", "mentions": [cid]}],
                "relations": [],
            }) + "</output>"

    result = extract_claims_to_run(run, "web", cfg, _PerBatch(), batch_size=1)
    assert result["batches"] == 2 and result["written"] == 2 and result["parse_failures"] == 0
    ids = [json.loads(x)["claim_id"] for x in (run / "claims.jsonl").read_text(encoding="utf-8").splitlines() if x]
    assert ids == ["b00_c1", "b01_c1"]  # prefixed across batches
    ents = [json.loads(x) for x in (run / "entities.jsonl").read_text(encoding="utf-8").splitlines() if x]
    assert len(ents) == 1 and set(ents[0]["mentions"]) == {"src-b:c01", "src-b:c02"}  # merged


def test_extract_reports_parse_failures_on_truncation(tmp_path):
    # A reasoning model that runs out of tokens mid-think returns no parseable
    # JSON; that must surface as parse_failures, not silently as "0 claims found".
    run = tmp_path / "research-runs" / "src-f"
    run.mkdir(parents=True)
    (run / "chunks.jsonl").write_text(json.dumps({"node_id": "src-f:c01", "text": "x"}) + "\n", encoding="utf-8")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")
    truncated = _FakeBackend("<think>reasoning ran on and never reached the output block")
    result = extract_claims_to_run(run, "web", cfg, truncated)
    assert result["written"] == 0 and result["parse_failures"] == 1


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
                    "supporting_evidence": [  # chunk[0:23] == "Hydra settles instantly"
                        {"locator": "src-9:c01", "start_char": 0, "end_char": 23, "url": None}]}],
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
    assert claims[0]["document_id"] == "src-9" and claims[0]["schema_version"] == "2.0"
    assert ents[0]["entity_id"] == "hydra" and ents[0]["schema_version"] == "2.0"
    assert rels[0]["predicate"] == "uses" and rels[0]["document_id"] == "src-9"
