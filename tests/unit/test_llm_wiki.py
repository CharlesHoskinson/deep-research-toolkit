import pytest

from deep_research_toolkit.llm.wiki import CitationError, write_wiki_body

CLAIMS = [
    {"claim_id": "c1", "claim": "Praos was introduced in 2018.",
     "supporting_evidence": [{"locator": "n1", "quote": "introduced in 2018"}]},
    {"claim_id": "c2", "claim": "Praos tolerates message delays.",
     "supporting_evidence": [{"locator": "n2", "quote": "tolerates delays"}]},
]


class StubBackend:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user, **kw):
        self.calls.append((system, user))
        return self.replies.pop(0)


def test_valid_body_passes_and_reports_coverage():
    body = "## Overview\n\nPraos arrived in 2018 [claim:c1] and tolerates delays [claim:c2].\n"
    out = write_wiki_body("Ouroboros Praos", "Concept", CLAIMS, StubBackend([body]))
    assert out["body"] == body
    assert out["citations"]["coverage"] == 1.0


def test_unknown_marker_retries_once_then_raises():
    bad = "Praos is fast [claim:nope]."
    backend = StubBackend([bad, bad])
    with pytest.raises(CitationError):
        write_wiki_body("Praos", "Concept", CLAIMS, backend)
    assert len(backend.calls) == 2
    assert "nope" in backend.calls[1][1]  # correction prompt names the bad id


def test_empty_claims_is_an_error():
    with pytest.raises(ValueError):
        write_wiki_body("Praos", "Concept", [], StubBackend(["x"]))
