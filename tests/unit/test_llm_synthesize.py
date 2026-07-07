import pytest

from deep_research_toolkit.llm.synthesize import CitationError, synthesize_thesis

DOSSIER = {
    "included": [
        {"claim_id": "c1", "claim": "Praos was introduced in 2018.",
         "evidence": [{"quote": "introduced in 2018", "source_id": "s1"}]},
        {"claim_id": "c2", "claim": "Praos tolerates delays.",
         "evidence": [{"quote": "tolerates delays", "source_id": "s1"}]},
    ],
    "rejected": [],
}


class StubBackend:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user, **kw):
        self.calls.append((system, user, kw))
        return self.replies.pop(0)


def test_valid_thesis_passes():
    reply = "Praos, introduced in 2018 [claim:c1], tolerates delays [claim:c2]."
    out = synthesize_thesis("How robust is Praos?", DOSSIER, StubBackend([reply]))
    assert out["thesis"] == reply
    assert out["citations"]["coverage"] == 1.0


def test_unknown_id_retries_then_raises():
    bad = "Praos is quantum-safe [claim:c9]."
    backend = StubBackend([bad, bad])
    with pytest.raises(CitationError):
        synthesize_thesis("q", DOSSIER, backend)
    assert len(backend.calls) == 2
    assert "c9" in backend.calls[1][1]


def test_empty_dossier_is_an_error():
    with pytest.raises(ValueError):
        synthesize_thesis("q", {"included": [], "rejected": []}, StubBackend(["x"]))


def test_zero_citation_thesis_is_rejected():
    with pytest.raises(ValueError, match="coverage"):
        synthesize_thesis("q", DOSSIER, StubBackend(["No markers here.", "Still no markers here."]))


def test_low_coverage_thesis_retries_once_then_succeeds():
    low = "No markers here."
    good = "Praos, introduced in 2018 [claim:c1], tolerates delays [claim:c2]."
    out = synthesize_thesis("q", DOSSIER, StubBackend([low, good]))
    assert out["thesis"] == good
    assert out["citations"]["coverage"] == 1.0


def test_marker_and_coverage_retries_are_bounded_and_use_temperature():
    bad_marker = "Praos is quantum-safe [claim:c9]."
    clean_but_low = "Praos exists."  # markers fixed, but zero coverage
    good = "Praos, introduced in 2018 [claim:c1], tolerates delays [claim:c2]."
    backend = StubBackend([bad_marker, clean_but_low, good])
    out = synthesize_thesis("q", DOSSIER, backend)
    assert out["citations"]["coverage"] == 1.0
    assert len(backend.calls) == 3
    assert backend.calls[0][2].get("temperature") is None
    assert backend.calls[1][2].get("temperature") == 0.25
    assert backend.calls[2][2].get("temperature") == 0.25


def test_fenced_reply_is_unwrapped():
    fenced = "```markdown\nPraos arrived in 2018 [claim:c1] and tolerates delays [claim:c2].\n```"
    out = synthesize_thesis("q", DOSSIER, StubBackend([fenced]))
    assert not out["thesis"].startswith("```")


def test_bare_marker_reply_is_normalized_and_passes():
    reply = "Praos arrived in 2018 [c1] and tolerates delays [c2]."
    out = synthesize_thesis("q", DOSSIER, StubBackend([reply]))
    assert out["citations"]["coverage"] == 1.0
    assert "[claim:c1]" in out["thesis"]


def test_repetition_loop_reply_raises_after_retry_also_loops():
    looping = "the same phrase " * 40  # 120 words of one repeated phrase
    backend = StubBackend([looping, looping])
    with pytest.raises(ValueError, match="repetition"):
        synthesize_thesis("q", DOSSIER, backend)
    assert len(backend.calls) == 2


def test_repetition_loop_then_good_reply_succeeds():
    looping = "the same phrase " * 40
    good = "Praos, introduced in 2018 [claim:c1], tolerates delays [claim:c2]."
    backend = StubBackend([looping, good])
    out = synthesize_thesis("q", DOSSIER, backend)
    assert out["thesis"] == good
    assert out["citations"]["coverage"] == 1.0
    assert len(backend.calls) == 2


def test_repetition_loop_on_citation_retry_reply_raises():
    # Call 1: unknown marker (no loop) -> marker retry. Call 2: the retry
    # loops -> repetition correction. Call 3: still loops -> ValueError.
    bad_marker = "Praos is quantum-safe [claim:c9]."
    looping = "the same phrase " * 40
    backend = StubBackend([bad_marker, looping, looping])
    with pytest.raises(ValueError, match="repetition"):
        synthesize_thesis("q", DOSSIER, backend)
    assert len(backend.calls) == 3
