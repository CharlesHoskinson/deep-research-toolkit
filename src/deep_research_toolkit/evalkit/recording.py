"""A raw-completion-recording Backend wrapper for the eval runner.

wiki_write/synthesize normalize a reply (unfence, bare-marker rewrite) before
the citation gate ever sees it (see `llm.response.generate_cited`), so the
gate-facing text can never reveal the marker-fidelity signal the design doc's
canary #4 and the eval report both track (design doc §3.2/§3.4: "bare-marker
rate before normalization"). `RecordingBackend` sits between the runner and
the real backend to capture the reply exactly as the model emitted it, before
any of that normalization runs.
"""
from __future__ import annotations


class RecordingBackend:
    """Wraps a Backend, recording every raw `complete()` reply into `self.raw`
    (in call order) before returning it unchanged. A pure pass-through
    otherwise -- sampling kwargs are forwarded as-is."""

    def __init__(self, backend) -> None:
        self.backend = backend
        self.raw: list[str] = []

    def complete(self, system: str, user: str, **sampling) -> str:
        reply = self.backend.complete(system, user, **sampling)
        self.raw.append(reply)
        return reply
