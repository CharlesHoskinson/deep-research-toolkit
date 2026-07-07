"""Eval-suite helpers: flake-rate statistics for the live-model tier (Task 5),
plus the extraction/prose/adjudication metrics, paired-bootstrap comparison,
and the raw-completion RecordingBackend wrapper consumed by
scripts/eval-pipeline.py (Task 7). Pure Python (RecordingBackend excepted,
which wraps whatever backend the runner gives it) -- no model calls of its
own; the live tier and the eval runner call into this package."""
from __future__ import annotations
