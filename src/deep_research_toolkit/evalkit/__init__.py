"""Eval-suite helpers: flake-rate statistics for the live-model tier (Task 5),
plus the extraction/prose/adjudication metrics and paired-bootstrap comparison
consumed by scripts/eval-pipeline.py (Tasks 6-7). Pure Python, no model calls
of its own -- the live tier and the eval runner call into this package."""
from __future__ import annotations
