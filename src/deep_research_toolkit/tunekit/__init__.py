"""Fine-tune meta-infrastructure (Phase 2, design doc §6): dataset generation,
run provenance, the eval-gated promotion registry, and the GGUF control-token
validator. Deliberately a minimal file-based loop (append-only JSONL + a
DuckDB view + pure gate functions) rather than an MLOps platform -- see
docs/superpowers/specs/2026-07-07-pipeline-hardening-and-finetune-meta-infra-design.md
§6-§8. Live model orchestration (teachers, training, serving) is out of scope
here; this package provides the scaffolding those steps plug into."""
from __future__ import annotations
