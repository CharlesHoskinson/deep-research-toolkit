"""manifest.json read/write: the append-per-stage contract every pipeline
stage honors, plus schema_version stamping.

The critical invariant (a real bug found and fixed during the original
pipeline's development): re-running an earlier stage against an existing
run directory must NOT wipe out later stages' recorded state. Every writer
here loads first, merges its own stage key in, and preserves everything
else -- this module exists so that invariant lives in exactly one place
instead of being reimplemented (and potentially re-broken) per skill.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "1.0"


def now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_manifest(run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def start_manifest(
    run_dir: Path,
    document_id: str,
    source_file: str,
    source_hash: str,
) -> dict[str, Any]:
    """Create manifest.json for a brand-new run directory. If one already
    exists, this preserves it (delegates to update_stage's merge behavior)
    rather than overwriting -- safe to call idempotently.
    """
    existing = load_manifest(run_dir)
    if existing is not None:
        existing["document_id"] = document_id
        existing["source_file"] = source_file
        existing["source_hash"] = source_hash
        existing.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
        existing.setdefault("stages", {})
        return existing

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "document_id": document_id,
        "source_file": source_file,
        "source_hash": source_hash,
        "created_at": now_iso(),
        "stages": {},
    }


def update_stage(run_dir: Path, stage_name: str, **fields: Any) -> None:
    """Merge fields into exactly one stage's entry in manifest.json,
    preserving every other stage's already-recorded state *and* any fields
    already recorded within this same stage.

    This must merge rather than replace `stages[stage_name]` wholesale: some
    stages are produced by more than one caller writing into the same stage
    key at different times (e.g. knowledge-extraction gets `table_count` from
    extract_tables, `figure_count` from extract_figures, and
    claim_count/entity_count/relation_count from Claude's own follow-up
    update -- see docs/contracts/pdf-ingestion-pipeline.md). Replacing the
    whole dict on each call would let a later call silently wipe out an
    earlier call's fields; merging is what keeps them all present together.
    Creates a minimal manifest (keyed by the run directory's own name) if
    none exists yet, so a stage can run standalone without requiring the
    router to have run first.
    """
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    if manifest is None:
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "document_id": run_dir.name,
            "stages": {},
        }
    manifest.setdefault("stages", {})
    existing_stage = manifest["stages"].get(stage_name, {})
    manifest["stages"][stage_name] = {**existing_stage, "completed_at": now_iso(), **fields}

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    manifest_path = Path(run_dir) / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
