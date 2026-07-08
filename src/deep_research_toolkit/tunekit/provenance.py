"""Per-run experiment tracking (design doc §6.4): a `run.json` written INTO
the run directory before any weights, plus one flat summary row appended to
a repo-level `runs.jsonl` -- the same "detailed report + flat append-only
ledger" split `scripts/eval-pipeline.py` uses for `eval-results/run-*.json` +
`eval-results/history.jsonl`.

`config_sha256` hashes the RESOLVED config (defaults/overrides already
merged) serialized with canonical (fully sorted, recursive) key order via
`json.dumps(..., sort_keys=True)`, so the same effective config always hashes
identically regardless of the order its keys were constructed in. Runs are
keyed by `(config_sha256, dataset_hash)` -- that pair is the idempotence key:
`run_exists` lets an orchestrator skip re-running a config against a dataset
it has already trained, and any eval number is reproducible from the triple
`{config_sha256, dataset_hash, git_commit}` (design doc §6.4)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

#: Default training seed (design doc §6.4: "seed=42").
DEFAULT_SEED = 42


def config_sha256(config: dict) -> str:
    """sha256 over the resolved config, serialized with fully sorted
    (canonical) key order at every nesting level -- so two dicts that are
    equal but were built with keys in a different order hash identically.
    Returns a ``"sha256:<hex>"`` string, matching the `corpus_version`/
    `prompt_version` join-key convention in `scripts/eval-pipeline.py`."""
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=True)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_run_record(config: dict, dataset_hash: str, versions: dict, *,
                     seed: int = DEFAULT_SEED, git_commit: str | None = None,
                     base_model_digest: str | None = None,
                     recipe_id: str | None = None,
                     timestamp: str | None = None) -> dict:
    """Assembles the run record dict (design doc §6.4 fields) without
    touching the filesystem -- the pure half of `write_run_record`, kept
    separate so `run_exists`-style keyed lookups and tests can construct a
    record without a run directory."""
    return {
        "git_commit": git_commit,
        "config_sha256": config_sha256(config),
        "dataset_hash": dataset_hash,
        "seed": seed,
        "base_model_digest": base_model_digest,
        "recipe_id": recipe_id,
        "versions": versions,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    }


def write_run_record(run_dir, config: dict, dataset_hash: str, versions: dict, *,
                     seed: int = DEFAULT_SEED, git_commit: str | None = None,
                     base_model_digest: str | None = None,
                     recipe_id: str | None = None,
                     runs_jsonl_path=None, timestamp: str | None = None) -> dict:
    """Writes `<run_dir>/run.json` (pretty-printed, the per-run detail) and
    appends one matching row to `runs_jsonl_path` (default:
    `<run_dir>.parent / "runs.jsonl"` -- a shared ledger across every run
    directory, mirroring `eval-results/history.jsonl`). Called BEFORE weights
    are written (design doc §6.4/§9: "incremental save" -- a run's provenance
    survives even if training itself is interrupted).

    Returns the record dict that was written, with `"run_dir"` added (the
    only field not in the pure `run.json` shape, since a directory knows its
    own path without needing to record it inside itself -- kept here purely
    for the caller's convenience)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    record = build_run_record(
        config, dataset_hash, versions, seed=seed, git_commit=git_commit,
        base_model_digest=base_model_digest, recipe_id=recipe_id, timestamp=timestamp)

    run_json_path = run_dir / "run.json"
    run_json_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    runs_path = Path(runs_jsonl_path) if runs_jsonl_path is not None else run_dir.parent / "runs.jsonl"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(runs_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({**record, "run_dir": str(run_dir)}, ensure_ascii=False) + "\n")

    return {**record, "run_dir": str(run_dir)}


def run_exists(runs_path, config_sha: str, dataset_hash: str) -> bool:
    """Keyed idempotence check (design doc §6.4: "keyed by (config_sha256,
    dataset_hash) -> idempotent skip-if-exists"): True iff `runs_path`
    already has a row matching BOTH keys. A missing `runs_path` (no runs
    recorded yet) is not an error -- it just means no run exists."""
    for row in _read_jsonl(Path(runs_path)):
        if row.get("config_sha256") == config_sha and row.get("dataset_hash") == dataset_hash:
            return True
    return False
