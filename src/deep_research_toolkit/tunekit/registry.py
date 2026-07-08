"""Eval-gated promotion registry (design doc §6.3): one git-tracked,
append-only `registry.jsonl` -- the source of truth, one row per PROMOTED
model version -- plus a DuckDB view over it joined with `runs.jsonl` (§6.4)
for SQL queries like "which tag beats baseline bait at corpus_hash X?". Tags
(e.g. `gemma4-extract:v3`) are mutable pointers a caller keeps elsewhere;
this module only ever appends immutable rows.

Every row is schema-validated before it is written: two IMMUTABLE ANCHORS
(`ollama_manifest_digest`, a 40-char `hf_commit_sha`), the PROVENANCE TRIPLE
(`config_sha256`, `dataset_hash`, `git_commit`), an EVAL block (recall,
bait_rejection, gate_pass, atomicity), and `corpus_hash`/`prompt_hash` are
MANDATORY -- "never store an eval number without the corpus_hash/prompt_hash
it was measured on" (design doc §6.3). A row missing any of these is
refused, never silently written with a null.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

#: 40 lowercase-hex characters -- a full (non-abbreviated) git/HF commit SHA.
#: An abbreviated SHA is not accepted: it can become ambiguous as a repo
#: grows, which defeats the point of an "immutable anchor".
_HF_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

#: The two immutable anchors (design doc §6.3): the served Ollama artifact's
#: manifest digest, and the exact Hugging Face commit the weights came from.
_ANCHOR_FIELDS = ("ollama_manifest_digest", "hf_commit_sha")

#: The provenance triple (design doc §6.3/§6.4): binds a registry row back to
#: the exact config, dataset, and code that produced it.
_PROVENANCE_FIELDS = ("config_sha256", "dataset_hash", "git_commit")

#: Mandatory per design doc §6.3: "never store an eval number without the
#: corpus_hash/prompt_hash it was measured on".
_HASH_FIELDS = ("corpus_hash", "prompt_hash")

#: The eval block's required sub-keys (design doc §6.3's `eval{...}`).
_EVAL_FIELDS = ("recall", "bait_rejection", "gate_pass", "atomicity")

#: Every row-level field a valid registry row must carry. `timestamp` is
#: REQUIRED (ISO-8601): a promoted version without a promotion time cannot be
#: audited, and the DuckDB view exposes it as `promoted_at`.
REQUIRED_FIELDS = (*_ANCHOR_FIELDS, *_PROVENANCE_FIELDS, *_HASH_FIELDS,
                   "eval", "status", "timestamp")


class RegistrySchemaError(ValueError):
    """Raised when a registry row is missing a mandatory field, or a field
    fails its format check (e.g. hf_commit_sha not 40 hex chars). Carries
    every violation found, not just the first, so a caller building a row
    programmatically sees the whole picture in one failure."""

    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__("invalid registry row: " + "; ".join(violations))


class DuplicateRegistryRowError(ValueError):
    """Raised by append_registry_row when a row's (config_sha256,
    dataset_hash, corpus_hash) triple already exists in the registry -- the
    same artifact measured on the same corpus must not be promoted twice
    (a duplicate would also fan out the DuckDB view's LEFT JOIN)."""


class RegistryCorruptionError(ValueError):
    """Raised by read_registry when a registry line is not valid JSON --
    fail CLOSED (an unreadable source of truth must not silently read as a
    shorter registry) but diagnosable: the message names the line number."""


class DuckDBNotInstalled(RuntimeError):
    def __init__(self):
        super().__init__(
            "duckdb is not installed. This skill needs the 'compiler' extra: "
            'pip install "deep-research-toolkit[compiler]"'
        )


def _iso_timestamp_ok(value) -> bool:
    """True iff `value` parses as an ISO-8601 timestamp. A trailing 'Z' is
    normalized to '+00:00' first (datetime.fromisoformat on Python 3.10
    rejects the bare 'Z' suffix)."""
    try:
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def validate_registry_row(row: dict) -> None:
    """Raises RegistrySchemaError listing every violation; returns None on a
    valid row. Checked, in order: every REQUIRED_FIELDS key is present and
    non-empty/non-None; `hf_commit_sha` is exactly 40 lowercase-hex chars;
    `timestamp` is an ISO-8601 string; `eval` is a dict carrying every
    _EVAL_FIELDS key (a missing eval metric is allowed to be `None` -- e.g.
    atomicity when reference is empty -- but the KEY itself must be present,
    so a caller can't silently omit a metric by leaving the key out)."""
    violations: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in row or row[field] in (None, ""):
            violations.append(f"missing required field {field!r}")

    hf_sha = row.get("hf_commit_sha")
    if hf_sha and not _HF_COMMIT_SHA_RE.match(str(hf_sha)):
        violations.append(
            f"hf_commit_sha {hf_sha!r} is not a 40-character lowercase-hex commit SHA "
            "(abbreviated SHAs are not immutable anchors)")

    ts = row.get("timestamp")
    if ts and not _iso_timestamp_ok(ts):
        violations.append(f"timestamp {ts!r} is not an ISO-8601 timestamp string")

    eval_block = row.get("eval")
    if eval_block is not None:
        if not isinstance(eval_block, dict):
            violations.append("'eval' must be a dict")
        else:
            for key in _EVAL_FIELDS:
                if key not in eval_block:
                    violations.append(f"'eval' is missing key {key!r}")

    if violations:
        raise RegistrySchemaError(violations)


def build_registry_row(*, ollama_manifest_digest: str, hf_commit_sha: str,
                       config_sha256: str, dataset_hash: str, git_commit: str,
                       eval: dict, corpus_hash: str, prompt_hash: str,
                       status: str = "promoted", timestamp: str | None = None) -> dict:
    """Assembles a registry row from the design doc §6.3 fields and validates
    it before returning. Pure -- does not touch the filesystem."""
    row = {
        "ollama_manifest_digest": ollama_manifest_digest,
        "hf_commit_sha": hf_commit_sha,
        "config_sha256": config_sha256,
        "dataset_hash": dataset_hash,
        "git_commit": git_commit,
        "eval": eval,
        "corpus_hash": corpus_hash,
        "prompt_hash": prompt_hash,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "status": status,
    }
    validate_registry_row(row)
    return row


def append_registry_row(registry_path, row: dict) -> dict:
    """Validates `row` (raising RegistrySchemaError on any violation --
    NOTHING is written on a bad row), refuses a duplicate of an existing
    row's (config_sha256, dataset_hash, corpus_hash) triple (raising
    DuplicateRegistryRowError naming the existing row's timestamp), then
    appends it as one JSON line to `registry_path`, creating the file (and
    its parent dir) if needed. Reading the existing registry also means a
    corrupted file blocks all further appends (RegistryCorruptionError) --
    fail closed, fix the file, then promote. Returns `row` unchanged, for
    chaining with `build_registry_row`."""
    validate_registry_row(row)
    registry_path = Path(registry_path)
    key = (row["config_sha256"], row["dataset_hash"], row["corpus_hash"])
    for existing in read_registry(registry_path):
        if (existing.get("config_sha256"), existing.get("dataset_hash"),
                existing.get("corpus_hash")) == key:
            raise DuplicateRegistryRowError(
                f"registry already has a row for config_sha256={key[0]!r}, "
                f"dataset_hash={key[1]!r}, corpus_hash={key[2]!r} "
                f"(promoted at {existing.get('timestamp')!r}) -- the registry is "
                "append-only and rows are unique per (config, dataset, corpus) triple")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def read_registry(registry_path) -> list[dict]:
    """Reads every row of `registry_path` back, in append order. A missing
    file reads as an empty registry, not an error (nothing promoted yet);
    an unparseable LINE raises RegistryCorruptionError naming its line
    number -- the source of truth must never silently read shorter than it
    is on disk."""
    path = Path(registry_path)
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RegistryCorruptionError(
                    f"registry {path} line {lineno} is not valid JSON: {e}") from e
    return rows


def registry_view_sql(runs_path, registry_path, view_name: str = "tunekit_runs") -> str:
    """Pure: returns the `CREATE OR REPLACE VIEW` SQL text joining
    `runs.jsonl` (every attempted run) with `registry.jsonl` (only PROMOTED
    runs) on `(config_sha256, dataset_hash)` -- the same keyed-idempotence
    pair `run_exists` uses. Does NOT import duckdb and does not require it to
    be installed: this function only builds SQL text, so it is unit-testable
    without the optional 'compiler' extra. Use `open_registry_duckdb` to
    actually execute it against a live connection.

    Paths are interpolated into single-quoted SQL string literals, so any
    single quote in a path is doubled (standard SQL escaping) -- a path like
    ``it's/runs.jsonl`` must not break (or worse, reshape) the statement."""
    runs_path = str(Path(runs_path)).replace("\\", "/").replace("'", "''")
    registry_path = str(Path(registry_path)).replace("\\", "/").replace("'", "''")
    return (
        f"CREATE OR REPLACE VIEW {view_name} AS\n"
        f"SELECT r.*, g.ollama_manifest_digest, g.hf_commit_sha, g.corpus_hash,\n"
        f"       g.prompt_hash, g.eval, g.status, g.timestamp AS promoted_at\n"
        f"FROM read_ndjson_auto('{runs_path}') AS r\n"
        f"LEFT JOIN read_ndjson_auto('{registry_path}') AS g\n"
        f"  ON r.config_sha256 = g.config_sha256 AND r.dataset_hash = g.dataset_hash;"
    )


def open_registry_duckdb(db_path, runs_path, registry_path, view_name: str = "tunekit_runs"):
    """Guarded duckdb import (optional 'compiler' extra, per
    tests/unit/test_dependency_boundary.py's pattern): opens/creates a DuckDB
    database at `db_path`, executes `registry_view_sql`, and returns the open
    connection. Raises DuckDBNotInstalled with an actionable install command
    if duckdb isn't available -- never a raw ImportError/traceback."""
    try:
        import duckdb
    except ImportError as e:
        raise DuckDBNotInstalled() from e
    con = duckdb.connect(str(db_path))
    con.execute(registry_view_sql(runs_path, registry_path, view_name=view_name))
    return con
