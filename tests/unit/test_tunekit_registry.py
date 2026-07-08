"""tunekit.registry: schema-validated append-only registry.jsonl (design doc
§6.3). Refuses any row missing an immutable anchor, the provenance triple,
the eval block, or corpus_hash/prompt_hash; the DuckDB view SQL is pure text
(no duckdb import) so it's testable without the optional 'compiler' extra."""
from __future__ import annotations

import json
import sys

import pytest

from deep_research_toolkit.tunekit.registry import (
    REQUIRED_FIELDS,
    DuckDBNotInstalled,
    DuplicateRegistryRowError,
    RegistryCorruptionError,
    RegistrySchemaError,
    append_registry_row,
    build_registry_row,
    open_registry_duckdb,
    read_registry,
    registry_view_sql,
    validate_registry_row,
)

VALID_HF_SHA = "a" * 40
VALID_EVAL = {"recall": 0.93, "bait_rejection": 0.97, "gate_pass": 0.99, "atomicity": 1.2}


def _valid_row(**overrides) -> dict:
    row = {
        "ollama_manifest_digest": "sha256:manifestdigest",
        "hf_commit_sha": VALID_HF_SHA,
        "config_sha256": "sha256:configabc",
        "dataset_hash": "sha256:datasetabc",
        "git_commit": "deadbeefcafe",
        "eval": dict(VALID_EVAL),
        "corpus_hash": "sha256:corpusabc",
        "prompt_hash": "sha256:promptabc",
        "status": "promoted",
        "timestamp": "2026-07-07T00:00:00+00:00",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# validate_registry_row
# ---------------------------------------------------------------------------

def test_valid_row_passes():
    validate_registry_row(_valid_row())  # no raise


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_missing_required_field_raises(field):
    row = _valid_row()
    del row[field]
    with pytest.raises(RegistrySchemaError) as exc_info:
        validate_registry_row(row)
    assert field in str(exc_info.value)


def test_missing_corpus_hash_raises():
    row = _valid_row()
    del row["corpus_hash"]
    with pytest.raises(RegistrySchemaError) as exc_info:
        validate_registry_row(row)
    assert "corpus_hash" in str(exc_info.value)


def test_missing_prompt_hash_raises():
    row = _valid_row()
    del row["prompt_hash"]
    with pytest.raises(RegistrySchemaError) as exc_info:
        validate_registry_row(row)
    assert "prompt_hash" in str(exc_info.value)


def test_short_hf_commit_sha_raises():
    row = _valid_row(hf_commit_sha="abc123")
    with pytest.raises(RegistrySchemaError) as exc_info:
        validate_registry_row(row)
    assert "hf_commit_sha" in str(exc_info.value)


def test_uppercase_hf_commit_sha_raises():
    row = _valid_row(hf_commit_sha="A" * 40)
    with pytest.raises(RegistrySchemaError):
        validate_registry_row(row)


def test_eval_block_missing_key_raises():
    row = _valid_row(eval={"recall": 0.9, "bait_rejection": 0.9, "gate_pass": 0.9})  # no atomicity
    with pytest.raises(RegistrySchemaError) as exc_info:
        validate_registry_row(row)
    assert "atomicity" in str(exc_info.value)


def test_eval_block_wrong_type_raises():
    row = _valid_row(eval="not-a-dict")
    with pytest.raises(RegistrySchemaError):
        validate_registry_row(row)


def test_multiple_violations_all_reported():
    row = _valid_row(hf_commit_sha="short")
    del row["corpus_hash"]
    with pytest.raises(RegistrySchemaError) as exc_info:
        validate_registry_row(row)
    assert len(exc_info.value.violations) >= 2


def test_timestamp_is_required():
    """I4: a promoted version without a promotion time cannot be audited."""
    row = _valid_row()
    del row["timestamp"]
    with pytest.raises(RegistrySchemaError) as exc_info:
        validate_registry_row(row)
    assert "timestamp" in str(exc_info.value)


def test_non_iso_timestamp_raises():
    row = _valid_row(timestamp="last tuesday, ish")
    with pytest.raises(RegistrySchemaError) as exc_info:
        validate_registry_row(row)
    assert "ISO-8601" in str(exc_info.value)


def test_z_suffixed_iso_timestamp_accepted():
    validate_registry_row(_valid_row(timestamp="2026-07-07T00:00:00Z"))  # no raise


# ---------------------------------------------------------------------------
# build_registry_row
# ---------------------------------------------------------------------------

def test_build_registry_row_happy_path():
    row = build_registry_row(
        ollama_manifest_digest="sha256:manifestdigest", hf_commit_sha=VALID_HF_SHA,
        config_sha256="sha256:configabc", dataset_hash="sha256:datasetabc",
        git_commit="deadbeefcafe", eval=dict(VALID_EVAL),
        corpus_hash="sha256:corpusabc", prompt_hash="sha256:promptabc")
    assert row["status"] == "promoted"
    assert "timestamp" in row


def test_build_registry_row_rejects_invalid():
    with pytest.raises(RegistrySchemaError):
        build_registry_row(
            ollama_manifest_digest="sha256:x", hf_commit_sha="tooshort",
            config_sha256="sha256:c", dataset_hash="sha256:d", git_commit="abc",
            eval=dict(VALID_EVAL), corpus_hash="sha256:corpus", prompt_hash="sha256:prompt")


# ---------------------------------------------------------------------------
# append_registry_row / read_registry
# ---------------------------------------------------------------------------

def test_append_and_read_round_trip(tmp_path):
    registry_path = tmp_path / "registry.jsonl"
    row1 = _valid_row(dataset_hash="sha256:d1")
    row2 = _valid_row(dataset_hash="sha256:d2")
    append_registry_row(registry_path, row1)
    append_registry_row(registry_path, row2)

    rows = read_registry(registry_path)
    assert len(rows) == 2
    assert rows[0]["dataset_hash"] == "sha256:d1"
    assert rows[1]["dataset_hash"] == "sha256:d2"

    # append-only: file has exactly 2 lines, each valid JSON.
    lines = registry_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # does not raise


def test_append_registry_row_refuses_incomplete_row(tmp_path):
    registry_path = tmp_path / "registry.jsonl"
    bad_row = _valid_row()
    del bad_row["corpus_hash"]
    with pytest.raises(RegistrySchemaError):
        append_registry_row(registry_path, bad_row)
    assert not registry_path.exists()  # nothing written on a refused row


def test_read_registry_missing_file_is_empty(tmp_path):
    assert read_registry(tmp_path / "nonexistent.jsonl") == []


def test_append_refuses_duplicate_triple_naming_existing_timestamp(tmp_path):
    """I3: the same (config_sha256, dataset_hash, corpus_hash) triple must
    not be promoted twice -- and the error names when it already was."""
    registry_path = tmp_path / "registry.jsonl"
    append_registry_row(registry_path, _valid_row(timestamp="2026-07-01T00:00:00+00:00"))
    dup = _valid_row(timestamp="2026-07-07T12:00:00+00:00")  # same triple, later time
    with pytest.raises(DuplicateRegistryRowError) as exc_info:
        append_registry_row(registry_path, dup)
    assert "2026-07-01T00:00:00+00:00" in str(exc_info.value)
    assert len(read_registry(registry_path)) == 1  # nothing was written


def test_append_allows_same_config_dataset_on_different_corpus(tmp_path):
    """The uniqueness key is the full triple: the same artifact re-measured
    on a NEW corpus is a legitimate new row, not a duplicate."""
    registry_path = tmp_path / "registry.jsonl"
    append_registry_row(registry_path, _valid_row(corpus_hash="sha256:corpus-v1"))
    append_registry_row(registry_path, _valid_row(corpus_hash="sha256:corpus-v2"))
    assert len(read_registry(registry_path)) == 2


def test_read_registry_corrupted_line_raises_with_line_number(tmp_path):
    """M4: an unreadable source of truth must fail CLOSED and diagnosable,
    never silently read as a shorter registry."""
    registry_path = tmp_path / "registry.jsonl"
    append_registry_row(registry_path, _valid_row())
    with open(registry_path, "a", encoding="utf-8") as f:
        f.write("{this is not json\n")
    with pytest.raises(RegistryCorruptionError) as exc_info:
        read_registry(registry_path)
    assert "line 2" in str(exc_info.value)


def test_corrupted_registry_blocks_further_appends(tmp_path):
    registry_path = tmp_path / "registry.jsonl"
    registry_path.write_text("not json at all\n", encoding="utf-8")
    with pytest.raises(RegistryCorruptionError):
        append_registry_row(registry_path, _valid_row())


# ---------------------------------------------------------------------------
# registry_view_sql (pure text, no duckdb import required)
# ---------------------------------------------------------------------------

def test_registry_view_sql_is_pure_text_and_needs_no_duckdb_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "duckdb", None)  # force ImportError if ever imported
    sql = registry_view_sql("runs/runs.jsonl", "registry.jsonl")
    assert "CREATE OR REPLACE VIEW tunekit_runs" in sql
    assert "read_ndjson_auto" in sql
    assert "config_sha256" in sql and "dataset_hash" in sql


def test_registry_view_sql_custom_view_name():
    sql = registry_view_sql("runs.jsonl", "registry.jsonl", view_name="my_view")
    assert "CREATE OR REPLACE VIEW my_view" in sql


def test_registry_view_sql_escapes_single_quotes_in_paths():
    """M3: a path containing a single quote must be SQL-escaped (doubled),
    not left to terminate the string literal early."""
    sql = registry_view_sql("it's-a-dir/runs.jsonl", "o'registry.jsonl")
    assert "it''s-a-dir/runs.jsonl" in sql
    assert "o''registry.jsonl" in sql
    assert "it's" not in sql  # no un-escaped copy anywhere


# ---------------------------------------------------------------------------
# open_registry_duckdb: guarded import
# ---------------------------------------------------------------------------

def test_open_registry_duckdb_without_duckdb_raises_specific_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "duckdb", None)
    with pytest.raises(DuckDBNotInstalled) as exc_info:
        open_registry_duckdb(tmp_path / "db.duckdb", tmp_path / "runs.jsonl", tmp_path / "registry.jsonl")
    assert "deep-research-toolkit[compiler]" in str(exc_info.value)


def test_open_registry_duckdb_queries_joined_view(tmp_path):
    duckdb = pytest.importorskip("duckdb")

    runs_path = tmp_path / "runs.jsonl"
    registry_path = tmp_path / "registry.jsonl"
    with open(runs_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"config_sha256": "sha256:c1", "dataset_hash": "sha256:d1",
                            "seed": 42, "git_commit": "abc"}) + "\n")
        f.write(json.dumps({"config_sha256": "sha256:c2", "dataset_hash": "sha256:d2",
                            "seed": 42, "git_commit": "def"}) + "\n")

    row = _valid_row(config_sha256="sha256:c1", dataset_hash="sha256:d1")
    append_registry_row(registry_path, row)

    con = open_registry_duckdb(tmp_path / "view.duckdb", runs_path, registry_path)
    try:
        result = con.execute(
            "SELECT config_sha256, status FROM tunekit_runs ORDER BY config_sha256").fetchall()
    finally:
        con.close()

    assert result[0] == ("sha256:c1", "promoted")  # joined to its registry row
    assert result[1] == ("sha256:c2", None)  # no matching registry row -> LEFT JOIN null
    assert isinstance(duckdb.__version__, str)


def test_view_does_not_fan_out_after_refused_duplicate(tmp_path):
    """I3: with the uniqueness triple enforced at append time, a run row
    can only ever join ONE registry row -- the LEFT JOIN cannot fan out."""
    pytest.importorskip("duckdb")

    runs_path = tmp_path / "runs.jsonl"
    registry_path = tmp_path / "registry.jsonl"
    runs_path.write_text(
        json.dumps({"config_sha256": "sha256:c1", "dataset_hash": "sha256:d1",
                    "seed": 42, "git_commit": "abc"}) + "\n", encoding="utf-8")

    row = _valid_row(config_sha256="sha256:c1", dataset_hash="sha256:d1")
    append_registry_row(registry_path, row)
    with pytest.raises(DuplicateRegistryRowError):
        append_registry_row(registry_path, _valid_row(
            config_sha256="sha256:c1", dataset_hash="sha256:d1",
            timestamp="2026-07-08T00:00:00+00:00"))

    con = open_registry_duckdb(tmp_path / "view.duckdb", runs_path, registry_path)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM tunekit_runs WHERE config_sha256 = 'sha256:c1'").fetchone()[0]
    finally:
        con.close()
    assert n == 1  # exactly one view row per run -- no fan-out
