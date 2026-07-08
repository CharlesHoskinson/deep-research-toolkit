"""tunekit.provenance: per-run run.json + runs.jsonl (design doc §6.4).
config_sha256 must be canonical-key-order-stable, run.json/runs.jsonl must
both be written on every call, and run_exists must be a correct keyed
idempotence check over (config_sha256, dataset_hash)."""
from __future__ import annotations

import json

from deep_research_toolkit.tunekit.provenance import (
    DEFAULT_SEED,
    build_run_record,
    config_sha256,
    run_exists,
    write_run_record,
)


def test_config_sha256_is_stable_across_key_order():
    a = {"lr": 2e-4, "recipe": "B", "lora": {"r": 16, "alpha": 32}}
    b = {"lora": {"alpha": 32, "r": 16}, "recipe": "B", "lr": 2e-4}
    assert config_sha256(a) == config_sha256(b)


def test_config_sha256_changes_with_content():
    a = {"lr": 2e-4}
    b = {"lr": 3e-4}
    assert config_sha256(a) != config_sha256(b)


def test_config_sha256_has_sha256_prefix():
    assert config_sha256({"x": 1}).startswith("sha256:")


def test_build_run_record_defaults_seed_to_42():
    record = build_run_record({"lr": 2e-4}, "sha256:dataset", {"torch": "2.11.0"})
    assert record["seed"] == DEFAULT_SEED


def test_write_run_record_writes_run_json_and_runs_jsonl(tmp_path):
    run_dir = tmp_path / "runs" / "run-0001"
    config = {"recipe": "B", "lr": 2e-4}
    versions = {"torch": "2.11.0+cu129", "transformers": "5.5.3", "unsloth": "2026.7.1"}

    record = write_run_record(run_dir, config, "sha256:dataset-abc", versions,
                              git_commit="deadbeef", base_model_digest="sha256:basemodel",
                              recipe_id="recipe-b")

    run_json_path = run_dir / "run.json"
    assert run_json_path.is_file()
    on_disk = json.loads(run_json_path.read_text(encoding="utf-8"))
    assert on_disk["config_sha256"] == config_sha256(config)
    assert on_disk["dataset_hash"] == "sha256:dataset-abc"
    assert on_disk["seed"] == DEFAULT_SEED
    assert on_disk["git_commit"] == "deadbeef"
    assert on_disk["base_model_digest"] == "sha256:basemodel"
    assert on_disk["recipe_id"] == "recipe-b"
    assert on_disk["versions"] == versions

    runs_jsonl_path = run_dir.parent / "runs.jsonl"
    assert runs_jsonl_path.is_file()
    rows = [json.loads(line) for line in runs_jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["config_sha256"] == config_sha256(config)
    assert rows[0]["run_dir"] == str(run_dir)

    assert record["config_sha256"] == config_sha256(config)
    assert record["run_dir"] == str(run_dir)


def test_write_run_record_appends_across_multiple_runs(tmp_path):
    versions = {"torch": "2.11.0+cu129"}
    write_run_record(tmp_path / "runs" / "run-0001", {"recipe": "A"}, "sha256:d1", versions)
    write_run_record(tmp_path / "runs" / "run-0002", {"recipe": "B"}, "sha256:d2", versions)

    runs_jsonl_path = tmp_path / "runs" / "runs.jsonl"
    rows = [json.loads(line) for line in runs_jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert {r["dataset_hash"] for r in rows} == {"sha256:d1", "sha256:d2"}


def test_write_run_record_honors_explicit_runs_jsonl_path(tmp_path):
    shared_ledger = tmp_path / "shared" / "runs.jsonl"
    write_run_record(tmp_path / "a", {"recipe": "A"}, "sha256:d1", {},
                     runs_jsonl_path=shared_ledger)
    write_run_record(tmp_path / "b", {"recipe": "B"}, "sha256:d2", {},
                     runs_jsonl_path=shared_ledger)
    rows = [json.loads(line) for line in shared_ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    # No stray runs.jsonl written at either run dir's own parent.
    assert not (tmp_path / "runs.jsonl").is_file()


def test_run_exists_true_only_for_matching_pair(tmp_path):
    runs_path = tmp_path / "runs.jsonl"
    config = {"recipe": "B"}
    sha = config_sha256(config)
    write_run_record(tmp_path / "run-0001", config, "sha256:dataset-x", {},
                     runs_jsonl_path=runs_path)

    assert run_exists(runs_path, sha, "sha256:dataset-x") is True
    assert run_exists(runs_path, sha, "sha256:dataset-DIFFERENT") is False
    assert run_exists(runs_path, "sha256:different-config", "sha256:dataset-x") is False


def test_run_exists_false_when_ledger_missing(tmp_path):
    assert run_exists(tmp_path / "nonexistent-runs.jsonl", "sha256:x", "sha256:y") is False
