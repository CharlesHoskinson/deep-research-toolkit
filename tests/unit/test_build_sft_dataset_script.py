"""Unit tests for scripts/build-sft-dataset.py's importable pure logic
(`parse_base_model_overrides`, `default_teachers`'s override wiring) plus
CLI wiring: `--help`, and an end-to-end subprocess run over a fake
`--teachers-module` (never a live model) exercising `--resume` after a
simulated crash. Live extraction against a real Ollama endpoint is NOT
unit-tested here (see the module's own TEACHER WIRING docstring) -- only
`default_teachers`'s construction-time wiring is pure enough to exercise
without one (constructing a `LocalOpenAIBackend` does no network I/O; only
`.complete()` would, and it's never called in these tests).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "build_sft_dataset_script", REPO_ROOT / "scripts" / "build-sft-dataset.py")
build_sft_dataset_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_sft_dataset_script)

SCRIPT_PATH = REPO_ROOT / "scripts" / "build-sft-dataset.py"


def _fake_local_config(tmp_path: Path):
    """A real Config (provider=local) built by writing a throwaway
    .deepresearch.yml and loading it -- avoids hand-constructing every
    dataclass field of deep_research_toolkit.config.Config."""
    from deep_research_toolkit.config import load_config

    (tmp_path / ".deepresearch.yml").write_text(
        "llm:\n  provider: local\n  local:\n    base_url: http://localhost:11434/v1\n"
        "    model: placeholder\n", encoding="utf-8")
    return load_config(start=tmp_path)


# ---------------------------------------------------------------------------
# parse_base_model_overrides
# ---------------------------------------------------------------------------

def test_parse_base_model_overrides_parses_repeated_pairs():
    overrides = build_sft_dataset_script.parse_base_model_overrides(
        ["gemma4:e4b=gemma4:e4b-q8_0", "qwen3:30b-a3b-instruct-2507-q4_K_M=qwen3:30b-a3b-instruct-2507-q8_0"])
    assert overrides == {
        "gemma4:e4b": "gemma4:e4b-q8_0",
        "qwen3:30b-a3b-instruct-2507-q4_K_M": "qwen3:30b-a3b-instruct-2507-q8_0",
    }


def test_parse_base_model_overrides_empty_input():
    assert build_sft_dataset_script.parse_base_model_overrides(None) == {}
    assert build_sft_dataset_script.parse_base_model_overrides([]) == {}


def test_parse_base_model_overrides_rejects_missing_equals():
    with pytest.raises(ValueError):
        build_sft_dataset_script.parse_base_model_overrides(["gemma4:e4b-no-equals-sign"])


def test_parse_base_model_overrides_rejects_empty_route_or_model():
    with pytest.raises(ValueError):
        build_sft_dataset_script.parse_base_model_overrides(["=gemma4:e4b"])
    with pytest.raises(ValueError):
        build_sft_dataset_script.parse_base_model_overrides(["gemma4:e4b="])


# ---------------------------------------------------------------------------
# default_teachers: override wiring (construction-only, no network I/O)
# ---------------------------------------------------------------------------

def test_default_teachers_keys_match_the_router_emitted_tags(tmp_path):
    config = _fake_local_config(tmp_path)
    teachers = build_sft_dataset_script.default_teachers(config, "web")
    assert set(teachers) == build_sft_dataset_script._LOCAL_MODEL_ROUTES
    assert "frontier" not in teachers


def test_default_teachers_uses_override_tag_for_a_wired_route(tmp_path, monkeypatch):
    requested_models = []

    class _FakeBackend:
        thinking = False

        def complete(self, system, user, **sampling):
            raise AssertionError("must never actually be called in this test")

    def fake_get_backend(config, role=None):
        requested_models.append(config.llm_roles["extract"]["model"])
        return _FakeBackend()

    monkeypatch.setattr(build_sft_dataset_script, "get_backend", fake_get_backend)
    config = _fake_local_config(tmp_path)

    teachers = build_sft_dataset_script.default_teachers(
        config, "web", base_model_overrides={"gemma4:e4b": "gemma4:e4b-q8_0"})

    assert set(teachers) == build_sft_dataset_script._LOCAL_MODEL_ROUTES  # keys unchanged
    assert "gemma4:e4b-q8_0" in requested_models   # override tag WAS requested
    assert "gemma4:e4b" not in requested_models    # the un-overridden route tag was NOT


def test_default_teachers_warns_and_ignores_an_unmatched_override(tmp_path, monkeypatch, capsys):
    def fake_get_backend(config, role=None):
        class _FakeBackend:
            thinking = False
        return _FakeBackend()

    monkeypatch.setattr(build_sft_dataset_script, "get_backend", fake_get_backend)
    config = _fake_local_config(tmp_path)

    teachers = build_sft_dataset_script.default_teachers(
        config, "web", base_model_overrides={"e4b": "gemma4:e4b"})  # stale pre-fix route name

    assert set(teachers) == build_sft_dataset_script._LOCAL_MODEL_ROUTES
    captured = capsys.readouterr()
    assert "e4b" in captured.err and "ignored" in captured.err


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def test_help_works_without_a_live_endpoint():
    result = subprocess.run([sys.executable, str(SCRIPT_PATH), "--help"],
                           capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert "--base-model-override" in result.stdout
    assert "--resume" in result.stdout


def _write_training_chunk(doc_dir: Path, locator: str, text: str) -> None:
    doc_dir.mkdir(parents=True, exist_ok=True)
    with open(doc_dir / "chunks.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"locator": locator, "text": text}) + "\n")


_FAKE_TEACHERS_MODULE = '''
import json
import os

def get_teachers(config, producer):
    raise_for = os.environ.get("SFT_TEST_RAISE_FOR", "")

    def teacher(chunk_batch, k, temperature):
        chunk = chunk_batch[0]
        loc = chunk.get("locator") or chunk.get("node_id")
        if raise_for and loc == raise_for:
            raise RuntimeError(f"simulated teacher outage for {loc}")
        text = chunk["text"]
        claim = {
            "claim_id": "c1",
            "claim": text.rstrip("."),
            "supporting_evidence": [{"locator": loc, "start_char": 0, "end_char": len(text) - 1}],
        }
        return [json.dumps({"claims": [claim], "entities": [], "relations": []})]

    return {"gemma4:e4b": teacher, "qwen3:30b-a3b-instruct-2507-q4_K_M": teacher, "frontier": teacher}
'''


def test_cli_resume_after_a_simulated_crash_completes_with_no_re_calls(tmp_path):
    """End-to-end (subprocess) proof that --resume actually works through the
    CLI, not just through build_sft_dataset() directly: a fake
    --teachers-module raises for the 3rd of 4 chunks on the first
    invocation (simulating a crash), then a --resume rerun with the same
    teacher healed completes and produces all 4 examples."""
    training_dir = tmp_path / "training-corpus"
    doc_dir = training_dir / "doc-a"
    chunks = [
        ("trainA#c001", "Validators exchange heartbeats every single cycle."),
        ("trainA#c002", "Auditors record every checkpoint each hour precisely."),
        ("trainA#c003", "Leaders broadcast proposals across the whole cluster."),
        ("trainA#c004", "Followers acknowledge proposals within one round trip."),
    ]
    for locator, text in chunks:
        _write_training_chunk(doc_dir, locator, text)

    eval_corpus_dir = tmp_path / "empty-eval-corpus"
    eval_corpus_dir.mkdir()
    out_dir = tmp_path / "out"

    (tmp_path / "fake_teachers_for_sft_cli.py").write_text(_FAKE_TEACHERS_MODULE, encoding="utf-8")

    base_env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    base_args = [
        sys.executable, str(SCRIPT_PATH), str(training_dir),
        "--out-dir", str(out_dir), "--eval-corpus-dir", str(eval_corpus_dir),
        "--k-ladder", "4", "--teachers-module", "fake_teachers_for_sft_cli",
    ]

    # First run: chunk 3 explodes -- simulates a crash partway through.
    crash_result = subprocess.run(
        base_args, capture_output=True, text=True, timeout=60, cwd=str(tmp_path),
        env={**base_env, "SFT_TEST_RAISE_FOR": "trainA#c003"})
    assert crash_result.returncode != 0
    assert not (out_dir / "train.jsonl").exists()  # never reached final assembly

    progress_rows = [json.loads(line)
                     for line in (out_dir / "progress.json").read_text(encoding="utf-8").splitlines()]
    assert {r["locator"] for r in progress_rows} == {"trainA#c001", "trainA#c002"}

    # Second run: --resume, teacher healed (no SFT_TEST_RAISE_FOR).
    resume_result = subprocess.run(
        [*base_args, "--resume"], capture_output=True, text=True, timeout=60, cwd=str(tmp_path),
        env={**base_env, "SFT_TEST_RAISE_FOR": ""})
    assert resume_result.returncode == 0, resume_result.stderr

    train_rows = [json.loads(line) for line in (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    val_rows = [json.loads(line) for line in (out_dir / "val.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train_rows) + len(val_rows) == 4
    all_locators = sorted(r["locator"] for r in train_rows + val_rows)
    assert all_locators == ["trainA#c001", "trainA#c002", "trainA#c003", "trainA#c004"]

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["resumed"] is True
    assert manifest["n_accepted"] == 4
