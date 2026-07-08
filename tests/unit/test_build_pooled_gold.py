"""Unit tests for scripts/build-pooled-gold.py's importable pure/file-I/O
logic: the --renamespace migration driver (Anomaly 1) and its CLI wiring.
Live extraction (main()'s per-model loop against a real endpoint) is NOT
unit-tested here, matching the module's own docstring -- only pool_gold and
now the renamespace helpers are pure enough to exercise without a model.
--renamespace itself needs no live endpoint, so it IS covered end-to-end.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "build_pooled_gold", REPO_ROOT / "scripts" / "build-pooled-gold.py")
build_pooled_gold = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_pooled_gold)


def _row(claim_id: str, claim: str, doc_id: str = "doc-a") -> dict:
    return {
        "schema_version": "2.0", "document_id": doc_id, "claim_id": claim_id,
        "claim": claim, "claim_type": "empirical", "confidence": "high",
        "supporting_evidence": [{"locator": f"{doc_id}#c001", "start_char": 0, "end_char": 5,
                                 "url": None, "quote": "hello"}],
        "citable": True,
    }


def _write_pooled_gold(root: Path, doc_id: str, rows: list[dict]) -> Path:
    doc_dir = root / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    path = doc_dir / "pooled-gold.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# renamespace_pooled_gold: the --renamespace driver over a tmp corpus
# ---------------------------------------------------------------------------

def test_renamespace_pooled_gold_disambiguates_and_reports_before_after(tmp_path):
    rows = [
        _row("b00_c_0001", "claim from model A"),
        _row("b00_c_0002", "unique claim, untouched"),
        _row("b00_c_0001", "claim from model B"),  # collides with row 0
    ]
    path = _write_pooled_gold(tmp_path, "doc-a", rows)

    summary = build_pooled_gold.renamespace_pooled_gold(
        tmp_path, ["gemma4:e4b", "qwen3:30b-a3b"])

    assert summary["docs"]["doc-a"]["before"] == {"dup_ids": 1, "dup_rows": 2}
    assert summary["docs"]["doc-a"]["after"] == {"dup_ids": 0, "dup_rows": 0}
    assert summary["total_before"] == 1
    assert summary["total_after"] == 0

    written = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["claim_id"] for r in written]
    assert ids == ["gemma4-e4b.b00_c_0001", "b00_c_0002", "qwen3-30b-a3b.b00_c_0001"]
    assert len(ids) == len(set(ids))
    # non-id fields (including the doc's other schema fields) are untouched
    assert written[0]["claim"] == "claim from model A"
    assert written[2]["claim"] == "claim from model B"
    assert written[0]["schema_version"] == "2.0"


def test_renamespace_pooled_gold_is_idempotent(tmp_path):
    rows = [_row("b00_c_0001", "A"), _row("b00_c_0001", "B")]
    path = _write_pooled_gold(tmp_path, "doc-a", rows)
    models = ["gemma4:e4b", "qwen3:30b-a3b"]

    first = build_pooled_gold.renamespace_pooled_gold(tmp_path, models)
    assert first["total_before"] == 1
    mtime_after_first = path.stat().st_mtime_ns
    content_after_first = path.read_text(encoding="utf-8")

    second = build_pooled_gold.renamespace_pooled_gold(tmp_path, models)
    assert second["total_before"] == 0  # nothing left to disambiguate
    assert second["total_after"] == 0
    # a clean doc is never rewritten
    assert path.read_text(encoding="utf-8") == content_after_first
    assert path.stat().st_mtime_ns == mtime_after_first


def test_renamespace_pooled_gold_skips_docs_without_a_pooled_gold_file(tmp_path):
    (tmp_path / "doc-no-pool").mkdir()
    (tmp_path / "doc-no-pool" / "chunks.jsonl").write_text("", encoding="utf-8")
    summary = build_pooled_gold.renamespace_pooled_gold(tmp_path, ["m1", "m2"])
    assert summary["docs"] == {}
    assert summary["total_before"] == 0


def test_renamespace_pooled_gold_leaves_clean_doc_untouched(tmp_path):
    rows = [_row("b00_c_0001", "A"), _row("b00_c_0002", "B")]
    path = _write_pooled_gold(tmp_path, "doc-a", rows)
    original = path.read_text(encoding="utf-8")
    summary = build_pooled_gold.renamespace_pooled_gold(tmp_path, ["m1", "m2"])
    assert summary["docs"]["doc-a"]["before"] == {"dup_ids": 0, "dup_rows": 0}
    assert path.read_text(encoding="utf-8") == original


def test_renamespace_pooled_gold_handles_multiple_docs_independently(tmp_path):
    _write_pooled_gold(tmp_path, "doc-a", [_row("b00_c_0001", "A1", "doc-a"),
                                           _row("b00_c_0001", "A2", "doc-a")])
    _write_pooled_gold(tmp_path, "doc-b", [_row("b00_c_0001", "B1", "doc-b")])  # no collision
    summary = build_pooled_gold.renamespace_pooled_gold(tmp_path, ["gemma4:e4b", "qwen3:30b-a3b"])
    assert summary["docs"]["doc-a"]["before"]["dup_ids"] == 1
    assert summary["docs"]["doc-b"]["before"]["dup_ids"] == 0
    assert summary["total_before"] == 1


# ---------------------------------------------------------------------------
# real committed corpus: the actual Anomaly-1 migration, run against a copy
# ---------------------------------------------------------------------------

def test_committed_corpus_pooled_gold_has_no_duplicate_ids(tmp_path):
    # The real corpus was migrated in place by `--renamespace` (Anomaly 1: 618
    # duplicate claim_id groups / 1,591 rows, all within-doc collisions
    # between the two pooled teacher models, resolved to 0). Running the
    # migration again over a COPY (never touches the repo's own fixtures)
    # must find nothing left to do -- the committed state stays a fixed point.
    import shutil
    src = REPO_ROOT / "tests" / "fixtures" / "eval-corpus"
    for doc_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        pooled = doc_dir / "pooled-gold.jsonl"
        if pooled.is_file():
            dst_dir = tmp_path / doc_dir.name
            dst_dir.mkdir()
            shutil.copyfile(pooled, dst_dir / "pooled-gold.jsonl")

    summary = build_pooled_gold.renamespace_pooled_gold(
        tmp_path, build_pooled_gold.LEGACY_POOL_MODELS.split(","))
    assert summary["total_before"] == 0
    assert summary["total_after"] == 0


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def test_help_works_without_a_live_endpoint():
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build-pooled-gold.py"), "--help"],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert "--renamespace" in result.stdout
    assert "--legacy-models" in result.stdout


def test_renamespace_cli_runs_without_a_live_endpoint(tmp_path):
    import subprocess
    import sys
    _write_pooled_gold(tmp_path, "doc-a",
                       [_row("b00_c_0001", "A"), _row("b00_c_0001", "B")])
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build-pooled-gold.py"),
         "--renamespace", "--corpus", str(tmp_path)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "1 dup id(s)" in result.stdout
    assert "0 dup id(s)" in result.stdout


def test_default_models_uses_pulled_qwen_tag():
    # Cosmetic fix: the bare "qwen3:30b-a3b" tag was never pulled -- the
    # default must name the instruct-2507 quant that actually is.
    assert "qwen3:30b-a3b-instruct-2507-q4_K_M" in build_pooled_gold.DEFAULT_MODELS
    assert build_pooled_gold.DEFAULT_MODELS.split(",")[0] == "gemma4:e4b"
