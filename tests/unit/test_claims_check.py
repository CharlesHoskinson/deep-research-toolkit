"""check_claims_file: mechanical write-time gate for agent-authored claims.jsonl."""
import json
import shutil
from pathlib import Path

from deep_research_toolkit.common.claims_check import check_claims_file

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "reference-run-hydra-settlement"


def _copy_fixture(tmp_path: Path) -> Path:
    run = tmp_path / FIXTURE.name
    shutil.copytree(FIXTURE, run)
    return run


def test_reference_run_passes_clean(tmp_path):
    report = check_claims_file(_copy_fixture(tmp_path))
    assert report["failures"] == []
    assert report["checked"] == report["ok"] > 0


def test_corrupted_quote_is_flagged(tmp_path):
    run = _copy_fixture(tmp_path)
    rows = [json.loads(l) for l in (run / "claims.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows[0]["supporting_evidence"][0]["quote"] = "this text appears in no chunk"
    (run / "claims.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    report = check_claims_file(run)
    assert len(report["failures"]) == 1
    assert report["failures"][0]["claim_id"] == rows[0]["claim_id"]
    assert "not a verbatim substring" in report["failures"][0]["reason"]


def test_missing_evidence_is_flagged(tmp_path):
    run = _copy_fixture(tmp_path)
    rows = [json.loads(l) for l in (run / "claims.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows[0]["supporting_evidence"] = []
    (run / "claims.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    report = check_claims_file(run)
    assert any("no supporting evidence" in f["reason"] for f in report["failures"])
