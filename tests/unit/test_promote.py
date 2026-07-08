"""Unit tests for scripts/promote.py's pure promotion gates (design doc
§6.3/§8), imported the same way tests/unit/test_eval_pipeline.py loads its
hyphenated sibling script. Exercises every CI-distinct exit code against
synthetic eval-report dicts shaped like scripts/eval-pipeline.py's
build_report output -- no live model, no real report files beyond what a
test writes to tmp_path for the CLI-level tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("promote", REPO_ROOT / "scripts" / "promote.py")
promote = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(promote)


CORPUS_HASH = "sha256:corpus-x"
PROMPT_HASH = "sha256:prompt-x"


def _join_keys(corpus_version=CORPUS_HASH, prompt_version=PROMPT_HASH):
    return {"corpus_version": corpus_version, "prompt_version": prompt_version}


def _per_doc(recalls: dict[str, float]) -> dict:
    return {doc: {"recall": r} for doc, r in recalls.items()}


def _report(*, recalls: dict, gate_pass_rate=0.99, bait_rejection=0.97, atomicity=1.1,
           model="e4b", join_keys=None, wiki_pass=0.90, synth_pass=0.96, adjudicate_acc=1.0,
           include_roles=("wiki_write", "synthesize", "conflict_adjudicate")) -> dict:
    roles = {
        "extract": {
            "models": {
                model: {
                    "recall": sum(recalls.values()) / len(recalls),
                    "gate_pass_rate": gate_pass_rate,
                    "bait_rejection": bait_rejection,
                    "atomicity": atomicity,
                    "per_doc": _per_doc(recalls),
                }
            }
        },
    }
    if "wiki_write" in include_roles:
        roles["wiki_write"] = {"model": "gemma4:31b", "mean_pass_rate": wiki_pass}
    if "synthesize" in include_roles:
        roles["synthesize"] = {"model": "gemma4:31b", "mean_pass_rate": synth_pass}
    if "conflict_adjudicate" in include_roles:
        roles["conflict_adjudicate"] = {"model": "gemma4:31b", "accuracy": adjudicate_acc}
    return {"join_keys": join_keys or _join_keys(), "roles": roles}


# A baseline that just barely misses 0.90 recall, so a candidate with strong,
# consistent per-doc recall clears BOTH "clears 0.90" and "beats baseline".
BASELINE_RECALLS = {f"doc{i}": 0.70 for i in range(1, 11)}
STRONG_CANDIDATE_RECALLS = {f"doc{i}": 0.96 for i in range(1, 11)}


def _baseline_report(**kwargs):
    return _report(recalls=dict(BASELINE_RECALLS), **kwargs)


def _strong_candidate_report(**kwargs):
    return _report(recalls=dict(STRONG_CANDIDATE_RECALLS), **kwargs)


# ---------------------------------------------------------------------------
# provenance_gate / exit 6
# ---------------------------------------------------------------------------

def test_provenance_gate_passes_when_corpus_and_prompt_match():
    result = promote.provenance_gate(_strong_candidate_report(), _baseline_report())
    assert result["passed"] is True


def test_provenance_gate_fails_on_mismatched_corpus():
    candidate = _strong_candidate_report(join_keys=_join_keys(corpus_version="sha256:OTHER"))
    baseline = _baseline_report()
    result = promote.provenance_gate(candidate, baseline)
    assert result["passed"] is False
    assert any("DIFFERENT corpora" in r for r in result["reasons"])


def test_provenance_gate_fails_on_missing_corpus_hash():
    candidate = _strong_candidate_report(join_keys=_join_keys(corpus_version=None))
    result = promote.provenance_gate(candidate, _baseline_report())
    assert result["passed"] is False


def test_evaluate_promotion_exit_6_on_provenance_failure():
    candidate = _strong_candidate_report(join_keys=_join_keys(corpus_version="sha256:OTHER"))
    exit_code, details = promote.evaluate_promotion(candidate, _baseline_report())
    assert exit_code == promote.EXIT_PROVENANCE == 6
    assert details["exit_code"] == 6
    assert "recall" not in details  # short-circuited before later gates ran


# ---------------------------------------------------------------------------
# recall_gate / exit 2
# ---------------------------------------------------------------------------

def test_recall_gate_passes_when_candidate_clears_floor_and_beats_baseline():
    result = promote.recall_gate(_strong_candidate_report(), _baseline_report())
    assert result["passed"] is True
    assert result["clears_floor"] is True
    assert result["beats_baseline"] is True
    assert result["candidate_ci95"][0] >= 0.90


def test_recall_gate_fails_when_candidate_below_floor():
    weak = {f"doc{i}": 0.50 for i in range(1, 11)}
    result = promote.recall_gate(_report(recalls=weak), _baseline_report())
    assert result["passed"] is False
    assert result["clears_floor"] is False


def test_recall_gate_fails_when_candidate_does_not_beat_baseline():
    # Candidate clears 0.90 on its own, but baseline is even higher -> "does
    # not beat baseline" should still fail the gate.
    candidate_recalls = {f"doc{i}": 0.91 for i in range(1, 11)}
    baseline_recalls = {f"doc{i}": 0.97 for i in range(1, 11)}
    result = promote.recall_gate(_report(recalls=candidate_recalls), _report(recalls=baseline_recalls))
    assert result["clears_floor"] is True
    assert result["beats_baseline"] is False
    assert result["passed"] is False


def test_evaluate_promotion_exit_2_on_recall_failure():
    weak = {f"doc{i}": 0.50 for i in range(1, 11)}
    exit_code, details = promote.evaluate_promotion(_report(recalls=weak), _baseline_report())
    assert exit_code == promote.EXIT_RECALL == 2
    assert details["recall"]["passed"] is False
    assert "bait" not in details  # short-circuited


# ---------------------------------------------------------------------------
# bait_gate / exit 3
# ---------------------------------------------------------------------------

def test_bait_gate_passes_when_no_regression():
    result = promote.bait_gate(
        _strong_candidate_report(bait_rejection=0.97), _baseline_report(bait_rejection=0.95))
    assert result["passed"] is True


def test_bait_gate_fails_on_any_regression():
    result = promote.bait_gate(
        _strong_candidate_report(bait_rejection=0.94), _baseline_report(bait_rejection=0.95))
    assert result["passed"] is False


def test_evaluate_promotion_exit_3_on_bait_regression():
    candidate = _strong_candidate_report(bait_rejection=0.80)
    baseline = _baseline_report(bait_rejection=0.95)
    exit_code, details = promote.evaluate_promotion(candidate, baseline)
    assert exit_code == promote.EXIT_BAIT == 3
    assert details["bait"]["passed"] is False
    assert "gate_pass" not in details


# ---------------------------------------------------------------------------
# gate_pass_gate / exit 4
# ---------------------------------------------------------------------------

def test_gate_pass_gate_passes_at_floor():
    result = promote.gate_pass_gate(_strong_candidate_report(gate_pass_rate=0.95))
    assert result["passed"] is True


def test_gate_pass_gate_fails_below_floor():
    result = promote.gate_pass_gate(_strong_candidate_report(gate_pass_rate=0.80))
    assert result["passed"] is False


def test_evaluate_promotion_exit_4_on_gate_pass_failure():
    candidate = _strong_candidate_report(gate_pass_rate=0.80)
    baseline = _baseline_report()
    exit_code, details = promote.evaluate_promotion(candidate, baseline)
    assert exit_code == promote.EXIT_GATE_PASS == 4
    assert details["gate_pass"]["passed"] is False
    assert "drift" not in details


# ---------------------------------------------------------------------------
# drift_gate / exit 5
# ---------------------------------------------------------------------------

def test_drift_gate_passes_when_roles_hold():
    result = promote.drift_gate(_strong_candidate_report(), _baseline_report())
    assert result["passed"] is True


def test_drift_gate_fails_on_wiki_write_regression():
    candidate = _strong_candidate_report(wiki_pass=0.70)
    baseline = _baseline_report(wiki_pass=0.90)
    result = promote.drift_gate(candidate, baseline)
    assert result["passed"] is False
    assert result["roles"]["wiki_write"]["passed"] is False


def test_drift_gate_fails_on_conflict_adjudicate_regression():
    candidate = _strong_candidate_report(adjudicate_acc=0.5)
    baseline = _baseline_report(adjudicate_acc=1.0)
    result = promote.drift_gate(candidate, baseline)
    assert result["passed"] is False
    assert result["roles"]["conflict_adjudicate"]["passed"] is False


def test_drift_gate_skips_role_absent_from_either_report():
    candidate = _strong_candidate_report(include_roles=("synthesize", "conflict_adjudicate"))
    baseline = _baseline_report(include_roles=("synthesize", "conflict_adjudicate"))
    result = promote.drift_gate(candidate, baseline)
    assert result["passed"] is True
    assert result["roles"]["wiki_write"]["skipped"] is True


def test_evaluate_promotion_exit_5_on_drift_failure():
    candidate = _strong_candidate_report(synth_pass=0.10)
    baseline = _baseline_report(synth_pass=0.96)
    exit_code, details = promote.evaluate_promotion(candidate, baseline)
    assert exit_code == promote.EXIT_DRIFT == 5
    assert details["drift"]["passed"] is False


# ---------------------------------------------------------------------------
# All-green -> exit 0
# ---------------------------------------------------------------------------

def test_evaluate_promotion_exit_0_when_all_gates_pass():
    exit_code, details = promote.evaluate_promotion(_strong_candidate_report(), _baseline_report())
    assert exit_code == promote.EXIT_PROMOTE == 0
    assert details["exit_code"] == 0
    assert all(details[g]["passed"] for g in ("provenance", "recall", "bait", "gate_pass", "drift"))


# ---------------------------------------------------------------------------
# extract_model_metrics: model disambiguation
# ---------------------------------------------------------------------------

def test_extract_model_metrics_auto_selects_sole_model():
    report = _strong_candidate_report(model="e4b")
    metrics = promote.extract_model_metrics(report)
    assert metrics["gate_pass_rate"] == 0.99


def test_extract_model_metrics_raises_on_ambiguous_report():
    report = _strong_candidate_report(model="e4b")
    report["roles"]["extract"]["models"]["qwen3:30b"] = dict(
        report["roles"]["extract"]["models"]["e4b"])
    with pytest.raises(ValueError):
        promote.extract_model_metrics(report)


def test_extract_model_metrics_explicit_model_name():
    report = _strong_candidate_report(model="e4b")
    metrics = promote.extract_model_metrics(report, model="e4b")
    assert metrics["gate_pass_rate"] == 0.99


def test_extract_model_metrics_unknown_model_raises_keyerror():
    report = _strong_candidate_report(model="e4b")
    with pytest.raises(KeyError):
        promote.extract_model_metrics(report, model="nonexistent")


# ---------------------------------------------------------------------------
# CLI (main): reads report files from disk, returns the matching exit code
# ---------------------------------------------------------------------------

def test_main_exit_0_and_json_output(tmp_path, capsys):
    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline.json"
    candidate_path.write_text(json.dumps(_strong_candidate_report()), encoding="utf-8")
    baseline_path.write_text(json.dumps(_baseline_report()), encoding="utf-8")

    exit_code = promote.main([str(candidate_path), "--baseline", str(baseline_path), "--json"])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["exit_code"] == 0


def test_main_exit_2_on_weak_candidate(tmp_path):
    weak = {f"doc{i}": 0.50 for i in range(1, 11)}
    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline.json"
    candidate_path.write_text(json.dumps(_report(recalls=weak)), encoding="utf-8")
    baseline_path.write_text(json.dumps(_baseline_report()), encoding="utf-8")

    exit_code = promote.main([str(candidate_path), "--baseline", str(baseline_path)])
    assert exit_code == 2
