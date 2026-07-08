"""Unit tests for scripts/promote.py's pure promotion gates (design doc
§6.3/§8), imported the same way tests/unit/test_eval_pipeline.py loads its
hyphenated sibling script. Exercises every CI-distinct exit code against
synthetic eval-report dicts shaped like scripts/eval-pipeline.py's
build_report output -- no live model, no real report files beyond what a
test writes to tmp_path for the CLI-level tests.

The recall gate runs at PER-CLAIM grain, reconstructed from each doc's
`recalled` count + `missed_claim_ids` list (see promote.py's GATE 1 GRAIN
docstring) -- the reviewer probes covering the per-doc-macro failure modes
are test_recall_gate_skewed_corpus_probe_fails_floor (macro 0.91 vs pooled
0.174) and test_recall_gate_near_tie_null_does_not_beat_baseline /
test_recall_gate_min_effect_rejects_hairline_delta (anti-conservative
paired check)."""
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


def _doc(n_ref: int, n_missed: int, prefix: str) -> dict:
    """One per-doc metrics dict carrying the per-claim-reconstructable
    fields the real report has: recall, recalled, missed_claim_ids. Missed
    ids are the FIRST n_missed of the doc's claim-id universe, so a
    candidate that misses fewer than the baseline has a missed set that is a
    subset of the baseline's -- exactly the improved-model shape."""
    missed_ids = [f"{prefix}_m{j:03d}" for j in range(n_missed)]
    return {
        "recall": (n_ref - n_missed) / n_ref,
        "recalled": n_ref - n_missed,
        "missed_claim_ids": missed_ids,
    }


def _docs(n_docs: int = 10, n_ref: int = 50, n_missed: int = 2) -> dict:
    return {f"doc{i}": _doc(n_ref, n_missed, f"doc{i}") for i in range(1, n_docs + 1)}


def _report(*, docs: dict, gate_pass_rate=0.99, bait_rejection=0.97, atomicity=1.1,
           model="e4b", join_keys=None, wiki_pass=0.90, synth_pass=0.96, adjudicate_acc=1.0,
           include_roles=("wiki_write", "synthesize", "conflict_adjudicate")) -> dict:
    n_recalled = sum(d["recalled"] for d in docs.values())
    n_ref = sum(d["recalled"] + len(d["missed_claim_ids"]) for d in docs.values())
    roles = {
        "extract": {
            "models": {
                model: {
                    "recall": (n_recalled / n_ref) if n_ref else None,
                    "gate_pass_rate": gate_pass_rate,
                    "bait_rejection": bait_rejection,
                    "atomicity": atomicity,
                    "per_doc": docs,
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


# Baseline: 10 docs x 50 claims, 15 missed each -> pooled recall 0.70.
# Strong candidate: 2 missed each -> pooled 0.96 (CI lower ~0.94 at n=500);
# paired deltas per doc: 13 claims recalled that baseline missed (+1), 2
# missed by both (0), 35 recalled by both (0) -> mean delta 0.26.
def _baseline_report(**kwargs):
    return _report(docs=_docs(n_missed=15), **kwargs)


def _strong_candidate_report(**kwargs):
    return _report(docs=_docs(n_missed=2), **kwargs)


# ---------------------------------------------------------------------------
# per-claim reconstruction helpers
# ---------------------------------------------------------------------------

def test_per_claim_indicators_reconstructs_ones_and_zeros():
    ind = promote.per_claim_indicators(_doc(10, 3, "d"))
    assert sorted(ind) == [0.0] * 3 + [1.0] * 7


def test_per_claim_indicators_empty_when_fields_missing():
    assert promote.per_claim_indicators({"recall": 0.9}) == []


def test_per_claim_deltas_signs_and_zero_padding():
    # candidate missed {m0}; baseline missed {m0, m1, m2}; R = 10.
    cand = {"recalled": 9, "missed_claim_ids": ["m0"]}
    base = {"recalled": 7, "missed_claim_ids": ["m0", "m1", "m2"]}
    deltas = promote.per_claim_deltas(cand, base)
    assert len(deltas) == 10
    assert deltas.count(1.0) == 2   # m1, m2: baseline-only misses
    assert deltas.count(-1.0) == 0
    assert deltas.count(0.0) == 8   # m0 both-missed + 7 both-recalled


def test_per_claim_deltas_candidate_regression_is_negative():
    cand = {"recalled": 7, "missed_claim_ids": ["m0", "m1", "m2"]}
    base = {"recalled": 9, "missed_claim_ids": ["m0"]}
    deltas = promote.per_claim_deltas(cand, base)
    assert deltas.count(-1.0) == 2


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
# recall_gate / exit 2 -- per-claim grain
# ---------------------------------------------------------------------------

def test_recall_gate_passes_when_candidate_clears_floor_and_beats_baseline():
    result = promote.recall_gate(_strong_candidate_report(), _baseline_report())
    assert result["passed"] is True
    assert result["clears_floor"] is True
    assert result["beats_baseline"] is True
    assert result["candidate_ci95"][0] >= 0.90
    assert result["n_claims"] == 500          # 10 docs x 50 claims, pooled
    assert result["n_paired_claims"] == 500


def test_recall_gate_fails_when_candidate_below_floor():
    weak = _report(docs=_docs(n_missed=25))   # pooled 0.50
    result = promote.recall_gate(weak, _baseline_report())
    assert result["passed"] is False
    assert result["clears_floor"] is False


def test_recall_gate_fails_when_candidate_does_not_beat_baseline():
    # Candidate pooled 0.94 clears the floor on its own, but baseline is
    # 0.98 -- per-claim deltas are 2x -1 per doc -> "does not beat baseline"
    # must still fail the gate.
    candidate = _report(docs=_docs(n_missed=3))   # pooled 0.94, CI lower ~0.92
    baseline = _report(docs=_docs(n_missed=1))    # pooled 0.98
    result = promote.recall_gate(candidate, baseline)
    assert result["clears_floor"] is True
    assert result["beats_baseline"] is False
    assert result["passed"] is False


def test_recall_gate_skewed_corpus_probe_fails_floor():
    """The reviewer's skewed-corpus probe: one 100-claim doc at 0.10 recall
    plus nine 1-claim docs at 1.0. The per-DOC macro mean is 0.91 (would
    pass a 0.90 floor); the true claim-weighted recall is 19/109 = 0.174.
    The per-claim gate MUST fail the floor."""
    docs = {"docA": _doc(100, 90, "docA")}
    docs.update({f"doc{c}": _doc(1, 0, f"doc{c}") for c in "BCDEFGHIJ"})
    macro_mean = sum(d["recall"] for d in docs.values()) / len(docs)
    assert macro_mean > 0.90  # the trap the old per-doc gate fell into

    candidate = _report(docs=docs)
    result = promote.recall_gate(candidate, _baseline_report())
    assert result["clears_floor"] is False
    assert result["passed"] is False
    assert result["n_claims"] == 109
    # pooled point estimate is ~0.174 -- nowhere near the floor
    assert result["candidate_ci95"][1] < 0.90


def test_recall_gate_near_tie_null_does_not_beat_baseline():
    """Near-tie null probe: candidate and baseline each miss 2 claims per
    doc, but DIFFERENT ones (+1/-1 in equal measure -> mean delta exactly
    0). The paired per-claim check must not fire."""
    cand_docs = {}
    base_docs = {}
    for i in range(1, 11):
        prefix = f"doc{i}"
        cand_docs[prefix] = {"recall": 0.96, "recalled": 48,
                             "missed_claim_ids": [f"{prefix}_m001", f"{prefix}_m003"]}
        base_docs[prefix] = {"recall": 0.96, "recalled": 48,
                             "missed_claim_ids": [f"{prefix}_m002", f"{prefix}_m004"]}
    candidate = _report(docs=cand_docs)
    baseline = _report(docs=base_docs)
    result = promote.recall_gate(candidate, baseline)
    assert result["clears_floor"] is True   # pooled 0.96 clears on its own
    assert result["beats_baseline"] is False
    assert result["passed"] is False


def test_recall_gate_min_effect_rejects_hairline_delta():
    """I2 probe: a uniform +0.003 pooled delta (baseline misses 3 claims of
    1000 that the candidate recalls; identical otherwise). Even if the delta
    CI lower bound is > 0, it cannot clear the 0.01 min-effect judge-noise
    floor -- the gate must reject."""
    cand_docs = _docs(n_docs=10, n_ref=100, n_missed=0)
    base_docs = _docs(n_docs=10, n_ref=100, n_missed=0)
    for i in (1, 2, 3):  # baseline misses one claim in 3 of the 10 docs
        base_docs[f"doc{i}"] = _doc(100, 1, f"doc{i}")
    candidate = _report(docs=cand_docs)
    baseline = _report(docs=base_docs)
    result = promote.recall_gate(candidate, baseline)
    assert result["clears_floor"] is True   # candidate recalls everything
    assert result["beats_baseline"] is False
    assert result["min_effect"] == promote.DEFAULT_MIN_EFFECT == 0.01
    assert result["passed"] is False


def test_evaluate_promotion_exit_2_on_recall_failure():
    weak = _report(docs=_docs(n_missed=25))
    exit_code, details = promote.evaluate_promotion(weak, _baseline_report())
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
# drift_gate / exit 5 -- fails CLOSED on one-sided absence
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


def test_drift_gate_skips_role_absent_from_both_reports():
    candidate = _strong_candidate_report(include_roles=("synthesize", "conflict_adjudicate"))
    baseline = _baseline_report(include_roles=("synthesize", "conflict_adjudicate"))
    result = promote.drift_gate(candidate, baseline)
    assert result["passed"] is True
    assert result["roles"]["wiki_write"]["skipped"] is True


def test_drift_gate_fails_closed_when_candidate_missing_a_measured_role():
    """I1 probe: baseline measured wiki_write; the candidate eval skipped
    it. The forgetting check cannot be confirmed -> the role FAILS with
    'role not measured', it is not skipped."""
    candidate = _strong_candidate_report(include_roles=("synthesize", "conflict_adjudicate"))
    baseline = _baseline_report()  # has all three
    result = promote.drift_gate(candidate, baseline)
    assert result["passed"] is False
    assert result["roles"]["wiki_write"]["passed"] is False
    assert "not measured on candidate" in result["roles"]["wiki_write"]["reason"]


def test_drift_gate_fails_closed_when_baseline_missing_a_measured_role():
    candidate = _strong_candidate_report()
    baseline = _baseline_report(include_roles=("wiki_write", "conflict_adjudicate"))
    result = promote.drift_gate(candidate, baseline)
    assert result["passed"] is False
    assert "not measured on baseline" in result["roles"]["synthesize"]["reason"]


def test_evaluate_promotion_exit_5_when_candidate_missing_all_capability_roles():
    """The reviewer's missing-roles probe: a candidate report with NO
    capability roles at all against a fully-measured baseline used to exit 0
    (fail open); it must exit 5."""
    candidate = _strong_candidate_report(include_roles=())
    baseline = _baseline_report()
    exit_code, details = promote.evaluate_promotion(candidate, baseline)
    assert exit_code == promote.EXIT_DRIFT == 5
    assert details["drift"]["passed"] is False


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
    candidate_path = tmp_path / "candidate.json"
    baseline_path = tmp_path / "baseline.json"
    candidate_path.write_text(json.dumps(_report(docs=_docs(n_missed=25))), encoding="utf-8")
    baseline_path.write_text(json.dumps(_baseline_report()), encoding="utf-8")

    exit_code = promote.main([str(candidate_path), "--baseline", str(baseline_path)])
    assert exit_code == 2
