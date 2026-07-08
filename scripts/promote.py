#!/usr/bin/env python3
"""Eval-gated promotion gate (design doc §6.3/§8): `promote <candidate>
--baseline <current>` runs the pre-registered assert gates over TWO ALREADY-
PRODUCED eval reports (the shape `scripts/eval-pipeline.py`'s `build_report`
writes to `eval-results/run-*.json`) and exits with a gate-distinct code.
Live model orchestration -- running the candidate/baseline models, merging
weights, GGUF export -- is OUT OF SCOPE here; `promote` only CONSUMES the two
reports and decides whether promotion is warranted. Every gate function below
is pure (report dicts in, a verdict dict out) so this whole module is
unit-testable with synthetic reports and never touches a backend or a file
other than the two report JSONs it's pointed at.

GATES (design doc §8, "trained adapter -> production adoption"), each mapped
to its own exit code so CI can tell failures apart at a glance:
  0  all gates passed -> promote
  2  recall gate failed (pooled per-claim lower-CI-bound < floor, OR the
     paired per-claim delta's lower CI bound does not clear min_effect)
  3  bait_rejection regressed vs baseline
  4  gate_pass_rate fell below its floor
  5  capability drift on wiki_write/synthesize/conflict_adjudicate (a role
     measured on only ONE side also fails -- the gate cannot confirm
     no-regress without both numbers, so absence fails CLOSED)
  6  provenance incomplete (missing/mismatched corpus_hash or prompt_hash --
     candidate and baseline MUST have been measured on the same corpus, or
     the whole comparison is invalid; checked FIRST, before any statistics)

GATE 1 GRAIN -- PER-CLAIM, NOT PER-DOC (design doc §6.3/§8): the eval report
exposes per-DOC metrics, but each doc's `recalled` count plus its
`missed_claim_ids` list lets the per-CLAIM indicator vector be reconstructed
EXACTLY: a doc with R = recalled + len(missed_claim_ids) reference claims
contributes `recalled` ones and len(missed) zeros. Pooling those indicators
across docs (the provenance gate has already guaranteed both reports scored
the SAME corpus) gives the n~=186-grain sample the spec's CI requirement is
calibrated to. A per-DOC macro mean must NOT be used here: it weights a
1-claim doc equally with a 100-claim doc (macro/micro divergence -- one
100-claim doc at 0.10 recall plus nine 1-claim docs at 1.0 has a 0.91 macro
mean but a true claim-weighted recall of 0.174), and a bootstrap over n=10
doc values is anti-conservative.

The two checks, both over per-claim quantities:
  (a) "clears 0.90": bootstrap CI (evalkit.bootstrap.paired_bootstrap,
      reused as a generic bootstrap-of-a-list) over the POOLED candidate
      indicator vector; the lower bound must be >= the floor.
  (b) "beats baseline": reconstruct the paired per-claim delta for every
      reference claim id -- id missed by baseline only -> +1, missed by
      candidate only -> -1, missed by both -> 0, plus one 0 per claim BOTH
      recalled -- pool across docs, bootstrap, and require the delta CI's
      lower bound > `min_effect` (default 0.01: the calibrated judge-noise
      floor -- spec §6.3, "at n=186 a 0.3-pt delta is judge noise", so a
      sub-point delta must not trigger promotion however tight its CI).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_research_toolkit.evalkit.bootstrap import paired_bootstrap  # noqa: E402

DEFAULT_RECALL_FLOOR = 0.90
DEFAULT_GATE_PASS_FLOOR = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_SEED = 7
#: Minimum paired per-claim delta the "beats baseline" CI lower bound must
#: clear -- the calibrated judge-noise floor (spec §6.3: at n=186 a 0.3-pt
#: delta is judge noise; requiring >1 point keeps hairline "wins" from
#: promoting).
DEFAULT_MIN_EFFECT = 0.01

#: Exit codes, one per gate (see module docstring).
EXIT_PROMOTE = 0
EXIT_RECALL = 2
EXIT_BAIT = 3
EXIT_GATE_PASS = 4
EXIT_DRIFT = 5
EXIT_PROVENANCE = 6

#: Capability-drift roles (design doc §8 gate 4) and the metric each report's
#: `roles.<role>` dict carries -- reused from scripts/eval-pipeline.py's own
#: field names (`mean_pass_rate` for wiki_write/synthesize, `accuracy` for
#: conflict_adjudicate's pair-claims-v2 protocol).
_DRIFT_ROLE_METRIC = {
    "wiki_write": "mean_pass_rate",
    "synthesize": "mean_pass_rate",
    "conflict_adjudicate": "accuracy",
}


# ---------------------------------------------------------------------------
# Report accessors
# ---------------------------------------------------------------------------

def extract_model_metrics(report: dict, model: str | None = None) -> dict:
    """Returns `report["roles"]["extract"]["models"][model]`. When `model` is
    omitted, the sole model present is used; a report with zero or more than
    one extract model and no explicit `model` raises ValueError (ambiguous --
    the caller must say which model is the candidate/baseline)."""
    models = ((report.get("roles") or {}).get("extract") or {}).get("models") or {}
    if model is not None:
        if model not in models:
            raise KeyError(f"model {model!r} not found in report (models present: {sorted(models)})")
        return models[model]
    if len(models) == 1:
        return next(iter(models.values()))
    raise ValueError(
        f"report has {len(models)} extract model(s) ({sorted(models)}) -- pass --model to disambiguate")


def _join_keys(report: dict) -> dict:
    return report.get("join_keys") or {}


# ---------------------------------------------------------------------------
# Gate 1 (exit 2): recall -- pooled PER-CLAIM grain (see module docstring)
# ---------------------------------------------------------------------------

def per_claim_indicators(doc_metrics: dict) -> list[float]:
    """One doc's per-claim recall indicator vector, reconstructed exactly
    from the report's own fields: `recalled` ones + one zero per
    `missed_claim_ids` entry (R = recalled + len(missed) reference claims).
    A doc missing either field contributes nothing (it carried no
    reconstructable claim-level signal)."""
    missed = doc_metrics.get("missed_claim_ids")
    recalled = doc_metrics.get("recalled")
    if missed is None or recalled is None:
        return []
    return [1.0] * int(recalled) + [0.0] * len(missed)


def per_claim_deltas(cand_metrics: dict, base_metrics: dict) -> list[float]:
    """One doc's paired per-claim (candidate - baseline) recall deltas,
    reconstructed by claim id: an id in the BASELINE's missed set only ->
    +1 (candidate recalled what baseline missed); in the CANDIDATE's missed
    set only -> -1; in both -> 0; plus one 0 for every reference claim BOTH
    recalled (R - |union of missed|). Iteration over the union is sorted so
    the delta vector -- and therefore the seeded bootstrap over it -- is
    deterministic across processes."""
    cand_missed_ids = cand_metrics.get("missed_claim_ids")
    base_missed_ids = base_metrics.get("missed_claim_ids")
    cand_recalled = cand_metrics.get("recalled")
    if cand_missed_ids is None or base_missed_ids is None or cand_recalled is None:
        return []
    cand_missed, base_missed = set(cand_missed_ids), set(base_missed_ids)
    r = int(cand_recalled) + len(cand_missed)
    union = cand_missed | base_missed
    deltas: list[float] = []
    for cid in sorted(union):
        if cid in base_missed and cid not in cand_missed:
            deltas.append(1.0)
        elif cid in cand_missed and cid not in base_missed:
            deltas.append(-1.0)
        else:
            deltas.append(0.0)
    deltas.extend([0.0] * max(0, r - len(union)))
    return deltas


def recall_gate(candidate: dict, baseline: dict, model: str | None = None,
                floor: float = DEFAULT_RECALL_FLOOR,
                min_effect: float = DEFAULT_MIN_EFFECT,
                resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES, seed: int = DEFAULT_SEED) -> dict:
    """See the module docstring's GATE 1 GRAIN section: both checks run over
    POOLED PER-CLAIM quantities (n~=186), never a per-doc macro mean.
    `clears_floor` bootstraps the pooled candidate indicator vector and
    requires its CI lower bound >= `floor`; `beats_baseline` bootstraps the
    pooled paired per-claim deltas and requires that CI's lower bound >
    `min_effect` (the judge-noise floor -- a statistically-tight hairline
    delta must still not promote). Returns {"passed", "clears_floor",
    "beats_baseline", "candidate_ci95", "delta_ci95", "floor", "min_effect",
    "n_claims", "n_paired_claims"}."""
    cand = extract_model_metrics(candidate, model)
    base = extract_model_metrics(baseline, model)
    cand_per_doc = cand.get("per_doc") or {}
    base_per_doc = base.get("per_doc") or {}

    indicators: list[float] = []
    for doc_id in sorted(cand_per_doc):
        indicators.extend(per_claim_indicators(cand_per_doc[doc_id]))
    cand_ci = paired_bootstrap(indicators, b=resamples, seed=seed)
    clears_floor = bool(indicators) and cand_ci["ci95"][0] >= floor

    deltas: list[float] = []
    for doc_id in sorted(set(cand_per_doc) & set(base_per_doc)):
        deltas.extend(per_claim_deltas(cand_per_doc[doc_id], base_per_doc[doc_id]))
    delta_ci = paired_bootstrap(deltas, b=resamples, seed=seed)
    beats_baseline = bool(deltas) and delta_ci["ci95"][0] > min_effect

    return {
        "passed": clears_floor and beats_baseline,
        "clears_floor": clears_floor,
        "beats_baseline": beats_baseline,
        "candidate_ci95": cand_ci["ci95"],
        "delta_ci95": delta_ci["ci95"],
        "floor": floor,
        "min_effect": min_effect,
        "n_claims": len(indicators),
        "n_paired_claims": len(deltas),
    }


# ---------------------------------------------------------------------------
# Gate 2 (exit 3): bait_rejection no-regress
# ---------------------------------------------------------------------------

def bait_gate(candidate: dict, baseline: dict, model: str | None = None) -> dict:
    """Any regression (candidate < baseline) blocks -- no tolerance band.
    Missing bait_rejection on either side also blocks: a gate can't confirm
    "does not regress" without both numbers."""
    cand_val = extract_model_metrics(candidate, model).get("bait_rejection")
    base_val = extract_model_metrics(baseline, model).get("bait_rejection")
    if cand_val is None or base_val is None:
        return {"passed": False, "candidate": cand_val, "baseline": base_val,
                "reason": "bait_rejection missing on candidate or baseline"}
    return {"passed": cand_val >= base_val, "candidate": cand_val, "baseline": base_val}


# ---------------------------------------------------------------------------
# Gate 3 (exit 4): gate_pass floor
# ---------------------------------------------------------------------------

def gate_pass_gate(candidate: dict, model: str | None = None,
                   floor: float = DEFAULT_GATE_PASS_FLOOR) -> dict:
    cand_val = extract_model_metrics(candidate, model).get("gate_pass_rate")
    if cand_val is None:
        return {"passed": False, "gate_pass_rate": None, "floor": floor,
                "reason": "gate_pass_rate missing"}
    return {"passed": cand_val >= floor, "gate_pass_rate": cand_val, "floor": floor}


# ---------------------------------------------------------------------------
# Gate 4 (exit 5): capability drift (wiki_write / synthesize / conflict_adjudicate)
# ---------------------------------------------------------------------------

def drift_gate(candidate: dict, baseline: dict) -> dict:
    """FAILS CLOSED on one-sided absence: a capability role measured in one
    report but not the other cannot be confirmed no-regress, so it FAILS the
    gate ("role not measured") -- most importantly, a candidate eval that
    simply skipped wiki_write/synthesize/adjudicate must not sail through the
    forgetting check. Only a role absent from BOTH reports is skipped (that
    role genuinely isn't part of this eval protocol). A role present in both
    with a missing metric VALUE also fails, for the same reason."""
    roles_out: dict = {}
    passed = True
    for role, metric_key in _DRIFT_ROLE_METRIC.items():
        cand_role = (candidate.get("roles") or {}).get(role)
        base_role = (baseline.get("roles") or {}).get(role)
        if cand_role is None and base_role is None:
            roles_out[role] = {"passed": True, "skipped": True,
                               "reason": "role not present in either report"}
            continue
        if cand_role is None or base_role is None:
            side = "candidate" if cand_role is None else "baseline"
            roles_out[role] = {"passed": False,
                               "reason": f"role not measured on {side}"}
            passed = False
            continue
        cand_val = cand_role.get(metric_key)
        base_val = base_role.get(metric_key)
        if cand_val is None or base_val is None:
            roles_out[role] = {"passed": False, "metric": metric_key,
                               "reason": f"{metric_key} missing on candidate or baseline"}
            passed = False
            continue
        role_passed = cand_val >= base_val
        roles_out[role] = {"passed": role_passed, "metric": metric_key,
                           "candidate": cand_val, "baseline": base_val}
        passed = passed and role_passed
    return {"passed": passed, "roles": roles_out}


# ---------------------------------------------------------------------------
# Gate 0 (exit 6): provenance -- same corpus, same prompt version
# ---------------------------------------------------------------------------

def provenance_gate(candidate: dict, baseline: dict) -> dict:
    """`promote` requires candidate and baseline to have been "run on the
    SAME corpus" (design doc §6.3) -- checked via `join_keys.corpus_version`
    (the eval-pipeline.py join key that IS the corpus_hash a registry row
    would be stamped with). Also requires both reports to carry a
    `prompt_version` (the prompt_hash equivalent) -- comparing eval numbers
    across different prompt versions is exactly the "provenance rot" risk
    the design doc calls out (§9). This runs FIRST: if it fails, the other
    gates' statistics are not meaningful."""
    cand_keys, base_keys = _join_keys(candidate), _join_keys(baseline)
    cand_corpus, base_corpus = cand_keys.get("corpus_version"), base_keys.get("corpus_version")
    cand_prompt, base_prompt = cand_keys.get("prompt_version"), base_keys.get("prompt_version")

    reasons = []
    if not cand_corpus:
        reasons.append("candidate report is missing join_keys.corpus_version (corpus_hash)")
    if not base_corpus:
        reasons.append("baseline report is missing join_keys.corpus_version (corpus_hash)")
    if not cand_prompt:
        reasons.append("candidate report is missing join_keys.prompt_version (prompt_hash)")
    if not base_prompt:
        reasons.append("baseline report is missing join_keys.prompt_version (prompt_hash)")
    if cand_corpus and base_corpus and cand_corpus != base_corpus:
        reasons.append(
            f"candidate and baseline were scored on DIFFERENT corpora "
            f"({cand_corpus!r} vs {base_corpus!r}) -- comparison is invalid")

    return {"passed": not reasons, "reasons": reasons,
            "corpus_hash": cand_corpus, "prompt_hash": cand_prompt}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate_promotion(candidate: dict, baseline: dict, model: str | None = None,
                       recall_floor: float = DEFAULT_RECALL_FLOOR,
                       min_effect: float = DEFAULT_MIN_EFFECT,
                       gate_pass_floor: float = DEFAULT_GATE_PASS_FLOOR,
                       resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
                       seed: int = DEFAULT_SEED) -> tuple[int, dict]:
    """Runs every gate in fixed order, stopping at the first failure (each
    gate's own statistics can be nonsensical once an earlier one has
    failed -- e.g. recall stats over mismatched corpora). Returns
    `(exit_code, details)`; `details` always carries every gate that WAS run
    (not just the failing one), keyed by gate name, plus `"exit_code"`."""
    details: dict = {}

    provenance = provenance_gate(candidate, baseline)
    details["provenance"] = provenance
    if not provenance["passed"]:
        details["exit_code"] = EXIT_PROVENANCE
        return EXIT_PROVENANCE, details

    recall = recall_gate(candidate, baseline, model=model, floor=recall_floor,
                         min_effect=min_effect, resamples=resamples, seed=seed)
    details["recall"] = recall
    if not recall["passed"]:
        details["exit_code"] = EXIT_RECALL
        return EXIT_RECALL, details

    bait = bait_gate(candidate, baseline, model=model)
    details["bait"] = bait
    if not bait["passed"]:
        details["exit_code"] = EXIT_BAIT
        return EXIT_BAIT, details

    gate_pass = gate_pass_gate(candidate, model=model, floor=gate_pass_floor)
    details["gate_pass"] = gate_pass
    if not gate_pass["passed"]:
        details["exit_code"] = EXIT_GATE_PASS
        return EXIT_GATE_PASS, details

    drift = drift_gate(candidate, baseline)
    details["drift"] = drift
    if not drift["passed"]:
        details["exit_code"] = EXIT_DRIFT
        return EXIT_DRIFT, details

    details["exit_code"] = EXIT_PROMOTE
    return EXIT_PROMOTE, details


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidate", help="Path to the candidate's eval-pipeline.py report JSON")
    parser.add_argument("--baseline", required=True, help="Path to the baseline's eval report JSON")
    parser.add_argument("--model", default=None,
                        help="extract model name to compare, if a report has more than one")
    parser.add_argument("--recall-floor", type=float, default=DEFAULT_RECALL_FLOOR)
    parser.add_argument("--min-effect", type=float, default=DEFAULT_MIN_EFFECT,
                        help="Minimum paired per-claim recall delta the beats-baseline CI lower "
                             f"bound must clear (judge-noise floor; default {DEFAULT_MIN_EFFECT})")
    parser.add_argument("--gate-pass-floor", type=float, default=DEFAULT_GATE_PASS_FLOOR)
    parser.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json", action="store_true", help="Print the full gate detail as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))

    exit_code, details = evaluate_promotion(
        candidate, baseline, model=args.model, recall_floor=args.recall_floor,
        min_effect=args.min_effect, gate_pass_floor=args.gate_pass_floor,
        resamples=args.resamples, seed=args.seed)

    if args.json:
        print(json.dumps(details, indent=2, ensure_ascii=False))
    else:
        verdict = "PROMOTE" if exit_code == EXIT_PROMOTE else "REJECT"
        print(f"{verdict} (exit {exit_code})")
        for gate_name, gate_result in details.items():
            if gate_name == "exit_code":
                continue
            print(f"  {gate_name}: {'pass' if gate_result.get('passed') else 'FAIL'}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
