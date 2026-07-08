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
  2  recall gate failed (lower-CI-bound < floor, OR does not beat baseline)
  3  bait_rejection regressed vs baseline
  4  gate_pass_rate fell below its floor
  5  capability drift on wiki_write/synthesize/conflict_adjudicate
  6  provenance incomplete (missing/mismatched corpus_hash or prompt_hash --
     candidate and baseline MUST have been measured on the same corpus, or
     the whole comparison is invalid; checked FIRST, before any statistics)

DESIGN DECISION (spec ambiguity -- see the design doc §6.3 gate 1 wording
"recall lower-CI-bound clears 0.90 AND beats baseline", immediately followed
by "bootstrap 95% CI ... on the per-chunk delta"): a CI computed on a DELTA
straddles zero by construction and cannot itself "clear 0.90", so gate 1 is
implemented as two independent checks that together satisfy both clauses:
  (a) "clears 0.90": bootstrap the CANDIDATE's own per-doc recall values
      (evalkit.bootstrap.paired_bootstrap, reused as a generic bootstrap-of-
      a-list -- it does not care whether the list is deltas or raw values)
      and require the resampled-mean CI's LOWER bound >= the floor.
  (b) "beats baseline": bootstrap the PER-DOC (candidate - baseline) deltas
      on the docs both reports scored, and require that CI's lower bound to
      be > 0 (strictly positive -- the same "CI excludes zero, in favor of
      the candidate" criterion scripts/eval-pipeline.py's --models A/B path
      already uses).
Per-CHUNK granularity per the design doc requires per-chunk metrics that the
current eval-pipeline.py report does not expose (it reports per-DOC recall in
`roles.extract.models.<model>.per_doc`); this reuses the finest granularity
the existing report actually carries. Revisit if a per-chunk report ever
ships.
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
# Gate 1 (exit 2): recall
# ---------------------------------------------------------------------------

def recall_gate(candidate: dict, baseline: dict, model: str | None = None,
                floor: float = DEFAULT_RECALL_FLOOR,
                resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES, seed: int = DEFAULT_SEED) -> dict:
    """See the module docstring's DESIGN DECISION for what "clears 0.90 AND
    beats baseline" is computed as. Returns
    {"passed", "clears_floor", "beats_baseline", "candidate_ci95",
     "delta_ci95", "n_docs", "n_common_docs"}."""
    cand = extract_model_metrics(candidate, model)
    base = extract_model_metrics(baseline, model)
    cand_per_doc = cand.get("per_doc") or {}
    base_per_doc = base.get("per_doc") or {}

    cand_recalls = [m["recall"] for m in cand_per_doc.values() if m.get("recall") is not None]
    cand_ci = paired_bootstrap(cand_recalls, b=resamples, seed=seed)
    clears_floor = cand_ci["ci95"][0] >= floor

    common_docs = sorted(set(cand_per_doc) & set(base_per_doc))
    deltas = [
        cand_per_doc[d]["recall"] - base_per_doc[d]["recall"]
        for d in common_docs
        if cand_per_doc[d].get("recall") is not None and base_per_doc[d].get("recall") is not None
    ]
    delta_ci = paired_bootstrap(deltas, b=resamples, seed=seed)
    beats_baseline = delta_ci["ci95"][0] > 0

    return {
        "passed": clears_floor and beats_baseline,
        "clears_floor": clears_floor,
        "beats_baseline": beats_baseline,
        "candidate_ci95": cand_ci["ci95"],
        "delta_ci95": delta_ci["ci95"],
        "floor": floor,
        "n_docs": len(cand_recalls),
        "n_common_docs": len(deltas),
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
    """A role absent from EITHER report is skipped (not run this eval, not a
    regression signal) rather than failing the gate -- but a role present in
    both with a missing metric value DOES fail (can't confirm no-regress)."""
    roles_out: dict = {}
    passed = True
    for role, metric_key in _DRIFT_ROLE_METRIC.items():
        cand_role = (candidate.get("roles") or {}).get(role)
        base_role = (baseline.get("roles") or {}).get(role)
        if cand_role is None or base_role is None:
            roles_out[role] = {"passed": True, "skipped": True,
                               "reason": "role not present in both reports"}
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
                         resamples=resamples, seed=seed)
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
        gate_pass_floor=args.gate_pass_floor, resamples=args.resamples, seed=args.seed)

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
