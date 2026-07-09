#!/usr/bin/env python3
"""Assemble two promote.py-ready eval reports (candidate + baseline) for the
Recipe-B fine-tune promotion gate.

promote.py compares ONE candidate extract model against ONE baseline extract
model, and its capability-drift gate (exit 5) requires wiki_write / synthesize /
conflict_adjudicate to be present in BOTH reports (absence fails closed). Only
the EXTRACT role was tuned in Recipe-B, so the three prose/adjudicate roles are
the identical stock models (gemma4:12b / gemma4:31b) in both the candidate and
baseline worlds -- measuring them once and injecting the SAME role dict into
both reports is the correct representation (same model, same corpus) and makes
the drift gate pass on equal values rather than on independent flake noise.

This script does NOT run any model. It only re-shapes already-produced
eval-pipeline.py run reports:
  * candidate extract metrics  <- --candidate-run  roles.extract.models[cand]
  * baseline  extract metrics  <- --baseline-run   roles.extract.models[base]
  * shared prose/adjudicate    <- --prose-run       roles.{wiki_write,
                                   synthesize,conflict_adjudicate}
Each output report keeps its own source run's join_keys (corpus_version +
prompt_version), which the provenance gate (exit 6) checks are identical across
candidate and baseline before any statistics run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROSE_ROLES = ("wiki_write", "synthesize", "conflict_adjudicate")


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_model(run: dict, model: str) -> dict:
    models = ((run.get("roles") or {}).get("extract") or {}).get("models") or {}
    if model not in models:
        raise SystemExit(f"model {model!r} not in {sorted(models)}")
    return models[model]


def _prose_roles(run: dict) -> dict:
    roles = run.get("roles") or {}
    out = {}
    for r in PROSE_ROLES:
        if r in roles:
            out[r] = roles[r]
    missing = [r for r in PROSE_ROLES if r not in out]
    if missing:
        raise SystemExit(f"prose-run missing roles: {missing}")
    return out


def build(extract_run: dict, model: str, prose: dict) -> dict:
    return {
        "ts": extract_run.get("ts"),
        "corpus_dir": extract_run.get("corpus_dir"),
        "join_keys": extract_run.get("join_keys"),
        "roles": {
            "extract": {"models": {model: _extract_model(extract_run, model)}},
            **prose,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-run", required=True)
    ap.add_argument("--candidate-model", required=True)
    ap.add_argument("--baseline-run", required=True)
    ap.add_argument("--baseline-model", required=True)
    ap.add_argument("--prose-run", required=True,
                    help="Run report carrying the stock wiki_write/synthesize/"
                         "conflict_adjudicate role results (injected into BOTH).")
    ap.add_argument("--out-candidate", required=True)
    ap.add_argument("--out-baseline", required=True)
    args = ap.parse_args()

    prose = _prose_roles(_load(args.prose_run))
    cand = build(_load(args.candidate_run), args.candidate_model, prose)
    base = build(_load(args.baseline_run), args.baseline_model, prose)

    Path(args.out_candidate).write_text(json.dumps(cand, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    Path(args.out_baseline).write_text(json.dumps(base, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(f"wrote {args.out_candidate}")
    print(f"wrote {args.out_baseline}")
    # Provenance sanity echo (promote.py enforces this).
    ck, bk = cand["join_keys"], base["join_keys"]
    print(f"candidate corpus={ck.get('corpus_version')}")
    print(f"baseline  corpus={bk.get('corpus_version')}")
    print(f"corpus match: {ck.get('corpus_version') == bk.get('corpus_version')}; "
          f"prompt match: {ck.get('prompt_version') == bk.get('prompt_version')}")


if __name__ == "__main__":
    main()
