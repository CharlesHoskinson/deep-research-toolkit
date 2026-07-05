#!/usr/bin/env python3
"""Manual validation harness for the local-LLM claim-extraction path.

NOT run in CI. This needs two things CI does not have:
  1. A .deepresearch.yml (in this repo or a parent directory) with
     `llm: {provider: local, local: {base_url: ..., model: ...}}`.
  2. A live OpenAI-compatible endpoint serving that model (e.g.
     `ollama serve` with Ornith-1.0-9B pulled).

What it does:
  - Copies the reference run (default: tests/fixtures/reference-run-hydra-settlement)
    into a temp directory, minus its claims.jsonl, so the fixture is never touched.
  - Runs extract_claims_to_run against the copy: the local model reads
    chunks.jsonl and proposes claims; any claim whose evidence quote is not a
    verbatim substring of the source is auto-dropped (the model can only
    under-produce, never corrupt).
  - Diffs the produced claims.jsonl against the reference claims.jsonl and
    reports how many reference claims were recovered (matched by overlapping
    verbatim evidence quotes), plus the verbatim-pass/drop summary.

Usage:
  python scripts/validate-local-llm.py
  python scripts/validate-local-llm.py --run-dir path/to/run --producer pdf
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_research_toolkit.config import load_config  # noqa: E402
from deep_research_toolkit.llm.backend import LLMBackendNotConfigured, get_backend  # noqa: E402
from deep_research_toolkit.llm.extract import extract_claims_to_run  # noqa: E402

DEFAULT_RUN = REPO_ROOT / "tests" / "fixtures" / "reference-run-hydra-settlement"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _quotes(claim: dict) -> list[str]:
    return [ev.get("quote") or "" for ev in claim.get("supporting_evidence") or []]


def _recovered(reference: list[dict], produced: list[dict]) -> tuple[list[dict], list[dict]]:
    """A reference claim counts as recovered when any produced claim carries an
    evidence quote that overlaps one of its quotes (either is a substring of the
    other). Both sides are verbatim-gated, so quote overlap is a strong match."""
    produced_quotes = [q for c in produced for q in _quotes(c) if q]
    hit, missed = [], []
    for ref in reference:
        ref_quotes = [q for q in _quotes(ref) if q]
        if any(q in p or p in q for q in ref_quotes for p in produced_quotes):
            hit.append(ref)
        else:
            missed.append(ref)
    return hit, missed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN),
                        help="Run directory containing chunks.jsonl (default: hydra fixture reference run)")
    parser.add_argument("--producer", choices=["pdf", "web"], default="pdf",
                        help="Producer type of the run (default: pdf, matching the fixture)")
    parser.add_argument("--reference-claims", default=None,
                        help="claims.jsonl to diff against (default: <run-dir>/claims.jsonl)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "chunks.jsonl").is_file():
        sys.exit(f"no chunks.jsonl in {run_dir}")
    reference_path = Path(args.reference_claims) if args.reference_claims else run_dir / "claims.jsonl"

    config = load_config()
    if config.llm_provider != "local":
        sys.exit(
            f"llm.provider is {config.llm_provider!r}, not 'local'. This harness exercises the "
            "programmatic local-model path only. Set in .deepresearch.yml:\n"
            "  llm:\n    provider: local\n    local:\n      base_url: http://localhost:11434/v1\n"
            "      model: Ornith-1.0-9B\n"
            "and make sure that endpoint is serving."
        )

    # Work on a copy so the fixture run (and its committed claims.jsonl) is never touched.
    tmp = Path(tempfile.mkdtemp(prefix="validate-local-llm-"))
    if args.producer == "pdf":
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        copy_name = manifest.get("document_id", run_dir.name)
    else:
        copy_name = run_dir.name
    work_run = tmp / copy_name
    shutil.copytree(run_dir, work_run)
    (work_run / "claims.jsonl").unlink(missing_ok=True)

    # Point the verbatim gate's source lookup at the copy.
    if args.producer == "pdf":
        config = dataclasses.replace(config, pdf_runs_path=tmp)
    else:
        config = dataclasses.replace(config, research_runs_path=tmp)

    try:
        # role="extract" mirrors the production call in extract_claims.py, so
        # the harness exercises the same roles.extract config (non-thinking,
        # temp 0, json response format) that a real run would use.
        backend = get_backend(config, role="extract")
        result = extract_claims_to_run(work_run, args.producer, config, backend)
    except LLMBackendNotConfigured as e:
        sys.exit(str(e))

    produced = _read_jsonl(work_run / "claims.jsonl")
    reference = _read_jsonl(reference_path)

    print(f"run: {run_dir}")
    print(f"model output: {result['written']} claim(s) passed the verbatim gate, "
          f"{len(result['dropped'])} dropped")
    if result["dropped"]:
        print(f"dropped claim_ids (non-verbatim or missing evidence): {result['dropped']}")
    if result.get("parse_failures"):
        print(f"parse failures (batches that never yielded parseable JSON): {result['parse_failures']}")

    stats = getattr(backend, "stats", None)
    if stats and stats.get("calls"):
        print(f"backend stats: {stats['calls']} call(s), "
              f"{stats['prompt_tokens']} prompt + {stats['completion_tokens']} completion tokens, "
              f"{stats['seconds']:.1f}s total ({stats['seconds']/stats['calls']:.1f}s/call)")

    if reference:
        hit, missed = _recovered(reference, produced)
        print(f"reference diff: recovered {len(hit)}/{len(reference)} reference claim(s) "
              f"from {reference_path}")
        for ref in missed:
            print(f"  MISSED: [{ref.get('claim_id')}] {ref.get('claim')}")
    else:
        print(f"reference diff: skipped (no reference claims at {reference_path})")

    print(f"produced claims left for inspection: {work_run / 'claims.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
