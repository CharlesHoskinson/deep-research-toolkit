#!/usr/bin/env python3
"""Pooled-gold generator (Task 12 of the Phase-1 pipeline-hardening plan).

NOT run in CI, and its live execution is not unit-tested (only the pure
`evalkit.pool_gold` helper is). This needs what CI does not have:
  1. A .deepresearch.yml with `llm: {provider: local, local: {base_url: ...}}`.
  2. A live OpenAI-compatible endpoint (e.g. `ollama serve`) with BOTH pooled
     models pulled (default: gemma4:e4b and qwen3:30b-a3b-instruct-2507-q4_K_M,
     the two recall-leading extractors).

What it does, per doc under tests/fixtures/eval-corpus/:
  - Copies the doc into a temp run dir (the committed chunks.jsonl /
    reference-claims.jsonl are never touched -- extraction writes only the
    copy's claims.jsonl), repoints research_runs_path at the copy's parent,
    and runs extract_claims_to_run with the extract role's model forced to
    each pooled model in turn -- the same copy/repoint pattern as
    scripts/validate-local-llm.py and scripts/eval-pipeline.py, and the same
    role-model override mechanism as eval-pipeline's --models A/B path.
  - Reads back each model's gate-passing claims.jsonl, namespaces every
    claim_id with that model's slug (evalkit.namespace_claim_ids -- Anomaly-1
    fix: both pooled models mint bNN_c_NNNN-style ids independently, and
    pool_gold dedups by claim TEXT not id, so without this an id can collide
    across models within a doc), then pools the namespaced lists with
    evalkit.pool_gold (union + dedup by selfconsistency.claim_key,
    min_support=1: any model's gate-passed claim is gold).
  - Writes the pooled set to <doc>/pooled-gold.jsonl -- the ONLY file this
    script adds to the corpus. This set is the fixed eval denominator and the
    Phase-2 SFT target.
  - Prints a per-doc summary plus the corpus_version and a sha256 over all
    pooled-gold.jsonl files (doc order), so a rebuild is a visible,
    hash-verifiable change.

Skips-with-reason (exit 0, like the live test tier) when the provider isn't
'local', the endpoint doesn't answer, or a pooled model isn't pulled.

`--renamespace` runs a separate, offline migration instead of extraction: it
rewrites claim_id in every already-committed <doc>/pooled-gold.jsonl (rows
pooled before namespace_claim_ids existed) so no id repeats within its doc --
see renamespace_pooled_gold's docstring for how source-model attribution is
recovered without a per-row provenance field. No live endpoint or config
needed; it never touches chunks.jsonl/reference-claims.jsonl.

Usage:
  python scripts/build-pooled-gold.py
  python scripts/build-pooled-gold.py --corpus path/to/corpus --models m1,m2
  python scripts/build-pooled-gold.py --renamespace
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_research_toolkit.config import load_config  # noqa: E402
from deep_research_toolkit.evalkit import (  # noqa: E402
    duplicate_claim_id_groups,
    namespace_claim_ids,
    pool_gold,
    renamespace_duplicate_claim_ids,
)
from deep_research_toolkit.llm.backend import LLMBackendNotConfigured, get_backend  # noqa: E402
from deep_research_toolkit.llm.extract import extract_claims_to_run  # noqa: E402

DEFAULT_CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "eval-corpus"
#: qwen3:30b-a3b (bare) was never pulled -- the endpoint only ever served the
#: instruct-2507 quant below, so that bare tag would always skip. Matches the
#: model actually recorded in the committed pool (see LEGACY_POOL_MODELS).
DEFAULT_MODELS = "gemma4:e4b,qwen3:30b-a3b-instruct-2507-q4_K_M"
#: The exact models, in pool order, that built the CURRENTLY COMMITTED
#: tests/fixtures/eval-corpus/*/pooled-gold.jsonl files -- back when
#: DEFAULT_MODELS still named the unpulled "qwen3:30b-a3b" tag. --renamespace
#: defaults to this (not DEFAULT_MODELS above) so migrating the existing
#: corpus attributes each disambiguated legacy row to the model that
#: actually produced it, not to today's corrected default.
LEGACY_POOL_MODELS = "gemma4:e4b,qwen3:30b-a3b"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _repoint_research_runs(config, path: Path):
    """dataclasses.replace for the real Config; shallow-copy+setattr fallback
    for lightweight doubles (mirrors scripts/eval-pipeline.py)."""
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.replace(config, research_runs_path=path)
    cfg2 = copy.copy(config)
    cfg2.research_runs_path = path
    return cfg2


def _repoint_extract_model(config, model: str):
    """Overrides only the extract role's model; every other role keeps its
    configured model untouched (mirrors scripts/eval-pipeline.py's --models
    A/B mechanism)."""
    roles = dict(config.llm_roles)
    spec = dict(roles.get("extract") or {})
    spec["model"] = model
    roles["extract"] = spec
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.replace(config, llm_roles=roles)
    cfg2 = copy.copy(config)
    cfg2.llm_roles = roles
    return cfg2


def _skip(reason: str) -> int:
    """Skip-with-reason, like the live tier's conftest gate: a clear message
    and a clean exit, never a crash/failure."""
    print(f"SKIP: {reason}")
    return 0


def _pulled_models(base_url: str, timeout: float = 2.0) -> set[str] | None:
    """Model names pulled on an Ollama endpoint (GET /api/tags). Returns None
    when the endpoint answers but isn't Ollama-shaped (can't verify -- caller
    proceeds and lets the backend error surface naturally)."""
    root = base_url[:-3] if base_url.endswith("/v1") else base_url
    url = root.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {m.get("name", "") for m in data.get("models") or []}
    except Exception:  # noqa: BLE001 -- unverifiable is not a skip on its own
        return None


def _dup_counts(rows: list[dict]) -> dict:
    """{"dup_ids": distinct claim_ids appearing >1x, "dup_rows": total rows
    among those groups} -- the --renamespace before/after report."""
    groups = duplicate_claim_id_groups(rows)
    return {"dup_ids": len(groups), "dup_rows": sum(len(v) for v in groups.values())}


def renamespace_pooled_gold(corpus_dir: Path, models: list[str]) -> dict:
    """--renamespace driver: walks every <doc>/pooled-gold.jsonl under
    corpus_dir, rewrites duplicate claim_ids via
    evalkit.renamespace_duplicate_claim_ids, and reports before/after
    duplicate counts. Pure file I/O over the committed corpus -- no live
    endpoint, no config, so it runs even when llm.provider isn't 'local'.
    Idempotent: a doc with no collisions (before["dup_ids"] == 0) is never
    rewritten, so re-running after a clean migration touches no files.

    Returns {"docs": {doc_id: {"before": counts, "after": counts}},
    "total_before": int, "total_after": int} (dup_ids summed across docs)."""
    summary: dict = {"docs": {}, "total_before": 0, "total_after": 0}
    for doc_dir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
        path = doc_dir / "pooled-gold.jsonl"
        if not path.is_file():
            continue
        rows = _read_jsonl(path)
        before = _dup_counts(rows)
        new_rows = renamespace_duplicate_claim_ids(rows, models)
        after = _dup_counts(new_rows)
        if after["dup_ids"]:
            raise RuntimeError(
                f"{doc_dir.name}: {after['dup_ids']} duplicate claim_id(s) remain after "
                f"renamespacing with models={models} -- refusing to write (a duplicate group "
                f"deeper than the model list can disambiguate; list more --legacy-models)")
        if before["dup_ids"]:
            _write_jsonl_rows(path, new_rows)
        summary["docs"][doc_dir.name] = {"before": before, "after": after}
        summary["total_before"] += before["dup_ids"]
        summary["total_after"] += after["dup_ids"]
    return summary


def extract_doc_with_model(doc_dir: Path, config, backend) -> list[dict]:
    """Copy `doc_dir` into a temp run dir, repoint research_runs_path at the
    copy's parent, run extract_claims_to_run against the copy, and return its
    gate-passing claims. The committed doc dir is never written to."""
    tmp = Path(tempfile.mkdtemp(prefix="pooled-gold-"))
    try:
        work_run = tmp / doc_dir.name
        shutil.copytree(doc_dir, work_run)
        (work_run / "claims.jsonl").unlink(missing_ok=True)
        cfg = _repoint_research_runs(config, tmp)
        extract_claims_to_run(work_run, "web", cfg, backend)
        return _read_jsonl(work_run / "claims.jsonl")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS_DIR),
                        help="Eval corpus directory (default: tests/fixtures/eval-corpus)")
    parser.add_argument("--models", default=DEFAULT_MODELS,
                        help=f"Comma-separated extract models to pool (default: {DEFAULT_MODELS})")
    parser.add_argument("--renamespace", action="store_true",
                        help="Idempotent migration (Anomaly 1): rewrite claim_id in every "
                             "committed <doc>/pooled-gold.jsonl under --corpus so no id "
                             "repeats within its doc, then exit. Offline -- no live endpoint "
                             "or config needed; never runs extraction.")
    parser.add_argument("--legacy-models", default=LEGACY_POOL_MODELS,
                        help="Comma-separated models, in pool order, that built the "
                             f"pooled-gold.jsonl files being migrated by --renamespace "
                             f"(default: {LEGACY_POOL_MODELS})")
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus).resolve()

    if args.renamespace:
        legacy_models = [m.strip() for m in args.legacy_models.split(",") if m.strip()]
        if not legacy_models:
            return _skip("no --legacy-models given to renamespace against")
        summary = renamespace_pooled_gold(corpus_dir, legacy_models)
        for doc_id, counts in summary["docs"].items():
            b, a = counts["before"], counts["after"]
            print(f"{doc_id}: {b['dup_ids']} dup id(s) ({b['dup_rows']} rows) -> "
                  f"{a['dup_ids']} dup id(s) ({a['dup_rows']} rows)")
        print(f"total: {summary['total_before']} -> {summary['total_after']} duplicate claim_id "
              f"group(s) across {len(summary['docs'])} doc(s) under {corpus_dir}")
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        return _skip("no models given to pool")

    config = load_config()
    if config.llm_provider != "local":
        return _skip(
            f"llm.provider is {config.llm_provider!r}, not 'local'. Pooled-gold generation "
            "needs a live local endpoint with the pooled models pulled. Set in .deepresearch.yml:\n"
            "  llm:\n    provider: local\n    local:\n      base_url: http://localhost:11434/v1\n"
            "      model: <model>\nand make sure that endpoint is serving.")

    base_url = config.llm_local.get("base_url", "")
    version_url = (base_url[:-3] if base_url.endswith("/v1") else base_url).rstrip("/") + "/api/version"
    try:
        urllib.request.urlopen(version_url, timeout=2)
    except Exception as e:  # noqa: BLE001 -- any failure means "no live endpoint"
        return _skip(f"no live endpoint at {base_url}: {e}")

    pulled = _pulled_models(base_url)
    if pulled is not None:
        missing = [m for m in models if m not in pulled and f"{m}:latest" not in pulled]
        if missing:
            return _skip(f"model(s) not pulled on {base_url}: {', '.join(missing)} "
                         f"(pull them, e.g. `ollama pull {missing[0]}`)")

    doc_dirs = sorted(p for p in corpus_dir.iterdir() if p.is_dir())
    if not doc_dirs:
        return _skip(f"no doc dirs under {corpus_dir}")

    try:
        backends = {m: get_backend(_repoint_extract_model(config, m), role="extract") for m in models}
    except LLMBackendNotConfigured as e:
        return _skip(str(e))

    digest = hashlib.sha256()
    total_pooled = 0
    for doc_dir in doc_dirs:
        per_model: dict[str, list[dict]] = {}
        for model in models:
            cfg = _repoint_extract_model(config, model)
            claims = extract_doc_with_model(doc_dir, cfg, backends[model])
            # Anomaly-1 fix: namespace ids by source model BEFORE pooling, so
            # two models minting the same raw bNN_c_NNNN id can never collide
            # in the pooled output (pool_gold dedups by claim text, not id).
            per_model[model] = namespace_claim_ids(claims, model)
        pooled = pool_gold([per_model[m] for m in models])
        out_path = doc_dir / "pooled-gold.jsonl"
        _write_jsonl_rows(out_path, pooled)
        digest.update(out_path.read_bytes())
        total_pooled += len(pooled)
        counts = ", ".join(f"{m}: {len(per_model[m])}" for m in models)
        print(f"{doc_dir.name}: {counts} -> pooled {len(pooled)} ({out_path})")

    index_path = corpus_dir / "corpus-index.json"
    corpus_version = None
    if index_path.is_file():
        corpus_version = json.loads(index_path.read_text(encoding="utf-8")).get("corpus_version")
    print(f"corpus_version: {corpus_version}")
    print(f"pooled-gold: {total_pooled} claim(s) across {len(doc_dirs)} doc(s), "
          f"sha256:{digest.hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
