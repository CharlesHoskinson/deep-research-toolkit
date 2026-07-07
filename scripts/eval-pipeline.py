#!/usr/bin/env python3
"""Live-model eval runner over tests/fixtures/eval-corpus/ (Task 7 of
docs/superpowers/plans/2026-07-06-eval-suite-implementation.md).

Per role (extract, wiki_write, synthesize, conflict_adjudicate) this drives
the same programmatic entry points the production pipeline uses --
extract_claims_to_run, write_wiki_body, synthesize_thesis,
adjudicate_candidates -- against a live `llm.provider: local` endpoint, scores
the result deterministically (evalkit.metrics), and writes one JSON report to
eval-results/run-<UTCstamp>.json plus one summary line per (model, role) to
eval-results/history.jsonl. `--compare <baseline.json>` diffs a run against a
committed baseline and exits 1 on an extract gate_pass_rate/recall regression
beyond `--tolerance`; `--write-baseline` promotes a run to that baseline.

The script's pure logic -- corpus loading/limiting, stratified sampling, the
adjudicate synthetic-candidate protocol, report assembly, history-line
construction, and baseline comparison -- lives in module-level functions with
no backend argument, so tests/unit/test_eval_pipeline.py exercises all of it
(plus the extraction wiring, via a fake in-process backend) without ever
touching a live model. Only backend *construction* (`get_backend`, which opens
a real HTTP client lazily) and the wiki_write/synthesize/conflict_adjudicate
live calls are reserved for the live tier / the Task 8 runbook.

ADJUDICATE SYNTHETIC-CANDIDATE PROTOCOL ("pair-claims-v2"):
corpus-index.json's `contradiction_pairs` are objects {a, b, claim_a,
claim_b, verdict, note} -- two chunk locators, the two gold claim_ids that
STATE the conflicting fact (one per side, pinned explicitly in the index and
enforced by scripts/check-eval-corpus.py: each must exist in its side's
reference-claims.jsonl and cite the pair's own chunk), a gold verdict, and a
human note describing the conflict (e.g. "MCB v2 release year: 2020 vs
2021"). There is no compiled relation graph in this eval (that lives
downstream of the knowledge compiler, out of scope here), so a
`kind: "relation"` candidate is synthesized directly from each pair's two
pinned gold claims instead of mechanically-extracted relations:
  - subject: the pair's full note, slugified and truncated (~60 chars),
    deduplicated against subject collisions across pairs.
  - predicate: the literal string "asserts".
  - objects: [claim_a.claim, claim_b.claim] -- the claim rows looked up by
    id across the corpus's reference-claims.jsonl files.
  - relation_ids: [claim_a, claim_b] (the two pinned gold claim_ids).
A pair whose claim_a/claim_b cannot be looked up is skipped (never a crash)
and recorded in the role result's `pair_warnings` list.
Scoring: model verdict == pair's gold verdict -> 1.0; "insufficient_evidence"
-> 0.5; anything else (wrong verdict, missing/invalid row) -> 0.0. Accuracy is
the mean over all scored pairs. Every adjudicate report row carries
`"adjudicate_protocol": "pair-claims-v2"` so a future, richer candidate
construction is a visible, versioned change rather than a silent drift.
(v1 derived the claims by keyword overlap with the note's pre-colon topic
phrase; that phrase drops the distinguishing values, so on half the corpus
pairs v1 picked claims that never stated the conflicting fact -- v2 pins the
gold explicitly instead.)
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import random
import re
import shutil
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_research_toolkit.config import load_config  # noqa: E402
from deep_research_toolkit.evalkit.bootstrap import paired_bootstrap  # noqa: E402
from deep_research_toolkit.evalkit.flake import run_flaky  # noqa: E402
from deep_research_toolkit.evalkit.metrics import (  # noqa: E402
    bait_rejection,
    extract_metrics,
    marker_rates,
)
from deep_research_toolkit.evalkit.recording import RecordingBackend  # noqa: E402
from deep_research_toolkit.llm.adjudicate import _SYSTEM as _ADJUDICATE_SYSTEM  # noqa: E402
from deep_research_toolkit.llm.adjudicate import adjudicate_candidates  # noqa: E402
from deep_research_toolkit.llm.backend import get_backend  # noqa: E402
from deep_research_toolkit.llm.extract import _SYSTEM as _EXTRACT_SYSTEM  # noqa: E402
from deep_research_toolkit.llm.extract import extract_claims_to_run  # noqa: E402
from deep_research_toolkit.llm.synthesize import _SYSTEM as _SYNTHESIZE_SYSTEM  # noqa: E402
from deep_research_toolkit.llm.synthesize import synthesize_thesis  # noqa: E402
from deep_research_toolkit.llm.wiki import _SYSTEM as _WIKI_SYSTEM  # noqa: E402
from deep_research_toolkit.llm.wiki import write_wiki_body  # noqa: E402

DEFAULT_CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "eval-corpus"
DEFAULT_ROLES = "extract,wiki_write,synthesize,conflict_adjudicate"
DEFAULT_TOLERANCE = 0.03
DEFAULT_RUNS = 3
STRATIFIED_K = 10


# ---------------------------------------------------------------------------
# Small filesystem/JSONL helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Corpus loading / limiting
# ---------------------------------------------------------------------------

def load_claims_by_chunk(corpus_dir: Path) -> dict[str, list[dict]]:
    """Map every chunk locator to the reference claims whose supporting
    evidence cites it, across every doc dir in the corpus. Read-only over the
    corpus -- never mutates it."""
    by_chunk: dict[str, list[dict]] = {}
    for doc_dir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
        for claim in _read_jsonl(doc_dir / "reference-claims.jsonl"):
            for ev in claim.get("supporting_evidence") or []:
                loc = ev.get("locator")
                if loc:
                    by_chunk.setdefault(loc, []).append(claim)
    return by_chunk


def select_docs_for_limit(corpus_dir: Path, limit: int | None) -> list[tuple[str, int | None]]:
    """Doc selection for `--limit` (a total-chunk cap across the whole extract
    run, for smoke runs): docs are visited in sorted order; a doc's chunk cap
    is None (use every chunk) unless the limit is exhausted partway through
    it, in which case that doc gets a numeric cap and every later doc is
    omitted entirely. `limit=None` selects every doc, uncapped."""
    doc_ids = sorted(p.name for p in corpus_dir.iterdir() if p.is_dir())
    if limit is None:
        return [(d, None) for d in doc_ids]
    selected: list[tuple[str, int | None]] = []
    remaining = limit
    for doc_id in doc_ids:
        if remaining <= 0:
            break
        n_chunks = len(_read_jsonl(corpus_dir / doc_id / "chunks.jsonl"))
        if n_chunks == 0:
            continue
        cap = min(n_chunks, remaining)
        selected.append((doc_id, cap if cap < n_chunks else None))
        remaining -= cap
    return selected


def stratified_sample_chunks(index: dict, k: int, seed: int = 7,
                             claims_by_chunk: dict | None = None) -> list[str]:
    """Deterministic round-robin sample of up to `k` chunk locators, drawing
    across every slice tag present in corpus-index.json's `chunks` map so the
    wiki_write/synthesize sample isn't accidentally all-prose. Each slice's
    pool is shuffled with `random.Random(seed)` before the round-robin draw,
    so the same (index, k, seed) always yields the same sample. When
    `claims_by_chunk` is given, chunks with zero gold claims are excluded up
    front -- a prose role can't be exercised on a chunk with nothing to cite,
    so sampling one would silently shrink the effective K."""
    chunks = index.get("chunks") or {}
    by_slice: dict[str, list[str]] = {}
    for locator, meta in chunks.items():
        if claims_by_chunk is not None and not claims_by_chunk.get(locator):
            continue
        for s in (meta or {}).get("slices") or ["prose"]:
            by_slice.setdefault(s, []).append(locator)

    rng = random.Random(seed)
    for pool in by_slice.values():
        pool.sort()
        rng.shuffle(pool)
    slice_names = sorted(by_slice)

    picked: list[str] = []
    seen: set[str] = set()
    max_len = max((len(v) for v in by_slice.values()), default=0)
    round_idx = 0
    while len(picked) < k and round_idx < max_len:
        for s in slice_names:
            pool = by_slice[s]
            if round_idx < len(pool):
                loc = pool[round_idx]
                if loc not in seen:
                    seen.add(loc)
                    picked.append(loc)
                    if len(picked) >= k:
                        break
        round_idx += 1
    return picked


# ---------------------------------------------------------------------------
# Extraction: copy/repoint pattern (mirrors scripts/validate-local-llm.py)
# ---------------------------------------------------------------------------

def _repoint_research_runs(config, path: Path):
    """dataclasses.replace for the real Config; a shallow-copy+setattr
    fallback for lightweight test doubles (SimpleNamespace etc.) that aren't
    dataclass instances."""
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.replace(config, research_runs_path=path)
    cfg2 = copy.copy(config)
    cfg2.research_runs_path = path
    return cfg2


def run_extract_for_doc(doc_dir: Path, config, backend, chunk_limit: int | None = None) -> dict:
    """Copy `doc_dir` into a temp run dir (keeping reference-claims.jsonl --
    extraction only ever writes claims.jsonl, which the corpus doesn't ship,
    so there is nothing to delete first), repoint research_runs_path at the
    copy's parent, and run extract_claims_to_run against it -- the exact
    copy/repoint pattern scripts/validate-local-llm.py uses, so the eval
    exercises the same code path a real run would.

    `chunk_limit` truncates the copy's chunks.jsonl to its first N rows
    (smoke runs); the returned `reference` set is filtered down to only
    claims whose evidence cites a retained chunk, so scoring never penalizes
    the model for claims it was never shown the source text for.

    The original `doc_dir` (part of the read-only committed corpus) is never
    modified -- the temp copy is removed again once extraction completes."""
    doc_id = doc_dir.name
    tmp = Path(tempfile.mkdtemp(prefix="eval-extract-"))
    try:
        work_run = tmp / doc_id
        shutil.copytree(doc_dir, work_run)

        retained_locators: set[str] | None = None
        if chunk_limit is not None:
            chunks = _read_jsonl(work_run / "chunks.jsonl")[:chunk_limit]
            _write_jsonl_rows(work_run / "chunks.jsonl", chunks)
            retained_locators = {c.get("locator") or c.get("node_id") for c in chunks}

        cfg = _repoint_research_runs(config, tmp)
        result = extract_claims_to_run(work_run, "web", cfg, backend)
        produced = _read_jsonl(work_run / "claims.jsonl")
        reference = _read_jsonl(doc_dir / "reference-claims.jsonl")
        if retained_locators is not None:
            reference = [
                r for r in reference
                if all((ev.get("locator") in retained_locators) for ev in (r.get("supporting_evidence") or []))
            ]
        return {
            "produced": produced,
            "dropped": result["dropped"],
            "parse_failures": result["parse_failures"],
            "written": result["written"],
            "reference": reference,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_extract_for_model(corpus_dir: Path, index: dict, config, backend,
                          doc_selection: list[tuple[str, int | None]]) -> dict:
    """Runs run_extract_for_doc across every selected doc, then scores the
    pooled produced/reference/dropped sets with evalkit.metrics.extract_metrics
    plus a corpus-wide bait_rejection rate (design doc §3.4). `per_doc` keeps
    the individual doc-level metrics too, for a future cross-model paired
    bootstrap (see the --models A/B path in main())."""
    all_produced: list[dict] = []
    all_reference: list[dict] = []
    all_dropped: list = []
    parse_failures_total = 0
    bait_scores: list[float] = []
    bait_sources = index.get("bait_sources") or {}
    per_doc: dict[str, dict] = {}

    for doc_id, chunk_cap in doc_selection:
        doc_dir = corpus_dir / doc_id
        result = run_extract_for_doc(doc_dir, config, backend, chunk_limit=chunk_cap)
        all_produced.extend(result["produced"])
        all_reference.extend(result["reference"])
        all_dropped.extend(result["dropped"])
        parse_failures_total += result["parse_failures"]
        per_doc[doc_id] = extract_metrics(
            result["produced"], result["reference"], result["dropped"], result["parse_failures"])

        chunk_texts = {c.get("locator"): c.get("text", "") for c in _read_jsonl(doc_dir / "chunks.jsonl")}
        for bait_loc in chunk_texts:
            source_loc = bait_sources.get(bait_loc)
            if not source_loc or source_loc not in chunk_texts:
                continue  # not a bait chunk, or its source fell outside a --limit truncation
            bait_claims = [
                c for c in result["produced"]
                if any(ev.get("locator") == bait_loc for ev in c.get("supporting_evidence") or [])
            ]
            rate = bait_rejection(bait_claims, chunk_texts[source_loc])
            if rate is not None:
                bait_scores.append(rate)

    metrics = extract_metrics(all_produced, all_reference, all_dropped, parse_failures_total)
    metrics["bait_rejection"] = (sum(bait_scores) / len(bait_scores)) if bait_scores else None
    metrics["docs"] = len(doc_selection)
    metrics["per_doc"] = per_doc
    stats = getattr(backend, "stats", None)
    if stats:
        metrics["backend_stats"] = stats
    return metrics


# ---------------------------------------------------------------------------
# Prose roles (wiki_write / synthesize): flake-run + raw-completion recording
# ---------------------------------------------------------------------------

def _dossier_from_claims(claims: list[dict]) -> dict:
    included = [
        {"claim_id": c.get("claim_id"), "claim": c.get("claim"),
         "evidence": [{"quote": ev.get("quote"), "locator": ev.get("locator")}
                      for ev in c.get("supporting_evidence") or []]}
        for c in claims
    ]
    return {"included": included, "rejected": []}


def run_prose_role_with_backend(role: str, sampled_locators: list[str], claims_by_chunk: dict,
                                backend, runs: int = DEFAULT_RUNS) -> dict:
    """Runs write_wiki_body (role="wiki_write") or synthesize_thesis
    (role="synthesize") once per sampled chunk, `runs` times each via
    evalkit.flake.run_flaky, over a RecordingBackend -- so bare-vs-prefixed
    marker rates (design doc §3.2/§3.4) can be measured on the raw,
    pre-normalization replies the model actually emitted."""
    rec = RecordingBackend(backend)
    coverages: list[float] = []
    per_chunk: dict[str, dict] = {}

    def _attempt(locator: str) -> bool:
        claims = claims_by_chunk.get(locator) or []
        if not claims:
            return False
        if role == "wiki_write":
            out = write_wiki_body(locator, "chunk", claims, rec)
        else:
            dossier = _dossier_from_claims(claims)
            out = synthesize_thesis(f"What does the evidence for {locator} establish?", dossier, rec)
        coverages.append(out["citations"]["coverage"])
        return True

    for locator in sampled_locators:
        per_chunk[locator] = run_flaky(lambda loc=locator: _attempt(loc), runs=runs)

    rates = marker_rates(rec.raw)
    pass_rates = [r["rate"] for r in per_chunk.values()]
    return {
        "per_chunk": per_chunk,
        "mean_pass_rate": (sum(pass_rates) / len(pass_rates)) if pass_rates else None,
        "mean_coverage": (sum(coverages) / len(coverages)) if coverages else None,
        "bare_marker_rate": rates["bare_rate"],
        "marker_counts": {"bare": rates["bare"], "prefixed": rates["prefixed"]},
    }


# ---------------------------------------------------------------------------
# conflict_adjudicate: the "pair-claims-v2" synthetic-candidate protocol
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_SUBJECT_MAX_LEN = 60


def slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-") or "topic"


def load_claims_by_id(corpus_dir: Path) -> dict[str, dict]:
    """Map every reference claim_id to its claim row, across every doc dir in
    the corpus -- the lookup the pair-claims-v2 candidates are built from.
    Read-only over the corpus."""
    by_id: dict[str, dict] = {}
    for doc_dir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
        for claim in _read_jsonl(doc_dir / "reference-claims.jsonl"):
            cid = claim.get("claim_id")
            if cid:
                by_id[cid] = claim
    return by_id


def build_adjudicate_candidates(pairs: list[dict],
                                claims_by_id: dict) -> tuple[list[dict], list[dict], list[str]]:
    """The "pair-claims-v2" protocol documented in the module docstring: one
    kind="relation" candidate per corpus-index.json contradiction pair, built
    from the pair's explicitly pinned gold claims (claim_a/claim_b looked up
    by id). A pair whose pinned claim id cannot be resolved is skipped and
    recorded as a warning -- never a crash. Returns (candidates for
    adjudicate_candidates(), parallel scoring metadata, warnings)."""
    used_subjects: set[str] = set()
    candidates: list[dict] = []
    meta: list[dict] = []
    warnings: list[str] = []
    for pair in pairs:
        note = pair.get("note") or ""
        claim_a = claims_by_id.get(pair.get("claim_a"))
        claim_b = claims_by_id.get(pair.get("claim_b"))
        if claim_a is None or claim_b is None:
            missing = [f"{key}={pair.get(key)!r}" for key in ("claim_a", "claim_b")
                       if claims_by_id.get(pair.get(key)) is None]
            warnings.append(
                f"pair ({pair.get('a')!r}, {pair.get('b')!r}): gold claim id(s) not found in "
                f"reference-claims: {', '.join(missing)} -- pair skipped")
            continue

        subject = slugify(note)[:_SUBJECT_MAX_LEN].rstrip("-")
        base, n = subject, 2
        while subject in used_subjects:
            subject = f"{base}-{n}"
            n += 1
        used_subjects.add(subject)

        candidates.append({
            "kind": "relation",
            "subject": subject,
            "predicate": "asserts",
            "objects": [claim_a.get("claim") or "", claim_b.get("claim") or ""],
            "relation_ids": [pair["claim_a"], pair["claim_b"]],
            "source_ids": [pair["a"], pair["b"]],
        })
        meta.append({"subject": subject, "predicate": "asserts",
                     "gold_verdict": pair["verdict"], "note": note})
    return candidates, meta, warnings


def score_adjudicate(result: dict, meta: list[dict]) -> dict:
    """Scores adjudicate_candidates()'s output against the pair-claims-v2
    metadata: exact verdict match -> 1.0, "insufficient_evidence" -> 0.5,
    anything else (wrong verdict, or no verdict at all -- invalid/parse-failed
    rows never reach `verdicts`) -> 0.0. `accuracy` is the mean over all
    scored pairs (None when there are no pairs to score)."""
    verdict_by_key = {(v.get("subject"), v.get("predicate")): v.get("verdict")
                      for v in result.get("verdicts") or []}
    scores = []
    for m in meta:
        verdict = verdict_by_key.get((m["subject"], m["predicate"]))
        if verdict == m["gold_verdict"]:
            scores.append(1.0)
        elif verdict == "insufficient_evidence":
            scores.append(0.5)
        else:
            scores.append(0.0)
    return {
        "accuracy": (sum(scores) / len(scores)) if scores else None,
        "n_pairs": len(meta),
        "schema_valid": len(result.get("verdicts") or []),
        "schema_invalid": len(result.get("invalid") or []),
        "parse_failures": result.get("parse_failures", 0),
        "adjudicate_protocol": "pair-claims-v2",
    }


def run_adjudicate_with_backend(pairs: list[dict], claims_by_id: dict, backend) -> dict:
    candidates, meta, warnings = build_adjudicate_candidates(pairs, claims_by_id)
    result = adjudicate_candidates(candidates, backend)
    scored = score_adjudicate(result, meta)
    scored["pair_warnings"] = warnings
    return scored


# ---------------------------------------------------------------------------
# Join keys
# ---------------------------------------------------------------------------

def prompt_version() -> str:
    """sha256 over the four judgment-caller roles' raw (unformatted) _SYSTEM
    strings -- a role prompt edit changes this hash, so a report's join keys
    make prompt drift visible instead of silently mixing runs across prompt
    versions (design doc §3.4)."""
    joined = _EXTRACT_SYSTEM + _WIKI_SYSTEM + _SYNTHESIZE_SYSTEM + _ADJUDICATE_SYSTEM
    return "sha256:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()


def get_ollama_version(base_url: str, timeout: float = 2.0) -> str:
    """GET <base_url minus '/v1'>/api/version -- Ollama's native version
    endpoint (the OpenAI-compatible /v1 route carries no version endpoint of
    its own). Never raises: any failure (no endpoint, timeout, non-Ollama
    server, bad JSON) reports "unknown" so a report can always be produced."""
    root = base_url[:-3] if base_url.endswith("/v1") else base_url
    url = root.rstrip("/") + "/api/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return str(data.get("version", "unknown"))
    except Exception:  # noqa: BLE001 -- join key must degrade, never fail the run
        return "unknown"


# ---------------------------------------------------------------------------
# Report assembly / history / baseline comparison
# ---------------------------------------------------------------------------

def build_report(corpus_dir: str, join_keys: dict, role_results: dict, ts: str | None = None) -> dict:
    return {
        "ts": ts or datetime.now(timezone.utc).isoformat(),
        "corpus_dir": str(corpus_dir),
        "join_keys": join_keys,
        "roles": role_results,
    }


#: Bulky per-run detail that belongs in the run report only -- history.jsonl
#: rows are the flat summary time series, so these keys are stripped there.
_HISTORY_EXCLUDED_KEYS = {"per_doc", "missed_claim_ids", "per_chunk"}


def history_lines(report: dict) -> list[dict]:
    """One flat summary row per (model, role) -- extract's per-model metrics
    fan out into one row each; every other role is already one model. Bulky
    detail structures (per_doc, missed_claim_ids, per_chunk) stay in the run
    report only. Appended, never rewritten, to eval-results/history.jsonl as
    the run time series."""
    rows: list[dict] = []
    ts = report.get("ts")
    join_keys = report.get("join_keys") or {}

    def _summary(metrics: dict) -> dict:
        return {k: v for k, v in metrics.items() if k not in _HISTORY_EXCLUDED_KEYS}

    for role, data in (report.get("roles") or {}).items():
        data = data or {}
        if "models" in data:
            for model, metrics in (data.get("models") or {}).items():
                rows.append({"ts": ts, "role": role, "model": model, **join_keys,
                             **_summary(metrics or {})})
        else:
            metrics = {k: v for k, v in data.items() if k != "model"}
            rows.append({"ts": ts, "role": role, "model": data.get("model"), **join_keys,
                         **_summary(metrics)})
    return rows


def compare_reports(current: dict, baseline: dict, tolerance: float = DEFAULT_TOLERANCE) -> list[str]:
    """Diffs the extract role's gate_pass_rate/recall per model against a
    baseline report; a drop beyond `tolerance` is a regression. A metric that
    was numeric in the baseline but is None in the current run also counts as
    a regression -- a metric silently degrading to unmeasurable must not read
    as "no drop". Models absent from the baseline (new additions) are never
    flagged. Returns human-readable regression strings, empty when the run is
    clean."""
    regressions: list[str] = []
    cur_models = (((current.get("roles") or {}).get("extract") or {}).get("models")) or {}
    base_models = (((baseline.get("roles") or {}).get("extract") or {}).get("models")) or {}
    for model, cur_metrics in cur_models.items():
        base_metrics = base_models.get(model)
        if not base_metrics:
            continue
        for metric in ("gate_pass_rate", "recall"):
            cur_v = cur_metrics.get(metric)
            base_v = base_metrics.get(metric)
            if base_v is None:
                continue
            if cur_v is None:
                regressions.append(
                    f"{model}/{metric}: baseline {base_v:.3f} -> current None "
                    f"(metric became unmeasurable)")
                continue
            drop = base_v - cur_v
            if drop > tolerance:
                regressions.append(
                    f"{model}/{metric}: baseline {base_v:.3f} -> current {cur_v:.3f} "
                    f"(dropped {drop:.3f} > tolerance {tolerance:.3f})")
    return regressions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _repoint_extract_model(config, model: str | None):
    """Overrides only the extract role's model (for --models A/B); every
    other role keeps its configured model untouched."""
    if model is None:
        return config
    roles = dict(config.llm_roles)
    spec = dict(roles.get("extract") or {})
    spec["model"] = model
    roles["extract"] = spec
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.replace(config, llm_roles=roles)
    cfg2 = copy.copy(config)
    cfg2.llm_roles = roles
    return cfg2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS_DIR),
                        help="Eval corpus directory (default: tests/fixtures/eval-corpus)")
    parser.add_argument("--roles", default=DEFAULT_ROLES,
                        help=f"Comma-separated roles to run (default: {DEFAULT_ROLES})")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model list overriding the extract role's model, for A/B "
                             "(default: whatever .deepresearch.yml's roles.extract maps to)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"Flake runs per wiki_write/synthesize sampled chunk (default: {DEFAULT_RUNS})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Chunk cap across the extract corpus, for smoke runs (default: no cap)")
    parser.add_argument("--compare", default=None,
                        help="Path to a baseline.json to diff this run against")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help=f"Max allowed extract gate_pass_rate/recall drop vs baseline (default: {DEFAULT_TOLERANCE})")
    parser.add_argument("--write-baseline", action="store_true",
                        help="Copy this run's report to eval-results/baseline.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    corpus_dir = Path(args.corpus).resolve()
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None

    config = load_config()
    if config.llm_provider != "local":
        print(
            f"llm.provider is {config.llm_provider!r}, not 'local'. scripts/eval-pipeline.py exercises "
            "the programmatic local-model path only. Set in .deepresearch.yml:\n"
            "  llm:\n    provider: local\n    local:\n      base_url: http://localhost:11434/v1\n"
            "      model: <model>\nand make sure that endpoint is serving.",
            file=sys.stderr,
        )
        return 1

    index = json.loads((corpus_dir / "corpus-index.json").read_text(encoding="utf-8"))
    claims_by_chunk = load_claims_by_chunk(corpus_dir)
    doc_selection = select_docs_for_limit(corpus_dir, args.limit)

    role_results: dict = {}

    if "extract" in roles:
        extract_models = models or [config.llm_roles["extract"]["model"]]
        per_model: dict[str, dict] = {}
        for model in extract_models:
            cfg = _repoint_extract_model(config, model if models else None)
            backend = get_backend(cfg, role="extract")
            per_model[model] = run_extract_for_model(corpus_dir, index, cfg, backend, doc_selection)
        role_results["extract"] = {"models": per_model}

        if models and len(extract_models) > 1:
            # Cross-model A/B: paired bootstrap over per-doc recall deltas
            # against the first named model (design doc §3.4). n here is the
            # doc count (~10), well below the plan's per-chunk granularity --
            # small-n bootstrap CIs are wide, so treat "significant" as a
            # strong signal and "not significant" as inconclusive rather than
            # as evidence of parity.
            baseline_model = extract_models[0]
            baseline_per_doc = per_model[baseline_model]["per_doc"]
            ab: dict[str, dict | None] = {}
            for model in extract_models[1:]:
                other_per_doc = per_model[model]["per_doc"]
                deltas = []
                for doc_id in sorted(set(baseline_per_doc) & set(other_per_doc)):
                    b = baseline_per_doc[doc_id].get("recall")
                    o = other_per_doc[doc_id].get("recall")
                    if b is not None and o is not None:
                        deltas.append(o - b)
                ab[model] = paired_bootstrap(deltas, b=2000, seed=7) if deltas else None
            role_results["extract"]["ab_vs_" + baseline_model] = ab

    k = min(STRATIFIED_K, len(index.get("chunks") or {}))
    sampled = stratified_sample_chunks(index, k=k, seed=7, claims_by_chunk=claims_by_chunk) if k else []

    if "wiki_write" in roles:
        backend = get_backend(config, role="wiki_write")
        result = run_prose_role_with_backend("wiki_write", sampled, claims_by_chunk, backend, runs=args.runs)
        role_results["wiki_write"] = {"model": config.llm_roles["wiki_write"]["model"], **result}

    if "synthesize" in roles:
        backend = get_backend(config, role="synthesize")
        result = run_prose_role_with_backend("synthesize", sampled, claims_by_chunk, backend, runs=args.runs)
        role_results["synthesize"] = {"model": config.llm_roles["synthesize"]["model"], **result}

    if "conflict_adjudicate" in roles:
        backend = get_backend(config, role="conflict_adjudicate")
        pairs = index.get("contradiction_pairs") or []
        claims_by_id = load_claims_by_id(corpus_dir)
        result = run_adjudicate_with_backend(pairs, claims_by_id, backend)
        role_results["conflict_adjudicate"] = {"model": config.llm_roles["conflict_adjudicate"]["model"], **result}

    join_keys = {
        "prompt_version": prompt_version(),
        "corpus_version": index.get("corpus_version"),
        "ollama_version": get_ollama_version(config.llm_local.get("base_url", "")),
    }
    report = build_report(str(corpus_dir), join_keys, role_results)

    results_dir = Path("eval-results")
    results_dir.mkdir(exist_ok=True)
    ts_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_path = results_dir / f"run-{ts_stamp}.json"
    run_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(results_dir / "history.jsonl", "a", encoding="utf-8") as f:
        for row in history_lines(report):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {run_path}")

    exit_code = 0
    if args.compare:
        baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        regressions = compare_reports(report, baseline, tolerance=args.tolerance)
        if regressions:
            print(f"{len(regressions)} regression(s) vs {args.compare}:")
            for r in regressions:
                print(f"  - {r}")
            exit_code = 1
        else:
            print(f"no regressions vs {args.compare} (tolerance {args.tolerance})")

    if args.write_baseline:
        baseline_path = results_dir / "baseline.json"
        baseline_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote baseline: {baseline_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
