#!/usr/bin/env python3
"""CLI wrapper around `deep_research_toolkit.tunekit.dataset.build_sft_dataset`
(design doc §6.1): reads a training corpus (the SAME per-doc `chunks.jsonl`
shape `tests/fixtures/eval-corpus` uses, but a DIFFERENT, non-test corpus --
never point this at the eval corpus itself), builds the hard contamination
guard from the eval corpus, runs the DART k-escalation sampler with a
teacher-per-slice router, and writes `train.jsonl` / `val.jsonl` /
`manifest.json` / `escalation-log.json` to an output directory.

TEACHER WIRING: local-model routes -- `tunekit.dataset.DEFAULT_ROUTER_TABLE` /
`DEFAULT_TEACHER_ROUTE` now name the REAL pulled Ollama tags directly
(`gemma4:e4b`, `qwen3:30b-a3b-instruct-2507-q4_K_M` -- a live run found the
router naming abstract labels like "e4b" that don't match any tag Ollama
actually serves, 404ing on the very first call; see that module's
DEFAULT_ROUTER_TABLE comment) -- are wired automatically against whatever
`.deepresearch.yml` resolves for the `extract` role, repointed at each
route's model tag. `--base-model-override ROUTE=MODEL` (repeatable) is the
escape hatch for a host whose pulled tags differ from those defaults: ROUTE
must name one of `tunekit.dataset`'s wired local route tags (see
`_LOCAL_MODEL_ROUTES` below; an override naming anything else is logged and
ignored, not silently swallowed) and MODEL is the real tag to actually
request from the endpoint instead, e.g.
`--base-model-override gemma4:e4b=gemma4:e4b-q8_0` to try a different
quantization without editing source. The "frontier" bait-slice teacher
(design doc §6.1: a Claude/GPT-class API teacher for the bait slice, since
every local model caps ~0.86-0.96 under the 0.95 bait bar) is DELIBERATELY
NOT WIRED HERE -- it needs API-key plumbing out of scope for this
meta-infra deliverable. Passing `--teachers-module a.module.path` (exposing
`get_teachers(config, producer) -> dict[str, TeacherFn]`) overrides/extends
the default wiring, so a caller CAN supply a frontier teacher without
changing this script.

CHECKPOINT/RESUME: every run appends each completed chunk's accepted
examples to `<out-dir>/accepted.partial.jsonl` and its escalation-log row to
`<out-dir>/progress.json` as it goes (both flushed per line), so a crash
partway through (a teacher call raising, the process being killed, ...)
never loses already-gated work. Re-running the SAME command with `--resume`
added skips every chunk already recorded in `<out-dir>/progress.json`
(its teacher is never called again) and reseeds the accumulator from
`accepted.partial.jsonl` before continuing -- final assembly (dedup, split,
manifest) still runs once at the end over the FULL accumulated set. Without
`--resume`, a fresh run first clears any stale checkpoint files already
sitting in `--out-dir` rather than silently inheriting a previous crash's
partial state.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_research_toolkit.config import load_config  # noqa: E402
from deep_research_toolkit.llm.backend import get_backend  # noqa: E402
from deep_research_toolkit.llm.extract import build_extraction_prompt  # noqa: E402
from deep_research_toolkit.tunekit.dataset import (  # noqa: E402
    DEFAULT_K_LADDER,
    DEFAULT_MAX_COMPLETIONS_PER_CHUNK,
    DEFAULT_ROUTER_TABLE,
    DEFAULT_TEACHER_ROUTE,
    DEFAULT_YIELD_FLOOR,
    BannedTeacherError,
    ContaminationError,
    build_sft_dataset,
    load_contamination_index,
)

DEFAULT_EVAL_CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "eval-corpus"

#: Router model names this script knows how to wire to a local backend.
#: "frontier" is intentionally absent -- see the module docstring.
_LOCAL_MODEL_ROUTES = {route.model for route in (*DEFAULT_ROUTER_TABLE.values(), DEFAULT_TEACHER_ROUTE)
                       if route.model != "frontier"}


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


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_training_chunks(training_corpus_dir: Path) -> list[dict]:
    """Reads every `<doc>/chunks.jsonl` under `training_corpus_dir` (the
    same per-doc-directory shape as `tests/fixtures/eval-corpus`), in
    sorted doc order, tagging each chunk's slices from the doc's own
    `corpus-index.json` if one exists alongside it (optional -- a training
    corpus need not carry the full eval-index machinery, just chunk text +
    locator; DART difficulty tagging degrades to "no slice tags" without
    it)."""
    chunks: list[dict] = []
    index_path = training_corpus_dir / "corpus-index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    slices_by_locator = {loc: (meta or {}).get("slices") or [] for loc, meta in (index.get("chunks") or {}).items()}

    for doc_dir in sorted(p for p in training_corpus_dir.iterdir() if p.is_dir()):
        for chunk in _read_jsonl(doc_dir / "chunks.jsonl"):
            locator = chunk.get("locator") or chunk.get("node_id")
            if locator in slices_by_locator:
                chunk = {**chunk, "slices": slices_by_locator[locator]}
            chunks.append(chunk)
    return chunks


def _repoint_model(config, model: str):
    """dataclasses.replace for the real Config; shallow-copy fallback for
    lightweight test doubles -- the same pattern scripts/eval-pipeline.py's
    `_repoint_extract_model` uses, duplicated here (not imported) since that
    function is a private helper of a hyphenated sibling script."""
    roles = dict(config.llm_roles)
    spec = dict(roles.get("extract") or {})
    spec["model"] = model
    roles["extract"] = spec
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.replace(config, llm_roles=roles)
    cfg2 = copy.copy(config)
    cfg2.llm_roles = roles
    return cfg2


def make_local_backend_teacher(config, model: str, producer: str):
    """Wraps `llm.backend.get_backend` as a TeacherFn: renders the chunk
    batch through `build_extraction_prompt` (byte-identical to production)
    and issues `k` independent `backend.complete()` calls at `temperature`,
    returning their raw replies for `dataset.py`'s
    `parse_claims_response` + span gate to filter."""
    cfg = _repoint_model(config, model)
    backend = get_backend(cfg, role="extract")

    def teacher(chunk_batch: list[dict], k: int, temperature: float) -> list[str]:
        system, user = build_extraction_prompt(
            chunk_batch, producer=producer, thinking=getattr(backend, "thinking", True))
        return [backend.complete(system, user, temperature=temperature) for _ in range(k)]

    return teacher


def parse_base_model_overrides(pairs: list[str] | None) -> dict[str, str]:
    """Parses repeated `--base-model-override ROUTE=MODEL` flags into a
    `{route: real_model_tag}` dict. Raises ValueError (caller turns this
    into a clean CLI error, not a traceback) on a malformed entry -- missing
    `=`, or an empty route/model either side of it."""
    overrides: dict[str, str] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--base-model-override must be ROUTE=MODEL, got {pair!r}")
        route, _, model = pair.partition("=")
        route, model = route.strip(), model.strip()
        if not route or not model:
            raise ValueError(f"--base-model-override must be ROUTE=MODEL, got {pair!r}")
        overrides[route] = model
    return overrides


def default_teachers(config, producer: str, base_model_overrides: dict[str, str] | None = None) -> dict:
    """Builds the default teacher map: every LOCAL router model gets a
    real `get_backend`-backed teacher, using `base_model_overrides`'s tag
    for that route if one was given (`--base-model-override`), else the
    route's own tag unchanged. The teacher is still keyed by the ROUTE
    (matching what `tunekit.dataset`'s router emits, so
    `build_sft_dataset`'s `teachers[route.model]` lookup keeps working) --
    only the tag actually requested from the endpoint changes. "frontier"
    is NOT included -- a dataset build that routes a chunk to it and finds
    no teacher wired is SKIPPED-with-reason now (see tunekit.dataset's
    n_skipped_unwired), not aborted.

    An override naming a route this script doesn't wire locally (not in
    `_LOCAL_MODEL_ROUTES`) is logged and ignored rather than silently
    dropped -- most likely a typo or a route name from before the router's
    defaults were switched to real tags."""
    overrides = dict(base_model_overrides or {})
    teachers = {}
    for route_model in _LOCAL_MODEL_ROUTES:
        real_model = overrides.pop(route_model, route_model)
        teachers[route_model] = make_local_backend_teacher(config, real_model, producer)
    for unmatched_route in overrides:
        print(f"--base-model-override {unmatched_route}=... does not match any locally-wired "
             f"route (known routes: {sorted(_LOCAL_MODEL_ROUTES)}) -- ignored", file=sys.stderr)
    return teachers


def load_teachers_from_module(module_path: str, config, producer: str) -> dict:
    module = importlib.import_module(module_path)
    return module.get_teachers(config, producer)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("training_corpus_dir", help="Directory of <doc>/chunks.jsonl dirs (NOT the eval corpus)")
    parser.add_argument("--out-dir", default="sft-dataset", help="Output directory for train/val/manifest")
    parser.add_argument("--eval-corpus-dir", default=str(DEFAULT_EVAL_CORPUS_DIR),
                        help="Eval corpus dir the contamination guard checks against")
    parser.add_argument("--producer", default="web", choices=["web", "pdf"])
    parser.add_argument("--k-ladder", default=",".join(str(k) for k in DEFAULT_K_LADDER),
                        help="Comma-separated DART escalation ladder, e.g. 4,16,64")
    parser.add_argument("--yield-floor", type=int, default=DEFAULT_YIELD_FLOOR)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-completions-per-chunk", type=int,
                        default=DEFAULT_MAX_COMPLETIONS_PER_CHUNK,
                        help="Hard per-chunk teacher-completion budget across the whole "
                             f"escalation ladder (default {DEFAULT_MAX_COMPLETIONS_PER_CHUNK})")
    parser.add_argument("--max-total-completions", type=int, default=None,
                        help="Global teacher-completion budget across the whole build; "
                             "chunks past the budget are skipped and logged "
                             "(default: unlimited, total usage logged)")
    parser.add_argument("--teachers-module", default=None,
                        help="Python module path exposing get_teachers(config, producer) -> dict; "
                             "overrides/extends the default local-backend wiring (needed for the "
                             "frontier bait-slice teacher)")
    parser.add_argument("--base-model-override", action="append", default=[], metavar="ROUTE=MODEL",
                        help="Repeatable. Maps a locally-wired route (see tunekit.dataset's "
                             "DEFAULT_ROUTER_TABLE/DEFAULT_TEACHER_ROUTE, e.g. gemma4:e4b) to the "
                             "actual model tag to request from the endpoint instead, e.g. "
                             "--base-model-override gemma4:e4b=gemma4:e4b-q8_0 -- for a host whose "
                             "pulled tags differ from this module's defaults")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previous build from <out-dir>/progress.json + "
                             "accepted.partial.jsonl: chunks already recorded there are skipped "
                             "(their teacher is never called again) instead of re-sampled")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    training_corpus_dir = Path(args.training_corpus_dir).resolve()
    eval_corpus_dir = Path(args.eval_corpus_dir).resolve()
    out_dir = Path(args.out_dir)
    k_ladder = tuple(int(k.strip()) for k in args.k_ladder.split(",") if k.strip())

    config = load_config()
    chunks = load_training_chunks(training_corpus_dir)
    if not chunks:
        print(f"no chunks found under {training_corpus_dir}", file=sys.stderr)
        return 1

    contamination_index = load_contamination_index(eval_corpus_dir)

    try:
        base_model_overrides = parse_base_model_overrides(args.base_model_override)
    except ValueError as e:
        print(f"invalid --base-model-override: {e}", file=sys.stderr)
        return 1

    teachers = default_teachers(config, args.producer, base_model_overrides=base_model_overrides)
    if args.teachers_module:
        teachers = {**teachers, **load_teachers_from_module(args.teachers_module, config, args.producer)}

    try:
        result = build_sft_dataset(
            chunks, teachers, contamination_index,
            k_ladder=k_ladder, yield_floor=args.yield_floor, temperature=args.temperature,
            producer=args.producer, val_fraction=args.val_fraction, seed=args.seed,
            source_corpus_hash=None,
            max_completions_per_chunk=args.max_completions_per_chunk,
            max_total_completions=args.max_total_completions,
            out_dir=out_dir, resume=args.resume)
    except ContaminationError as e:
        print(f"contamination guard tripped: {e}", file=sys.stderr)
        return 1
    except BannedTeacherError as e:
        print(f"banned teacher: {e}", file=sys.stderr)
        return 1

    _write_jsonl(out_dir / "train.jsonl", result["train"])
    _write_jsonl(out_dir / "val.jsonl", result["val"])
    (out_dir / "manifest.json").write_text(
        json.dumps(result["manifest"], indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "escalation-log.json").write_text(
        json.dumps(result["escalation_log"], indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {len(result['train'])} train / {len(result['val'])} val examples to {out_dir}")
    print(json.dumps(result["manifest"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
