#!/usr/bin/env python3
"""Assembles the FINAL Recipe-B SFT dataset (`sft-final/train.jsonl` /
`val.jsonl` / `manifest.json`) from the three pools already on disk:

1. **Local self-distillation set** (`sft-dataset-recipe-b/{train,val}.jsonl`,
   612 non-bait chunk records, built by `scripts/build-sft-dataset.py`).
   These records over-produce ~39 claims/record (the raw self-consistency
   union, ~15x reference atomicity) -- CURATED here via
   `tunekit.assemble.curate_local_record` (see that module's docstring for
   why this is a heuristic cap, not the ideal per-claim support>=2 filter:
   support counts do not survive into either the records or
   `escalation-log.json`).
2. **Frontier bait set** (`datasets/frontier-bait/candidates/*.json`, 90
   chunks, k=3 authored candidates each, already gate-verified at 1.0 pass
   rate) -- MERGED here via `tunekit.assemble.merge_all_bait` (union+dedup
   across each chunk's 3 candidates, rendered through the SAME
   `to_conversation_record` the local records were built with).
3. **General instruction data** (design doc §6.1: "mix 10-20% general
   instruction data against forgetting") -- no cached permissively-licensed
   set was found in this repo, so `tunekit.assemble.synthesize_general_instructions`
   provides 100 self-authored, non-extraction records (summarize / casual
   rewrite / Q&A / prose-to-bullets over 25 short generic topics, disjoint
   from every training-corpus/eval-corpus subject).

The combined pool is then RE-SPLIT 90/10 train/val (seed 42, stratified per
component so bait/general are held out proportionally -- not just
concatenating the local set's ORIGINAL 551/61 split with fresh bait/general
splits), and a provenance manifest is written to `sft-final/manifest.json`.

`sft-final/` is gitignored (large data files stay out of git); only this
script and `sft-final/manifest.json` are meant to be committed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_research_toolkit.tunekit.assemble import (  # noqa: E402
    CURATION_METHOD,
    DEFAULT_JACCARD_THRESHOLD,
    DEFAULT_PER_RECORD_CAP,
    atomicity_stats,
    build_generator_digests,
    component_hash,
    curate_local_record,
    final_dataset_hash,
    load_corpus_chunk_texts,
    load_jsonl,
    merge_all_bait,
    record_claim_count,
    source_corpus_hash,
    stratified_split,
    synthesize_general_instructions,
    write_jsonl,
)

VERBATIM_GATE_VERSION = "span-v2.0"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--local-dir", default=str(REPO_ROOT / "sft-dataset-recipe-b"),
                   help="Dir with local train.jsonl/val.jsonl/manifest.json/escalation-log.json")
    p.add_argument("--bait-dir", default=str(REPO_ROOT / "datasets" / "frontier-bait"),
                   help="Dir with candidates/*.json + manifest.json for the frontier bait set")
    p.add_argument("--training-corpus-dir", default=str(REPO_ROOT / "training-corpus"),
                   help="Full (local+bait) committed training corpus, for bait chunk text lookup "
                        "and the source_corpus_hash")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "sft-final"),
                   help="Output dir for train.jsonl/val.jsonl/manifest.json (gitignored)")
    p.add_argument("--jaccard-threshold", type=float, default=DEFAULT_JACCARD_THRESHOLD)
    p.add_argument("--cap", type=int, default=DEFAULT_PER_RECORD_CAP)
    p.add_argument("--general-n", type=int, default=100)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    local_dir = Path(args.local_dir)
    bait_dir = Path(args.bait_dir)
    training_corpus_dir = Path(args.training_corpus_dir)
    out_dir = Path(args.out_dir)

    # --- 1. Local set: load + curate -------------------------------------
    local_raw = load_jsonl(local_dir / "train.jsonl") + load_jsonl(local_dir / "val.jsonl")
    if not local_raw:
        print(f"no local records found under {local_dir}", file=sys.stderr)
        return 1
    escalation_log = json.loads((local_dir / "escalation-log.json").read_text(encoding="utf-8"))

    atomicity_before = atomicity_stats(local_raw)
    local_curated = [
        curate_local_record(r, jaccard_threshold=args.jaccard_threshold, cap=args.cap)
        for r in local_raw
    ]
    atomicity_after = atomicity_stats(local_curated)

    # --- 2. Frontier bait: merge ------------------------------------------
    bait_manifest = json.loads((bait_dir / "manifest.json").read_text(encoding="utf-8"))
    chunk_by_locator = load_corpus_chunk_texts(training_corpus_dir)
    bait_records = merge_all_bait(bait_dir / "candidates", chunk_by_locator, producer=bait_manifest.get("producer", "web"))
    bait_atomicity = atomicity_stats(bait_records)

    # --- 3. General instruction data ---------------------------------------
    general_records = synthesize_general_instructions(n=args.general_n, seed=args.seed)

    # --- 4. Re-split (stratified per component) over the COMBINED pool ----
    records_by_component = {"local": local_curated, "bait": bait_records, "general": general_records}
    train, val = stratified_split(records_by_component, val_fraction=args.val_fraction, seed=args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)

    # --- 5. Provenance manifest --------------------------------------------
    dataset_hash = final_dataset_hash(out_dir / "train.jsonl")
    src_corpus_hash = source_corpus_hash(training_corpus_dir)
    generator_digests = build_generator_digests(
        local_curated, escalation_log,
        bait_n_claims=sum(record_claim_count(r) for r in bait_records),
        general_n_records=len(general_records),
    )

    def _n(component: str, split: list[dict]) -> int:
        return sum(1 for r in split if r.get("component") == component)

    manifest = {
        "dataset_hash": dataset_hash,
        "component_hashes": {
            "local": component_hash(local_curated),
            "frontier_bait": component_hash(bait_records),
            "general": component_hash(general_records),
        },
        "n_total": {"train": len(train), "val": len(val)},
        "n_by_component": {
            "local": {"train": _n("local", train), "val": _n("local", val),
                     "total": len(local_curated)},
            "bait": {"train": _n("bait", train), "val": _n("bait", val), "total": len(bait_records)},
            "general": {"train": _n("general", train), "val": _n("general", val),
                       "total": len(general_records)},
        },
        "atomicity": {
            "local_before_curation": atomicity_before,
            "local_after_curation": atomicity_after,
            "bait_after_merge": bait_atomicity,
        },
        "verbatim_gate_version": VERBATIM_GATE_VERSION,
        "source_corpus_hash": src_corpus_hash,
        "generator_digests": generator_digests,
        "curation_method": CURATION_METHOD,
        "curation_params": {
            "jaccard_threshold": args.jaccard_threshold,
            "per_record_cap": args.cap,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "general_n": args.general_n,
        },
        "general_source": (
            "Self-authored by the assembling agent (no cached permissively-licensed "
            "instruction dataset was found in this repo) -- 25 short generic topics "
            "(cooking, science, civics-adjacent how-tos, health, etc.), each deliberately "
            "disjoint from every training-corpus/eval-corpus subject, x 4 task types "
            "(summarize / casual rewrite / Q&A / prose-to-bullets) = 100 records. See "
            "deep_research_toolkit.tunekit.assemble.GENERAL_TOPICS for the source content."
        ),
        "sources": {
            "local_raw_manifest": str((local_dir / "manifest.json").resolve()),
            "bait_manifest": str((bait_dir / "manifest.json").resolve()),
            "training_corpus_dir": str(training_corpus_dir.resolve()),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {len(train)} train / {len(val)} val records to {out_dir}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
