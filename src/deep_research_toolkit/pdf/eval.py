"""Health-check a pdf-runs/<document_id>/ directory before its outputs are
trusted -- the rag-eval-harness stage of the PDF ingestion pipeline (see
docs/contracts/pdf-ingestion-pipeline.md).

Every check here is read-only and purely mechanical -- no LLM calls --
which is deliberate: this is the gate that runs right before a document's
outputs get trusted downstream, so it needs to be cheap enough to run every
time and deterministic enough that a failure always means something real
broke, never a matter of taste. Q/A retrieval probes from the original
research (ask N questions, check the right node comes back) are a
genuinely useful *additional* check but need an LLM call per probe -- see
this skill's SKILL.md for that as an optional manual step; it is
intentionally NOT part of this module's pass rate.

Designed to run against a partial pipeline too: any check whose upstream
file(s) don't exist yet is skipped (reported as passed with a "skipped -"
detail) rather than failing the whole run, so run_eval() stays usable as
each stage lands independently.
"""
from __future__ import annotations

import datetime
import json
import string
from pathlib import Path
from typing import Any

from ..common.manifest import load_manifest, update_stage

EVAL_SCHEMA_VERSION = "1.0"

# "printable ASCII + common punctuation/whitespace" per the contract's
# no_ocr_garbage heuristic. string.printable is exactly digits + ascii
# letters + punctuation + whitespace (space, \t, \n, \r, \v, \f).
ALLOWED_CHARS = set(string.printable)
GARBAGE_RATIO_THRESHOLD = 0.05


def _read_jsonl(path: Path) -> list[dict[str, Any]] | None:
    """Returns None if the file doesn't exist (caller should skip its
    check), or a list of parsed rows (possibly empty) if it does."""
    if not path.is_file():
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _skip(name: str, reason: str) -> dict[str, Any]:
    return {"name": name, "passed": True, "detail": f"skipped - {reason}"}


def check_headings_recovered(run_dir: Path) -> dict[str, Any]:
    name = "headings_recovered"
    canonical = _read_text(run_dir / "canonical.md")
    chunks = _read_jsonl(run_dir / "chunks.jsonl")
    if canonical is None or chunks is None:
        return _skip(name, "canonical.md or chunks.jsonl not present yet")

    headings = []
    seen = set()
    for line in canonical.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            text = stripped.lstrip("#").strip()
            if text and text not in seen:
                seen.add(text)
                headings.append(text)

    section_path_entries = set()
    for node in chunks:
        for entry in node.get("section_path") or []:
            section_path_entries.add(entry)

    if not headings:
        return {"name": name, "passed": True, "detail": "no headings found in canonical.md"}

    missing = [h for h in headings if h not in section_path_entries]
    passed = not missing
    detail = (
        f"{len(headings) - len(missing)}/{len(headings)} headings from canonical.md "
        f"appear as section_path entries in chunks.jsonl"
    )
    if missing:
        detail += f"; missing: {missing}"
    return {"name": name, "passed": passed, "detail": detail}


def check_tables_present(run_dir: Path) -> dict[str, Any]:
    name = "tables_present"
    provenance = _read_jsonl(run_dir / "provenance.jsonl")
    if provenance is None:
        return _skip(name, "provenance.jsonl not present yet")

    table_units = [u for u in provenance if u.get("unit_type") == "table"]
    tables_dir = run_dir / "tables"
    csv_files = sorted(tables_dir.glob("*.csv")) if tables_dir.is_dir() else []

    passed = len(table_units) == len(csv_files)
    detail = (
        f"{len(table_units)} table unit(s) detected in provenance.jsonl, "
        f"{len(csv_files)} table CSV(s) in tables/"
    )
    return {"name": name, "passed": passed, "detail": detail}


def check_page_citations_valid(run_dir: Path) -> dict[str, Any]:
    name = "page_citations_valid"
    claims = _read_jsonl(run_dir / "claims.jsonl")
    provenance = _read_jsonl(run_dir / "provenance.jsonl")
    if claims is None or provenance is None:
        return _skip(name, "claims.jsonl or provenance.jsonl not present yet")

    valid_pages = {u.get("page") for u in provenance}
    total = 0
    bad = []
    for claim in claims:
        for ev in claim.get("supporting_evidence") or []:
            total += 1
            if ev.get("page") not in valid_pages:
                bad.append(f"{claim.get('claim_id')} cites page {ev.get('page')} (not in provenance.jsonl)")

    if total == 0:
        return {"name": name, "passed": True, "detail": "no supporting_evidence citations found"}

    passed = not bad
    detail = f"{total - len(bad)}/{total} claim citations point at pages that exist in provenance.jsonl"
    if bad:
        detail += "; invalid: " + "; ".join(bad)
    return {"name": name, "passed": passed, "detail": detail}


def check_evidence_quotes_verbatim(run_dir: Path) -> dict[str, Any]:
    """The most important check: every claim's supporting quote must be a
    verbatim substring of the provenance text on its cited page. This is
    what keeps claims auditable instead of merely plausible-sounding --
    do not weaken this to a fuzzy/normalized match."""
    name = "evidence_quotes_verbatim"
    claims = _read_jsonl(run_dir / "claims.jsonl")
    provenance = _read_jsonl(run_dir / "provenance.jsonl")
    if claims is None or provenance is None:
        return _skip(name, "claims.jsonl or provenance.jsonl not present yet")

    text_by_page: dict[Any, list[str]] = {}
    for u in provenance:
        text_by_page.setdefault(u.get("page"), []).append(u.get("text") or "")

    total = 0
    bad = []
    for claim in claims:
        for ev in claim.get("supporting_evidence") or []:
            total += 1
            quote = ev.get("quote") or ""
            page_texts = text_by_page.get(ev.get("page"), [])
            if not quote or not any(quote in t for t in page_texts):
                bad.append(f"{claim.get('claim_id')} quote not verbatim on page {ev.get('page')}: {quote!r}")

    if total == 0:
        return {"name": name, "passed": True, "detail": "no supporting_evidence quotes found"}

    passed = not bad
    detail = f"{total - len(bad)}/{total} supporting_evidence quotes found verbatim on their cited page"
    if bad:
        detail += "; failing: " + "; ".join(bad)
    return {"name": name, "passed": passed, "detail": detail}


def check_figures_accounted_for(run_dir: Path) -> dict[str, Any]:
    name = "figures_accounted_for"
    captions = _read_jsonl(run_dir / "figures" / "captions.jsonl")
    if captions is None:
        return _skip(name, "figures/captions.jsonl not present yet")

    figures_dir = run_dir / "figures"
    total = len(captions)
    bad = []
    for entry in captions:
        fid = entry.get("figure_id")
        if "extracted" not in entry:
            bad.append(f"{fid}: no 'extracted' field (would be silently missing status)")
            continue
        if entry.get("extracted"):
            candidates = list(figures_dir.glob(f"{fid}.*")) if fid else []
            image_candidates = [c for c in candidates if c.suffix.lower() in (".png", ".jpg", ".jpeg")]
            if not image_candidates:
                bad.append(f"{fid}: marked extracted but no image file found in figures/")
        # extracted: false is fine as long as the field is present -- that's
        # the "flagged, not silently dropped" case the contract calls out.

    if total == 0:
        return {"name": name, "passed": True, "detail": "no figures referenced in captions.jsonl"}

    passed = not bad
    detail = f"{total - len(bad)}/{total} figures extracted or explicitly marked missing"
    if bad:
        detail += "; problems: " + "; ".join(bad)
    return {"name": name, "passed": passed, "detail": detail}


def check_no_ocr_garbage(run_dir: Path) -> dict[str, Any]:
    name = "no_ocr_garbage"
    provenance = _read_jsonl(run_dir / "provenance.jsonl")
    if provenance is None:
        return _skip(name, "provenance.jsonl not present yet")

    text_by_page: dict[Any, list[str]] = {}
    for u in provenance:
        text_by_page.setdefault(u.get("page"), []).append(u.get("text") or "")

    bad_pages = []
    checked_pages = 0
    for page, texts in text_by_page.items():
        combined = "".join(texts)
        if not combined:
            continue
        checked_pages += 1
        non_printable = sum(1 for c in combined if c not in ALLOWED_CHARS)
        ratio = non_printable / len(combined)
        if ratio > GARBAGE_RATIO_THRESHOLD:
            bad_pages.append(f"page {page} ({ratio:.1%} non-printable)")

    if checked_pages == 0:
        return {"name": name, "passed": True, "detail": "no page text to check"}

    passed = not bad_pages
    if passed:
        detail = f"checked {checked_pages} page(s); no page has >5% non-printable/mojibake characters"
    else:
        detail = f"pages exceeding 5% non-printable: {', '.join(bad_pages)}"
    return {"name": name, "passed": passed, "detail": detail}


CHECKS = [
    check_headings_recovered,
    check_tables_present,
    check_page_citations_valid,
    check_evidence_quotes_verbatim,
    check_figures_accounted_for,
    check_no_ocr_garbage,
]


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Eval report: {report['document_id']}",
        "",
        f"Generated: {report['generated_at']}",
        f"Pass rate: {report['pass_rate']:.0%}",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for c in report["checks"]:
        result = "PASS" if c["passed"] else "FAIL"
        detail = c["detail"].replace("|", "\\|")
        lines.append(f"| {c['name']} | {result} | {detail} |")
    lines.append("")
    return "\n".join(lines)


def run_eval(run_dir: str | Path) -> dict[str, Any]:
    """Run all six mechanical checks against <run_dir>, write
    eval_report.json + eval_report.md into it, record the rag-eval-harness
    stage in manifest.json via common.manifest.update_stage, and return the
    report dict.

    Safe to call against a partial pipeline: any check whose upstream
    file(s) don't exist yet is skipped (reported as passed with a
    "skipped - <reason>" detail) rather than raising or failing the run.
    """
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    document_id = (manifest or {}).get("document_id", run_dir.name)

    checks = [check_fn(run_dir) for check_fn in CHECKS]
    pass_rate = round(sum(1 for c in checks if c["passed"]) / len(checks), 4)

    report = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "document_id": document_id,
        "generated_at": _now_iso(),
        "checks": checks,
        "pass_rate": pass_rate,
    }

    report_json_path = run_dir / "eval_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    report_md_path = run_dir / "eval_report.md"
    report_md_path.write_text(render_markdown(report), encoding="utf-8")

    update_stage(run_dir, "rag-eval-harness", pass_rate=pass_rate)

    print(f"wrote {report_json_path}")
    print(f"wrote {report_md_path}")
    print(f"pass_rate: {pass_rate}")

    return report
