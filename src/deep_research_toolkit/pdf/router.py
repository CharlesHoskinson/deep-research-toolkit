"""PDF classification: compute signals, decide a route, and start a
pdf-runs/<document_id>/ working directory (classification.json + manifest.json).

Ported from the agentictrading prototype's classify_pdf.py -- the logic is
unchanged, just moved into an importable package (see
docs/contracts/pdf-ingestion-pipeline.md's "What changed" section) so it's
unit-testable without a subprocess and shared with anything else that wants
to classify a PDF (e.g. a future batch-ingest tool).

Thresholds are deliberately simple and documented in
skills/pdf-ingest-router/references/routing-table.md rather than tuned
precisely -- a reasonable first cut a human can override, not a classifier
that needs to be perfect on the first pass.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from deep_research_toolkit.common.hashing import file_hash
from deep_research_toolkit.common.manifest import (
    start_manifest,
    update_stage,
    write_manifest,
)

CLASSIFICATION_SCHEMA_VERSION = "1.0"

# Common math/LaTeX-ish symbols. Presence of these per page is a rough proxy
# for "this document has real math in it" -- not a proof, just a signal.
MATH_SYMBOL_RE = re.compile(
    r"[∑∫√≤≥≠±∞∂Δ∇×÷→↔∈∉⊂⊆∪∩∀∃π∝≈≡]"
    r"|\\(frac|sum|int|alpha|beta|gamma|delta|theta|lambda|sigma|omega|partial|nabla|infty)\b"
    r"|[αβγδεζηθικλμνξοπρστυφχψω]"
)

# A page counts as "image-only" (no usable extractable text) below this many
# non-whitespace characters.
IMAGE_ONLY_CHAR_THRESHOLD = 10

# Routing thresholds. See skills/pdf-ingest-router/references/routing-table.md
# for the reasoning.
SCANNED_IMAGE_RATIO = 0.6
FINANCIAL_LEGAL_TABLE_RATIO = 0.6
SLIDE_LIKE_MAX_AVG_CHARS = 150
MATH_DENSITY_MEDIUM = 1.0
MATH_DENSITY_HIGH = 5.0

# route -> backend recommended "in principle" per the routing table,
# independent of what's actually implemented yet.
IDEAL_BACKEND = {
    "digital-text": "docling",
    "financial-legal": "docling",
    "form": "docling",
    "scientific-math": "docling (marker fallback if formulas look mangled)",
    "scanned": "docling (OCR mode)",
    "slide-like": "docling (render pages + describe figures)",
}

IMPLEMENTED_BACKENDS = {"docling"}


class PdfDepsNotInstalled(RuntimeError):
    """Raised when pypdf is missing. This skill needs the 'pdf' extra --
    lazy-import + a specific, actionable error, per the pattern in
    deep_research_toolkit.web.fetch.
    """

    def __init__(self):
        super().__init__(
            "pypdf is not installed. This skill needs the 'pdf' extra: "
            'pip install "deep-research-toolkit[pdf]"'
        )


def _import_pypdf():
    try:
        import pypdf
    except ImportError as e:
        raise PdfDepsNotInstalled() from e
    return pypdf


def _import_pdfplumber():
    """pdfplumber is optional at the signal-computation level (unlike pypdf,
    which is required just to open the file and count pages): if it's
    missing, compute_signals falls back to conservative defaults instead of
    raising, matching the original prototype's behavior.
    """
    try:
        import pdfplumber
    except ImportError:
        return None
    return pdfplumber


def slugify(stem: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug or "document"


def has_acroform_fields(reader) -> bool:
    try:
        fields = reader.get_fields()
        return bool(fields)
    except Exception:
        return False


def compute_signals(pdf_path, page_count: int) -> dict[str, Any]:
    """Return the signals dict, computed with pdfplumber where possible.

    Falls back to conservative defaults (as if the signal weren't present)
    if pdfplumber can't open the file at all -- pypdf's page count and
    AcroForm check still work independently in that case.
    """
    signals: dict[str, Any] = {
        "avg_extractable_chars_per_page": 0.0,
        "has_acroform_fields": False,
        "image_only_page_ratio": 0.0,
        "table_like_page_ratio": 0.0,
        "detected_math_density": "low",
    }
    pdfplumber = _import_pdfplumber()
    if pdfplumber is None or page_count == 0:
        return signals

    total_chars = 0
    image_only_pages = 0
    table_pages = 0
    math_hits_total = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                total_chars += len(text)
                if len(text.strip()) < IMAGE_ONLY_CHAR_THRESHOLD:
                    image_only_pages += 1

                try:
                    tables = page.find_tables()
                except Exception:
                    tables = []
                if tables:
                    table_pages += 1

                math_hits_total += len(MATH_SYMBOL_RE.findall(text))
    except Exception:
        # Malformed/unreadable-by-pdfplumber file -- report defaults rather
        # than crashing the whole classification.
        return signals

    signals["avg_extractable_chars_per_page"] = round(total_chars / page_count, 2)
    signals["image_only_page_ratio"] = round(image_only_pages / page_count, 4)
    signals["table_like_page_ratio"] = round(table_pages / page_count, 4)

    avg_math_per_page = math_hits_total / page_count
    if avg_math_per_page >= MATH_DENSITY_HIGH:
        signals["detected_math_density"] = "high"
    elif avg_math_per_page >= MATH_DENSITY_MEDIUM:
        signals["detected_math_density"] = "medium"
    else:
        signals["detected_math_density"] = "low"

    return signals


def decide_route(signals: dict[str, Any]) -> tuple[str, str]:
    """Pick a route + free-text note, in priority order (see
    skills/pdf-ingest-router/references/routing-table.md for the full
    explanation of why this order).
    """
    if signals["has_acroform_fields"]:
        return "form", "Document has AcroForm fields, so it is fillable-form-shaped regardless of its text/table signals."

    if signals["image_only_page_ratio"] >= SCANNED_IMAGE_RATIO:
        return "scanned", (
            f"{signals['image_only_page_ratio']:.0%} of pages have no extractable text — "
            "treated as scanned/image-based and routed to Docling's OCR mode."
        )

    if signals["detected_math_density"] == "high":
        return "scientific-math", "High density of math/LaTeX-ish symbols detected across pages."

    if signals["table_like_page_ratio"] >= FINANCIAL_LEGAL_TABLE_RATIO:
        return "financial-legal", (
            f"{signals['table_like_page_ratio']:.0%} of pages contain table-like structure with digital text — "
            "treated as a financial/legal document (statements, schedules, contracts)."
        )

    if (
        signals["avg_extractable_chars_per_page"] <= SLIDE_LIKE_MAX_AVG_CHARS
        and signals["image_only_page_ratio"] < SCANNED_IMAGE_RATIO
    ):
        return "slide-like", (
            f"Sparse text ({signals['avg_extractable_chars_per_page']} chars/page average) without being "
            "majority image-only — consistent with a slide deck rather than prose."
        )

    return "digital-text", "Plain extractable digital text, no strong signal for any of the other routes."


def build_notes(route: str, extra_note: str) -> str:
    ideal = IDEAL_BACKEND[route]
    ideal_backend_name = ideal.split(" ")[0]
    note = extra_note
    if ideal_backend_name not in IMPLEMENTED_BACKENDS or ideal != "docling":
        note += (
            f" Ideal backend per the routing table is '{ideal}', but only plain Docling is wired up "
            "in this pass, so recommended_backend is 'docling' as a best-effort fallback."
        )
    return note


def classify(pdf_path: str | Path, runs_dir: str | Path) -> Path:
    """Classify one PDF and start its <runs_dir>/<document_id>/ working
    directory: writes classification.json, starts manifest.json (via
    common.manifest), and appends the pdf-ingest-router stage entry.

    Returns the run directory path.
    """
    pypdf = _import_pypdf()

    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"no such file: {pdf_path}")

    source_hash = file_hash(pdf_path)
    stem = pdf_path.stem
    document_id = f"{slugify(stem)}-{source_hash[:8]}"

    run_dir = Path(runs_dir) / document_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        reader = pypdf.PdfReader(str(pdf_path))
        page_count = len(reader.pages)
    except Exception as e:
        raise RuntimeError(f"could not open {pdf_path} with pypdf: {e}") from e

    signals = compute_signals(pdf_path, page_count)
    signals["has_acroform_fields"] = has_acroform_fields(reader)

    route, note = decide_route(signals)
    notes = build_notes(route, note)

    classification = {
        "schema_version": CLASSIFICATION_SCHEMA_VERSION,
        "document_id": document_id,
        "source_file": str(pdf_path),
        "page_count": page_count,
        "route": route,
        "signals": signals,
        "recommended_backend": "docling",
        "notes": notes,
    }

    with open(run_dir / "classification.json", "w", encoding="utf-8") as f:
        json.dump(classification, f, indent=2)
        f.write("\n")

    # start_manifest is idempotent (preserves an existing manifest's other
    # stage entries on rerun) -- see common.manifest's module docstring for
    # why that invariant matters.
    manifest = start_manifest(run_dir, document_id, str(pdf_path), f"sha256:{source_hash}")
    write_manifest(run_dir, manifest)
    update_stage(run_dir, "pdf-ingest-router", route=route)

    return run_dir
