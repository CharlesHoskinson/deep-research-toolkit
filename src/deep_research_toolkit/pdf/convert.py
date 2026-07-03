"""Convert a run's source PDF to canonical markdown + Docling's raw
structured JSON -- the pdf-to-canonical-markdown stage of the PDF
ingestion pipeline (see docs/contracts/pdf-ingestion-pipeline.md).

Reads manifest.json's source_file, runs it through Docling with table
structure recognition on, writes canonical.md and docling_raw.json into
the run directory, and records the stage via common.manifest.update_stage
(never hand-rolled -- see that module's docstring for why).
"""
from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

from deep_research_toolkit.common.manifest import load_manifest, update_stage

# Docling's default DocumentConverter() tries to fetch OCR/layout models
# even for a plain digital-text PDF. On a flaky connection that surfaces as
# requests.exceptions.ChunkedEncodingError (or a bare urllib3 ProtocolError)
# partway through the download -- a genuine transient failure hit and fixed
# during the original prototype's development, not a hypothetical. The fix
# is NOT to disable OCR -- this pipeline needs OCR for the `scanned` route
# later -- it's to retry the whole conversion once before giving up. Do not
# simplify this away.
TRANSIENT_EXCEPTIONS = []
try:
    import requests.exceptions

    TRANSIENT_EXCEPTIONS += [
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    ]
except ImportError:
    pass
try:
    import urllib3.exceptions

    TRANSIENT_EXCEPTIONS.append(urllib3.exceptions.ProtocolError)
except ImportError:
    pass
if not TRANSIENT_EXCEPTIONS:
    TRANSIENT_EXCEPTIONS = [OSError]
TRANSIENT_EXCEPTIONS = tuple(TRANSIENT_EXCEPTIONS)


class DoclingNotInstalled(RuntimeError):
    def __init__(self):
        super().__init__(
            "docling is not installed. This skill needs the 'pdf' extra: "
            'pip install "deep-research-toolkit[pdf]"'
        )


def build_converter():
    """Docling DocumentConverter with table-structure recognition explicitly
    enabled. Docling is imported lazily, here and only here, so importing
    this module never requires docling to be installed unless a conversion
    is actually run.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as e:
        raise DoclingNotInstalled() from e

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def convert_with_retry(converter, source_file: str):
    """Run converter.convert once, retrying a single time on a transient
    network error (e.g. a flaky OCR/layout model download), rather than
    giving up or silently disabling OCR. See the TRANSIENT_EXCEPTIONS
    comment above -- this behavior is real and hard-won, preserve it as-is.
    """
    try:
        return converter.convert(source_file)
    except TRANSIENT_EXCEPTIONS as e:
        print(
            f"pdf-to-canonical-markdown: transient error during Docling conversion "
            f"({e!r}), retrying once...",
            file=sys.stderr,
        )
        return converter.convert(source_file)


def docling_version() -> str:
    try:
        import docling

        return docling.__version__
    except AttributeError:
        pass
    try:
        return importlib.metadata.version("docling")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def convert(run_dir: str | Path) -> None:
    """Convert <run_dir>'s manifest-recorded source_file into canonical.md
    and docling_raw.json, written into run_dir, and record the
    pdf-to-canonical-markdown stage in manifest.json.

    Raises FileNotFoundError if manifest.json or source_file is missing,
    ValueError if manifest.json has no source_file recorded.
    """
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    if manifest is None:
        raise FileNotFoundError(
            f"no manifest.json in {run_dir} -- run pdf-ingest-router first"
        )

    source_file = manifest.get("source_file")
    if not source_file:
        raise ValueError(f"manifest.json in {run_dir} has no source_file")
    if not Path(source_file).is_file():
        raise FileNotFoundError(f"source_file does not exist: {source_file}")

    converter = build_converter()
    result = convert_with_retry(converter, source_file)
    doc = result.document

    canonical_md_path = run_dir / "canonical.md"
    canonical_md_path.write_text(doc.export_to_markdown(), encoding="utf-8")

    docling_raw_path = run_dir / "docling_raw.json"
    with open(docling_raw_path, "w", encoding="utf-8") as f:
        json.dump(doc.export_to_dict(), f, indent=2, default=str)

    update_stage(
        run_dir,
        "pdf-to-canonical-markdown",
        parser="docling",
        parser_version=docling_version(),
    )

    print(f"wrote {canonical_md_path}")
    print(f"wrote {docling_raw_path}")
