# Environment

## Python

3.10 minimum. Verify against Docling's actual current floor before
releasing (it trends toward requiring recent Python) rather than assuming
this stays accurate indefinitely.

## OS

Developed and originally verified on Windows 11 with a `.venv`. CI targets
Ubuntu and Windows (see `.github/workflows/ci.yml`); macOS is not
CI-covered yet (added if maintainer bandwidth allows).

Windows-specific notes:
- Activate with `.venv\Scripts\activate` (vs `source .venv/bin/activate`
  on Linux/macOS).
- Playwright's browser-binary download (`scrapling install`, part of the
  `web` extra) has historically had rougher edges on Windows than Linux CI
  (antivirus flagging the download, longer-path issues) — don't assume a
  green Linux CI run means Windows is equally clean.
- `manifest.json`'s `source_file` field is host-machine-absolute — see
  `docs/contracts/pdf-ingestion-pipeline.md`'s note on this being a known,
  deferred portability limitation. Don't share a `pdf-runs/` corpus across
  machines/OSes and expect `source_file` to resolve.

## First-run network + disk requirements (separate from `pip install`)

Two dependencies do real work on **first use**, not at install time:

- **Docling** (`pdf` extra) downloads layout-detection and OCR models from
  HuggingFace/ModelScope on first conversion — expect low-single-digit GB
  and real network access requirements. If you're in a restricted-network
  environment, look up Docling's documented offline/pre-cached-model mode
  before you need it mid-pipeline.
- **Playwright** (`web` extra, via `scrapling install`) downloads browser
  binaries (~300MB+) separately from `pip install`.

Run `drt doctor --warm` after installing an extra to trigger these
downloads proactively with progress output, rather than being surprised by
them mid-pipeline the first time you actually run a skill.

## Dependency tiers

```
pip install "deep-research-toolkit[web]"       # Scrapling-based web research only
pip install "deep-research-toolkit[pdf]"       # PDF ingestion pipeline only
pip install "deep-research-toolkit[compiler]"  # knowledge compiler layer (not yet built)
pip install "deep-research-toolkit[full]"      # everything
```

A skill's own `SKILL.md` states which extra it needs. Every script fails
with a specific, actionable message naming the missing extra
(`ImportError` caught at the point of use) rather than a raw traceback —
this is a hard requirement for any new skill script, not a nice-to-have.
`reportlab` is a `dev`-only dependency (used solely to regenerate
`tests/fixtures/*.pdf`) — it is never a runtime dependency of any shipped
skill.
