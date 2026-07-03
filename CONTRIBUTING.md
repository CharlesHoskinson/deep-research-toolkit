# Contributing

## Setup

```
python -m venv .venv
.venv\Scripts\activate   # Windows; source .venv/bin/activate on Linux/macOS
pip install -e ".[dev,full]"
```

See `docs/environment.md` for Python version floor, OS-specific notes, and
first-run network/disk requirements (Docling and Playwright both download
models/binaries on first use, separately from `pip install`).

## Repo layout

- `src/deep_research_toolkit/` — the actual installable package. All real
  logic lives here as plain, unit-testable functions.
- `skills/` — Claude Code / Codex skill definitions (`SKILL.md` + thin
  `scripts/*.py` CLI shims that call into `src/deep_research_toolkit/`).
  Skill bodies describe *actions*, never a specific tool name, so the same
  `SKILL.md` works unmodified on both platforms.
- `docs/contracts/` — the on-disk data-format contracts every skill agrees
  on (manifest.json, OKF frontmatter, JSONL schemas, schema versioning).
- `tests/unit/` — fast, dependency-light tests against pure functions. Runs
  on every push.
- `tests/integration/` — marked `@pytest.mark.heavy`; real Docling/Scrapling
  calls against `tests/fixtures/`. Runs on a schedule, not every push.

## Before opening a PR

- `pytest -m "not heavy"` must pass.
- If you touch any on-disk format (`manifest.json` keys, JSONL row shapes,
  OKF frontmatter fields), bump the relevant `schema_version` in
  `docs/contracts/schema-versions.md` and add a `CHANGELOG.md` entry under
  "Schema changes."
- `python scripts/check-manifests-in-sync.py` must pass if you touched
  `.claude-plugin/plugin.json` or `.codex-plugin/plugin.json`.
