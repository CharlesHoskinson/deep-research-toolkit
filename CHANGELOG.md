# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).
Versioning: one semver number for the whole suite (see `docs/contracts/schema-versions.md`
for how suite versions map to on-disk schema versions).

## [Unreleased]

### Added
- Initial extraction and generalization of the web-research and PDF-ingestion
  skill stacks from the private `agentictrading` research repo.
- `.deepresearch.yml` project-level configuration.
- `drt` CLI (`init`, `upgrade`, `doctor`, `migrate`).
- Dual Claude Code / Codex plugin manifests over one shared `skills/` tree.
- `schema_version` fields on `manifest.json`, `classification.json`, and OKF
  frontmatter (new -- the original repo had none).
