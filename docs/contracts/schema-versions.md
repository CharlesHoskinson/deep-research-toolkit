# Schema Version Registry

This suite versions itself with one semver number for the whole package
(`deep_research_toolkit.__version__`), because the skills are tightly
coupled through shared on-disk file formats — independent per-skill
versioning would imply compatibility promises that don't actually exist.

Semver meaning, enforced in review:
- **Patch**: bug fixes, no on-disk schema or CLI-flag changes.
- **Minor**: new skills, new optional/additive fields, new CLI flags —
  never a change that makes an existing on-disk file fail to read.
- **Major**: any change that would make a file written by an older suite
  version fail validation, or require an existing field to change
  meaning/type/be removed.

Each on-disk artifact type carries its own `schema_version` field
(independent of the suite version) because these evolve as real needs
appear — this table is the registry mapping suite version to the schema
versions it produces/accepts.

| Suite version | manifest.json | classification.json | provenance.jsonl | chunks.jsonl | claims/entities/relations.jsonl | OKF frontmatter (`okf_version`) |
|---|---|---|---|---|---|---|
| 0.1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

## Migrating

`drt migrate <path>` reads whatever `schema_version`/`okf_version` a file
declares and reports whether it matches what the installed suite version
expects. In this first release there is nothing to migrate *from* — the
command exists now, before it's needed, so the discipline of checking is
established before there's ever a real breaking change to handle. When a
future release does introduce a breaking schema change:

1. Bump the relevant column in the table above.
2. Add a "Schema changes" entry to `CHANGELOG.md` describing exactly what
   changed and why.
3. Extend `drt migrate` with the actual field-rewrite logic for that
   specific transition (old version → new version), not a generic
   migration framework speculatively built ahead of a real need.
