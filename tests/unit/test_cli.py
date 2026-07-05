"""Unit tests for the `drt` CLI (src/deep_research_toolkit/cli.py).

Covers init / upgrade / doctor / migrate. All tests are hermetic: they run
against a tmp_path cwd (via monkeypatch.chdir), touch no network, and load
no LLM. Driven through `main([...])` (the real argparse entry point) except
where noted.
"""
from __future__ import annotations

import json

from deep_research_toolkit.cli import INSTALL_MANIFEST_NAME, DRT_STATE_DIR, TIER_FEATURES, main
from deep_research_toolkit.common.frontmatter import OKF_SCHEMA_VERSION, write_okf
from deep_research_toolkit.common.manifest import MANIFEST_SCHEMA_VERSION
from deep_research_toolkit.config import CONFIG_FILENAME, load_config


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def test_init_writes_valid_config_matching_tier(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rc = main(["init", "--tier", "pdf", "--topic-name", "T", "--scope-hint", "S"])

    assert rc == 0
    config_path = tmp_path / CONFIG_FILENAME
    assert config_path.exists()

    cfg = load_config(tmp_path)
    assert cfg.topic_name == "T"
    # scope_hint is written into a YAML folded ('>') scalar, which appends a
    # trailing newline on parse -- strip before comparing.
    assert cfg.scope_hint.strip() == "S"
    assert cfg.features == TIER_FEATURES["pdf"]


def test_init_writes_the_local_qwen_stack(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert main(["init", "--tier", "full", "--topic-name", "T", "--scope-hint", "S"]) == 0

    cfg = load_config(tmp_path)
    assert cfg.llm_provider == "local"
    assert cfg.embedding_model == "qwen3-embedding:8b"
    assert cfg.llm_local["model"] == "qwen2.5:7b-instruct"
    assert cfg.llm_roles["extract"]["model"] == "qwen2.5:7b-instruct"
    assert cfg.llm_roles["wiki_write"]["model"] == "qwen3.6:35b-a3b"
    assert cfg.llm_roles["synthesize"]["model"] == "qwen3.6:27b"
    assert cfg.llm_roles["code_agent"]["model"] == "Ornith-1.0-9B"


def test_init_scaffolds_knowledge_base(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rc = main(["init", "--tier", "pdf", "--topic-name", "T", "--scope-hint", "S"])

    assert rc == 0
    index_md = tmp_path / "knowledge_base" / "index.md"
    sources_index = tmp_path / "knowledge_base" / "sources" / "index.md"
    assert index_md.exists()
    assert sources_index.exists()
    assert "Knowledge Base Index" in index_md.read_text(encoding="utf-8")
    assert "Sources" in sources_index.read_text(encoding="utf-8")


def test_init_copies_skills_into_both_platform_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rc = main(["init", "--tier", "full", "--topic-name", "T", "--scope-hint", "S"])

    assert rc == 0
    claude_skill = tmp_path / ".claude" / "skills" / "knowledge-compiler" / "SKILL.md"
    agents_skill = tmp_path / ".agents" / "skills" / "knowledge-compiler" / "SKILL.md"
    assert claude_skill.exists()
    assert agents_skill.exists()
    assert claude_skill.read_text(encoding="utf-8") == agents_skill.read_text(encoding="utf-8")


def test_init_writes_install_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rc = main(["init", "--tier", "full", "--topic-name", "T", "--scope-hint", "S"])

    assert rc == 0
    manifest_path = tmp_path / DRT_STATE_DIR / INSTALL_MANIFEST_NAME
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "suite_version" in manifest
    assert manifest["files"]  # non-empty {relative_path: sha256} map
    # every recorded path should exist on disk with a real 64-hex-char sha256
    for rel_path, digest in manifest["files"].items():
        assert (tmp_path / rel_path).exists()
        assert len(digest) == 64


def test_init_without_force_refuses_when_config_exists(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc1 = main(["init", "--tier", "pdf", "--topic-name", "T", "--scope-hint", "S"])
    assert rc1 == 0

    rc2 = main(["init", "--tier", "pdf", "--topic-name", "Other", "--scope-hint", "Other"])

    assert rc2 == 1
    out = capsys.readouterr().out
    assert "already exists" in out
    # confirm the original config was NOT clobbered
    cfg = load_config(tmp_path)
    assert cfg.topic_name == "T"


def test_init_with_force_rescaffolds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc1 = main(["init", "--tier", "pdf", "--topic-name", "T", "--scope-hint", "S"])
    assert rc1 == 0

    rc2 = main(["init", "--tier", "full", "--force", "--topic-name", "Other", "--scope-hint", "Other"])

    assert rc2 == 0
    cfg = load_config(tmp_path)
    assert cfg.topic_name == "Other"
    assert cfg.features == TIER_FEATURES["full"]


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def test_doctor_returns_int_and_lists_expected_modules(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    rc = main(["doctor"])

    assert isinstance(rc, int)
    assert rc in (0, 1)
    out = capsys.readouterr().out
    for module_name in (
        "scrapling",
        "docling",
        "pypdf",
        "pdfplumber",
        "duckdb",
        "lancedb",
        "sentence_transformers",
        "openai",
    ):
        assert module_name in out
    assert "No .deepresearch.yml found" in out


def test_doctor_reports_found_config(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["init", "--tier", "pdf", "--topic-name", "T", "--scope-hint", "S"])
    capsys.readouterr()  # drain init's own output

    rc = main(["doctor"])

    assert isinstance(rc, int)
    out = capsys.readouterr().out
    assert ".deepresearch.yml found at" in out


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------


def test_migrate_reports_no_mismatches_for_current_versions(tmp_path):
    write_okf(
        tmp_path / "page.md",
        {"type": "Index", "title": "T", "timestamp": "2026-01-01T00:00:00Z", "status": "researched"},
        "# T\n",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": MANIFEST_SCHEMA_VERSION, "sources": []}),
        encoding="utf-8",
    )

    rc = main(["migrate", str(tmp_path)])

    assert rc == 0


def test_migrate_reports_mismatch_for_stale_versions(tmp_path, capsys):
    # OKF page with an old okf_version.
    page_path = tmp_path / "stale_page.md"
    write_okf(
        page_path,
        {"type": "Index", "title": "T", "timestamp": "2026-01-01T00:00:00Z", "status": "researched"},
        "# T\n",
    )
    stale_text = page_path.read_text(encoding="utf-8").replace(
        f"okf_version: '{OKF_SCHEMA_VERSION}'", "okf_version: '0.1'"
    )
    # write_okf's yaml.safe_dump may or may not quote the version string;
    # fall back to a plain string replace on the unquoted form too.
    if "okf_version: '0.1'" not in stale_text:
        stale_text = page_path.read_text(encoding="utf-8").replace(
            f"okf_version: {OKF_SCHEMA_VERSION}", "okf_version: 0.1"
        )
    page_path.write_text(stale_text, encoding="utf-8")

    # manifest.json with a mismatched schema_version.
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": "0.1", "sources": []}),
        encoding="utf-8",
    )

    rc = main(["migrate", str(tmp_path)])

    assert rc == 1
    out = capsys.readouterr().out
    assert "schema_version=" in out
    assert "okf_version=" in out
    assert "mismatch" in out


# ---------------------------------------------------------------------------
# upgrade
# ---------------------------------------------------------------------------


def test_upgrade_preserves_locally_modified_files(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc_init = main(["init", "--tier", "full", "--topic-name", "T", "--scope-hint", "S"])
    assert rc_init == 0
    capsys.readouterr()  # drain init's output

    # Work in raw bytes (not read_text/write_text) so this doesn't get
    # tripped up by Windows universal-newline translation, which would
    # silently rewrite \n <-> \r\n and make our own byte-for-byte hash
    # comparisons below meaningless.
    edited = tmp_path / ".claude" / "skills" / "knowledge-compiler" / "SKILL.md"
    original_bytes = edited.read_bytes()
    edited.write_bytes(original_bytes + b"\n<!-- local edit -->\n")

    rc_upgrade = main(["upgrade"])

    assert rc_upgrade == 0
    out = capsys.readouterr().out
    assert "skipped 1 file(s) with local edits" in out
    rel_key = ".claude/skills/knowledge-compiler/SKILL.md"
    assert rel_key in out
    # the local edit must not have been clobbered
    assert edited.read_bytes() == original_bytes + b"\n<!-- local edit -->\n"
    # The manifest keeps recording the ORIGINAL (pre-edit) hash for a
    # skipped file rather than the edited-on-disk hash -- this is what makes
    # the skip durable across repeated `upgrade` runs (the comparison next
    # time is still edited-on-disk vs. original-recorded, so it's skipped
    # again instead of silently being "accepted" as the new baseline).
    manifest = json.loads((tmp_path / DRT_STATE_DIR / INSTALL_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["files"][rel_key] == _sha256_bytes(original_bytes)


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
