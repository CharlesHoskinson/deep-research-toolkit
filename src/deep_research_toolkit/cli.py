"""`drt` CLI: project lifecycle commands (init / upgrade / doctor / migrate).

This is deliberately narrow in scope -- it does NOT expose every pipeline
stage as a subcommand. Pipeline stages stay invoked as
`skills/<name>/scripts/<script>.py <args>`, matching the existing skill
convention Claude/Codex already follow from SKILL.md instructions. This CLI
only handles: scaffolding a new project, upgrading installed skill files,
checking the environment, and schema-version reporting.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import json
import shutil
import sys
from pathlib import Path

import yaml

from . import __version__
from .config import CONFIG_FILENAME, find_config

INSTALL_MANIFEST_NAME = ".install-manifest.json"
DRT_STATE_DIR = ".deepresearch"

DEFAULT_YAML_TEMPLATE = """\
# .deepresearch.yml -- deep-research-toolkit project configuration
version: 1

knowledge_base:
  path: {kb_path}
  pdf_runs_dir: {pdf_runs_dir}
  research_runs_dir: {research_runs_dir}
  index_dir: {index_dir}

topic:
  name: "{topic_name}"
  scope_hint: >
    {scope_hint}
  tags: []

features:
  web_research: {web_research}
  pdf_ingestion: {pdf_ingestion}
  knowledge_compiler: {knowledge_compiler}

llm:
  # Local, role-routed Qwen stack served by Ollama is the default. It needs a
  # running Ollama endpoint (see llm.local.base_url) with the models below
  # pulled. To run without local models -- letting the in-session agent do the
  # extraction by hand instead -- set: provider: agent
  provider: local
  embedding_model: qwen3-embedding:8b
  local:
    base_url: http://localhost:11434/v1
    model: qwen2.5:7b-instruct   # flat fallback (role=None, and any role below without its own model)
    api_key_env: OPENAI_API_KEY
    temperature: 0.6
    top_p: 0.95
    top_k: 20
    max_tokens: 16000
  # Per-phase models. extract stays a true instruct model (qwen2.5:7b-instruct),
  # NOT qwen3.5:9b -- under the Ollama builds tested it ignored non-thinking mode
  # and produced nothing on extraction.
  roles:
    extract:
      model: qwen2.5:7b-instruct
    wiki_write:
      model: qwen3.6:35b-a3b
    conflict_adjudicate:
      model: qwen3.6:27b
    synthesize:
      model: qwen3.6:27b
    code_agent:
      model: Ornith-1.0-9B

scrapling:
  default_mode: http
  rate_limit_seconds: 1.0
"""

TIER_FEATURES = {
    "web": {"web_research": True, "pdf_ingestion": False, "knowledge_compiler": False},
    "pdf": {"web_research": False, "pdf_ingestion": True, "knowledge_compiler": False},
    "compiler": {"web_research": False, "pdf_ingestion": False, "knowledge_compiler": True},
    "full": {"web_research": True, "pdf_ingestion": True, "knowledge_compiler": True},
}


def _skill_templates_root() -> Path:
    """Locate the shipped skill templates -- works both for an editable
    install (skill_templates/ synced via scripts/sync-skill-templates.py)
    and a real installed wheel (package data).
    """
    return Path(importlib.resources.files("deep_research_toolkit")) / "skill_templates"


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_starter_knowledge_base(kb_path: Path) -> None:
    from .common.frontmatter import write_okf

    kb_path.mkdir(parents=True, exist_ok=True)
    index_path = kb_path / "index.md"
    if not index_path.exists():
        write_okf(
            index_path,
            {"type": "Index", "title": "Knowledge Base Index", "timestamp": _now_iso(), "status": "researched"},
            "# Knowledge Base Index\n\nPages will be linked here as they're added.\n",
        )

    sources_index = kb_path / "sources" / "index.md"
    if not sources_index.exists():
        write_okf(
            sources_index,
            {"type": "Index", "title": "Source Records", "timestamp": _now_iso(), "status": "researched"},
            "# Sources\n\n"
            "Each ingest run should add one row here per raw source consulted.\n\n"
            "| id | resource | fetched | notes |\n"
            "|----|----------|---------|-------|\n",
        )


def _now_iso() -> str:
    import datetime

    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _copy_skills(project_root: Path) -> dict[str, str]:
    """Copy every skill from the shipped templates into .claude/skills/ and
    .agents/skills/ (copy, never symlink -- see docs/decisions/0001-architecture.md
    on why: Windows symlink permissions, zip distribution, naive backup tools).
    Returns a {relative_path: content_hash} map for the install manifest.
    """
    templates_root = _skill_templates_root()
    if not templates_root.is_dir():
        sys.exit(
            f"No skill templates found at {templates_root}. This is a packaging bug -- "
            "skill_templates/ should have been synced from skills/ at build time."
        )

    installed: dict[str, str] = {}
    for platform_dir in [".claude/skills", ".agents/skills"]:
        dest_root = project_root / platform_dir
        for skill_dir in sorted(templates_root.iterdir()):
            if not skill_dir.is_dir():
                continue
            dest = dest_root / skill_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for src_file in skill_dir.rglob("*"):
                if src_file.is_dir():
                    continue
                rel = src_file.relative_to(skill_dir)
                dest_file = dest / rel
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest_file)
                installed[str(dest_file.relative_to(project_root)).replace("\\", "/")] = _file_hash(dest_file)
    return installed


def cmd_init(args: argparse.Namespace) -> int:
    project_root = Path.cwd()
    config_path = project_root / CONFIG_FILENAME

    if config_path.exists() and not args.force:
        print(f"{config_path} already exists. Pass --force to re-scaffold (this will NOT touch your "
              "knowledge base or overwrite unmodified-vs-installed skill files -- see `drt upgrade` for that).")
        return 1

    features = TIER_FEATURES.get(args.tier, TIER_FEATURES["full"])
    kb_path = args.knowledge_base or "knowledge_base"
    pdf_runs_dir = args.pdf_runs_dir or "pdf-runs"
    research_runs_dir = args.research_runs_dir or "research-runs"

    yaml_text = DEFAULT_YAML_TEMPLATE.format(
        kb_path=kb_path,
        pdf_runs_dir=pdf_runs_dir,
        research_runs_dir=research_runs_dir,
        index_dir=".deepresearch/index",
        topic_name=args.topic_name or "(unnamed project)",
        scope_hint=args.scope_hint or "Describe what this project's research is about here.",
        web_research=str(features["web_research"]).lower(),
        pdf_ingestion=str(features["pdf_ingestion"]).lower(),
        knowledge_compiler=str(features["knowledge_compiler"]).lower(),
    )
    config_path.write_text(yaml_text, encoding="utf-8")
    print(f"wrote {config_path}")

    _write_starter_knowledge_base(project_root / kb_path)
    print(f"scaffolded {kb_path}/ (index.md, sources/index.md)")

    installed = _copy_skills(project_root)
    print(f"installed {len(installed)} skill files into .claude/skills/ and .agents/skills/")

    state_dir = project_root / DRT_STATE_DIR
    state_dir.mkdir(exist_ok=True)
    (state_dir / INSTALL_MANIFEST_NAME).write_text(
        json.dumps({"suite_version": __version__, "files": installed}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\ndeep-research-toolkit {__version__} initialized. Next: install the '{args.tier}' extra if you "
          f'haven\'t: pip install "deep-research-toolkit[{args.tier}]"')
    if args.tier in ("web", "full"):
        print("  then run: scrapling install   (Playwright browser binaries for the web-research skill)")
    return 0


def cmd_upgrade(args: argparse.Namespace) -> int:
    project_root = Path.cwd()
    state_dir = project_root / DRT_STATE_DIR
    manifest_path = state_dir / INSTALL_MANIFEST_NAME

    if not manifest_path.exists():
        sys.exit(f"No {manifest_path} found -- run `drt init` first, this project wasn't installed via drt.")

    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_files: dict[str, str] = old_manifest.get("files", {})

    templates_root = _skill_templates_root()
    skipped = []
    updated = []

    for platform_dir in [".claude/skills", ".agents/skills"]:
        dest_root = project_root / platform_dir
        for skill_dir in sorted(templates_root.iterdir()):
            if not skill_dir.is_dir():
                continue
            for src_file in skill_dir.rglob("*"):
                if src_file.is_dir():
                    continue
                rel_in_skill = src_file.relative_to(skill_dir)
                dest_file = dest_root / skill_dir.name / rel_in_skill
                rel_key = str(dest_file.relative_to(project_root)).replace("\\", "/")

                if dest_file.exists():
                    current_hash = _file_hash(dest_file)
                    recorded_hash = old_files.get(rel_key)
                    if recorded_hash is not None and current_hash != recorded_hash:
                        skipped.append(rel_key)
                        continue

                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest_file)
                updated.append(rel_key)

    new_files = {**old_files}
    for platform_dir in [".claude/skills", ".agents/skills"]:
        dest_root = project_root / platform_dir
        for f in dest_root.rglob("*"):
            if f.is_file():
                key = str(f.relative_to(project_root)).replace("\\", "/")
                if key not in skipped:
                    new_files[key] = _file_hash(f)

    manifest_path.write_text(
        json.dumps({"suite_version": __version__, "files": new_files}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"deep-research-toolkit: {old_manifest.get('suite_version', '?')} -> {__version__}")
    print(f"updated {len(updated)} file(s)")
    if skipped:
        print(f"skipped {len(skipped)} file(s) with local edits (not overwritten):")
        for s in skipped:
            print(f"  - {s}")
    print("\n.deepresearch.yml and your knowledge base were not touched.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    checks = [
        ("web", "scrapling", "scrapling"),
        ("pdf", "docling", "docling"),
        ("pdf", "pypdf", "pypdf"),
        ("pdf", "pdfplumber", "pdfplumber"),
        ("compiler", "duckdb", "duckdb"),
        ("compiler", "lancedb", "lancedb"),
        ("compiler", "sentence_transformers", "sentence-transformers"),
        ("compiler", "openai", "openai"),
    ]
    missing_tiers = set()
    for tier, module_name, pip_name in checks:
        try:
            __import__(module_name)
            print(f"  [ok]      {module_name} ({tier})")
        except ImportError:
            print(f"  [missing] {module_name} ({tier}) -- pip install \"deep-research-toolkit[{tier}]\"")
            missing_tiers.add(tier)

    config_path = find_config()
    if config_path:
        print(f"\n.deepresearch.yml found at {config_path}")
    else:
        print("\nNo .deepresearch.yml found in this directory or any parent -- run `drt init`.")

    if args.warm:
        print("\nWarming up installed extras...")
        try:
            import scrapling.fetchers  # noqa: F401

            print("  scrapling.fetchers imports OK (browser binaries still need `scrapling install` if not run yet)")
        except ImportError:
            pass
        try:
            import docling  # noqa: F401

            print("  docling imports OK (models download on first real conversion, not here)")
        except ImportError:
            pass
        print(
            "  note: a `local` LLM provider needs a running Ollama/vLLM endpoint"
            " at llm.local.base_url (not checked here)"
        )

    return 1 if missing_tiers else 0


#: Directories that never contain OKF knowledge-base pages, even though
#: SKILL.md files inside them also happen to start with "---" YAML
#: frontmatter -- excluded so `drt migrate` doesn't try to parse skill
#: definitions as if they were knowledge-base content.
_MIGRATE_EXCLUDED_DIR_NAMES = {".claude", ".agents", "skills", "skill_templates", ".git", ".venv"}


def _is_excluded_from_migrate_scan(path: Path, root: Path) -> bool:
    return any(part in _MIGRATE_EXCLUDED_DIR_NAMES for part in path.relative_to(root).parts[:-1])


def cmd_migrate(args: argparse.Namespace) -> int:
    from .common.frontmatter import OKF_SCHEMA_VERSION, parse_okf
    from .common.manifest import MANIFEST_SCHEMA_VERSION

    path = Path(args.path)
    if not path.exists():
        sys.exit(f"no such path: {path}")

    if path.is_file():
        manifest_targets = [path] if path.name == "manifest.json" else []
        md_targets = [path] if path.suffix == ".md" else []
    else:
        manifest_targets = [f for f in sorted(path.rglob("manifest.json")) if not _is_excluded_from_migrate_scan(f, path)]
        md_targets = [f for f in sorted(path.rglob("*.md")) if not _is_excluded_from_migrate_scan(f, path)]

    mismatches = 0
    checked = 0

    for f in manifest_targets:
        checked += 1
        data = json.loads(f.read_text(encoding="utf-8"))
        found = data.get("schema_version", "(none -- predates versioning)")
        if found != MANIFEST_SCHEMA_VERSION:
            mismatches += 1
            print(f"  {f}: schema_version={found!r}, current={MANIFEST_SCHEMA_VERSION!r}")

    for f in md_targets:
        text = f.read_text(encoding="utf-8")
        try:
            page = parse_okf(text, path=f)
        except yaml.YAMLError:
            continue  # not a real OKF page (or malformed) -- not this command's job to flag that
        if page is None or "type" not in page.frontmatter:
            continue  # doesn't look like an OKF knowledge-base page (e.g. a plain doc) -- skip
        checked += 1
        found = page.frontmatter.get("okf_version", "1 (predates versioning)")
        if str(found) != OKF_SCHEMA_VERSION:
            mismatches += 1
            print(f"  {f}: okf_version={found!r}, current={OKF_SCHEMA_VERSION!r}")

    print(f"\nchecked {checked} file(s), {mismatches} schema-version mismatch(es)")
    if mismatches:
        print("See docs/contracts/schema-versions.md for the migration guide.")
    return 1 if mismatches else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drt", description=__doc__)
    parser.add_argument("--version", action="version", version=f"deep-research-toolkit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Scaffold .deepresearch.yml and install skills into this project")
    p_init.add_argument("--tier", choices=["web", "pdf", "compiler", "full"], default="full")
    p_init.add_argument("--knowledge-base", help="Path for the knowledge base (default: knowledge_base/)")
    p_init.add_argument("--pdf-runs-dir", help="Path for PDF pipeline run directories (default: pdf-runs/)")
    p_init.add_argument("--research-runs-dir", help="Path for web-research run directories (default: research-runs/)")
    p_init.add_argument("--topic-name", help="Short name for this project's research topic")
    p_init.add_argument("--scope-hint", help="One-paragraph description of what's in/out of scope")
    p_init.add_argument("--force", action="store_true", help="Re-scaffold even if .deepresearch.yml exists")
    p_init.set_defaults(func=cmd_init)

    p_upgrade = sub.add_parser("upgrade", help="Update installed skill files to the current package version")
    p_upgrade.set_defaults(func=cmd_upgrade)

    p_doctor = sub.add_parser("doctor", help="Check which optional extras are installed")
    p_doctor.add_argument("--warm", action="store_true", help="Trigger first-run downloads proactively")
    p_doctor.set_defaults(func=cmd_doctor)

    p_migrate = sub.add_parser("migrate", help="Report schema_version mismatches under a path (detect-only in v1)")
    p_migrate.add_argument("path")
    p_migrate.set_defaults(func=cmd_migrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
