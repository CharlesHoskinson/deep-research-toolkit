"""Project-level configuration: find and load .deepresearch.yml.

Discovery mirrors .git: walk upward from cwd (or a given start path) looking
for .deepresearch.yml. Every skill script should call find_config()/load_config()
at startup instead of hardcoding paths like "knowledge/" -- this is what lets
the same skill run unmodified across different consuming projects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILENAME = ".deepresearch.yml"
DEFAULT_KNOWLEDGE_BASE_PATH = "knowledge_base"
DEFAULT_PDF_RUNS_PATH = "pdf-runs"
DEFAULT_RESEARCH_RUNS_PATH = "research-runs"
DEFAULT_INDEX_DIR = ".deepresearch/index"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

#: Per-phase model roles for a local model stack -- extraction wants a fast,
#: non-thinking, schema-constrained model; synthesis/adjudication want a
#: reasoning model. Each role is overridable under `llm.roles.<role>` in
#: .deepresearch.yml; any field left unset falls back to the flat `llm.local`
#: model, so a single-model setup still works (back-compat) while a real stack
#: routes each phase to the right model.
ROLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "extract":             {"thinking": False, "temperature": 0.0, "max_tokens": 2000,  "response_format": "json"},
    "wiki_write":          {"thinking": False, "temperature": 0.2, "max_tokens": 4096,  "response_format": None},
    "conflict_adjudicate": {"thinking": True,  "temperature": 0.2, "max_tokens": 8192,  "response_format": None},
    "synthesize":          {"thinking": True,  "temperature": 0.4, "max_tokens": 12000, "response_format": None},
    "code_agent":          {"thinking": True,  "temperature": 0.6, "max_tokens": 16000, "response_format": None},
}


def _resolve_roles(roles_raw: dict[str, Any], local: dict[str, Any]) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}
    for name, d in ROLE_DEFAULTS.items():
        r = roles_raw.get(name) or {}
        resolved[name] = {
            "model": r.get("model", local["model"]),
            "base_url": r.get("base_url", local["base_url"]),
            "api_key_env": r.get("api_key_env", local["api_key_env"]),
            "thinking": bool(r.get("thinking", d["thinking"])),
            "temperature": float(r.get("temperature", d["temperature"])),
            "top_p": float(r.get("top_p", local.get("top_p", 0.95))),
            "top_k": int(r.get("top_k", local.get("top_k", 20))),
            "max_tokens": int(r.get("max_tokens", d["max_tokens"])),
            "response_format": r.get("response_format", d["response_format"]),
        }
    return resolved


def find_config(start: Path | None = None) -> Path | None:
    """Walk upward from `start` (default: cwd) looking for .deepresearch.yml.

    Same discovery model as `.git`: check `start`, then each parent, stop at
    the first match, return None if the filesystem root is reached with no
    match found.
    """
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        candidate = parent / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


@dataclass
class Config:
    """Resolved project configuration. Paths are absolute, already resolved
    relative to the config file's own location -- never relative to cwd,
    since a skill script may be invoked from any subdirectory.
    """

    config_path: Path | None
    project_root: Path
    knowledge_base_path: Path
    pdf_runs_path: Path
    research_runs_path: Path
    index_dir: Path
    embedding_model: str
    llm_local: dict[str, Any]
    llm_roles: dict[str, dict[str, Any]]
    topic_name: str
    scope_hint: str
    tags: list[str]
    features: dict[str, bool]
    llm_provider: str
    llm_model: str
    llm_api_key_env: str
    scrapling_default_mode: str
    scrapling_rate_limit_seconds: float
    raw: dict[str, Any] = field(default_factory=dict)

    def feature_enabled(self, name: str) -> bool:
        return bool(self.features.get(name, False))


def _default_config(project_root: Path) -> Config:
    """Zero-config fallback: no .deepresearch.yml found anywhere. Used so a
    bare `git clone && run` still does something sensible for quick
    exploration, per the CLI-first, no-magic-install philosophy -- this is
    NOT a substitute for `drt init`, which is required for real project use
    (it's what makes features.* opt-in explicit rather than silently
    defaulting to "everything on").
    """
    return Config(
        config_path=None,
        project_root=project_root,
        knowledge_base_path=project_root / DEFAULT_KNOWLEDGE_BASE_PATH,
        pdf_runs_path=project_root / DEFAULT_PDF_RUNS_PATH,
        research_runs_path=project_root / DEFAULT_RESEARCH_RUNS_PATH,
        index_dir=project_root / DEFAULT_INDEX_DIR,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        llm_local=(_default_local := {"base_url": "http://localhost:11434/v1", "model": "Ornith-1.0-9B",
                   "api_key_env": "OPENAI_API_KEY", "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                   "max_tokens": 16000}),
        llm_roles=_resolve_roles({}, _default_local),
        topic_name="(unconfigured project)",
        scope_hint="No .deepresearch.yml found -- run `drt init` to configure this project's research scope.",
        tags=[],
        features={"web_research": False, "pdf_ingestion": False, "knowledge_compiler": False},
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-5",
        llm_api_key_env="ANTHROPIC_API_KEY",
        scrapling_default_mode="http",
        scrapling_rate_limit_seconds=1.0,
        raw={},
    )


def load_config(start: Path | None = None) -> Config:
    """Find and parse .deepresearch.yml, or return the zero-config default
    if none exists. Never raises on a missing file -- only on a malformed one.
    """
    path = find_config(start)
    if path is None:
        return _default_config((start or Path.cwd()).resolve())

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    root = path.parent

    kb = raw.get("knowledge_base", {}) or {}
    topic = raw.get("topic", {}) or {}
    features = raw.get("features", {}) or {}
    llm = raw.get("llm", {}) or {}
    scrapling = raw.get("scrapling", {}) or {}

    local = (llm.get("local") or {})
    llm_local = {
        "base_url": local.get("base_url", "http://localhost:11434/v1"),
        "model": local.get("model", "Ornith-1.0-9B"),
        "api_key_env": local.get("api_key_env", "OPENAI_API_KEY"),
        "temperature": float(local.get("temperature", 0.6)),
        "top_p": float(local.get("top_p", 0.95)),
        "top_k": int(local.get("top_k", 20)),
        "max_tokens": int(local.get("max_tokens", 16000)),
    }
    llm_roles = _resolve_roles(llm.get("roles") or {}, llm_local)

    return Config(
        config_path=path,
        project_root=root,
        knowledge_base_path=(root / kb.get("path", DEFAULT_KNOWLEDGE_BASE_PATH)).resolve(),
        pdf_runs_path=(root / kb.get("pdf_runs_dir", DEFAULT_PDF_RUNS_PATH)).resolve(),
        research_runs_path=(root / kb.get("research_runs_dir", DEFAULT_RESEARCH_RUNS_PATH)).resolve(),
        llm_roles=llm_roles,
        index_dir=(root / kb.get("index_dir", DEFAULT_INDEX_DIR)).resolve(),
        embedding_model=llm.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        llm_local=llm_local,
        topic_name=topic.get("name", "(unnamed project)"),
        scope_hint=topic.get("scope_hint", ""),
        tags=list(topic.get("tags", [])),
        features={
            "web_research": bool(features.get("web_research", False)),
            "pdf_ingestion": bool(features.get("pdf_ingestion", False)),
            "knowledge_compiler": bool(features.get("knowledge_compiler", False)),
        },
        llm_provider=llm.get("provider", "anthropic"),
        llm_model=llm.get("model", "claude-sonnet-4-5"),
        llm_api_key_env=llm.get("api_key_env", "ANTHROPIC_API_KEY"),
        scrapling_default_mode=scrapling.get("default_mode", "http"),
        scrapling_rate_limit_seconds=float(scrapling.get("rate_limit_seconds", 1.0)),
        raw=raw,
    )


def resolve_path(cli_value: str | None, config_value: Path, fallback: str) -> Path:
    """Three-tier resolution used by every skill script: explicit CLI flag
    wins, then the loaded config, then a hardcoded fallback (only reached in
    the zero-config case) -- so scripts work both zero-config (quick
    exploration) and fully configured (real project use).
    """
    if cli_value is not None:
        return Path(cli_value).resolve()
    if config_value is not None:
        return config_value
    return Path(fallback).resolve()
