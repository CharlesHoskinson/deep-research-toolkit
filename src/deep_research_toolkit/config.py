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
#: qwen3-embedding served through Ollama -- a materially stronger retrieval
#: embedder than the old sentence-transformers default. Has a "name:tag" shape,
#: so compiler.embed routes it to the Ollama endpoint automatically.
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:8b"
#: Flat fallback model (role=None, and the base for any role without its own
#: model). A true instruct model, deliberately NOT a reasoning/hybrid one:
#: under the Ollama builds tested, qwen3.5:9b and Ornith ignored non-thinking
#: requests and reasoned to the token ceiling with no output on extraction.
DEFAULT_LOCAL_MODEL = "qwen2.5:7b-instruct"

#: Per-phase model roles for the local Qwen stack. Extraction is high-volume
#: and wants a fast, non-thinking, schema-constrained instruct model; wiki
#: writing wants a larger instruct model; adjudication and synthesis want a
#: reasoning model; code-agent work wants an agentic-coding model. Each role's
#: `model` ships as a default here; every field is overridable per role under
#: `llm.roles.<role>` in .deepresearch.yml. If a project sets a flat
#: `llm.local.model`, that wins for any role it doesn't name (single-model
#: back-compat); otherwise each role uses its shipped model below.
ROLE_DEFAULTS: dict[str, dict[str, Any]] = {
    "extract":             {"model": "qwen2.5:7b-instruct", "thinking": False, "temperature": 0.0, "max_tokens": 3000,  "response_format": "json"},
    "wiki_write":          {"model": "qwen3.6:35b-a3b",     "thinking": False, "temperature": 0.2, "max_tokens": 4096,  "response_format": None},
    "conflict_adjudicate": {"model": "qwen3.6:27b",         "thinking": True,  "temperature": 0.2, "max_tokens": 8192,  "response_format": None},
    "synthesize":          {"model": "qwen3.6:27b",         "thinking": True,  "temperature": 0.4, "max_tokens": 12000, "response_format": None},
    "code_agent":          {"model": "Ornith-1.0-9B",       "thinking": True,  "temperature": 0.6, "max_tokens": 16000, "response_format": None},
}


def _resolve_roles(
    roles_raw: dict[str, Any], local: dict[str, Any], flat_model_explicit: bool = True
) -> dict[str, dict[str, Any]]:
    """Resolve each role's full spec. Model precedence for a role that does not
    name its own model: the flat `llm.local.model` when the project set one
    explicitly (`flat_model_explicit`, single-model back-compat), otherwise the
    role's shipped `ROLE_DEFAULTS` model (the Qwen stack). Non-model fields
    always fall back to `ROLE_DEFAULTS`, then the flat local block.
    """
    resolved: dict[str, dict[str, Any]] = {}
    for name, d in ROLE_DEFAULTS.items():
        r = roles_raw.get(name) or {}
        model = r.get("model")
        if model is None:
            model = local["model"] if flat_model_explicit else d.get("model", local["model"])
        resolved[name] = {
            "model": model,
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
        llm_local=(_default_local := {"base_url": "http://localhost:11434/v1", "model": DEFAULT_LOCAL_MODEL,
                   "api_key_env": "OPENAI_API_KEY", "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                   "max_tokens": 16000}),
        # No flat model set by a project here, so each role uses its shipped Qwen model.
        llm_roles=_resolve_roles({}, _default_local, flat_model_explicit=False),
        topic_name="(unconfigured project)",
        scope_hint="No .deepresearch.yml found -- run `drt init` to configure this project's research scope.",
        tags=[],
        features={"web_research": False, "pdf_ingestion": False, "knowledge_compiler": False},
        llm_provider="local",
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
        "model": local.get("model", DEFAULT_LOCAL_MODEL),
        "api_key_env": local.get("api_key_env", "OPENAI_API_KEY"),
        "temperature": float(local.get("temperature", 0.6)),
        "top_p": float(local.get("top_p", 0.95)),
        "top_k": int(local.get("top_k", 20)),
        "max_tokens": int(local.get("max_tokens", 16000)),
    }
    # If the project pins a flat llm.local.model, it wins for roles it doesn't
    # name (single-model back-compat); otherwise each role uses its Qwen default.
    llm_roles = _resolve_roles(llm.get("roles") or {}, llm_local, flat_model_explicit="model" in local)

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
        llm_provider=llm.get("provider", "local"),
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
