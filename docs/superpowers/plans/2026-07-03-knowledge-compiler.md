# Knowledge Compiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the knowledge-compiler layer (hybrid DuckDB + LanceDB index + 8 retrieval tools), add web claim-extraction, and add an opt-in local LLM (Ornith-1.0-9B) backend — finishing the deep-research-toolkit skill suite per `docs/superpowers/specs/2026-07-03-knowledge-compiler-design.md`.

**Architecture:** Three producers (`knowledge_base/`, `pdf-runs/`, `research-runs/`) are normalized to a producer-agnostic `evidence_ref` and compiled into DuckDB (FTS/BM25 + graph via recursive CTEs) and LanceDB (vectors). A `retrieval-planner` skill exposes 8 cheap, deterministic tools; hybrid search is RRF-fused; `compose_dossier` enforces a verbatim-quote gate. A pluggable `llm` backend (`agent` default | `local` Ornith) can optionally automate the extraction steps, still gated by the verbatim check.

**Tech Stack:** Python ≥3.10, DuckDB (`fts` extension), LanceDB, sentence-transformers (`all-MiniLM-L6-v2`), OpenAI-compatible client for the local backend, pytest.

## Global Constraints

- Suite version is bumped to **0.2.0** everywhere it appears: `src/deep_research_toolkit/__init__.py`, `pyproject.toml`, `.claude-plugin/plugin.json`, `.codex-plugin/plugin.json`.
- **No AI/Claude attribution in commits.** Sole author is Charles Hoskinson; never add a `Co-Authored-By` trailer. Commit messages are imperative sentences (matching existing history style, e.g. "Add the DuckDB index schema"), no `feat:`/`fix:` prefixes.
- **Config-driven paths only.** Every script resolves paths via `deep_research_toolkit.config` (`load_config`, `resolve_path`) — never hardcode `knowledge_base/` or `pdf-runs/`. New config key: `knowledge_base.index_dir` (default `.deepresearch/index`).
- **Skill scripts are thin CLI shims** over `deep_research_toolkit.*` package modules (argparse + one call), matching the existing pattern in `skills/rag-eval-harness/scripts/run_eval.py`.
- **Heavy deps fail with a specific, actionable error** naming the exact extra (`deep-research-toolkit[compiler]`), never a raw traceback — mirror `deep_research_toolkit.web.fetch.ScraplingNotInstalled` / `pdf.router.PdfDepsNotInstalled`.
- **The verbatim-quote gate stays exact-substring** (`quote in source_text`). Never weaken to fuzzy/normalized matching — this is the suite's load-bearing invariant (see `pdf.eval.check_evidence_quotes_verbatim`).
- **Fast-tier tests use no torch, no network, no model download.** They install `duckdb`+`lancedb` (pip wheels) and inject the deterministic `FakeEmbedder`. Only the single heavy-tier integration test uses the real embedding model.
- **After changing anything under `skills/`, run `python scripts/sync-skill-templates.py`** so `src/deep_research_toolkit/skill_templates/` stays in sync (CI enforces this via `scripts/check-skill-templates-in-sync.py`).
- **Both `plugin.json` manifests must keep their shared fields identical** (CI enforces via `scripts/check-manifests-in-sync.py`; shared fields include `version`).

---

## File Structure

**New package modules**
- `src/deep_research_toolkit/compiler/__init__.py` — package marker
- `src/deep_research_toolkit/compiler/schema.py` — `EvidenceRef`, evidence normalizers, DDL + `create_tables`, `INDEX_SCHEMA_VERSION`
- `src/deep_research_toolkit/compiler/embed.py` — `Embedder` protocol, `FakeEmbedder`, `SentenceTransformerEmbedder`, `get_embedder`, `EmbedderNotInstalled`
- `src/deep_research_toolkit/compiler/ingest.py` — walk the 3 producers into normalized row dicts
- `src/deep_research_toolkit/compiler/build.py` — `compile_index(config, embedder=None)`
- `src/deep_research_toolkit/compiler/search.py` — `rrf_fuse`, `fts_search`, `vector_search`, `hybrid_search`
- `src/deep_research_toolkit/compiler/graph.py` — `neighbors`, `wiki_link_neighbors`
- `src/deep_research_toolkit/compiler/contradictions.py` — `find_candidates`
- `src/deep_research_toolkit/compiler/dossier.py` — `verbatim_ok`, `compose_dossier`
- `src/deep_research_toolkit/compiler/tools.py` — `Index` handle + the 8 tool functions
- `src/deep_research_toolkit/llm/__init__.py` — package marker
- `src/deep_research_toolkit/llm/backend.py` — `Backend` protocol, `get_backend`, `LLMBackendNotConfigured`
- `src/deep_research_toolkit/llm/agent.py` — `AgentBackend`
- `src/deep_research_toolkit/llm/local.py` — `LocalOpenAIBackend`, `strip_think`, `LocalLLMNotInstalled`
- `src/deep_research_toolkit/web/research_run.py` — `web_source_id`, `chunk_markdown`, `start_research_run`

**New skills**
- `skills/knowledge-compiler/SKILL.md` + `scripts/compile.py`
- `skills/retrieval-planner/SKILL.md` + `scripts/query.py` + `references/tool-contracts.md`
- `skills/knowledge-extraction/scripts/extract_claims.py` (added to existing skill)
- `skills/research-knowledge-graph/scripts/start_research_run.py`, `scripts/extract_claims.py` (added), and an updated `SKILL.md` + `references/web-claim-extraction.md`

**Modified**
- `src/deep_research_toolkit/config.py` — `index_dir`, `llm.local`, `embedding_model` fields
- `src/deep_research_toolkit/cli.py` — `DEFAULT_YAML_TEMPLATE`, `cmd_doctor` checks
- `pyproject.toml`, `__init__.py`, both `plugin.json` — version 0.2.0
- `docs/contracts/schema-versions.md`, `CHANGELOG.md`, `README.md`
- `.github/workflows/ci.yml` — install `duckdb lancedb` in the fast job
- `scripts/validate-local-llm.py` (new, repo-level manual harness)

**New docs**
- `docs/contracts/knowledge-compiler.md`
- `docs/decisions/0002-knowledge-compiler.md`

**New fixtures**
- `tests/fixtures/reference-kb/` — small OKF knowledge base
- `tests/fixtures/reference-run-web-ows/` — synthetic web research run

---

## Task 1: Config and version scaffolding

**Files:**
- Modify: `src/deep_research_toolkit/config.py`
- Modify: `src/deep_research_toolkit/__init__.py`
- Modify: `pyproject.toml:7` (version)
- Modify: `.claude-plugin/plugin.json:4`, `.codex-plugin/plugin.json:3`
- Modify: `src/deep_research_toolkit/cli.py` (`DEFAULT_YAML_TEMPLATE`)
- Test: `tests/unit/test_config.py` (extend)

**Interfaces:**
- Produces: `Config.index_dir: Path`, `Config.embedding_model: str`, `Config.llm_local: dict` (keys `base_url`, `model`, `api_key_env`, `temperature`, `top_p`, `top_k`). `Config.llm_provider` now accepts `"agent" | "anthropic" | "local"` (`anthropic` treated as agent-equivalent).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_config.py`:

```python
def test_index_dir_defaults_and_resolves(tmp_path):
    from deep_research_toolkit.config import load_config
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\nknowledge_base:\n  path: kb\n", encoding="utf-8"
    )
    cfg = load_config(tmp_path)
    assert cfg.index_dir == (tmp_path / ".deepresearch/index").resolve()
    assert cfg.embedding_model == "all-MiniLM-L6-v2"


def test_llm_local_block_parsed(tmp_path):
    from deep_research_toolkit.config import load_config
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\n"
        "llm:\n"
        "  provider: local\n"
        "  local:\n"
        "    base_url: http://localhost:11434/v1\n"
        "    model: Ornith-1.0-9B\n",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.llm_provider == "local"
    assert cfg.llm_local["base_url"] == "http://localhost:11434/v1"
    assert cfg.llm_local["model"] == "Ornith-1.0-9B"
    assert cfg.llm_local["temperature"] == 0.6  # default filled in
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py -k "index_dir or llm_local" -v`
Expected: FAIL — `AttributeError: 'Config' object has no attribute 'index_dir'`.

- [ ] **Step 3: Implement the config changes**

In `src/deep_research_toolkit/config.py`, add a constant near the other defaults:

```python
DEFAULT_INDEX_DIR = ".deepresearch/index"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

Add fields to the `Config` dataclass (after `research_runs_path`):

```python
    index_dir: Path
    embedding_model: str
    llm_local: dict[str, Any]
```

In `_default_config`, add:

```python
        index_dir=project_root / DEFAULT_INDEX_DIR,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        llm_local={"base_url": "http://localhost:11434/v1", "model": "Ornith-1.0-9B",
                   "api_key_env": "OPENAI_API_KEY", "temperature": 0.6, "top_p": 0.95, "top_k": 20},
```

In `load_config`, after the existing block parsing, add:

```python
    local = (llm.get("local") or {})
    llm_local = {
        "base_url": local.get("base_url", "http://localhost:11434/v1"),
        "model": local.get("model", "Ornith-1.0-9B"),
        "api_key_env": local.get("api_key_env", "OPENAI_API_KEY"),
        "temperature": float(local.get("temperature", 0.6)),
        "top_p": float(local.get("top_p", 0.95)),
        "top_k": int(local.get("top_k", 20)),
    }
```

and pass to the `Config(...)` constructor:

```python
        index_dir=(root / kb.get("index_dir", DEFAULT_INDEX_DIR)).resolve(),
        embedding_model=llm.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        llm_local=llm_local,
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_config.py -v`
Expected: PASS (all existing + 2 new).

- [ ] **Step 5: Bump versions and update the YAML template**

Set `__version__ = "0.2.0"` in `src/deep_research_toolkit/__init__.py`.
Set `version = "0.2.0"` in `pyproject.toml`. Set `"version": "0.2.0"` in both `.claude-plugin/plugin.json` and `.codex-plugin/plugin.json`.

In `cli.py`, extend `DEFAULT_YAML_TEMPLATE`'s `knowledge_base:` block with `  index_dir: {index_dir}\n` and the `llm:` block to include the `local:` sub-block and `embedding_model:`. Add `index_dir=".deepresearch/index"` to the `.format(...)` call in `cmd_init`. Verify the template still formats:

Run: `python -c "from deep_research_toolkit.cli import DEFAULT_YAML_TEMPLATE as t; print('index_dir' in t and 'local' in t)"`
Expected: `True`.

- [ ] **Step 6: Verify manifest sync and commit**

Run: `python scripts/check-manifests-in-sync.py`
Expected: `OK: 6 shared fields match ...`

```bash
git add -A
git commit -m "Add index_dir, embedding_model, and llm.local config; bump suite to 0.2.0"
```

---

## Task 2: evidence_ref normalization and index schema

**Files:**
- Create: `src/deep_research_toolkit/compiler/__init__.py`
- Create: `src/deep_research_toolkit/compiler/schema.py`
- Test: `tests/unit/test_compiler_schema.py`

**Interfaces:**
- Produces:
  - `INDEX_SCHEMA_VERSION: str = "1.0"`
  - `@dataclass EvidenceRef(producer: str, source_id: str, locator: str | None, quote: str, page: int | None = None, url: str | None = None)`
  - `normalize_evidence(claim: dict, producer: str, source_id: str) -> list[EvidenceRef]` — reads `claim["supporting_evidence"]`; PDF entries use `node_id`+`page`, web entries use `locator`+`url`.
  - `create_tables(con) -> None` — creates all DuckDB tables (idempotent: `CREATE TABLE IF NOT EXISTS`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_compiler_schema.py
import duckdb
from deep_research_toolkit.compiler.schema import (
    EvidenceRef, normalize_evidence, create_tables, INDEX_SCHEMA_VERSION,
)


def test_normalize_pdf_evidence_maps_node_id_and_page():
    claim = {"supporting_evidence": [{"node_id": "doc:n005", "quote": "Hydra can be used", "page": 1}]}
    refs = normalize_evidence(claim, producer="pdf", source_id="doc")
    assert refs == [EvidenceRef(producer="pdf", source_id="doc", locator="doc:n005",
                                quote="Hydra can be used", page=1, url=None)]


def test_normalize_web_evidence_maps_locator_and_url():
    claim = {"supporting_evidence": [{"locator": "src:c02", "quote": "OWS delegates signing", "url": "https://x/y"}]}
    refs = normalize_evidence(claim, producer="web", source_id="src")
    assert refs == [EvidenceRef(producer="web", source_id="src", locator="src:c02",
                                quote="OWS delegates signing", page=None, url="https://x/y")]


def test_create_tables_makes_all_expected_tables():
    con = duckdb.connect(":memory:")
    create_tables(con)
    names = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    assert {"wiki_pages", "wiki_links", "claims", "claim_evidence",
            "entities", "entity_mentions", "relations", "meta"} <= names
    assert INDEX_SCHEMA_VERSION == "1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_compiler_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deep_research_toolkit.compiler'`.

- [ ] **Step 3: Implement the schema module**

Create `src/deep_research_toolkit/compiler/__init__.py`:

```python
"""Knowledge compiler: index knowledge_base/ + *-runs/ into DuckDB + LanceDB."""
```

Create `src/deep_research_toolkit/compiler/schema.py`:

```python
"""Index schema: producer-agnostic evidence_ref + the DuckDB table DDL.

The compiler normalizes PDF- and web-sourced evidence into one EvidenceRef
shape at index time; on-disk producer files keep their native shapes (see
docs/contracts/knowledge-compiler.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

INDEX_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class EvidenceRef:
    producer: str          # "pdf" | "web"
    source_id: str         # document_id (pdf) or web source_id
    locator: str | None    # node_id (pdf) or research chunk_id (web)
    quote: str
    page: int | None = None
    url: str | None = None


def normalize_evidence(claim: dict[str, Any], producer: str, source_id: str) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    for ev in claim.get("supporting_evidence") or []:
        refs.append(
            EvidenceRef(
                producer=producer,
                source_id=source_id,
                locator=ev.get("node_id") if producer == "pdf" else ev.get("locator"),
                quote=ev.get("quote") or "",
                page=ev.get("page") if producer == "pdf" else None,
                url=ev.get("url") if producer == "web" else None,
            )
        )
    return refs


_DDL = """
CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR);
CREATE TABLE IF NOT EXISTS wiki_pages (
    path VARCHAR PRIMARY KEY, type VARCHAR, title VARCHAR, status VARCHAR,
    timestamp VARCHAR, body VARCHAR, frontmatter_json VARCHAR
);
CREATE TABLE IF NOT EXISTS wiki_links (from_path VARCHAR, to_path VARCHAR);
CREATE TABLE IF NOT EXISTS claims (
    claim_id VARCHAR, producer VARCHAR, source_id VARCHAR, claim VARCHAR,
    claim_type VARCHAR, confidence VARCHAR
);
CREATE TABLE IF NOT EXISTS claim_evidence (
    claim_id VARCHAR, producer VARCHAR, source_id VARCHAR, locator VARCHAR,
    page INTEGER, url VARCHAR, quote VARCHAR
);
CREATE TABLE IF NOT EXISTS entities (
    entity_id VARCHAR, name VARCHAR, type VARCHAR, aliases_json VARCHAR,
    producer VARCHAR, source_id VARCHAR
);
CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_id VARCHAR, locator VARCHAR, producer VARCHAR, source_id VARCHAR
);
CREATE TABLE IF NOT EXISTS relations (
    relation_id VARCHAR, subject VARCHAR, predicate VARCHAR, object VARCHAR,
    supporting_claim VARCHAR, producer VARCHAR, source_id VARCHAR
);
"""


def create_tables(con) -> None:
    con.execute(_DDL)
    con.execute("INSERT OR REPLACE INTO meta VALUES ('index_schema_version', ?)", [INDEX_SCHEMA_VERSION])
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_compiler_schema.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/compiler/ tests/unit/test_compiler_schema.py
git commit -m "Add evidence_ref normalization and the DuckDB index schema"
```

---

## Task 3: Embedder interface

**Files:**
- Create: `src/deep_research_toolkit/compiler/embed.py`
- Test: `tests/unit/test_compiler_embed.py`
- Test: `tests/unit/test_dependency_boundary.py` (extend)

**Interfaces:**
- Produces:
  - `class Embedder(Protocol): def embed(self, texts: list[str]) -> list[list[float]]: ...`
  - `class FakeEmbedder: dim: int = 16; def embed(...)` — deterministic hash-based unit-ish vectors, no deps.
  - `class SentenceTransformerEmbedder: def __init__(self, model_name: str); def embed(...)` — lazy import.
  - `EmbedderNotInstalled(RuntimeError)`
  - `get_embedder(model_name: str) -> Embedder` — returns the real one.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_compiler_embed.py
from deep_research_toolkit.compiler.embed import FakeEmbedder


def test_fake_embedder_is_deterministic_and_fixed_dim():
    e = FakeEmbedder()
    a = e.embed(["hydra", "cardano"])
    b = e.embed(["hydra", "cardano"])
    assert a == b
    assert len(a) == 2 and all(len(v) == e.dim for v in a)


def test_fake_embedder_distinguishes_texts():
    e = FakeEmbedder()
    v1, v2 = e.embed(["alpha", "completely different text"])
    assert v1 != v2
```

Extend `tests/unit/test_dependency_boundary.py`:

```python
def test_sentence_transformer_embedder_without_dep_raises_specific_error(monkeypatch):
    import sys
    from deep_research_toolkit.compiler.embed import SentenceTransformerEmbedder, EmbedderNotInstalled
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with __import__("pytest").raises(EmbedderNotInstalled) as exc:
        SentenceTransformerEmbedder("all-MiniLM-L6-v2").embed(["x"])
    assert "deep-research-toolkit[compiler]" in str(exc.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_compiler_embed.py tests/unit/test_dependency_boundary.py -k "embedder or Embedder" -v`
Expected: FAIL — module/attribute not found.

- [ ] **Step 3: Implement embed.py**

```python
"""Embeddings for the vector index. Real path uses sentence-transformers;
tests inject FakeEmbedder so both index engines run without torch."""
from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable


class EmbedderNotInstalled(RuntimeError):
    pass


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """Deterministic, dependency-free embedder for tests. Not for production
    (no semantic meaning) -- production requires SentenceTransformerEmbedder."""

    dim = 16

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            vals = [((digest[i % len(digest)] / 255.0) * 2 - 1) for i in range(self.dim)]
            norm = math.sqrt(sum(v * v for v in vals)) or 1.0
            out.append([v / norm for v in vals])
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise EmbedderNotInstalled(
                    "sentence-transformers is required for the knowledge compiler. "
                    'Install it with: pip install "deep-research-toolkit[compiler]"'
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        return [list(map(float, v)) for v in model.encode(texts, normalize_embeddings=True)]


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    return SentenceTransformerEmbedder(model_name)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_compiler_embed.py tests/unit/test_dependency_boundary.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/compiler/embed.py tests/unit/test_compiler_embed.py tests/unit/test_dependency_boundary.py
git commit -m "Add Embedder interface with deterministic test fake and lazy sentence-transformers backend"
```

---

## Task 4: Ingest — walk producers into normalized rows

**Files:**
- Create: `src/deep_research_toolkit/compiler/ingest.py`
- Test: `tests/unit/test_compiler_ingest.py`

**Interfaces:**
- Consumes: `EvidenceRef`, `normalize_evidence` (Task 2); `common.frontmatter.read_okf`, `find_links`, `resolve_link`.
- Produces:
  - `iter_wiki_pages(kb_dir: Path) -> list[dict]` — rows `{path, type, title, status, timestamp, body, frontmatter_json, links: list[str]}` (`path` is kb-root-relative, POSIX slashes; `links` are resolved kb-relative target paths).
  - `iter_run_claims(run_dir: Path, producer: str) -> tuple[list[dict], list[EvidenceRef-rows]]` — reads `claims.jsonl`; returns claim rows + flattened evidence rows (evidence row = `{claim_id, producer, source_id, locator, page, url, quote}`). `source_id` = `manifest.json`'s `document_id` (pdf) or the run dir name (web).
  - `iter_run_entities(run_dir, producer) -> tuple[list[dict], list[dict]]` — entity rows + mention rows.
  - `iter_run_relations(run_dir, producer) -> list[dict]`.
  - `discover_runs(runs_root: Path) -> list[Path]` — subdirectories containing a `claims.jsonl`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_compiler_ingest.py
import json
from deep_research_toolkit.common.frontmatter import write_okf
from deep_research_toolkit.compiler import ingest


def test_iter_wiki_pages_reads_frontmatter_and_links(tmp_path):
    kb = tmp_path / "kb"
    write_okf(kb / "index.md", {"type": "Index", "title": "Index", "timestamp": "t"}, "[A](/concepts/a.md)\n")
    write_okf(kb / "concepts/a.md", {"type": "Concept", "title": "A", "timestamp": "t", "status": "seed"}, "body A\n")
    rows = {r["path"]: r for r in ingest.iter_wiki_pages(kb)}
    assert set(rows) == {"index.md", "concepts/a.md"}
    assert rows["index.md"]["links"] == ["concepts/a.md"]
    assert rows["concepts/a.md"]["title"] == "A"


def test_iter_run_claims_normalizes_pdf_evidence(tmp_path):
    run = tmp_path / "doc-abc"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"document_id": "doc-abc"}), encoding="utf-8")
    (run / "claims.jsonl").write_text(json.dumps({
        "claim_id": "c1", "claim": "X", "claim_type": "architectural", "confidence": "high",
        "supporting_evidence": [{"node_id": "doc-abc:n5", "quote": "X", "page": 2}],
    }) + "\n", encoding="utf-8")
    claim_rows, ev_rows = ingest.iter_run_claims(run, producer="pdf")
    assert claim_rows[0]["source_id"] == "doc-abc"
    assert ev_rows[0]["locator"] == "doc-abc:n5" and ev_rows[0]["page"] == 2


def test_discover_runs_finds_only_dirs_with_claims(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "a" / "claims.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "b").mkdir()
    assert [p.name for p in ingest.discover_runs(tmp_path)] == ["a"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_compiler_ingest.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement ingest.py**

```python
"""Walk the three producers (knowledge_base/, pdf-runs/, research-runs/)
into flat row dicts ready to insert into the DuckDB index."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..common.frontmatter import read_okf, resolve_link
from .schema import normalize_evidence


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_wiki_pages(kb_dir: Path) -> list[dict[str, Any]]:
    kb_dir = Path(kb_dir)
    rows: list[dict[str, Any]] = []
    for md in sorted(kb_dir.rglob("*.md")):
        page = read_okf(md)
        if page is None or "type" not in page.frontmatter:
            continue
        rel = md.relative_to(kb_dir).as_posix()
        links = []
        for target in page.links:
            resolved = resolve_link(target, md, kb_dir)
            try:
                links.append(resolved.relative_to(kb_dir).as_posix())
            except ValueError:
                continue  # link points outside the kb; not a graph edge
        rows.append({
            "path": rel,
            "type": page.frontmatter.get("type"),
            "title": page.frontmatter.get("title"),
            "status": page.frontmatter.get("status"),
            "timestamp": str(page.frontmatter.get("timestamp", "")),
            "body": page.body,
            "frontmatter_json": json.dumps(page.frontmatter, default=str),
            "links": links,
        })
    return rows


def _source_id(run_dir: Path, producer: str) -> str:
    if producer == "pdf":
        manifest = run_dir / "manifest.json"
        if manifest.is_file():
            return json.loads(manifest.read_text(encoding="utf-8")).get("document_id", run_dir.name)
    return run_dir.name


def iter_run_claims(run_dir: Path, producer: str) -> tuple[list[dict], list[dict]]:
    run_dir = Path(run_dir)
    sid = _source_id(run_dir, producer)
    claim_rows, ev_rows = [], []
    for claim in _read_jsonl(run_dir / "claims.jsonl"):
        cid = claim.get("claim_id")
        claim_rows.append({
            "claim_id": cid, "producer": producer, "source_id": sid,
            "claim": claim.get("claim", ""), "claim_type": claim.get("claim_type"),
            "confidence": claim.get("confidence"),
        })
        for ref in normalize_evidence(claim, producer, sid):
            ev_rows.append({
                "claim_id": cid, "producer": ref.producer, "source_id": ref.source_id,
                "locator": ref.locator, "page": ref.page, "url": ref.url, "quote": ref.quote,
            })
    return claim_rows, ev_rows


def iter_run_entities(run_dir: Path, producer: str) -> tuple[list[dict], list[dict]]:
    run_dir = Path(run_dir)
    sid = _source_id(run_dir, producer)
    entity_rows, mention_rows = [], []
    for ent in _read_jsonl(run_dir / "entities.jsonl"):
        eid = ent.get("entity_id")
        entity_rows.append({
            "entity_id": eid, "name": ent.get("name"), "type": ent.get("type"),
            "aliases_json": json.dumps(ent.get("aliases") or []), "producer": producer, "source_id": sid,
        })
        for locator in ent.get("mentions") or []:
            mention_rows.append({"entity_id": eid, "locator": locator, "producer": producer, "source_id": sid})
    return entity_rows, mention_rows


def iter_run_relations(run_dir: Path, producer: str) -> list[dict]:
    run_dir = Path(run_dir)
    sid = _source_id(run_dir, producer)
    rows = []
    for rel in _read_jsonl(run_dir / "relations.jsonl"):
        rows.append({
            "relation_id": rel.get("relation_id"), "subject": rel.get("subject"),
            "predicate": rel.get("predicate"), "object": rel.get("object"),
            "supporting_claim": rel.get("supporting_claim"), "producer": producer, "source_id": sid,
        })
    return rows


def discover_runs(runs_root: Path) -> list[Path]:
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        return []
    return sorted(p for p in runs_root.iterdir() if p.is_dir() and (p / "claims.jsonl").is_file())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_compiler_ingest.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/compiler/ingest.py tests/unit/test_compiler_ingest.py
git commit -m "Add ingest: walk knowledge base and run directories into index rows"
```

---

## Task 5: Build the index (DuckDB + LanceDB)

**Files:**
- Create: `src/deep_research_toolkit/compiler/build.py`
- Test: `tests/unit/test_compiler_build.py`

**Interfaces:**
- Consumes: `schema.create_tables`, all `ingest.*` functions, `Embedder` (Task 3), `Config` (Task 1).
- Produces:
  - `compile_index(config, embedder: Embedder | None = None) -> dict` — builds `<index_dir>/knowledge.duckdb` and `<index_dir>/lancedb/`; returns stats `{"wiki_pages": n, "claims": n, "entities": n, "relations": n, "wiki_chunks": n, "claim_vectors": n}`. When `embedder` is None, calls `get_embedder(config.embedding_model)` (real). Full rebuild: drops and recreates tables each call.
  - `open_duckdb(index_dir: Path)` and `open_lancedb(index_dir: Path)` helpers (used by tools in Task 10).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_compiler_build.py
import json
from deep_research_toolkit.common.frontmatter import write_okf
from deep_research_toolkit.compiler.build import compile_index, open_duckdb
from deep_research_toolkit.compiler.embed import FakeEmbedder
from deep_research_toolkit.config import load_config


def _project(tmp_path):
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\nknowledge_base:\n  path: kb\n  pdf_runs_dir: pdf-runs\n"
        "  research_runs_dir: research-runs\n  index_dir: idx\n", encoding="utf-8")
    kb = tmp_path / "kb"
    write_okf(kb / "concepts/hydra.md",
              {"type": "Concept", "title": "Hydra", "timestamp": "t", "status": "draft"}, "Hydra body\n")
    run = tmp_path / "pdf-runs" / "doc-abc"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"document_id": "doc-abc"}), encoding="utf-8")
    (run / "claims.jsonl").write_text(json.dumps({
        "claim_id": "c1", "claim": "Hydra settles synchronously", "claim_type": "architectural",
        "confidence": "high", "supporting_evidence": [{"node_id": "doc-abc:n5", "quote": "settles", "page": 1}],
    }) + "\n", encoding="utf-8")
    return load_config(tmp_path)


def test_compile_index_populates_both_engines(tmp_path):
    cfg = _project(tmp_path)
    stats = compile_index(cfg, embedder=FakeEmbedder())
    assert stats["wiki_pages"] == 1 and stats["claims"] == 1
    con = open_duckdb(cfg.index_dir)
    assert con.execute("SELECT count(*) FROM claims").fetchone()[0] == 1
    assert con.execute("SELECT quote FROM claim_evidence WHERE claim_id='c1'").fetchone()[0] == "settles"


def test_compile_index_is_idempotent(tmp_path):
    cfg = _project(tmp_path)
    compile_index(cfg, embedder=FakeEmbedder())
    stats = compile_index(cfg, embedder=FakeEmbedder())  # second run must not double rows
    assert stats["wiki_pages"] == 1 and stats["claims"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_compiler_build.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement build.py**

```python
"""Compile knowledge_base/ + *-runs/ into a DuckDB + LanceDB index.

v1 is a full rebuild each call (drop + recreate). Incremental compilation is
deferred (see docs/decisions/0002-knowledge-compiler.md)."""
from __future__ import annotations

import shutil
from pathlib import Path

import duckdb

from . import ingest
from .embed import Embedder, get_embedder
from .schema import create_tables


def open_duckdb(index_dir: Path):
    con = duckdb.connect(str(Path(index_dir) / "knowledge.duckdb"))
    con.execute("INSTALL fts; LOAD fts;")
    return con


def open_lancedb(index_dir: Path):
    import lancedb
    return lancedb.connect(str(Path(index_dir) / "lancedb"))


def _insert(con, table: str, rows: list[dict], columns: list[str]) -> None:
    if not rows:
        return
    placeholders = ", ".join("?" for _ in columns)
    con.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        [[r.get(c) for c in columns] for r in rows],
    )


def compile_index(config, embedder: Embedder | None = None) -> dict:
    index_dir = Path(config.index_dir)
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    embedder = embedder or get_embedder(config.embedding_model)

    con = open_duckdb(index_dir)
    create_tables(con)

    wiki = ingest.iter_wiki_pages(config.knowledge_base_path)
    _insert(con, "wiki_pages", wiki,
            ["path", "type", "title", "status", "timestamp", "body", "frontmatter_json"])
    link_rows = [{"from_path": w["path"], "to_path": t} for w in wiki for t in w["links"]]
    _insert(con, "wiki_links", link_rows, ["from_path", "to_path"])

    claims, evidence, entities, mentions, relations = [], [], [], [], []
    for producer, root in [("pdf", config.pdf_runs_path), ("web", config.research_runs_path)]:
        for run in ingest.discover_runs(root):
            c, e = ingest.iter_run_claims(run, producer); claims += c; evidence += e
            en, mn = ingest.iter_run_entities(run, producer); entities += en; mentions += mn
            relations += ingest.iter_run_relations(run, producer)

    _insert(con, "claims", claims, ["claim_id", "producer", "source_id", "claim", "claim_type", "confidence"])
    _insert(con, "claim_evidence", evidence,
            ["claim_id", "producer", "source_id", "locator", "page", "url", "quote"])
    _insert(con, "entities", entities, ["entity_id", "name", "type", "aliases_json", "producer", "source_id"])
    _insert(con, "entity_mentions", mentions, ["entity_id", "locator", "producer", "source_id"])
    _insert(con, "relations", relations,
            ["relation_id", "subject", "predicate", "object", "supporting_claim", "producer", "source_id"])

    if wiki:
        con.execute("PRAGMA create_fts_index('wiki_pages', 'path', 'body', overwrite=1)")
    if claims:
        con.execute("PRAGMA create_fts_index('claims', 'claim_id', 'claim', overwrite=1)")

    # LanceDB vector tables
    wiki_vecs = _build_vectors(embedder, [(w["path"], w["body"]) for w in wiki])
    claim_vecs = _build_vectors(embedder, [(c["claim_id"], c["claim"]) for c in claims])
    if wiki_vecs or claim_vecs:
        db = open_lancedb(index_dir)
        if wiki_vecs:
            db.create_table("wiki_chunks", data=wiki_vecs, mode="overwrite")
        if claim_vecs:
            db.create_table("claim_vectors", data=claim_vecs, mode="overwrite")

    con.close()
    return {
        "wiki_pages": len(wiki), "claims": len(claims), "entities": len(entities),
        "relations": len(relations), "wiki_chunks": len(wiki_vecs), "claim_vectors": len(claim_vecs),
    }


def _build_vectors(embedder: Embedder, id_text: list[tuple[str, str]]) -> list[dict]:
    if not id_text:
        return []
    vectors = embedder.embed([t for _, t in id_text])
    return [{"id": i, "text": t, "vector": v} for (i, t), v in zip(id_text, vectors)]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_compiler_build.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/compiler/build.py tests/unit/test_compiler_build.py
git commit -m "Add compile_index: build DuckDB tables, FTS indexes, and LanceDB vectors"
```

---

## Task 6: Search — FTS, vector, RRF fusion

**Files:**
- Create: `src/deep_research_toolkit/compiler/search.py`
- Test: `tests/unit/test_compiler_search.py`

**Interfaces:**
- Consumes: an open DuckDB connection, an open LanceDB handle, `Embedder`.
- Produces:
  - `rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]` — pure; ids ordered best-first per list; returns fused `(id, score)` best-first.
  - `fts_search(con, table: str, id_col: str, text_col: str, query: str, k: int) -> list[str]` — ids best-first via `match_bm25`.
  - `vector_search(lancedb_handle, table: str, embedder, query: str, k: int) -> list[str]` — ids best-first; returns `[]` if the table is absent.
  - `hybrid_search(con, lancedb_handle, embedder, *, table, id_col, text_col, vec_table, query, k) -> list[str]` — RRF over the two id lists.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_compiler_search.py
from deep_research_toolkit.compiler.search import rrf_fuse


def test_rrf_rewards_agreement_across_lists():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] == "b"          # top of one, second of the other -> best combined
    assert set(ids[:2]) == {"a", "b"}


def test_rrf_handles_single_list_and_unique_ids():
    fused = rrf_fuse([["x", "y"]])
    assert [i for i, _ in fused] == ["x", "y"]


def test_rrf_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_compiler_search.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement search.py**

```python
"""Lexical (DuckDB FTS/BM25) + vector (LanceDB) search, fused with
Reciprocal Rank Fusion. RRF constant k=60 is the standard default."""
from __future__ import annotations


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def fts_search(con, table: str, id_col: str, text_col: str, query: str, k: int) -> list[str]:
    sql = (
        f"SELECT {id_col}, fts_main_{table}.match_bm25({id_col}, ?) AS score "
        f"FROM {table} WHERE score IS NOT NULL ORDER BY score DESC LIMIT {int(k)}"
    )
    return [row[0] for row in con.execute(sql, [query]).fetchall()]


def vector_search(lancedb_handle, table: str, embedder, query: str, k: int) -> list[str]:
    try:
        tbl = lancedb_handle.open_table(table)
    except Exception:
        return []
    qvec = embedder.embed([query])[0]
    rows = tbl.search(qvec).limit(k).to_list()
    return [r["id"] for r in rows]


def hybrid_search(con, lancedb_handle, embedder, *, table, id_col, text_col, vec_table, query, k) -> list[str]:
    lexical = fts_search(con, table, id_col, text_col, query, k)
    vector = vector_search(lancedb_handle, vec_table, embedder, query, k) if lancedb_handle else []
    return [i for i, _ in rrf_fuse([lexical, vector])][:k]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_compiler_search.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/compiler/search.py tests/unit/test_compiler_search.py
git commit -m "Add FTS + vector search with reciprocal rank fusion"
```

---

## Task 7: Graph — entity and wiki-link neighbors

**Files:**
- Create: `src/deep_research_toolkit/compiler/graph.py`
- Test: `tests/unit/test_compiler_graph.py`

**Interfaces:**
- Consumes: an open DuckDB connection with the `relations` and `wiki_links` tables populated.
- Produces:
  - `neighbors(con, entity: str, depth: int = 1) -> list[dict]` — recursive walk over `relations` treating `(subject, object)` as undirected edges; returns `[{"node": str, "depth": int}]` for nodes reachable within `depth`, excluding the start node, nearest depth first.
  - `wiki_link_neighbors(con, path: str, depth: int = 1) -> list[dict]` — same over `wiki_links` (directed `from_path -> to_path`, walked as undirected for reachability).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_compiler_graph.py
import duckdb
from deep_research_toolkit.compiler.schema import create_tables
from deep_research_toolkit.compiler.graph import neighbors


def _con():
    con = duckdb.connect(":memory:")
    create_tables(con)
    con.executemany(
        "INSERT INTO relations (relation_id, subject, predicate, object) VALUES (?, ?, ?, ?)",
        [("r1", "hydra", "serves_as", "settlement"),
         ("r2", "hydra", "open_question", "ows"),
         ("r3", "ows", "defined_by", "spec")],
    )
    return con


def test_neighbors_depth_1():
    got = {n["node"] for n in neighbors(_con(), "hydra", depth=1)}
    assert got == {"settlement", "ows"}


def test_neighbors_depth_2_reaches_further():
    got = {n["node"] for n in neighbors(_con(), "hydra", depth=2)}
    assert "spec" in got  # hydra -> ows -> spec


def test_neighbors_excludes_self():
    assert all(n["node"] != "hydra" for n in neighbors(_con(), "hydra", depth=3))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_compiler_graph.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement graph.py**

```python
"""Graph walks over the relation and wiki-link tables via DuckDB recursive
CTEs. Edges are treated as undirected for reachability. UNION (not UNION ALL)
terminates on cycles."""
from __future__ import annotations


def _walk(con, edge_sql: str, start: str, depth: int) -> list[dict]:
    sql = f"""
    WITH RECURSIVE edges(a, b) AS ({edge_sql}),
    walk(node, depth) AS (
        SELECT ?, 0
        UNION
        SELECT CASE WHEN e.a = w.node THEN e.b ELSE e.a END, w.depth + 1
        FROM walk w JOIN edges e ON (e.a = w.node OR e.b = w.node)
        WHERE w.depth < ?
    )
    SELECT node, min(depth) AS depth FROM walk WHERE node <> ?
    GROUP BY node ORDER BY depth, node
    """
    return [{"node": r[0], "depth": r[1]} for r in con.execute(sql, [start, int(depth), start]).fetchall()]


def neighbors(con, entity: str, depth: int = 1) -> list[dict]:
    return _walk(con, "SELECT subject, object FROM relations WHERE subject IS NOT NULL AND object IS NOT NULL",
                 entity, depth)


def wiki_link_neighbors(con, path: str, depth: int = 1) -> list[dict]:
    return _walk(con, "SELECT from_path, to_path FROM wiki_links", path, depth)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_compiler_graph.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/compiler/graph.py tests/unit/test_compiler_graph.py
git commit -m "Add recursive-CTE graph walks for entities and wiki links"
```

---

## Task 8: Contradictions — mechanical candidate detector

**Files:**
- Create: `src/deep_research_toolkit/compiler/contradictions.py`
- Test: `tests/unit/test_compiler_contradictions.py`

**Interfaces:**
- Consumes: an open DuckDB connection with `relations`, `wiki_pages` populated.
- Produces:
  - `find_candidates(con) -> list[dict]` — returns candidate contradiction pairs. Two kinds: (a) `{"kind": "relation", "subject", "predicate", "objects": [o1, o2], "relation_ids": [...], "source_ids": [...]}` when the same `(subject, predicate)` has ≥2 distinct `object` values; (b) `{"kind": "conflicted_page", "path": ...}` for each `wiki_pages` row with `status = 'conflicted'`. Confirmation is a downstream LLM step, never done here.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_compiler_contradictions.py
import duckdb
from deep_research_toolkit.compiler.schema import create_tables
from deep_research_toolkit.compiler.contradictions import find_candidates


def test_conflicting_objects_are_flagged():
    con = duckdb.connect(":memory:"); create_tables(con)
    con.executemany(
        "INSERT INTO relations (relation_id, subject, predicate, object, source_id) VALUES (?,?,?,?,?)",
        [("r1", "hydra", "throughput", "1000 TPS", "docA"),
         ("r2", "hydra", "throughput", "500 TPS", "docB"),
         ("r3", "hydra", "phase_count", "4", "docA")],
    )
    cands = [c for c in find_candidates(con) if c["kind"] == "relation"]
    assert len(cands) == 1
    assert cands[0]["subject"] == "hydra" and set(cands[0]["objects"]) == {"1000 TPS", "500 TPS"}


def test_conflicted_status_pages_are_flagged():
    con = duckdb.connect(":memory:"); create_tables(con)
    con.execute("INSERT INTO wiki_pages (path, status) VALUES ('c/x.md', 'conflicted')")
    con.execute("INSERT INTO wiki_pages (path, status) VALUES ('c/y.md', 'draft')")
    paths = [c["path"] for c in find_candidates(con) if c["kind"] == "conflicted_page"]
    assert paths == ["c/x.md"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_compiler_contradictions.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement contradictions.py**

```python
"""Mechanical contradiction *candidate* detection at compile/query time.
Confirming a candidate is a real contradiction is a batched LLM step driven
by the retrieval-planner SKILL.md -- never done in this module (ADR 0001
decision #3: tools stay cheap and deterministic)."""
from __future__ import annotations


def find_candidates(con) -> list[dict]:
    candidates: list[dict] = []

    rows = con.execute("""
        SELECT subject, predicate,
               list(DISTINCT object)  AS objects,
               list(relation_id)      AS relation_ids,
               list(DISTINCT source_id) AS source_ids
        FROM relations
        WHERE subject IS NOT NULL AND predicate IS NOT NULL AND object IS NOT NULL
        GROUP BY subject, predicate
        HAVING count(DISTINCT object) > 1
    """).fetchall()
    for subject, predicate, objects, relation_ids, source_ids in rows:
        candidates.append({
            "kind": "relation", "subject": subject, "predicate": predicate,
            "objects": list(objects), "relation_ids": list(relation_ids),
            "source_ids": list(source_ids),
        })

    for (path,) in con.execute(
        "SELECT path FROM wiki_pages WHERE status = 'conflicted' ORDER BY path"
    ).fetchall():
        candidates.append({"kind": "conflicted_page", "path": path})

    return candidates
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_compiler_contradictions.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/compiler/contradictions.py tests/unit/test_compiler_contradictions.py
git commit -m "Add mechanical contradiction candidate detector"
```

---

## Task 9: Dossier — verbatim gate

**Files:**
- Create: `src/deep_research_toolkit/compiler/dossier.py`
- Test: `tests/unit/test_compiler_dossier.py`

**Interfaces:**
- Consumes: an open DuckDB connection (`claims`, `claim_evidence`), and source-text lookup for the verbatim gate.
- Produces:
  - `verbatim_ok(quote: str, source_text: str) -> bool` — exact `quote in source_text` after nothing but requiring a non-empty quote. Same semantics as `pdf.eval.check_evidence_quotes_verbatim`.
  - `source_text_for(evidence_row: dict, config) -> str` — resolves the source text a quote must appear in: PDF → the text on that `page` in `<pdf_runs>/<source_id>/provenance.jsonl`; web → `<research_runs>/<source_id>/source.md`. Returns `""` if unavailable.
  - `compose_dossier(con, config, claim_ids: list[str]) -> dict` — returns `{"included": [...], "rejected": [...]}`, where each claim's evidence is checked; a claim is `included` only if **every** evidence quote passes the gate, else `rejected` with a reason.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_compiler_dossier.py
import json
import duckdb
from types import SimpleNamespace
from deep_research_toolkit.compiler.schema import create_tables
from deep_research_toolkit.compiler.dossier import verbatim_ok, compose_dossier


def test_verbatim_ok_is_exact_substring():
    assert verbatim_ok("settles instantly", "text where it settles instantly here")
    assert not verbatim_ok("settles quickly", "text where it settles instantly here")
    assert not verbatim_ok("", "anything")


def test_compose_dossier_drops_paraphrased_claim(tmp_path):
    # web source.md is the quote target
    run = tmp_path / "research-runs" / "src-1"
    run.mkdir(parents=True)
    (run / "source.md").write_text("Hydra settles instantly among participants.", encoding="utf-8")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")

    con = duckdb.connect(":memory:"); create_tables(con)
    con.executemany("INSERT INTO claims (claim_id, producer, source_id, claim) VALUES (?,?,?,?)",
                    [("c1", "web", "src-1", "good"), ("c2", "web", "src-1", "bad")])
    con.executemany(
        "INSERT INTO claim_evidence (claim_id, producer, source_id, locator, page, url, quote) VALUES (?,?,?,?,?,?,?)",
        [("c1", "web", "src-1", "src-1:c0", None, "u", "settles instantly"),
         ("c2", "web", "src-1", "src-1:c0", None, "u", "settles very fast")],  # paraphrase -> not verbatim
    )
    result = compose_dossier(con, cfg, ["c1", "c2"])
    assert [c["claim_id"] for c in result["included"]] == ["c1"]
    assert [c["claim_id"] for c in result["rejected"]] == ["c2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_compiler_dossier.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement dossier.py**

```python
"""compose_dossier: assemble claims + citations into an evidence dossier,
gated by the verbatim-quote invariant. A claim whose quote is not a verbatim
substring of its source text is dropped into `rejected`, never emitted as
if verified. This reuses the exact-substring semantics of
pdf.eval.check_evidence_quotes_verbatim; do not weaken it."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def verbatim_ok(quote: str, source_text: str) -> bool:
    return bool(quote) and quote in source_text


def _pdf_page_text(run_dir: Path, page: int | None) -> str:
    prov = run_dir / "provenance.jsonl"
    if not prov.is_file():
        return ""
    parts = []
    with open(prov, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            unit = json.loads(line)
            if unit.get("page") == page:
                parts.append(unit.get("text") or "")
    return "\n".join(parts)


def source_text_for(evidence_row: dict[str, Any], config) -> str:
    producer, source_id = evidence_row["producer"], evidence_row["source_id"]
    if producer == "pdf":
        return _pdf_page_text(Path(config.pdf_runs_path) / source_id, evidence_row.get("page"))
    if producer == "web":
        src = Path(config.research_runs_path) / source_id / "source.md"
        return src.read_text(encoding="utf-8") if src.is_file() else ""
    return ""


def compose_dossier(con, config, claim_ids: list[str]) -> dict:
    included, rejected = [], []
    for cid in claim_ids:
        claim_row = con.execute(
            "SELECT claim_id, producer, source_id, claim, claim_type, confidence FROM claims WHERE claim_id = ?",
            [cid],
        ).fetchone()
        if claim_row is None:
            rejected.append({"claim_id": cid, "reason": "claim_id not found in index"})
            continue
        ev_rows = con.execute(
            "SELECT claim_id, producer, source_id, locator, page, url, quote FROM claim_evidence WHERE claim_id = ?",
            [cid],
        ).fetchall()
        cols = ["claim_id", "producer", "source_id", "locator", "page", "url", "quote"]
        evidence = [dict(zip(cols, r)) for r in ev_rows]

        failures = []
        for ev in evidence:
            if not verbatim_ok(ev["quote"], source_text_for(ev, config)):
                failures.append(ev["quote"])

        entry = {
            "claim_id": cid, "claim": claim_row[3], "claim_type": claim_row[4],
            "confidence": claim_row[5], "evidence": evidence,
        }
        if failures or not evidence:
            entry["reason"] = "no evidence" if not evidence else f"non-verbatim quote(s): {failures}"
            rejected.append(entry)
        else:
            included.append(entry)
    return {"included": included, "rejected": rejected}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_compiler_dossier.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/compiler/dossier.py tests/unit/test_compiler_dossier.py
git commit -m "Add compose_dossier with the verbatim-quote hard gate"
```

---

## Task 10: Tools — the Index handle and 8 tool functions

**Files:**
- Create: `src/deep_research_toolkit/compiler/tools.py`
- Test: `tests/unit/test_compiler_tools.py`

**Interfaces:**
- Consumes: `build.open_duckdb/open_lancedb`, `search.*`, `graph.*`, `contradictions.find_candidates`, `dossier.compose_dossier`, `Embedder`, `Config`.
- Produces:
  - `class Index` with `open(config, embedder=None) -> Index` classmethod, holding the duckdb con + lancedb handle + embedder + config, and methods: `search_wiki(query, k=8)`, `read_page(path)`, `search_claims(query, k=8, producer=None)`, `get_entity(name_or_id)`, `neighbors(entity, depth=1)`, `get_sources(page=None, claim=None)`, `find_contradictions()`, `compose_dossier(query=None, claim_ids=None, k=12)`, and `close()`. Each returns plain dict/list (JSON-serializable). `compose_dossier` with a `query` first runs `search_claims` to pick claim_ids.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_compiler_tools.py
import json
from deep_research_toolkit.common.frontmatter import write_okf
from deep_research_toolkit.compiler.build import compile_index
from deep_research_toolkit.compiler.embed import FakeEmbedder
from deep_research_toolkit.compiler.tools import Index
from deep_research_toolkit.config import load_config


def _project(tmp_path):
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\nknowledge_base:\n  path: kb\n  pdf_runs_dir: pdf-runs\n"
        "  research_runs_dir: research-runs\n  index_dir: idx\n", encoding="utf-8")
    write_okf(tmp_path / "kb" / "concepts/hydra.md",
              {"type": "Concept", "title": "Hydra", "timestamp": "t", "status": "draft"},
              "Hydra is a settlement layer.\n")
    run = tmp_path / "pdf-runs" / "doc-abc"; run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"document_id": "doc-abc"}), encoding="utf-8")
    (run / "provenance.jsonl").write_text(json.dumps(
        {"page": 1, "text": "Hydra settles synchronously among participants."}) + "\n", encoding="utf-8")
    (run / "claims.jsonl").write_text(json.dumps({
        "claim_id": "c1", "claim": "Hydra settles synchronously", "claim_type": "architectural",
        "confidence": "high",
        "supporting_evidence": [{"node_id": "doc-abc:n5", "quote": "settles synchronously", "page": 1}],
    }) + "\n", encoding="utf-8")
    (run / "entities.jsonl").write_text(json.dumps(
        {"entity_id": "hydra", "name": "Hydra", "type": "protocol", "aliases": [], "mentions": ["doc-abc:n5"]}
    ) + "\n", encoding="utf-8")
    cfg = load_config(tmp_path)
    compile_index(cfg, embedder=FakeEmbedder())
    return cfg


def test_search_claims_and_read_page(tmp_path):
    cfg = _project(tmp_path)
    idx = Index.open(cfg, embedder=FakeEmbedder())
    assert any(c["claim_id"] == "c1" for c in idx.search_claims("settlement"))
    page = idx.read_page("concepts/hydra.md")
    assert page["frontmatter"]["title"] == "Hydra"
    idx.close()


def test_get_entity_and_compose_dossier_gate(tmp_path):
    cfg = _project(tmp_path)
    idx = Index.open(cfg, embedder=FakeEmbedder())
    ent = idx.get_entity("hydra")
    assert ent["name"] == "Hydra"  # entity resolves by id
    dossier = idx.compose_dossier(claim_ids=["c1"])
    assert [c["claim_id"] for c in dossier["included"]] == ["c1"]  # verbatim passes
    idx.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_compiler_tools.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement tools.py**

```python
"""The retrieval-planner tool surface: an Index handle plus 8 cheap,
deterministic tools. No tool makes an LLM call (ADR 0001 decision #3)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import search as search_mod
from . import graph as graph_mod
from .build import open_duckdb, open_lancedb
from .contradictions import find_candidates
from .dossier import compose_dossier as _compose_dossier
from .embed import Embedder, get_embedder


class Index:
    def __init__(self, con, lancedb_handle, embedder, config) -> None:
        self.con = con
        self.lance = lancedb_handle
        self.embedder = embedder
        self.config = config

    @classmethod
    def open(cls, config, embedder: Embedder | None = None) -> "Index":
        index_dir = Path(config.index_dir)
        if not (index_dir / "knowledge.duckdb").is_file():
            raise FileNotFoundError(
                f"No index at {index_dir}. Run the knowledge-compiler skill's compile.py first."
            )
        con = open_duckdb(index_dir)
        try:
            lance = open_lancedb(index_dir)
        except Exception:
            lance = None
        return cls(con, lance, embedder or get_embedder(config.embedding_model), config)

    def close(self) -> None:
        self.con.close()

    def search_wiki(self, query: str, k: int = 8) -> list[dict]:
        ids = search_mod.hybrid_search(self.con, self.lance, self.embedder,
                                       table="wiki_pages", id_col="path", text_col="body",
                                       vec_table="wiki_chunks", query=query, k=k)
        out = []
        for path in ids:
            row = self.con.execute("SELECT path, title, type, status, body FROM wiki_pages WHERE path = ?",
                                   [path]).fetchone()
            if row:
                out.append({"path": row[0], "title": row[1], "type": row[2], "status": row[3],
                            "snippet": (row[4] or "")[:200]})
        return out

    def read_page(self, path: str) -> dict:
        row = self.con.execute("SELECT path, body, frontmatter_json FROM wiki_pages WHERE path = ?",
                               [path]).fetchone()
        if not row:
            return {"path": path, "error": "not found"}
        return {"path": row[0], "body": row[1], "frontmatter": json.loads(row[2] or "{}")}

    def search_claims(self, query: str, k: int = 8, producer: str | None = None) -> list[dict]:
        ids = search_mod.hybrid_search(self.con, self.lance, self.embedder,
                                       table="claims", id_col="claim_id", text_col="claim",
                                       vec_table="claim_vectors", query=query, k=k)
        out = []
        for cid in ids:
            row = self.con.execute(
                "SELECT claim_id, producer, source_id, claim, claim_type, confidence FROM claims WHERE claim_id = ?",
                [cid]).fetchone()
            if not row or (producer and row[1] != producer):
                continue
            ev = self.con.execute(
                "SELECT producer, source_id, locator, page, url, quote FROM claim_evidence WHERE claim_id = ?",
                [cid]).fetchall()
            out.append({"claim_id": row[0], "producer": row[1], "source_id": row[2], "claim": row[3],
                        "claim_type": row[4], "confidence": row[5],
                        "evidence": [dict(zip(["producer", "source_id", "locator", "page", "url", "quote"], e))
                                     for e in ev]})
        return out

    def get_entity(self, name_or_id: str) -> dict:
        row = self.con.execute(
            "SELECT entity_id, name, type, aliases_json FROM entities "
            "WHERE entity_id = ? OR lower(name) = lower(?) LIMIT 1", [name_or_id, name_or_id]).fetchone()
        if not row:
            return {"query": name_or_id, "error": "entity not found"}
        eid = row[0]
        mentions = [r[0] for r in self.con.execute(
            "SELECT locator FROM entity_mentions WHERE entity_id = ?", [eid]).fetchall()]
        relations = [dict(zip(["relation_id", "subject", "predicate", "object"], r)) for r in self.con.execute(
            "SELECT relation_id, subject, predicate, object FROM relations WHERE subject = ? OR object = ?",
            [eid, eid]).fetchall()]
        return {"entity_id": eid, "name": row[1], "type": row[2],
                "aliases": json.loads(row[3] or "[]"), "mentions": mentions, "relations": relations}

    def neighbors(self, entity: str, depth: int = 1) -> list[dict]:
        return graph_mod.neighbors(self.con, entity, depth)

    def get_sources(self, page: str | None = None, claim: str | None = None) -> dict:
        if page:
            row = self.con.execute("SELECT frontmatter_json FROM wiki_pages WHERE path = ?", [page]).fetchone()
            fm = json.loads(row[0]) if row else {}
            return {"page": page, "source": fm.get("source"), "source_docs": fm.get("source_docs"),
                    "resource": fm.get("resource")}
        if claim:
            ev = self.con.execute(
                "SELECT DISTINCT producer, source_id, url FROM claim_evidence WHERE claim_id = ?",
                [claim]).fetchall()
            return {"claim": claim,
                    "sources": [dict(zip(["producer", "source_id", "url"], e)) for e in ev]}
        return {"error": "pass page= or claim="}

    def find_contradictions(self) -> list[dict]:
        return find_candidates(self.con)

    def compose_dossier(self, query: str | None = None, claim_ids: list[str] | None = None, k: int = 12) -> dict:
        if claim_ids is None:
            claim_ids = [c["claim_id"] for c in self.search_claims(query or "", k=k)]
        return _compose_dossier(self.con, self.config, claim_ids)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_compiler_tools.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/compiler/tools.py tests/unit/test_compiler_tools.py
git commit -m "Add the Index handle and 8 retrieval-planner tools"
```

---

## Task 11: knowledge-compiler skill

**Files:**
- Create: `skills/knowledge-compiler/SKILL.md`
- Create: `skills/knowledge-compiler/scripts/compile.py`
- Test: `tests/integration/test_compile_cli.py`

**Interfaces:**
- Consumes: `compile_index` (Task 5), `load_config`.
- Produces: a runnable `python skills/knowledge-compiler/scripts/compile.py [--index-dir DIR]` that builds the index and prints stats.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_compile_cli.py
import json, subprocess, sys
from pathlib import Path
from deep_research_toolkit.common.frontmatter import write_okf

REPO = Path(__file__).resolve().parents[2]


def test_compile_script_builds_index(tmp_path):
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\nknowledge_base:\n  path: kb\n  index_dir: idx\n", encoding="utf-8")
    write_okf(tmp_path / "kb" / "a.md", {"type": "Concept", "title": "A", "timestamp": "t"}, "body\n")
    # Force the deterministic embedder so the test needs no model download.
    env = {"DRT_FAKE_EMBEDDER": "1"}
    import os
    full_env = {**os.environ, **env}
    script = REPO / "skills" / "knowledge-compiler" / "scripts" / "compile.py"
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=full_env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "idx" / "knowledge.duckdb").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_compile_cli.py -v`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Implement the script and SKILL.md**

Create `skills/knowledge-compiler/scripts/compile.py`:

```python
#!/usr/bin/env python3
"""Thin CLI shim: build the knowledge index via deep_research_toolkit.compiler.build.

python scripts/compile.py [--index-dir DIR]

Set DRT_FAKE_EMBEDDER=1 to use the deterministic test embedder (no model
download) -- for CI and smoke tests only, never for a real corpus.
"""
import argparse
import os
import sys

from deep_research_toolkit.compiler.build import compile_index
from deep_research_toolkit.compiler.embed import FakeEmbedder
from deep_research_toolkit.config import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-dir", default=None, help="Override knowledge_base.index_dir")
    args = parser.parse_args()

    config = load_config()
    if args.index_dir:
        config.index_dir = resolve_path(args.index_dir, config.index_dir, ".deepresearch/index")

    embedder = FakeEmbedder() if os.environ.get("DRT_FAKE_EMBEDDER") == "1" else None
    stats = compile_index(config, embedder=embedder)
    print("compiled index:", stats)
    print(f"index at: {config.index_dir}")


if __name__ == "__main__":
    sys.exit(main())
```

Create `skills/knowledge-compiler/SKILL.md`:

```markdown
---
name: knowledge-compiler
description: Build or refresh the queryable knowledge index (DuckDB FTS + graph, LanceDB vectors) over everything the web and PDF pipelines have produced. Use after ingesting new sources, or before a research session that will query the accumulated knowledge base with the retrieval-planner skill. Requires the compiler extra.
---

# Knowledge Compiler

Compiles `knowledge_base/`, `pdf-runs/`, and `research-runs/` into a hybrid
index the `retrieval-planner` skill queries. See
`docs/contracts/knowledge-compiler.md` for the index schema and
`docs/decisions/0002-knowledge-compiler.md` for the design.

## First: configuration and dependencies

Read `.deepresearch.yml` (walk up from cwd, like `.git`). The index lives at
`knowledge_base.index_dir` (default `.deepresearch/index/`). Install the
extra once: `pip install "deep-research-toolkit[compiler]"`.

## Build the index

```
python scripts/compile.py [--index-dir DIR]
```

Full rebuild each run (idempotent). It walks all three producers, normalizes
their evidence into a producer-agnostic `evidence_ref`, builds the DuckDB
FTS + graph tables, and embeds wiki pages and claims into LanceDB with the
configured `embedding_model` (`all-MiniLM-L6-v2` by default). Prints row
counts on success.

The first run downloads the sentence-transformers model (a one-time,
offline-after cost, like Docling's models). Everything after is local.
```

- [ ] **Step 4: Run the test, then sync templates**

Run: `pytest tests/integration/test_compile_cli.py -v`
Expected: PASS.
Run: `python scripts/sync-skill-templates.py && python scripts/check-skill-templates-in-sync.py`
Expected: sync report, then OK.

- [ ] **Step 5: Commit**

```bash
git add skills/knowledge-compiler src/deep_research_toolkit/skill_templates tests/integration/test_compile_cli.py
git commit -m "Add the knowledge-compiler skill and compile.py shim"
```

---

## Task 12: retrieval-planner skill

**Files:**
- Create: `skills/retrieval-planner/SKILL.md`
- Create: `skills/retrieval-planner/scripts/query.py`
- Create: `skills/retrieval-planner/references/tool-contracts.md`
- Test: `tests/integration/test_query_cli.py`

**Interfaces:**
- Consumes: `Index` (Task 10), `load_config`, `FakeEmbedder`.
- Produces: `python scripts/query.py <subcommand> ...` printing JSON. Subcommands: `search-wiki`, `read-page`, `search-claims`, `get-entity`, `neighbors`, `get-sources`, `find-contradictions`, `compose-dossier`.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_query_cli.py
import json, os, subprocess, sys
from pathlib import Path
from deep_research_toolkit.common.frontmatter import write_okf

REPO = Path(__file__).resolve().parents[2]


def _run(cwd, *args):
    env = {**os.environ, "DRT_FAKE_EMBEDDER": "1"}
    script = REPO / "skills" / "retrieval-planner" / "scripts" / "query.py"
    return subprocess.run([sys.executable, str(script), *args], cwd=cwd, env=env,
                          capture_output=True, text=True)


def test_search_wiki_after_compile(tmp_path):
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\nknowledge_base:\n  path: kb\n  index_dir: idx\n", encoding="utf-8")
    write_okf(tmp_path / "kb" / "hydra.md",
              {"type": "Concept", "title": "Hydra", "timestamp": "t"}, "Hydra settlement layer.\n")
    compile_script = REPO / "skills" / "knowledge-compiler" / "scripts" / "compile.py"
    subprocess.run([sys.executable, str(compile_script)], cwd=tmp_path,
                   env={**os.environ, "DRT_FAKE_EMBEDDER": "1"}, check=True, capture_output=True)

    result = _run(tmp_path, "search-wiki", "settlement")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert any(r["path"] == "hydra.md" for r in payload)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_query_cli.py -v`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Implement query.py, SKILL.md, references**

Create `skills/retrieval-planner/scripts/query.py`:

```python
#!/usr/bin/env python3
"""Thin CLI shim over deep_research_toolkit.compiler.tools.Index. Prints JSON.

Subcommands: search-wiki, read-page, search-claims, get-entity, neighbors,
get-sources, find-contradictions, compose-dossier.

Set DRT_FAKE_EMBEDDER=1 to use the deterministic test embedder (CI/smoke only).
"""
import argparse
import json
import os
import sys

from deep_research_toolkit.compiler.embed import FakeEmbedder
from deep_research_toolkit.compiler.tools import Index
from deep_research_toolkit.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("find-contradictions")
    p = sub.add_parser("search-wiki"); p.add_argument("query"); p.add_argument("--k", type=int, default=8)
    p = sub.add_parser("read-page"); p.add_argument("path")
    p = sub.add_parser("search-claims"); p.add_argument("query"); p.add_argument("--k", type=int, default=8)
    p.add_argument("--producer", choices=["pdf", "web"], default=None)
    p = sub.add_parser("get-entity"); p.add_argument("name_or_id")
    p = sub.add_parser("neighbors"); p.add_argument("entity"); p.add_argument("--depth", type=int, default=1)
    p = sub.add_parser("get-sources"); p.add_argument("--page", default=None); p.add_argument("--claim", default=None)
    p = sub.add_parser("compose-dossier"); p.add_argument("query", nargs="?", default=None)
    p.add_argument("--claims", default=None, help="comma-separated claim_ids"); p.add_argument("--k", type=int, default=12)
    args = parser.parse_args()

    embedder = FakeEmbedder() if os.environ.get("DRT_FAKE_EMBEDDER") == "1" else None
    idx = Index.open(load_config(), embedder=embedder)
    try:
        if args.cmd == "search-wiki":
            out = idx.search_wiki(args.query, k=args.k)
        elif args.cmd == "read-page":
            out = idx.read_page(args.path)
        elif args.cmd == "search-claims":
            out = idx.search_claims(args.query, k=args.k, producer=args.producer)
        elif args.cmd == "get-entity":
            out = idx.get_entity(args.name_or_id)
        elif args.cmd == "neighbors":
            out = idx.neighbors(args.entity, depth=args.depth)
        elif args.cmd == "get-sources":
            out = idx.get_sources(page=args.page, claim=args.claim)
        elif args.cmd == "find-contradictions":
            out = idx.find_contradictions()
        elif args.cmd == "compose-dossier":
            claim_ids = args.claims.split(",") if args.claims else None
            out = idx.compose_dossier(query=args.query, claim_ids=claim_ids, k=args.k)
        else:
            parser.error("unknown command")
        print(json.dumps(out, indent=2, ensure_ascii=False))
    finally:
        idx.close()


if __name__ == "__main__":
    sys.exit(main())
```

Create `skills/retrieval-planner/SKILL.md`:

```markdown
---
name: retrieval-planner
description: Query the compiled knowledge index to answer research questions from what's already been gathered, before scraping or re-reading anything. Provides 8 tools (search_wiki, read_page, search_claims, get_entity, neighbors, get_sources, find_contradictions, compose_dossier). Use when answering a question the knowledge base may already cover, or when assembling a cited evidence dossier. Requires a built index (run knowledge-compiler first).
---

# Retrieval Planner

Deterministic, LLM-free tools over the index built by `knowledge-compiler`.
Full tool contracts are in `references/tool-contracts.md`. Every command
prints JSON to stdout.

## First

Ensure the index exists (`knowledge-compiler`'s `compile.py`), then read
`.deepresearch.yml` for paths. All commands:

```
python scripts/query.py search-wiki "<query>" [--k N]
python scripts/query.py read-page <kb-relative-path>
python scripts/query.py search-claims "<query>" [--k N] [--producer pdf|web]
python scripts/query.py get-entity <name-or-id>
python scripts/query.py neighbors <entity-id> [--depth D]
python scripts/query.py get-sources (--page P | --claim C)
python scripts/query.py find-contradictions
python scripts/query.py compose-dossier "<query>" [--claims c1,c2] [--k N]
```

## compose-dossier and the verbatim gate

`compose-dossier` returns `{included, rejected}`. A claim reaches `included`
only if **every** supporting quote is a verbatim substring of its source
(PDF page text or web `source.md`). Non-verbatim claims land in `rejected`
with a reason -- never silently included. Trust `included`; treat `rejected`
as a signal that an extraction pass needs fixing.

## find-contradictions is candidates, not verdicts

`find-contradictions` lists *mechanical* candidates (same subject+predicate
with different objects; `status: conflicted` pages). Confirming a candidate
is a real contradiction is your job as the agent: read the two claims and
their quotes via `search-claims`/`compose-dossier` and decide. Do this in a
single batched pass over all candidates, not one model call each.
```

Create `skills/retrieval-planner/references/tool-contracts.md` documenting each tool's exact input args and JSON output shape (one section per tool, copied from the `Index` method signatures in Task 10).

- [ ] **Step 4: Run the test, then sync templates**

Run: `pytest tests/integration/test_query_cli.py -v`
Expected: PASS.
Run: `python scripts/sync-skill-templates.py && python scripts/check-skill-templates-in-sync.py`
Expected: OK.

- [ ] **Step 5: Commit**

```bash
git add skills/retrieval-planner src/deep_research_toolkit/skill_templates tests/integration/test_query_cli.py
git commit -m "Add the retrieval-planner skill with the 8-tool query CLI"
```

---

## Task 13: Web research runs and claim-extraction

**Files:**
- Create: `src/deep_research_toolkit/web/research_run.py`
- Create: `skills/research-knowledge-graph/scripts/start_research_run.py`
- Create: `skills/research-knowledge-graph/references/web-claim-extraction.md`
- Modify: `skills/research-knowledge-graph/SKILL.md`
- Test: `tests/unit/test_research_run.py`

**Interfaces:**
- Consumes: `common.hashing.content_hash`, `common.manifest`.
- Produces:
  - `web_source_id(url: str, content: str) -> str` — `slug(host+path)-<8 hex>` (hash of content).
  - `chunk_markdown(text: str, source_id: str) -> list[dict]` — one node per heading section (fallback: whole doc as one node); node shape `{node_id, source_id, type, title, text, content_hash}` with `node_id = f"{source_id}:c{NN}"`.
  - `start_research_run(url: str, content: str, research_runs_dir: Path) -> Path` — creates `<dir>/<source_id>/`, writes `source.md`, `chunks.jsonl`, and `manifest.json` (producer web); returns the run dir.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_research_run.py
import json
from deep_research_toolkit.web.research_run import web_source_id, chunk_markdown, start_research_run


def test_web_source_id_stable_and_content_sensitive():
    a = web_source_id("https://ex.com/ows", "content one")
    b = web_source_id("https://ex.com/ows", "content one")
    c = web_source_id("https://ex.com/ows", "different content")
    assert a == b            # same url+content -> same id
    assert a != c            # content change -> different id
    assert a.startswith("ex-com-ows-")   # _slug turns "ex.com/ows" into "ex-com-ows"


def test_chunk_markdown_splits_on_headings():
    nodes = chunk_markdown("# A\nalpha text\n## B\nbeta text\n", "src-1")
    assert [n["title"] for n in nodes] == ["A", "B"]
    assert nodes[0]["node_id"] == "src-1:c01"
    assert "alpha" in nodes[0]["text"]


def test_start_research_run_writes_layout(tmp_path):
    run = start_research_run("https://ex.com/ows", "# OWS\nThe Open Wallet Standard.\n", tmp_path)
    assert (run / "source.md").read_text(encoding="utf-8").startswith("# OWS")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["producer"] == "web" and manifest["source_url"] == "https://ex.com/ows"
    assert (run / "chunks.jsonl").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_research_run.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement research_run.py**

```python
"""Web research runs: research-runs/<source_id>/ mirrors a PDF run so the
knowledge compiler indexes web- and PDF-sourced claims uniformly. See
docs/contracts/knowledge-compiler.md."""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

from ..common.hashing import content_hash

MANIFEST_SCHEMA_VERSION = "1.0"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "source"


def web_source_id(url: str, content: str) -> str:
    parsed = urlparse(url)
    base = _slug(f"{parsed.netloc}{parsed.path}")
    return f"{base}-{content_hash(content, length=8).split(':')[1]}"


def chunk_markdown(text: str, source_id: str) -> list[dict]:
    sections: list[tuple[str, list[str]]] = []
    current_title, current_lines = None, []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            if current_title is not None or current_lines:
                sections.append((current_title or "", current_lines))
            current_title = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_title is not None or current_lines:
        sections.append((current_title or "", current_lines))
    if not sections:
        sections = [("", text.splitlines())]

    nodes = []
    for i, (title, lines) in enumerate(sections, start=1):
        body = "\n".join(lines).strip()
        node_text = (title + "\n\n" + body).strip() if title else body
        nodes.append({
            "schema_version": "1.0",
            "node_id": f"{source_id}:c{str(i).zfill(2)}",
            "source_id": source_id, "type": "section", "title": title,
            "text": node_text, "content_hash": content_hash(node_text),
        })
    return nodes


def _now_iso() -> str:
    import datetime
    return (datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z"))


def start_research_run(url: str, content: str, research_runs_dir: Path) -> Path:
    research_runs_dir = Path(research_runs_dir)
    source_id = web_source_id(url, content)
    run_dir = research_runs_dir / source_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "source.md").write_text(content, encoding="utf-8")

    nodes = chunk_markdown(content, source_id)
    with open(run_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for node in nodes:
            f.write(json.dumps(node, ensure_ascii=False) + "\n")

    (run_dir / "manifest.json").write_text(json.dumps({
        "schema_version": MANIFEST_SCHEMA_VERSION, "producer": "web", "document_id": source_id,
        "source_url": url, "content_hash": content_hash(content), "fetched_at": _now_iso(),
        "chunk_count": len(nodes),
    }, indent=2) + "\n", encoding="utf-8")
    return run_dir
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_research_run.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Add the shim, references, and SKILL.md section, then sync**

Create `skills/research-knowledge-graph/scripts/start_research_run.py`:

```python
#!/usr/bin/env python3
"""Thin CLI shim: scaffold a web research run for claim extraction.

python scripts/start_research_run.py <url> --content-file PATH [--research-runs-dir DIR]
"""
import argparse
import sys
from pathlib import Path

from deep_research_toolkit.config import load_config, resolve_path
from deep_research_toolkit.web.research_run import start_research_run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("--content-file", required=True, help="Path to the fetched, cleaned markdown/text")
    parser.add_argument("--research-runs-dir", default=None)
    args = parser.parse_args()

    config = load_config()
    runs_dir = resolve_path(args.research_runs_dir, config.research_runs_path, "research-runs")
    content = Path(args.content_file).read_text(encoding="utf-8")
    run_dir = start_research_run(args.url, content, runs_dir)
    print(f"research run: {run_dir}")


if __name__ == "__main__":
    sys.exit(main())
```

Create `skills/research-knowledge-graph/references/web-claim-extraction.md` mirroring `knowledge-extraction`'s Part 2 rules (one assertion per claim; **quote is a verbatim substring of `source.md`, not a page**; merge entity mentions; don't force it), and giving the web-shaped `supporting_evidence` schema: `{"locator": "<source_id>:cNN", "quote": "...", "url": "<source_url>"}`.

Add a "Turning a source into claims" section to `skills/research-knowledge-graph/SKILL.md` that: (1) points to `start_research_run.py`, (2) tells the agent to read `chunks.jsonl` and write `claims.jsonl`/`entities.jsonl`/`relations.jsonl` into the run dir following `references/web-claim-extraction.md`, and (3) notes these feed the `knowledge-compiler` index alongside PDF runs.

Run: `python scripts/sync-skill-templates.py && python scripts/check-skill-templates-in-sync.py`
Expected: OK.

- [ ] **Step 6: Commit**

```bash
git add src/deep_research_toolkit/web/research_run.py skills/research-knowledge-graph src/deep_research_toolkit/skill_templates tests/unit/test_research_run.py
git commit -m "Add web research runs and claim-extraction for research-knowledge-graph"
```

---

## Task 14: LLM backend (agent default | local Ornith)

**Files:**
- Create: `src/deep_research_toolkit/llm/__init__.py`
- Create: `src/deep_research_toolkit/llm/backend.py`
- Create: `src/deep_research_toolkit/llm/agent.py`
- Create: `src/deep_research_toolkit/llm/local.py`
- Test: `tests/unit/test_llm_backend.py`

**Interfaces:**
- Consumes: `Config` (`llm_provider`, `llm_local`).
- Produces:
  - `class Backend(Protocol): def complete(self, system: str, user: str, **sampling) -> str: ...`
  - `class LLMBackendNotConfigured(RuntimeError)`, `class LocalLLMNotInstalled(RuntimeError)`
  - `get_backend(config) -> Backend` — `agent`/`anthropic` → `AgentBackend`; `local` → `LocalOpenAIBackend`.
  - `AgentBackend.complete(...)` raises `LLMBackendNotConfigured` with guidance.
  - `strip_think(text: str) -> str` — removes `<think>...</think>` blocks.
  - `LocalOpenAIBackend(base_url, model, api_key, temperature, top_p, top_k)` — lazy `openai` import; `complete` calls chat completions and returns `strip_think(content)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_llm_backend.py
import pytest
from types import SimpleNamespace
from deep_research_toolkit.llm.backend import get_backend, LLMBackendNotConfigured
from deep_research_toolkit.llm.agent import AgentBackend
from deep_research_toolkit.llm.local import LocalOpenAIBackend, strip_think


def _cfg(provider):
    return SimpleNamespace(llm_provider=provider, llm_local={
        "base_url": "http://localhost:11434/v1", "model": "Ornith-1.0-9B",
        "api_key_env": "OPENAI_API_KEY", "temperature": 0.6, "top_p": 0.95, "top_k": 20})


def test_agent_backend_is_default_and_refuses_programmatic_call():
    backend = get_backend(_cfg("anthropic"))
    assert isinstance(backend, AgentBackend)
    with pytest.raises(LLMBackendNotConfigured) as exc:
        backend.complete("sys", "user")
    assert "provider: local" in str(exc.value)


def test_local_provider_selects_local_backend():
    assert isinstance(get_backend(_cfg("local")), LocalOpenAIBackend)


def test_strip_think_removes_reasoning_blocks():
    assert strip_think("<think>reasoning here</think>\nFinal answer.") == "Final answer."
    assert strip_think("no think tags") == "no think tags"


def test_local_backend_parses_response(monkeypatch):
    backend = LocalOpenAIBackend(base_url="http://x/v1", model="Ornith-1.0-9B", api_key="k",
                                 temperature=0.6, top_p=0.95, top_k=20)

    class _Msg: content = "<think>x</think>hello"
    class _Choice: message = _Msg()
    class _Resp: choices = [_Choice()]

    monkeypatch.setattr(backend, "_client_complete", lambda system, user, **kw: _Resp())
    assert backend.complete("s", "u") == "hello"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_llm_backend.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the llm package**

`src/deep_research_toolkit/llm/__init__.py`:

```python
"""Pluggable LLM backend: the in-session agent by default (no programmatic
call), or an optional local OpenAI-compatible endpoint (e.g. Ornith-1.0-9B)."""
```

`src/deep_research_toolkit/llm/backend.py`:

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable


class LLMBackendNotConfigured(RuntimeError):
    pass


@runtime_checkable
class Backend(Protocol):
    def complete(self, system: str, user: str, **sampling) -> str: ...


def get_backend(config) -> Backend:
    provider = getattr(config, "llm_provider", "agent")
    if provider in ("agent", "anthropic"):
        from .agent import AgentBackend
        return AgentBackend()
    if provider == "local":
        import os
        from .local import LocalOpenAIBackend
        local = config.llm_local
        return LocalOpenAIBackend(
            base_url=local["base_url"], model=local["model"],
            api_key=os.environ.get(local.get("api_key_env", "OPENAI_API_KEY"), "not-needed"),
            temperature=local["temperature"], top_p=local["top_p"], top_k=local["top_k"],
        )
    raise LLMBackendNotConfigured(f"unknown llm.provider: {provider!r} (use agent | anthropic | local)")
```

`src/deep_research_toolkit/llm/agent.py`:

```python
from __future__ import annotations

from .backend import LLMBackendNotConfigured


class AgentBackend:
    """Default backend: the judgment steps are done by the in-session agent
    reading files per SKILL.md (ADR 0001 decision #4). There is no
    programmatic model to call here -- invoking complete() is a usage error."""

    def complete(self, system: str, user: str, **sampling) -> str:
        raise LLMBackendNotConfigured(
            "llm.provider is 'agent': extraction/synthesis is done by the in-session agent "
            "following the skill's SKILL.md, not by a programmatic call. To automate it with a "
            "local model instead, set 'provider: local' in .deepresearch.yml and run a local "
            "OpenAI-compatible endpoint (e.g. Ollama serving Ornith-1.0-9B)."
        )
```

`src/deep_research_toolkit/llm/local.py`:

```python
from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LocalLLMNotInstalled(RuntimeError):
    pass


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


class LocalOpenAIBackend:
    """Talks to an OpenAI-compatible endpoint (Ollama :11434/v1, vLLM
    :8000/v1) serving a local model such as Ornith-1.0-9B."""

    def __init__(self, base_url: str, model: str, api_key: str,
                 temperature: float, top_p: float, top_k: int) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self._client = None

    def _load_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise LocalLLMNotInstalled(
                    "The local LLM backend needs an OpenAI-compatible client. "
                    'Install it with: pip install "deep-research-toolkit[compiler]" '
                    "(or: pip install openai)."
                ) from e
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _client_complete(self, system: str, user: str, **kw):
        client = self._load_client()
        return client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=kw.get("temperature", self.temperature),
            top_p=kw.get("top_p", self.top_p),
            extra_body={"top_k": kw.get("top_k", self.top_k)},
        )

    def complete(self, system: str, user: str, **sampling) -> str:
        resp = self._client_complete(system, user, **sampling)
        return strip_think(resp.choices[0].message.content or "")
```

Add `openai>=1.40` to the `compiler` extra in `pyproject.toml` (so the local backend's client ships with that extra).

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_llm_backend.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/deep_research_toolkit/llm tests/unit/test_llm_backend.py pyproject.toml
git commit -m "Add pluggable LLM backend with agent default and local Ornith option"
```

---

## Task 15: Programmatic extract_claims path + validation harness

**Files:**
- Create: `src/deep_research_toolkit/llm/extract.py`
- Create: `skills/knowledge-extraction/scripts/extract_claims.py`
- Create: `skills/research-knowledge-graph/scripts/extract_claims.py`
- Create: `scripts/validate-local-llm.py`
- Test: `tests/unit/test_llm_extract.py`

**Interfaces:**
- Consumes: `Backend` (Task 14), `dossier.verbatim_ok`, `dossier.source_text_for`.
- Produces:
  - `build_extraction_prompt(chunks: list[dict]) -> tuple[str, str]` — returns `(system, user)` embedding the four extraction rules and the JSON output schema.
  - `parse_claims_response(text: str) -> list[dict]` — parses the model's JSON array (tolerant of a leading/trailing prose line).
  - `extract_claims_to_run(run_dir, producer, config, backend) -> dict` — reads `chunks.jsonl`, calls `backend.complete`, parses claims, **drops any claim with a non-verbatim quote**, writes `claims.jsonl`, returns `{"written": n, "dropped": [...]}`. Raises via `AgentBackend` if provider is `agent`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_llm_extract.py
import json
from types import SimpleNamespace
from deep_research_toolkit.llm.extract import parse_claims_response, extract_claims_to_run


class _FakeBackend:
    def __init__(self, payload): self.payload = payload
    def complete(self, system, user, **kw): return self.payload


def test_parse_claims_tolerates_surrounding_prose():
    text = 'Here are the claims:\n[{"claim_id": "c1", "claim": "x"}]\nDone.'
    assert parse_claims_response(text)[0]["claim_id"] == "c1"


def test_extract_drops_non_verbatim_quotes(tmp_path):
    run = tmp_path / "research-runs" / "src-1"; run.mkdir(parents=True)
    (run / "source.md").write_text("Hydra settles instantly.", encoding="utf-8")
    (run / "chunks.jsonl").write_text(json.dumps(
        {"node_id": "src-1:c01", "text": "Hydra settles instantly."}) + "\n", encoding="utf-8")
    cfg = SimpleNamespace(pdf_runs_path=tmp_path / "pdf-runs", research_runs_path=tmp_path / "research-runs")
    payload = json.dumps([
        {"claim_id": "c1", "claim": "good", "supporting_evidence": [
            {"locator": "src-1:c01", "quote": "settles instantly", "url": "u"}]},
        {"claim_id": "c2", "claim": "bad", "supporting_evidence": [
            {"locator": "src-1:c01", "quote": "settles very fast", "url": "u"}]},
    ])
    result = extract_claims_to_run(run, "web", cfg, _FakeBackend(payload))
    written = [json.loads(l) for l in (run / "claims.jsonl").read_text(encoding="utf-8").splitlines() if l]
    assert [c["claim_id"] for c in written] == ["c1"]
    assert result["dropped"] and result["written"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_llm_extract.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement extract.py**

```python
"""Optional programmatic claim extraction via a Backend (only meaningful
under llm.provider=local). The verbatim gate is applied here too, so an
off-label local model can only under-produce, never corrupt the corpus."""
from __future__ import annotations

import json
from pathlib import Path

from ..compiler.dossier import verbatim_ok, source_text_for
from ..compiler.schema import normalize_evidence

_SYSTEM = (
    "You extract atomic, evidence-backed claims from research text. Rules: "
    "(1) one checkable assertion per claim; (2) every supporting_evidence quote MUST be "
    "copied verbatim (an exact substring) from the chunk text -- never paraphrase; "
    "(3) merge entity mentions that refer to the same thing; (4) do not force a claim "
    "the text does not support. Output ONLY a JSON array of claim objects."
)


def build_extraction_prompt(chunks: list[dict]) -> tuple[str, str]:
    schema = ('[{"claim_id": "c_0001", "claim": "...", "claim_type": "architectural|empirical|'
              'definitional|comparative", "confidence": "high|medium|low", "supporting_evidence": '
              '[{"locator": "<node_id>", "quote": "<verbatim substring>", "url": "<source_url or null>"}]}]')
    body = "\n\n".join(f"[{c.get('node_id') or c.get('locator')}]\n{c.get('text','')}" for c in chunks)
    user = f"Output schema:\n{schema}\n\nChunks:\n{body}"
    return _SYSTEM, user


def parse_claims_response(text: str) -> list[dict]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    return json.loads(text[start:end + 1])


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_claims_to_run(run_dir, producer: str, config, backend) -> dict:
    run_dir = Path(run_dir)
    source_id = run_dir.name if producer == "web" else \
        json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")).get("document_id", run_dir.name)
    chunks = _read_jsonl(run_dir / "chunks.jsonl")
    system, user = build_extraction_prompt(chunks)
    claims = parse_claims_response(backend.complete(system, user))

    kept, dropped = [], []
    for claim in claims:
        refs = normalize_evidence(claim, producer, source_id)
        ok = bool(refs) and all(
            verbatim_ok(ref.quote, source_text_for(
                {"producer": ref.producer, "source_id": ref.source_id, "page": ref.page}, config))
            for ref in refs
        )
        (kept if ok else dropped).append(claim)

    with open(run_dir / "claims.jsonl", "w", encoding="utf-8") as f:
        for claim in kept:
            claim.setdefault("schema_version", "1.0")
            f.write(json.dumps(claim, ensure_ascii=False) + "\n")
    return {"written": len(kept), "dropped": [c.get("claim_id") for c in dropped]}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/unit/test_llm_extract.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Add the two shims and the validation harness, then sync**

Create `skills/knowledge-extraction/scripts/extract_claims.py` and `skills/research-knowledge-graph/scripts/extract_claims.py` — both thin shims: load config, `get_backend(config)`, call `extract_claims_to_run(run_dir, producer, config, backend)` (producer `"pdf"` and `"web"` respectively), print the result. Under the default `agent` provider these print the `LLMBackendNotConfigured` guidance and exit non-zero.

Example (`knowledge-extraction/scripts/extract_claims.py`):

```python
#!/usr/bin/env python3
"""Optional: extract claims.jsonl from chunks.jsonl using the configured LLM
backend (only under llm.provider=local). Under the default agent provider,
do this by hand following SKILL.md instead.

python scripts/extract_claims.py <run_dir>
"""
import argparse
import sys
from pathlib import Path

from deep_research_toolkit.config import load_config
from deep_research_toolkit.llm.backend import get_backend, LLMBackendNotConfigured
from deep_research_toolkit.llm.extract import extract_claims_to_run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    args = parser.parse_args()
    config = load_config()
    try:
        result = extract_claims_to_run(Path(args.run_dir), "pdf", config, get_backend(config))
    except LLMBackendNotConfigured as e:
        sys.exit(str(e))
    print(result)


if __name__ == "__main__":
    sys.exit(main())
```

Create repo-level `scripts/validate-local-llm.py`: given a `--run-dir` (default the hydra fixture reference run) and a running local endpoint, run `extract_claims_to_run` against a *copy* of the run's `chunks.jsonl`, diff the produced `claims.jsonl` against `tests/fixtures/reference-run-hydra-settlement/claims.jsonl` (report recovered claim count), and print the verbatim-pass/drop summary. Document at the top that it needs `.deepresearch.yml` with `provider: local` and a live model; it is a manual tool, not CI.

Run: `python scripts/sync-skill-templates.py && python scripts/check-skill-templates-in-sync.py`
Expected: OK.

- [ ] **Step 6: Commit**

```bash
git add src/deep_research_toolkit/llm/extract.py skills/knowledge-extraction skills/research-knowledge-graph scripts/validate-local-llm.py src/deep_research_toolkit/skill_templates tests/unit/test_llm_extract.py
git commit -m "Add programmatic extract_claims path and local-LLM validation harness"
```

---

## Task 16: Fixtures + heavy end-to-end integration test

**Files:**
- Create: `tests/fixtures/reference-kb/concepts/hydra-settlement.md`
- Create: `tests/fixtures/reference-kb/index.md`
- Create: `tests/fixtures/reference-run-web-ows/` (`manifest.json`, `source.md`, `chunks.jsonl`, `claims.jsonl`, `entities.jsonl`, `relations.jsonl`)
- Create: `tests/integration/test_full_compiler_pipeline.py`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Build the fixtures**

Create `tests/fixtures/reference-kb/index.md` (OKF Index linking to the concept page) and `tests/fixtures/reference-kb/concepts/hydra-settlement.md` (`type: Concept`, `status: draft`, `source_docs: [hydra-settlement-test-fixture-4edb3c3c]`, body mentioning Hydra settlement, cross-linking `/index.md`). Create `tests/fixtures/reference-run-web-ows/` as a synthetic **web** run: `source.md` containing a sentence like "The Open Wallet Standard (OWS) delegates signing to autonomous agents.", `chunks.jsonl` from that text, one `claims.jsonl` claim whose `supporting_evidence[0].quote` is a verbatim substring of `source.md` with `locator`/`url`, plus matching `entities.jsonl`/`relations.jsonl`. Keep it small (mirror the hydra reference run's scale).

- [ ] **Step 2: Write the heavy integration test**

```python
# tests/integration/test_full_compiler_pipeline.py
import shutil
from pathlib import Path

import pytest

from deep_research_toolkit.config import load_config
from deep_research_toolkit.compiler.build import compile_index
from deep_research_toolkit.compiler.tools import Index

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.heavy
def test_full_pipeline_with_real_embeddings(tmp_path):
    # Assemble a project from the shipped fixtures.
    (tmp_path / ".deepresearch.yml").write_text(
        "version: 1\nknowledge_base:\n  path: kb\n  pdf_runs_dir: pdf-runs\n"
        "  research_runs_dir: research-runs\n  index_dir: idx\n", encoding="utf-8")
    shutil.copytree(FIXTURES / "reference-kb", tmp_path / "kb")
    shutil.copytree(FIXTURES / "reference-run-hydra-settlement", tmp_path / "pdf-runs" / "hydra-settlement-test-fixture-4edb3c3c")
    shutil.copytree(FIXTURES / "reference-run-web-ows", tmp_path / "research-runs" / "reference-run-web-ows")

    cfg = load_config(tmp_path)
    stats = compile_index(cfg)  # real sentence-transformers embeddings
    assert stats["claims"] >= 6  # 5 pdf + >=1 web
    assert stats["wiki_chunks"] >= 1 and stats["claim_vectors"] >= 6

    idx = Index.open(cfg)
    try:
        assert idx.search_wiki("settlement")
        assert idx.search_claims("throughput")
        assert idx.get_entity("hydra-head")["name"]
        assert any(n["node"] for n in idx.neighbors("hydra-head", depth=2))
        # Every real fixture claim has a verbatim quote -> all included, none rejected on verbatim grounds.
        dossier = idx.compose_dossier(query="hydra settlement", k=12)
        assert dossier["included"]
    finally:
        idx.close()
```

- [ ] **Step 3: Run the heavy test locally (requires the compiler extra)**

Run: `pip install -e ".[dev,pdf,compiler]"`
Run: `pytest tests/integration/test_full_compiler_pipeline.py -m heavy -v`
Expected: PASS (first run downloads `all-MiniLM-L6-v2`).

- [ ] **Step 4: Confirm the fast suite still ignores it**

Run: `pytest tests -m "not heavy" -q`
Expected: PASS; the heavy test is deselected.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/reference-kb tests/fixtures/reference-run-web-ows tests/integration/test_full_compiler_pipeline.py
git commit -m "Add compiler fixtures and the heavy end-to-end integration test"
```

---

## Task 17: CI, drt doctor, and dependency-tier wiring

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `src/deep_research_toolkit/cli.py` (`cmd_doctor`)
- Test: `tests/unit/test_config.py` (doctor smoke, optional)

- [ ] **Step 1: Wire duckdb+lancedb into the fast CI job**

In `.github/workflows/ci.yml`, the `fast` job's install step, change to install the pip-wheel compiler deps (no torch) so the compiler fast tests run:

```yaml
      - name: Install package (dev + pdf; duckdb/lancedb for compiler fast tests)
        run: pip install -e ".[dev,pdf]" "duckdb>=1.0" "lancedb>=0.15"
```

In the `heavy` job's install step, add the full compiler extra:

```yaml
      - run: pip install -e ".[dev,pdf,compiler]"
```

- [ ] **Step 2: Extend `drt doctor`**

In `cli.py`'s `cmd_doctor`, add to the `checks` list:

```python
        ("compiler", "sentence_transformers", "sentence-transformers"),
        ("compiler", "openai", "openai"),
```

Under `--warm`, add an optional local-endpoint note (no hard dependency): print that a `local` provider needs a running Ollama/vLLM endpoint at `llm.local.base_url`.

- [ ] **Step 3: Run the doctor and the full fast suite**

Run: `python -m deep_research_toolkit.cli doctor`
Expected: lists duckdb/lancedb/sentence_transformers/openai with ok/missing, exit reflects missing tiers.
Run: `pytest tests -m "not heavy" -q`
Expected: PASS.
Run: `ruff check src/ skills/`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml src/deep_research_toolkit/cli.py
git commit -m "Wire compiler deps into fast CI and extend drt doctor checks"
```

---

## Task 18: Documentation, ADR, and changelog

**Files:**
- Create: `docs/contracts/knowledge-compiler.md`
- Create: `docs/decisions/0002-knowledge-compiler.md`
- Modify: `docs/contracts/schema-versions.md`
- Modify: `CHANGELOG.md`
- Modify: `README.md`

- [ ] **Step 1: Write the contract doc**

Create `docs/contracts/knowledge-compiler.md` covering: the `evidence_ref` shape and the PDF/web normalization mapping; the DuckDB table list (from `schema.py`); the two LanceDB tables; the RRF definition (`k=60`); each of the 8 tool contracts (input args + JSON output shape, matching `tools.py`); the `compose_dossier` verbatim gate; the `research-runs/<id>/` layout; and an "LLM backends" section (agent default vs local Ornith, and the `extract_claims.py` path with its auto-drop behavior).

- [ ] **Step 2: Write ADR 0002**

Create `docs/decisions/0002-knowledge-compiler.md` (Status: Accepted) recording the three build-time decisions: (a) full vector stack required + injectable test embedder for fast CI; (b) web `evidence_ref` shape and index-time normalization (PDF files unchanged); (c) the opt-in `local` extension of ADR 0001 decision #4 (agent stays default). Note full-rebuild-not-incremental and git-ignored index as accepted trade-offs.

- [ ] **Step 3: Update the schema registry and changelog**

In `docs/contracts/schema-versions.md`, add a `0.2.0` row to the table (research-run `claims/entities/relations.jsonl` reuse `1.0`; note the DuckDB `index_schema_version` is internal, not a portable on-disk contract). In `CHANGELOG.md` under `[Unreleased]` → rename to a `0.2.0` section and add: knowledge-compiler + retrieval-planner skills, web claim-extraction, the pluggable LLM backend with optional local Ornith, and the new `[compiler]` deps.

- [ ] **Step 4: Update the README**

In `README.md`: move the knowledge compiler from "Designed, not yet built" to "Built and tested"; redraw the diagram's dashed lower half (`knowledge compiler`, `retrieval-planner tools`, `evidence dossier`) as solid; add a short section describing the two new skills, the web claim-extraction step, and the optional local-Ornith backend (noting it is opt-in and gated by the verbatim check). Keep prose in the repo's existing 3-5-paragraph style.

- [ ] **Step 5: Final full verification**

Run: `python scripts/check-manifests-in-sync.py && python scripts/check-skill-templates-in-sync.py`
Expected: both OK.
Run: `pytest tests -m "not heavy" -q`
Expected: PASS.
Run: `ruff check src/ skills/`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add docs/ CHANGELOG.md README.md
git commit -m "Document the knowledge compiler: contract doc, ADR 0002, schema registry, README"
```

---

## Self-Review notes (for the executor)

- **Spec coverage:** every spec section maps to a task — evidence_ref (T2), web extraction (T13), index (T2/T4/T5), 8 tools (T6–T10), package layout (T2–T10,T14), embeddings/CI (T3,T16,T17), LLM backend (T14,T15), docs/versioning (T1,T17,T18).
- **Fast-CI invariant:** Tasks 2–10 tests inject `FakeEmbedder` and use in-memory/temp DuckDB + LanceDB — no torch, no network. Only Task 16's test carries `@pytest.mark.heavy`.
- **Verbatim gate:** defined once in `dossier.verbatim_ok` (T9), reused by `tools.compose_dossier` (T10) and `llm.extract` (T15) — the load-bearing invariant lives in one place.
- **Naming consistency:** `EvidenceRef`, `normalize_evidence`, `create_tables`, `compile_index`, `open_duckdb`, `rrf_fuse`, `hybrid_search`, `neighbors`, `find_candidates`, `compose_dossier`, `Index`, `get_backend`, `strip_think`, `extract_claims_to_run`, `start_research_run` are used identically across the tasks that define and consume them.
