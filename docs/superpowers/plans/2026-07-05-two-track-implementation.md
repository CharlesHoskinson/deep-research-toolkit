# Two-Track Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved two-track design (docs/superpowers/specs/2026-07-05-two-track-pipeline-gemma4-design.md): wire the three unwired local-LLM roles (Track B), add the write-time verbatim gate + portability hardening to the pure-skill path (Track A), and add latency/token metrics to the eval harness.

**Architecture:** New per-phase judgment callers (`llm/wiki.py`, `llm/adjudicate.py`, `llm/synthesize.py`) mirror `llm/extract.py`'s pattern — task-brief prompt, mechanical post-validation, never self-certifying. Prose roles are gated by a citation-marker check (`[claim:<id>]` markers must resolve to supplied gate-passed claims); adjudication is gated by schema/enum validation. Track A gets a `check_claims.py` write-time gate shared via `common/claims_check.py`. Skill scripts stay thin shims; scripts consume JSON from the deterministic `query.py` tools rather than opening DuckDB.

**Tech Stack:** Python 3.10+, stdlib + existing deps only (openai already optional). Tests: pytest, stub Backend objects (the `Backend` protocol is one method: `complete(system, user, **sampling) -> str`). Venv: `C:\deep-research-toolkit\.venv` (`.venv/Scripts/python`).

**Conventions that apply to every task:**
- Run tests with: `.venv/Scripts/python -m pytest tests/unit/<file> -q`
- Commit after each task. NO Co-Authored-By trailer.
- Scripts in `skills/*/scripts/` follow the `extract_claims.py` shim pattern: import from `deep_research_toolkit`, `sys.exit(str(e))` on `LLMBackendNotConfigured` (so `provider: agent` fails with guidance, per ADR 0001 decision 4).
- `skills/` is canonical; `src/deep_research_toolkit/skill_templates/` is synced only in Task 9 — do not hand-edit templates.

---

### Task 1: Backend usage stats + harness latency/token metrics

**Files:**
- Modify: `src/deep_research_toolkit/llm/local.py`
- Modify: `scripts/validate-local-llm.py`
- Test: `tests/unit/test_llm_local_stats.py` (new)

- [ ] **Step 1.1: Write the failing test**

```python
"""LocalOpenAIBackend accumulates per-call usage stats."""
from types import SimpleNamespace

from deep_research_toolkit.llm.local import LocalOpenAIBackend


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        msg = SimpleNamespace(content="hello")
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


def _backend():
    b = LocalOpenAIBackend(
        base_url="http://localhost:11434/v1", model="m", api_key="x",
        temperature=0.0, top_p=0.95, top_k=20,
    )
    b._client = _FakeClient()
    return b


def test_stats_accumulate_across_calls():
    b = _backend()
    b.complete("sys", "user")
    b.complete("sys", "user")
    assert b.stats["calls"] == 2
    assert b.stats["prompt_tokens"] == 22
    assert b.stats["completion_tokens"] == 14
    assert b.stats["seconds"] >= 0


def test_stats_survive_missing_usage():
    b = _backend()
    b._client.chat.completions.create = lambda **kw: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="x"))], usage=None
    )
    b.complete("sys", "user")
    assert b.stats["calls"] == 1
    assert b.stats["prompt_tokens"] == 0
```

- [ ] **Step 1.2: Run to verify failure** — `.venv/Scripts/python -m pytest tests/unit/test_llm_local_stats.py -q` → FAIL (`AttributeError: ... no attribute 'stats'`).

- [ ] **Step 1.3: Implement.** In `LocalOpenAIBackend.__init__`, after `self._client = None`, add:

```python
        # Cumulative usage over this backend's lifetime; read by the eval
        # harness to report cost/latency per model. Never reset internally.
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
```

Replace `complete()` with:

```python
    def complete(self, system: str, user: str, **sampling) -> str:
        import time
        t0 = time.perf_counter()
        resp = self._client_complete(system, user, **sampling)
        self.stats["calls"] += 1
        self.stats["seconds"] += time.perf_counter() - t0
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self.stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        return strip_think(resp.choices[0].message.content or "")
```

- [ ] **Step 1.4: Run to verify pass** — same command → 2 passed. Also run `.venv/Scripts/python -m pytest tests/unit/test_llm_backend.py tests/unit/test_llm_extract.py -q` (no regressions).

- [ ] **Step 1.5: Harness output.** In `scripts/validate-local-llm.py`, after the `parse_failures` print added earlier, add:

```python
    stats = getattr(backend, "stats", None)
    if stats and stats.get("calls"):
        print(f"backend stats: {stats['calls']} call(s), "
              f"{stats['prompt_tokens']} prompt + {stats['completion_tokens']} completion tokens, "
              f"{stats['seconds']:.1f}s total ({stats['seconds']/stats['calls']:.1f}s/call)")
```

- [ ] **Step 1.6: Commit** — `git add src/deep_research_toolkit/llm/local.py scripts/validate-local-llm.py tests/unit/test_llm_local_stats.py && git commit -m "Report per-model latency and token usage from the eval harness"`

---

### Task 2: Write-time claims gate — `common/claims_check.py` + skill CLIs

**Files:**
- Create: `src/deep_research_toolkit/common/claims_check.py`
- Create: `skills/knowledge-extraction/scripts/check_claims.py`
- Create: `skills/research-knowledge-graph/scripts/check_claims.py` (identical copy — the two-skill convention `extract_claims.py` already follows)
- Test: `tests/unit/test_claims_check.py` (new)

- [ ] **Step 2.1: Write the failing test**

```python
"""check_claims_file: mechanical write-time gate for agent-authored claims.jsonl."""
import json
import shutil
from pathlib import Path

from deep_research_toolkit.common.claims_check import check_claims_file

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "reference-run-hydra-settlement"


def _copy_fixture(tmp_path: Path) -> Path:
    run = tmp_path / FIXTURE.name
    shutil.copytree(FIXTURE, run)
    return run


def test_reference_run_passes_clean(tmp_path):
    report = check_claims_file(_copy_fixture(tmp_path))
    assert report["failures"] == []
    assert report["checked"] == report["ok"] > 0


def test_corrupted_quote_is_flagged(tmp_path):
    run = _copy_fixture(tmp_path)
    rows = [json.loads(l) for l in (run / "claims.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows[0]["supporting_evidence"][0]["quote"] = "this text appears in no chunk"
    (run / "claims.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    report = check_claims_file(run)
    assert len(report["failures"]) == 1
    assert report["failures"][0]["claim_id"] == rows[0]["claim_id"]
    assert "not a verbatim substring" in report["failures"][0]["reason"]


def test_missing_evidence_is_flagged(tmp_path):
    run = _copy_fixture(tmp_path)
    rows = [json.loads(l) for l in (run / "claims.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows[0]["supporting_evidence"] = []
    (run / "claims.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    report = check_claims_file(run)
    assert any("no supporting evidence" in f["reason"] for f in report["failures"])
```

- [ ] **Step 2.2: Run to verify failure** — `ModuleNotFoundError: ... claims_check`.

- [ ] **Step 2.3: Implement `src/deep_research_toolkit/common/claims_check.py`:**

```python
"""Mechanical write-time gate for a run directory's claims.jsonl.

Same invariant as extraction's inline gate and the eval harness: every
evidence quote must be a verbatim substring of the chunk its locator names.
This module exists so the pure-skill (provider: agent) track can gate
agent-authored claims at write time instead of only at compile/eval time.
Purely deterministic -- no LLM, no network (ADR 0001 decision 4)."""
from __future__ import annotations

import json
from pathlib import Path

from .verbatim import chunk_text_by_locator, verbatim_ok


def check_claims_file(run_dir: Path | str) -> dict:
    """Validate <run_dir>/claims.jsonl against <run_dir>/chunks.jsonl.

    Returns {"checked": int, "ok": int, "failures": [{"claim_id", "reason"}]}.
    Evidence rows may use `locator` (web runs) or `node_id` (pdf runs) as the
    chunk key -- the same two shapes extraction emits."""
    run_dir = Path(run_dir)
    chunks = chunk_text_by_locator(run_dir)
    claims_path = run_dir / "claims.jsonl"
    if not claims_path.is_file():
        return {"checked": 0, "ok": 0, "failures": [{"claim_id": None, "reason": f"no claims.jsonl in {run_dir}"}]}

    checked, ok, failures = 0, 0, []
    for line in claims_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        checked += 1
        try:
            claim = json.loads(line)
        except json.JSONDecodeError as e:
            failures.append({"claim_id": None, "reason": f"line {checked}: invalid JSON ({e})"})
            continue
        cid = claim.get("claim_id")
        evidence = claim.get("supporting_evidence") or []
        if not evidence:
            failures.append({"claim_id": cid, "reason": "no supporting evidence"})
            continue
        bad = []
        for ev in evidence:
            key = ev.get("locator") or ev.get("node_id") or ""
            quote = ev.get("quote") or ""
            if not quote:
                bad.append(f"empty quote (locator {key!r})")
            elif not verbatim_ok(quote, chunks.get(key, "")):
                bad.append(f"quote not a verbatim substring of chunk {key!r}")
        if bad:
            failures.append({"claim_id": cid, "reason": "; ".join(bad)})
        else:
            ok += 1
    return {"checked": checked, "ok": ok, "failures": failures}
```

- [ ] **Step 2.4: Run to verify pass** — 3 passed.

- [ ] **Step 2.5: CLI shim** at `skills/knowledge-extraction/scripts/check_claims.py`; then copy byte-identical to `skills/research-knowledge-graph/scripts/check_claims.py`:

```python
#!/usr/bin/env python3
"""Write-time verbatim gate for agent-authored claims.

Run after appending each extraction batch to claims.jsonl:

  python scripts/check_claims.py <run_dir>

Exit 0: every claim's every quote is a verbatim substring of its chunk.
Exit 1: failures listed as JSON on stdout -- fix or drop those claims
before extracting the next batch. Deterministic; no model, no network."""
import json
import sys

from deep_research_toolkit.common.claims_check import check_claims_file


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    report = check_claims_file(sys.argv[1])
    print(json.dumps(report, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2.6: Smoke both shims** — `.venv/Scripts/python skills/knowledge-extraction/scripts/check_claims.py tests/fixtures/reference-run-hydra-settlement` → exit 0, `"failures": []`. Same for the research-knowledge-graph copy against `tests/fixtures/reference-run-web-ows`.

- [ ] **Step 2.7: Commit** — `git add src/deep_research_toolkit/common/claims_check.py skills/*/scripts/check_claims.py tests/unit/test_claims_check.py && git commit -m "Add write-time verbatim gate for agent-authored claims (Track A)"`

---

### Task 3: Shared response helpers — `llm/response.py`

**Files:**
- Create: `src/deep_research_toolkit/llm/response.py`
- Test: `tests/unit/test_llm_response.py` (new)

- [ ] **Step 3.1: Write the failing test**

```python
from deep_research_toolkit.llm.response import extract_claim_ids, parse_json_block, validate_citations


def test_extract_claim_ids_in_order_with_dupes_removed():
    text = "Praos is adaptive [claim:c1]. It tolerates delay [claim:c2] [claim:c1]."
    assert extract_claim_ids(text) == ["c1", "c2"]


def test_validate_citations_flags_unknown_and_reports_coverage():
    text = "A [claim:c1]. B [claim:zz]."
    rep = validate_citations(text, allowed_ids=["c1", "c2", "c3"])
    assert rep["cited"] == ["c1"]
    assert rep["unknown"] == ["zz"]
    assert rep["coverage"] == 1 / 3


def test_parse_json_block_prefers_output_tags():
    text = 'noise {"a": 1} noise <output>[{"b": 2}]</output>'
    assert parse_json_block(text) == [{"b": 2}]


def test_parse_json_block_falls_back_to_bracket_slice():
    assert parse_json_block('prose... [{"x": 1}, {"x": 2}] trailing') == [{"x": 1}, {"x": 2}]


def test_parse_json_block_returns_none_on_garbage():
    assert parse_json_block("no json here") is None
```

- [ ] **Step 3.2: Run to verify failure.**

- [ ] **Step 3.3: Implement `src/deep_research_toolkit/llm/response.py`:**

```python
"""Shared mechanical validation for programmatic judgment callers.

Citation markers: prose roles (wiki_write, synthesize) must tag claim-bearing
sentences with [claim:<id>]. The marker check is the prose analogue of the
verbatim gate -- it cannot prove the prose is faithful, but it proves every
cited id exists in the gate-passed set supplied to the model, so nothing can
cite evidence the corpus does not hold."""
from __future__ import annotations

import json
import re

CLAIM_MARKER_RE = re.compile(r"\[claim:([A-Za-z0-9_\-\.]+)\]")
_OUTPUT_RE = re.compile(r"<output>(.*?)</output>", re.DOTALL)


def extract_claim_ids(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in CLAIM_MARKER_RE.finditer(text):
        seen.setdefault(m.group(1))
    return list(seen)


def validate_citations(text: str, allowed_ids: list[str]) -> dict:
    allowed = set(allowed_ids)
    cited_all = extract_claim_ids(text)
    cited = [c for c in cited_all if c in allowed]
    unknown = [c for c in cited_all if c not in allowed]
    return {
        "cited": cited,
        "unknown": unknown,
        "coverage": (len(cited) / len(allowed)) if allowed else 0.0,
    }


def parse_json_block(text: str):
    """JSON from a model reply: prefer the last <output>...</output> block,
    else the widest [...] or {...} slice. None if nothing parses."""
    blocks = _OUTPUT_RE.findall(text)
    candidates = [blocks[-1]] if blocks else []
    for opener, closer in ("[]", "{}"):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None
```

- [ ] **Step 3.4: Run to verify pass** — 5 passed.
- [ ] **Step 3.5: Commit** — `git commit -m "Add shared citation-marker and JSON-block helpers for judgment callers"` (add both files).

---

### Task 4: Programmatic wiki writer — `llm/wiki.py` + skill script

**Files:**
- Create: `src/deep_research_toolkit/llm/wiki.py`
- Create: `skills/llm-wiki-writer/scripts/write_wiki_page.py`
- Test: `tests/unit/test_llm_wiki.py` (new)

- [ ] **Step 4.1: Write the failing test**

```python
import pytest

from deep_research_toolkit.llm.wiki import CitationError, write_wiki_body

CLAIMS = [
    {"claim_id": "c1", "claim": "Praos was introduced in 2018.",
     "supporting_evidence": [{"locator": "n1", "quote": "introduced in 2018"}]},
    {"claim_id": "c2", "claim": "Praos tolerates message delays.",
     "supporting_evidence": [{"locator": "n2", "quote": "tolerates delays"}]},
]


class StubBackend:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user, **kw):
        self.calls.append((system, user))
        return self.replies.pop(0)


def test_valid_body_passes_and_reports_coverage():
    body = "## Overview\n\nPraos arrived in 2018 [claim:c1] and tolerates delays [claim:c2].\n"
    out = write_wiki_body("Ouroboros Praos", "Concept", CLAIMS, StubBackend([body]))
    assert out["body"] == body
    assert out["citations"]["coverage"] == 1.0


def test_unknown_marker_retries_once_then_raises():
    bad = "Praos is fast [claim:nope]."
    backend = StubBackend([bad, bad])
    with pytest.raises(CitationError):
        write_wiki_body("Praos", "Concept", CLAIMS, backend)
    assert len(backend.calls) == 2
    assert "nope" in backend.calls[1][1]  # correction prompt names the bad id


def test_empty_claims_is_an_error():
    with pytest.raises(ValueError):
        write_wiki_body("Praos", "Concept", [], StubBackend(["x"]))
```

- [ ] **Step 4.2: Run to verify failure.**

- [ ] **Step 4.3: Implement `src/deep_research_toolkit/llm/wiki.py`:**

```python
"""Programmatic wiki-body writer (role: wiki_write) for provider: local.

Mirrors extract.py's stance: the model proposes prose; a mechanical check
(citation markers resolving to supplied gate-passed claims) decides whether
it is accepted. Under provider: agent the backend raises -- the in-session
agent writes wiki pages itself per the llm-wiki-writer SKILL.md."""
from __future__ import annotations

import json

from .response import validate_citations

_SYSTEM = """You write one wiki page body for a research knowledge base.

GOAL: synthesize the supplied claims into clear, well-organized markdown
prose for the page named in the task. Use only the supplied claims -- no
outside knowledge, no speculation.

OUTPUT CONTRACT:
- Markdown body only. No frontmatter, no code fences around the whole reply.
- Every sentence that states a fact MUST end with the marker of the claim it
  came from, formatted exactly: [claim:<claim_id>]
- A sentence may carry several markers. Do not invent claim ids.
- Organize with ## sections when the material warrants it; otherwise a single
  coherent body. Neutral, precise register. No filler.
"""

_CORRECTION = (
    "Your previous body cited unknown claim id(s): {bad}. Every [claim:...] "
    "marker must use one of the supplied claim_ids exactly. Rewrite the full "
    "body now, fixing or removing the offending sentences."
)


class CitationError(ValueError):
    pass


def _task(title: str, page_type: str, claims: list[dict]) -> str:
    rows = [
        {"claim_id": c.get("claim_id"), "claim": c.get("claim"),
         "quotes": [ev.get("quote") for ev in (c.get("supporting_evidence") or [])]}
        for c in claims
    ]
    return (
        f"PAGE: {title} (type: {page_type})\n\nCLAIMS (the only permitted sources):\n"
        + json.dumps(rows, ensure_ascii=False, indent=1)
    )


def write_wiki_body(title: str, page_type: str, claims: list[dict], backend,
                    min_coverage: float = 0.5) -> dict:
    """Returns {"body": str, "citations": validate_citations report}.

    Raises CitationError if the model cites unknown ids twice, or ValueError
    if called with no claims / coverage below min_coverage after retry."""
    if not claims:
        raise ValueError("write_wiki_body needs at least one gate-passed claim")
    allowed = [c.get("claim_id") for c in claims]
    user = _task(title, page_type, claims)
    body = backend.complete(_SYSTEM, user).strip()
    report = validate_citations(body, allowed)
    if report["unknown"]:
        body = backend.complete(_SYSTEM, user + "\n\n" + _CORRECTION.format(bad=report["unknown"])).strip()
        report = validate_citations(body, allowed)
        if report["unknown"]:
            raise CitationError(f"model cited unknown claim ids after retry: {report['unknown']}")
    if report["coverage"] < min_coverage:
        raise ValueError(
            f"body cites {len(report['cited'])}/{len(allowed)} claims "
            f"(coverage {report['coverage']:.2f} < {min_coverage}) -- refusing a page that ignores its sources"
        )
    return {"body": body, "citations": report}
```

- [ ] **Step 4.4: Run to verify pass** — 3 passed.

- [ ] **Step 4.5: Skill script** `skills/llm-wiki-writer/scripts/write_wiki_page.py`:

```python
#!/usr/bin/env python3
"""Programmatic wiki page writer (provider: local, role: wiki_write).

Reads gate-passed claims from <run_dir>/claims.jsonl (optionally filtered to
one entity), asks the local wiki_write model for a cited body, mechanically
validates the [claim:<id>] markers, then scaffolds the page and records it in
the run's audit trail. Under provider: agent this exits with guidance --
write the page yourself per SKILL.md.

  python scripts/write_wiki_page.py <run_dir> <knowledge_path> \\
      --type Concept --title "Ouroboros Praos" [--entity "Praos"] [--force]
"""
import argparse
import json
import sys
from pathlib import Path

from deep_research_toolkit.common.scaffold import PageAlreadyExists, scaffold_page
from deep_research_toolkit.config import load_config
from deep_research_toolkit.llm.backend import LLMBackendNotConfigured, get_backend
from deep_research_toolkit.llm.wiki import CitationError, write_wiki_body
from deep_research_toolkit.pdf.wiki_writer import record_wiki_page


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("knowledge_path")
    parser.add_argument("--type", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--entity", help="only claims mentioning this entity (case-insensitive substring)")
    parser.add_argument("--description")
    parser.add_argument("--tags", help="comma-separated")
    parser.add_argument("--source-docs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    claims_path = run_dir / "claims.jsonl"
    if not claims_path.is_file():
        sys.exit(f"no claims.jsonl in {run_dir} -- run extraction first")
    claims = [json.loads(l) for l in claims_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.entity:
        needle = args.entity.lower()
        claims = [c for c in claims
                  if needle in (c.get("claim") or "").lower()
                  or any(needle in (e or "").lower() for e in (c.get("entities") or []))]
    if not claims:
        sys.exit("no claims matched -- nothing to synthesize")

    config = load_config()
    try:
        backend = get_backend(config, role="wiki_write")
        out = write_wiki_body(args.title, args.type, claims, backend)
    except (LLMBackendNotConfigured, CitationError, ValueError) as e:
        sys.exit(str(e))

    given = Path(args.knowledge_path)
    path = given if given.is_absolute() else config.knowledge_base_path / given
    try:
        scaffold_page(
            path, type=args.type, title=args.title, description=args.description,
            tags=[t.strip() for t in args.tags.split(",")] if args.tags else None,
            source_docs=[d.strip() for d in args.source_docs.split(",")] if args.source_docs else None,
            status="draft", body=out["body"], force=args.force,
        )
    except PageAlreadyExists as e:
        sys.exit(str(e))
    record_wiki_page(args.run_dir, args.knowledge_path)
    rep = out["citations"]
    print(f"wrote {path} ({len(rep['cited'])} claim(s) cited, coverage {rep['coverage']:.2f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.6: Smoke the agent-provider guard** — in a temp dir with no `.deepresearch.yml` (default provider), run the script against the hydra fixture: expect clean exit with the AgentBackend guidance message, no page written. Command: `cd $TMP && .venv/Scripts/python <repo>/skills/llm-wiki-writer/scripts/write_wiki_page.py <repo>/tests/fixtures/reference-run-hydra-settlement concepts/smoke.md --type Concept --title Smoke` → exits 1 with "provider" guidance.

- [ ] **Step 4.7: Commit** — `git commit -m "Wire the wiki_write role: programmatic cited wiki-body writer"` (add all three files).

---

### Task 5: Programmatic contradiction adjudicator — `llm/adjudicate.py` + skill script

**Files:**
- Create: `src/deep_research_toolkit/llm/adjudicate.py`
- Create: `skills/retrieval-planner/scripts/adjudicate_contradictions.py`
- Test: `tests/unit/test_llm_adjudicate.py` (new)

- [ ] **Step 5.1: Write the failing test**

```python
import json

from deep_research_toolkit.llm.adjudicate import adjudicate_candidates

CANDS = [
    {"kind": "relation", "subject": "praos", "predicate": "introduced_in",
     "objects": ["2017", "2018"], "relation_ids": ["r1", "r2"], "source_ids": ["s1", "s2"]},
]


class StubBackend:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user, **kw):
        return self.reply


def _reply(verdicts):
    return "<output>" + json.dumps(verdicts) + "</output>"


def test_valid_verdict_accepted():
    reply = _reply([{"subject": "praos", "predicate": "introduced_in",
                     "verdict": "contradiction", "rationale": "years differ",
                     "relation_ids": ["r1", "r2"]}])
    out = adjudicate_candidates(CANDS, StubBackend(reply))
    assert len(out["verdicts"]) == 1
    assert out["invalid"] == []


def test_bad_enum_and_foreign_relation_ids_are_invalid():
    reply = _reply([{"subject": "praos", "predicate": "introduced_in",
                     "verdict": "maybe", "rationale": "?", "relation_ids": ["r9"]}])
    out = adjudicate_candidates(CANDS, StubBackend(reply))
    assert out["verdicts"] == []
    assert len(out["invalid"]) == 1


def test_unparseable_reply_counts_as_parse_failure():
    out = adjudicate_candidates(CANDS, StubBackend("no json at all"))
    assert out["parse_failures"] == 1
```

- [ ] **Step 5.2: Run to verify failure.**

- [ ] **Step 5.3: Implement `src/deep_research_toolkit/llm/adjudicate.py`:**

```python
"""Programmatic contradiction adjudication (role: conflict_adjudicate).

Consumes the mechanical candidates from compiler.contradictions.find_candidates
(via query.py find-contradictions) and asks a reasoning model for verdicts.
Verdicts are proposals: schema-validated here, stored with rationale, never
silently applied to the corpus. Under provider: agent the backend raises and
the retrieval-planner SKILL.md batched pass applies instead."""
from __future__ import annotations

import json

from .response import parse_json_block

VERDICTS = ("contradiction", "not_contradiction", "insufficient_evidence")

_SYSTEM = """You adjudicate contradiction candidates from a knowledge graph.

Each candidate groups relations sharing (subject, predicate) but disagreeing
on object. Some are real contradictions; many are benign (aliases, different
granularity, non-exclusive values, time-qualified facts).

OUTPUT CONTRACT: reply with exactly one <output>...</output> block containing
a JSON array, one element per candidate, each:
  {"subject": str, "predicate": str,
   "verdict": "contradiction" | "not_contradiction" | "insufficient_evidence",
   "rationale": str (one or two sentences),
   "relation_ids": [ids copied from that candidate]}
Judge only from the supplied data; when the objects could coexist, say
not_contradiction; when you cannot tell, say insufficient_evidence.
"""


def adjudicate_candidates(candidates: list[dict], backend) -> dict:
    """Returns {"verdicts": [...], "invalid": [...], "parse_failures": int}."""
    relation_cands = [c for c in candidates if c.get("kind") == "relation"]
    if not relation_cands:
        return {"verdicts": [], "invalid": [], "parse_failures": 0}
    user = "CANDIDATES:\n" + json.dumps(relation_cands, ensure_ascii=False, indent=1)
    reply = backend.complete(_SYSTEM, user)
    data = parse_json_block(reply)
    if not isinstance(data, list):
        return {"verdicts": [], "invalid": [], "parse_failures": 1}
    allowed_ids = {rid for c in relation_cands for rid in (c.get("relation_ids") or [])}
    verdicts, invalid = [], []
    for row in data:
        if not isinstance(row, dict):
            invalid.append({"row": row, "reason": "not an object"})
            continue
        problems = []
        if row.get("verdict") not in VERDICTS:
            problems.append(f"verdict {row.get('verdict')!r} not in {VERDICTS}")
        rids = row.get("relation_ids") or []
        foreign = [r for r in rids if r not in allowed_ids]
        if not rids or foreign:
            problems.append(f"relation_ids missing or unknown: {foreign or 'empty'}")
        if not (row.get("rationale") or "").strip():
            problems.append("empty rationale")
        if problems:
            invalid.append({"row": row, "reason": "; ".join(problems)})
        else:
            verdicts.append(row)
    return {"verdicts": verdicts, "invalid": invalid, "parse_failures": 0}
```

- [ ] **Step 5.4: Run to verify pass** — 3 passed.

- [ ] **Step 5.5: Skill script** `skills/retrieval-planner/scripts/adjudicate_contradictions.py`:

```python
#!/usr/bin/env python3
"""Adjudicate contradiction candidates with the local conflict_adjudicate model.

Pipe candidates in from the deterministic tool, write verdicts JSONL out:

  python scripts/query.py find-contradictions > candidates.json
  python scripts/adjudicate_contradictions.py candidates.json --out adjudications.jsonl

Verdicts are proposals with rationales -- review them before acting on the
corpus. Under provider: agent this exits with guidance; run the SKILL.md
batched adjudication pass instead."""
import argparse
import json
import sys
from pathlib import Path

from deep_research_toolkit.config import load_config
from deep_research_toolkit.llm.adjudicate import adjudicate_candidates
from deep_research_toolkit.llm.backend import LLMBackendNotConfigured, get_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("candidates", help="JSON file from `query.py find-contradictions` ('-' for stdin)")
    parser.add_argument("--out", default="adjudications.jsonl")
    args = parser.parse_args()

    raw = sys.stdin.read() if args.candidates == "-" else Path(args.candidates).read_text(encoding="utf-8")
    candidates = json.loads(raw)
    if not candidates:
        print("no candidates -- nothing to adjudicate")
        return

    try:
        backend = get_backend(load_config(), role="conflict_adjudicate")
        result = adjudicate_candidates(candidates, backend)
    except LLMBackendNotConfigured as e:
        sys.exit(str(e))

    with open(args.out, "w", encoding="utf-8") as f:
        for v in result["verdicts"]:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    print(f"{len(result['verdicts'])} verdict(s) -> {args.out}; "
          f"{len(result['invalid'])} invalid row(s); {result['parse_failures']} parse failure(s)")
    if result["invalid"]:
        print(json.dumps(result["invalid"], indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.6: Commit** — `git commit -m "Wire the conflict_adjudicate role: schema-validated contradiction verdicts"` (add all three files).

---

### Task 6: Programmatic dossier synthesis — `llm/synthesize.py` + skill script

**Files:**
- Create: `src/deep_research_toolkit/llm/synthesize.py`
- Create: `skills/retrieval-planner/scripts/synthesize_dossier.py`
- Test: `tests/unit/test_llm_synthesize.py` (new)

- [ ] **Step 6.1: Write the failing test**

```python
import pytest

from deep_research_toolkit.llm.synthesize import CitationError, synthesize_thesis

DOSSIER = {
    "included": [
        {"claim_id": "c1", "claim": "Praos was introduced in 2018.",
         "evidence": [{"quote": "introduced in 2018", "source_id": "s1"}]},
        {"claim_id": "c2", "claim": "Praos tolerates delays.",
         "evidence": [{"quote": "tolerates delays", "source_id": "s1"}]},
    ],
    "rejected": [],
}


class StubBackend:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, system, user, **kw):
        return self.replies.pop(0)


def test_valid_thesis_passes():
    reply = "Praos, introduced in 2018 [claim:c1], tolerates delays [claim:c2]."
    out = synthesize_thesis("How robust is Praos?", DOSSIER, StubBackend([reply]))
    assert out["thesis"] == reply
    assert out["citations"]["coverage"] == 1.0


def test_unknown_id_retries_then_raises():
    bad = "Praos is quantum-safe [claim:c9]."
    with pytest.raises(CitationError):
        synthesize_thesis("q", DOSSIER, StubBackend([bad, bad]))


def test_empty_dossier_is_an_error():
    with pytest.raises(ValueError):
        synthesize_thesis("q", {"included": [], "rejected": []}, StubBackend(["x"]))
```

- [ ] **Step 6.2: Run to verify failure.**

- [ ] **Step 6.3: Implement `src/deep_research_toolkit/llm/synthesize.py`:**

```python
"""Programmatic thesis synthesis over a composed dossier (role: synthesize).

Input is compose_dossier output: claims that already passed the verbatim
gate. The model writes an argued thesis; the mechanical check is the same
citation-marker gate as wiki_write -- every claim-bearing sentence must cite
a claim_id from the dossier's included set."""
from __future__ import annotations

import json

from .response import validate_citations
from .wiki import CitationError

_SYSTEM = """You write the synthesis section of an evidence dossier.

GOAL: answer the question using only the included claims below -- weigh
them, connect them, state what they establish and what remains open.

OUTPUT CONTRACT:
- Markdown only, no title heading (the dossier supplies one).
- Every sentence that rests on a claim MUST end with [claim:<claim_id>]
  markers referencing the supplied claims. Do not invent ids.
- If the claims cannot answer the question, say exactly what is missing.
"""

_CORRECTION = (
    "Your previous synthesis cited unknown claim id(s): {bad}. Only supplied "
    "claim_ids may appear in [claim:...] markers. Rewrite the full synthesis."
)


def synthesize_thesis(question: str, dossier: dict, backend,
                      min_coverage: float = 0.3) -> dict:
    included = dossier.get("included") or []
    if not included:
        raise ValueError("dossier has no included claims -- nothing to synthesize")
    allowed = [c.get("claim_id") for c in included]
    rows = [{"claim_id": c.get("claim_id"), "claim": c.get("claim"),
             "quotes": [e.get("quote") for e in (c.get("evidence") or [])]}
            for c in included]
    user = f"QUESTION: {question}\n\nINCLUDED CLAIMS:\n" + json.dumps(rows, ensure_ascii=False, indent=1)
    thesis = backend.complete(_SYSTEM, user).strip()
    report = validate_citations(thesis, allowed)
    if report["unknown"]:
        thesis = backend.complete(_SYSTEM, user + "\n\n" + _CORRECTION.format(bad=report["unknown"])).strip()
        report = validate_citations(thesis, allowed)
        if report["unknown"]:
            raise CitationError(f"model cited unknown claim ids after retry: {report['unknown']}")
    if report["coverage"] < min_coverage:
        raise ValueError(
            f"synthesis cites {len(report['cited'])}/{len(allowed)} included claims "
            f"(coverage {report['coverage']:.2f} < {min_coverage})"
        )
    return {"thesis": thesis, "citations": report}
```

- [ ] **Step 6.4: Run to verify pass** — 3 passed.

- [ ] **Step 6.5: Skill script** `skills/retrieval-planner/scripts/synthesize_dossier.py`:

```python
#!/usr/bin/env python3
"""Synthesize a cited thesis over a composed dossier with the local model.

  python scripts/query.py compose-dossier --claims c1,c2 --format json > dossier.json
  python scripts/synthesize_dossier.py dossier.json --question "..." [--out thesis.md]

Output: the model's thesis (citation-gated) followed by the standard
self-citing dossier appendix. Under provider: agent this exits with
guidance; synthesize in-session instead."""
import argparse
import json
import sys
from pathlib import Path

from deep_research_toolkit.compiler.dossier import render_dossier_markdown
from deep_research_toolkit.config import load_config
from deep_research_toolkit.llm.backend import LLMBackendNotConfigured, get_backend
from deep_research_toolkit.llm.synthesize import synthesize_thesis
from deep_research_toolkit.llm.wiki import CitationError


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dossier", help="JSON file from `query.py compose-dossier --format json` ('-' for stdin)")
    parser.add_argument("--question", required=True)
    parser.add_argument("--out", help="write markdown here instead of stdout")
    args = parser.parse_args()

    raw = sys.stdin.read() if args.dossier == "-" else Path(args.dossier).read_text(encoding="utf-8")
    dossier = json.loads(raw)
    try:
        backend = get_backend(load_config(), role="synthesize")
        out = synthesize_thesis(args.question, dossier, backend)
    except (LLMBackendNotConfigured, CitationError, ValueError) as e:
        sys.exit(str(e))

    doc = (f"# Synthesis: {args.question}\n\n{out['thesis']}\n\n---\n\n"
           + render_dossier_markdown(dossier))
    if args.out:
        Path(args.out).write_text(doc, encoding="utf-8")
        print(f"wrote {args.out} (coverage {out['citations']['coverage']:.2f})")
    else:
        print(doc)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.6: Commit** — `git commit -m "Wire the synthesize role: citation-gated thesis over composed dossiers"` (add all three files).

---

### Task 7: Track A hardening — SKILL.md wording, gate loop, batching/resume

**Files:**
- Modify: `skills/knowledge-extraction/SKILL.md`
- Modify: `skills/research-knowledge-graph/SKILL.md`
- Modify: `skills/llm-wiki-writer/SKILL.md`, `skills/retrieval-planner/SKILL.md` (mention the new local-provider scripts)
- Modify: `README.md` (agent-wording instances only)

No unit tests — prose. Verification is grep-based.

- [ ] **Step 7.1: De-agent-brand the wording.** `grep -n "Claude" skills/*/SKILL.md skills/*/references/*.md` — for each hit that means "the in-session model" (not "Claude Code" the product), replace with "the agent" or imperative "you". Product names ("Claude Code", "Codex") stay. Same pass over README.md sections "The skills" and "The optional local LLM backend" (e.g. the line "the default worker is the in-session agent itself" pattern is already right; fix any "Claude does this directly" phrasing).

- [ ] **Step 7.2: Insert the write-time gate loop** into `skills/knowledge-extraction/SKILL.md` (in the by-hand extraction procedure, after the step that writes claims) and the equivalent spot in `skills/research-knowledge-graph/SKILL.md`:

```markdown
**Gate every batch before moving on.** After appending a batch of claims to
`claims.jsonl`, run:

    python scripts/check_claims.py <run_dir>

Exit 1 lists each failing claim and why (non-verbatim quote, missing
evidence). Fix or drop those claims now — re-quote from the chunk text,
never paraphrase — and re-run until it exits 0. Do not extract the next
batch over unfixed failures: the compile-time gate would reject them later
anyway, after you have lost the context to repair them.
```

- [ ] **Step 7.3: Insert batching/resume discipline** into the same two SKILL.md files (before the extraction procedure):

```markdown
**Work in batches; keep progress on disk.** Process `chunks.jsonl` in order,
10–20 chunks at a time; append each batch's output before reading the next.
After each gated batch, record the last chunk you finished in
`manifest.json` under `stages.<skill-name>.last_chunk`. If you are resuming
(after compaction, a crash, or a new session), re-read this SKILL.md, read
`last_chunk`, and continue from the next chunk — never restart a run that
has gated output. If your environment supports parallel subagents, you may
split the remaining chunks into contiguous ranges (one subagent per range,
each returning claims JSONL for you to gate and merge) — an optimization,
never a requirement.
```

- [ ] **Step 7.4: Mention the local-provider scripts** — one short paragraph each in `llm-wiki-writer/SKILL.md` ("Under `provider: local`, `scripts/write_wiki_page.py` drafts a citation-gated body with the `wiki_write` model; review the draft page like any other") and `retrieval-planner/SKILL.md` (same for `adjudicate_contradictions.py` and `synthesize_dossier.py`, noting verdicts/theses are proposals to review).

- [ ] **Step 7.5: Description budget check** — run:

```bash
.venv/Scripts/python - <<'EOF'
from pathlib import Path
import re
for p in sorted(Path("skills").glob("*/SKILL.md")):
    fm = p.read_text(encoding="utf-8").split("---")[1]
    m = re.search(r"^description:\s*(.+?)(?=^\w+:|\Z)", fm, re.M | re.S)
    desc = " ".join(m.group(1).split()) if m else ""
    print(f"{len(desc):5d}  {p.parent.name}")
EOF
```

Trim any description over 500 chars, preserving its trigger phrases (Codex preloads all skills' metadata into an 8,000-char shared budget).

- [ ] **Step 7.6: Verify** — `grep -rn "Claude" skills/` returns only product-name references; re-run `.venv/Scripts/python -m pytest tests/unit/test_lint.py tests/unit/test_frontmatter.py -q` (frontmatter still parses).

- [ ] **Step 7.7: Commit** — `git commit -m "Harden the pure-skill track: neutral wording, write-time gate loop, batch/resume discipline"`

---

### Task 8: Config template + README local-models update

**Files:**
- Modify: `src/deep_research_toolkit/cli.py` (DEFAULT_YAML_TEMPLATE)
- Modify: `README.md` ("Running local models" section)
- Test: existing `tests/unit/test_cli.py` (update if it asserts the exact template)

- [ ] **Step 8.1: Add a commented roles block** to `DEFAULT_YAML_TEMPLATE` in `cli.py`, inside the `llm:` mapping after the `local:` block:

```yaml
#   roles:                      # optional per-phase model routing (provider: local)
#     extract:     {model: gemma4:e4b}     # high-volume JSON; non-thinking
#     wiki_write:  {model: gemma4:12b}
#     conflict_adjudicate: {model: gemma4:31b}
#     synthesize:  {model: gemma4:31b}
```

(Match the template's existing comment style and indentation exactly; keep it commented out so `drt init` behavior is unchanged.)

- [ ] **Step 8.2: Run** `.venv/Scripts/python -m pytest tests/unit/test_cli.py -q`; if a test asserts the template verbatim, update its expected text to include the new commented block.

- [ ] **Step 8.3: README "Running local models":** replace the illustrative `qwen3.6`/`ornith` example roles block with the validated Gemma 4 mapping from the design doc §5.1, and add one paragraph after the "non-thinking switch" discussion:

```markdown
Gemma 4 sharpens this point: under Ollama its thinking mode is on by
default, and on the OpenAI-compatible endpoint `think: false` is silently
ignored (ollama/ollama#15288) — the switch that works there is
`reasoning_effort: "none"`, which the local backend now sends automatically
whenever a role sets `thinking: false`. Measured on the hydra fixture, all
Gemma 4 dense sizes pass the verbatim gate at 100% with reasoning
suppressed, 4–5× faster than with it left on.
```

- [ ] **Step 8.4: Commit** — `git commit -m "Document the validated Gemma 4 role map; surface llm.roles in the init template"`

---

### Task 9: Template sync + full verification

**Files:**
- Modify (generated): `src/deep_research_toolkit/skill_templates/**`

- [ ] **Step 9.1: Sync** — `.venv/Scripts/python scripts/sync-skill-templates.py`
- [ ] **Step 9.2: Check sync guards** — `.venv/Scripts/python scripts/check-skill-templates-in-sync.py && .venv/Scripts/python scripts/check-manifests-in-sync.py` → both clean.
- [ ] **Step 9.3: Full test suite** — `.venv/Scripts/python -m pip install -q duckdb lancedb` (enables the compiler tests skipped so far), then `.venv/Scripts/python -m pytest tests/unit -q` → all pass (sentence-transformers-dependent tests may skip; that is expected — note any skips in the commit message).
- [ ] **Step 9.4: Lint** — `.venv/Scripts/python -m pip install -q ruff && .venv/Scripts/python -m ruff check src skills scripts tests` → clean.
- [ ] **Step 9.5: Commit** — `git add -A && git commit -m "Sync skill templates for the two-track wiring"`

---

## Out of scope (deliberately)

- The ~200-chunk eval corpus (needs the user's chosen source documents; it is the fine-tune go/no-go gate, not code).
- A Codex-produced reference run (needs a Codex session).
- vLLM serving configs and any fine-tuning work (design doc §7 — gated on the 200-chunk results).
- Live end-to-end runs of the three new role scripts against Ollama (manual validation, same status as `validate-local-llm.py`; the harness for extract already covers the serving path they share).
