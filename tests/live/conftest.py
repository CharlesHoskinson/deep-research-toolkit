"""Fixtures for the live-model test tier.

Every test under tests/live/ needs a real OpenAI-compatible serving endpoint
(Ollama, vLLM). `live_backend_config` is the single skip gate: it loads the
project config and skips with a clear reason -- never fails -- when the
provider isn't 'local' or the endpoint doesn't answer. `canary_report`
collects whatever the canaries choose to record and dumps it once, at session
teardown, to eval-results/canaries-<UTC timestamp>.json for later inspection
(gate mismatches, context-ceiling findings, marker-fidelity rates, etc.)."""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pytest

from deep_research_toolkit.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "eval-results"


@pytest.fixture(scope="session")
def live_backend_config():
    """The project's loaded config, gated on a real serving endpoint.

    Skips (does not fail) the whole live tier when llm.provider isn't
    'local', or when a 2s probe of the configured base_url's /api/version
    doesn't answer -- e.g. no server running, wrong port, endpoint down."""
    config = load_config()
    if config.llm_provider != "local":
        pytest.skip("llm.provider is not 'local' -- live tests need a serving endpoint")
    base_url = config.llm_local["base_url"]
    version_url = base_url.replace("/v1", "") + "/api/version"
    try:
        urllib.request.urlopen(version_url, timeout=2)
    except Exception as e:  # noqa: BLE001 -- any failure means "no live endpoint"
        pytest.skip(f"no live endpoint at {base_url}: {e}")
    return config


@pytest.fixture(scope="session")
def canary_report():
    """A dict canaries append findings into, keyed by canary name. Dumped once
    at session teardown; nothing is written if no canary ran (e.g. the whole
    tier was skipped) or none of them recorded anything."""
    report: dict = {}
    yield report
    if not report:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"canaries-{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
