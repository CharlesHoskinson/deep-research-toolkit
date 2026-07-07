"""Six live-model canaries against a real local serving endpoint.

Each targets one Gemma-4-on-Ollama failure mode from the eval-suite design
(docs/superpowers/specs/2026-07-06-comprehensive-eval-and-flow-hardening-design.md
§3.2): reasoning leakage under think:false, context-window truncation, JSON
mode under think:false, wiki marker-citation fidelity, the 31B long-prompt
flash-attention hang (ollama/ollama#15368), and call-to-call determinism.

All state a real serving endpoint would produce is *recorded* into
canary_report (see conftest.py) rather than hard-asserted, except where the
canary has a genuine pass/fail contract (documented per test below)."""
from __future__ import annotations

import concurrent.futures
import json
import re
import urllib.request

import pytest

from deep_research_toolkit.llm.backend import get_backend
from deep_research_toolkit.llm.wiki import _SYSTEM as _WIKI_SYSTEM
from deep_research_toolkit.llm.wiki import _task as _wiki_task

pytestmark = pytest.mark.live_model

# A ~30-token filler sentence (rough English 1.3 tokens/word), repeated to pad
# the sentinel prompt out to an approximate target token count -- there's no
# tokenizer dependency in this repo, so sizing is an estimate, not exact.
_FILLER_SENTENCE = (
    "The autonomous survey drone logged routine telemetry from the mid-ocean "
    "sensor array, then continued its patrol before returning to the charging "
    "dock for the night. "
)
_TOKENS_PER_FILLER = 32


def _raw_chat(base_url: str, payload: dict, timeout: float = 60.0) -> dict:
    """POST to the OpenAI-compatible /chat/completions route directly (no
    `openai` client), for probes that need fields Backend.complete() doesn't
    forward (e.g. reasoning_effort on a non-role call, or a raw prompt)."""
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _filler_block(target_tokens: int) -> str:
    repeats = max(1, round(target_tokens / _TOKENS_PER_FILLER))
    return _FILLER_SENTENCE * repeats


def _sentinel_prompt(target_tokens: int, codeword: str = "ZEPHYR-KUMQUAT-42") -> str:
    return (
        f"SECRET CODEWORD: {codeword}.\n\n"
        + _filler_block(target_tokens)
        + "\n\nWhat is the secret codeword stated at the very beginning of this "
        "message? Reply with only the codeword, nothing else."
    )


def _find_31b_model(config) -> str | None:
    """Scan config.llm_roles values + llm_local for a model tag containing
    '31b' -- whichever role a real project maps to the 31B model, if any."""
    candidates = [config.llm_local.get("model", "")]
    candidates += [spec.get("model", "") for spec in config.llm_roles.values()]
    for model in candidates:
        if "31b" in model.lower():
            return model
    return None


def test_reasoning_suppression(live_backend_config, canary_report):
    """extract-role calls must actually answer (thinking=False), and a raw
    reasoning_effort:"none" call must not leak a non-empty reasoning field."""
    config = live_backend_config
    backend = get_backend(config, role="extract")
    reply = backend.complete("Reply with the word OK only.", "Say OK.")
    assert reply.strip() != ""

    model = config.llm_roles["extract"]["model"]
    resp = _raw_chat(config.llm_local["base_url"], {
        "model": model,
        "messages": [{"role": "user", "content": "Say OK."}],
        "reasoning_effort": "none",
        "max_tokens": 20,
    })
    message = resp["choices"][0]["message"]
    reasoning = message.get("reasoning")
    canary_report["reasoning_suppression"] = {"model": model, "reasoning_field": reasoning}
    assert not reasoning


def test_context_ceiling(live_backend_config, canary_report):
    """The ~6k-token sentinel probe must find the codeword (hard assertion).
    12k/20k/40k probes are recorded only -- this is where a real truncation
    ceiling shows up (see README "Serving knobs that matter")."""
    config = live_backend_config
    model = config.llm_roles["extract"]["model"]
    base_url = config.llm_local["base_url"]

    def _probe(target_tokens: int) -> bool:
        prompt = _sentinel_prompt(target_tokens)
        resp = _raw_chat(base_url, {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "none",
            "temperature": 0.0,
            "max_tokens": 50,
        }, timeout=120.0)
        content = resp["choices"][0]["message"].get("content") or ""
        return "ZEPHYR" in content.upper()

    ceiling = {6000: _probe(6000)}
    assert ceiling[6000], "6k-token sentinel probe failed to recover the codeword"
    for size in (12000, 20000, 40000):
        ceiling[size] = _probe(size)
    canary_report["context_ceiling"] = ceiling


def test_structured_output_with_think_off(live_backend_config):
    """extract-role JSON mode (response_format=json_object, thinking off)
    must produce a reply that parses as JSON."""
    backend = get_backend(live_backend_config, role="extract")
    reply = backend.complete(
        "Reply with strict JSON only, no commentary, no code fences.",
        'Emit a JSON object with exactly this shape: {"status": "ok"}',
    )
    parsed = json.loads(reply)
    assert isinstance(parsed, dict)


def test_marker_fidelity_smoke(live_backend_config, canary_report):
    """A wiki-role call over 2 synthetic claims must cite at least one known
    id, in bare ([c1]) or prefixed ([claim:c1]) form; the bare/prefixed split
    is recorded (the exact failure mode Task 1's normalize_claim_markers
    exists to correct)."""
    config = live_backend_config
    backend = get_backend(config, role="wiki_write")
    claims = [
        {"claim_id": "c1", "claim": "The settlement layer confirms transactions instantly.",
         "supporting_evidence": [{"locator": "docA#c001", "quote": "confirms transactions instantly"}]},
        {"claim_id": "c2", "claim": "The reconciliation phase runs once per epoch.",
         "supporting_evidence": [{"locator": "docA#c002", "quote": "runs once per epoch"}]},
    ]
    allowed = ["c1", "c2"]
    user = _wiki_task("Test Page", "concept", claims)
    raw = backend.complete(_WIKI_SYSTEM, user)

    bare_re = re.compile(r"\[(" + "|".join(allowed) + r")\]")
    prefixed_re = re.compile(r"\[claim:(" + "|".join(allowed) + r")\]")
    bare_matches = bare_re.findall(raw)
    prefixed_matches = prefixed_re.findall(raw)
    bare_n, prefixed_n = len(bare_matches), len(prefixed_matches)
    total = bare_n + prefixed_n
    canary_report["marker_fidelity"] = {
        "bare": bare_n, "prefixed": prefixed_n,
        "bare_rate": (bare_n / total) if total else None,
    }
    cited_ids = set(bare_matches) | set(prefixed_matches)
    assert cited_ids & set(allowed), f"reply cited neither known id (bare or prefixed): {raw!r}"


def test_long_prompt_liveness_31b(live_backend_config, canary_report):
    """An ~8k-token prompt to a configured :31b model must return within 180s
    wall clock. A hang here is the documented flash-attention issue
    (ollama/ollama#15368), not a correctness bug -- fail loud with the fix."""
    config = live_backend_config
    model = _find_31b_model(config)
    if not model:
        pytest.skip("no configured role maps to a :31b model -- nothing to probe")
    base_url = config.llm_local["base_url"]
    prompt = _filler_block(8000) + "\n\nSummarize the above in one short sentence."

    def _call():
        return _raw_chat(base_url, {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
        }, timeout=170.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_call)
        try:
            resp = future.result(timeout=180)
        except concurrent.futures.TimeoutError:
            pytest.fail(
                "31B long-prompt call exceeded 180s -- see ollama/ollama#15368 "
                "(flash-attention hang); try OLLAMA_FLASH_ATTENTION=0"
            )
    content = resp["choices"][0]["message"].get("content") or ""
    canary_report["long_prompt_liveness_31b"] = {"model": model, "non_empty": bool(content.strip())}
    assert content.strip() != ""


def test_determinism_smoke(live_backend_config, canary_report):
    """Two identical temp-0 extract-role calls with the same tiny prompt.
    The backend doesn't expose a seed override, so this only records whether
    the endpoint happens to be deterministic at temp 0 -- both replies must
    still be non-empty."""
    backend = get_backend(live_backend_config, role="extract")
    system = "Reply with strict JSON only, no commentary."
    user = 'Emit a JSON object with exactly this shape: {"status": "ok"}'
    reply_a = backend.complete(system, user, temperature=0.0)
    reply_b = backend.complete(system, user, temperature=0.0)
    canary_report["determinism"] = {"identical": reply_a == reply_b}
    assert reply_a.strip() != ""
    assert reply_b.strip() != ""
