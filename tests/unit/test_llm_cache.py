from types import SimpleNamespace

from deep_research_toolkit.config import _resolve_roles
from deep_research_toolkit.llm.backend import get_backend
from deep_research_toolkit.llm.cache import CachingBackend, cache_key
from deep_research_toolkit.llm.local import LocalOpenAIBackend


class CountingBackend:
    thinking = False
    def __init__(self): self.calls = 0
    def complete(self, system, user, **kw): self.calls += 1; return "REPLY"


def test_cache_key_stable_and_param_sensitive():
    k1 = cache_key("m", "extract", "s", "u", {"temperature": 0.0}, None)
    k2 = cache_key("m", "extract", "s", "u", {"temperature": 0.0}, None)
    k3 = cache_key("m", "extract", "s", "u", {"temperature": 0.25}, None)
    assert k1 == k2 and k1 != k3


def test_caching_backend_hits_disk(tmp_path):
    inner = CountingBackend()
    cb = CachingBackend(inner, cache_dir=tmp_path, enabled=True)
    a = cb.complete("s", "u", temperature=0.0)
    b = cb.complete("s", "u", temperature=0.0)
    assert a == b == "REPLY" and inner.calls == 1  # second call served from cache


def test_disabled_is_passthrough(tmp_path):
    inner = CountingBackend()
    cb = CachingBackend(inner, cache_dir=tmp_path, enabled=False)
    cb.complete("s", "u"); cb.complete("s", "u")
    assert inner.calls == 2


def _cfg(provider, llm_cache=False, index_dir=None):
    local = {"base_url": "http://localhost:11434/v1", "model": "Ornith-1.0-9B",
             "api_key_env": "OPENAI_API_KEY", "temperature": 0.6, "top_p": 0.95, "top_k": 20,
             "max_tokens": 16000}
    return SimpleNamespace(llm_provider=provider, llm_local=local,
                           llm_roles=_resolve_roles({}, local),
                           llm_cache=llm_cache, index_dir=index_dir)


def test_get_backend_wraps_local_in_cache_when_enabled(tmp_path):
    cfg = _cfg("local", llm_cache=True, index_dir=tmp_path / ".deepresearch" / "index")
    backend = get_backend(cfg, role="extract")
    assert isinstance(backend, CachingBackend)
    assert isinstance(backend.inner, LocalOpenAIBackend)
    assert backend.role == "extract"


def test_get_backend_returns_plain_backend_when_cache_off(tmp_path):
    cfg = _cfg("local", llm_cache=False, index_dir=tmp_path / ".deepresearch" / "index")
    backend = get_backend(cfg)
    assert isinstance(backend, LocalOpenAIBackend)
    assert not isinstance(backend, CachingBackend)


class FakeInner:
    """Mimics LocalOpenAIBackend's constructor-time generation attributes,
    which callers never pass through **sampling."""
    def __init__(self, model="m", temperature=0.0, top_p=0.95, top_k=20,
                 max_tokens=3000, thinking=False, response_format=None):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.response_format = response_format
        self.last_finish_reason = None
        self.calls = 0

    def complete(self, system, user, **kw):
        self.calls += 1
        return "REPLY"


def test_cache_key_is_config_sensitive(tmp_path):
    # Two caches sharing one JSONL file, inner backends differing ONLY in a
    # constructor-time setting (max_tokens) never passed via sampling. A entry
    # warmed by backend A must NOT be served to backend B.
    a_inner = FakeInner(max_tokens=3000)
    b_inner = FakeInner(max_tokens=9000)
    a = CachingBackend(a_inner, cache_dir=tmp_path, enabled=True, role="extract")
    a.complete("s", "u", temperature=0.0)
    assert a_inner.calls == 1
    b = CachingBackend(b_inner, cache_dir=tmp_path, enabled=True, role="extract")
    b.complete("s", "u", temperature=0.0)
    assert b_inner.calls == 1  # miss: differing max_tokens -> different key

    # And response_format likewise participates in the key (via schema).
    c_inner = FakeInner(response_format="json")
    c = CachingBackend(c_inner, cache_dir=tmp_path, enabled=True, role="extract")
    c.complete("s", "u", temperature=0.0)
    assert c_inner.calls == 1


class FinishReasonInner:
    thinking = False
    def __init__(self):
        self.model = "m"
        self.calls = 0
        self.last_finish_reason = None
    def complete(self, system, user, **kw):
        self.calls += 1
        self.last_finish_reason = "length"
        return "REPLY"


def test_last_finish_reason_proxies_on_miss_and_resets_on_hit(tmp_path):
    inner = FinishReasonInner()
    cb = CachingBackend(inner, cache_dir=tmp_path, enabled=True, role="extract")
    assert cb.last_finish_reason is None  # initialized
    cb.complete("s", "u", temperature=0.0)  # MISS -> real inner call
    assert cb.last_finish_reason == "length"
    cb.complete("s", "u", temperature=0.0)  # HIT -> no model call this turn
    assert inner.calls == 1
    assert cb.last_finish_reason is None


class MetaInner(FinishReasonInner):
    """Inner backend that exposes the per-call meta channel."""
    def complete_with_meta(self, system, user, **kw):
        return self.complete(system, user, **kw), "length"


def test_complete_with_meta_miss_then_hit(tmp_path):
    inner = MetaInner()
    cb = CachingBackend(inner, cache_dir=tmp_path, enabled=True, role="extract")
    text, reason = cb.complete_with_meta("s", "u", temperature=0.0)  # MISS
    assert (text, reason) == ("REPLY", "length")
    assert cb.last_finish_reason == "length"
    text, reason = cb.complete_with_meta("s", "u", temperature=0.0)  # HIT
    assert (text, reason) == ("REPLY", None)  # cached: no model call, no reason
    assert cb.last_finish_reason is None
    assert inner.calls == 1


def test_complete_with_meta_falls_back_for_plain_inner(tmp_path):
    # Inner without complete_with_meta (older fakes, other providers): the
    # cache falls back to complete() + last_finish_reason.
    inner = FinishReasonInner()
    cb = CachingBackend(inner, cache_dir=tmp_path, enabled=True, role="extract")
    assert cb.complete_with_meta("s", "u", temperature=0.0) == ("REPLY", "length")
    disabled = CachingBackend(FinishReasonInner(), cache_dir=tmp_path, enabled=False)
    assert disabled.complete_with_meta("s", "u") == ("REPLY", "length")
