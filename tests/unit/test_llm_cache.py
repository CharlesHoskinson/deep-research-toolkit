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
