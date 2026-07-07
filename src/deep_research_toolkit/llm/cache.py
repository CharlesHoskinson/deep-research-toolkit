"""Content-hash response cache: deterministic pipeline (reasoning_effort=none)
means an identical (model, role, prompt, params, schema) always yields the same
reply, so a sha256-keyed JSONL cache makes --runs N and halved-batch re-runs
nearly free. Opt-in via llm.cache: true."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def cache_key(model, role, system, user, params: dict, schema: dict | None) -> str:
    blob = json.dumps({"model": model, "role": role, "system": system, "user": user,
                       "params": params or {}, "schema": schema},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CachingBackend:
    def __init__(self, inner, cache_dir: Path, enabled: bool = True,
                 role: str = "", model: str | None = None):
        self.inner = inner
        self.enabled = enabled
        self.role = role
        self.model = model or getattr(inner, "model", "")
        self.thinking = getattr(inner, "thinking", True)
        # Proxied from the inner backend after each real call; Task 8's
        # truncation counter (extract.py) reads it. None on a cache hit
        # (no model call this turn) and until the first miss.
        self.last_finish_reason: str | None = None
        self._path = Path(cache_dir) / "llm-cache.jsonl"
        self._mem: dict[str, str] = {}
        if enabled and self._path.is_file():
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self._mem[row["key"]] = row["reply"]

    #: Generation settings that live on the inner backend as constructor-time
    #: attributes (never passed through **sampling by production callers). They
    #: must feed the cache key, else a `.deepresearch.yml` edit (e.g. a role's
    #: max_tokens or response_format) silently returns a stale cached reply.
    _GEN_ATTRS = ("temperature", "top_p", "top_k", "max_tokens", "thinking")

    def complete(self, system, user, **sampling) -> str:
        if not self.enabled:
            return self.inner.complete(system, user, **sampling)
        # Resolve the effective params from the inner backend's defaults,
        # overlaid with any per-call sampling overrides. response_format is
        # keyed separately as `schema`, so keep it out of the params sub-dict.
        params = {a: getattr(self.inner, a, None) for a in self._GEN_ATTRS}
        params.update({k: v for k, v in sampling.items() if k != "response_format"})
        schema = sampling.get("response_format", getattr(self.inner, "response_format", None))
        key = cache_key(self.model, self.role, system, user, params, schema)
        if key in self._mem:
            self.last_finish_reason = None  # served from cache, no model call
            return self._mem[key]
        reply = self.inner.complete(system, user, **sampling)
        self.last_finish_reason = getattr(self.inner, "last_finish_reason", None)
        self._mem[key] = reply
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "reply": reply}, ensure_ascii=False) + "\n")
        return reply
