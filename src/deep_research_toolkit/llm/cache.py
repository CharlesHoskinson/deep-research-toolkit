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
        self._path = Path(cache_dir) / "llm-cache.jsonl"
        self._mem: dict[str, str] = {}
        if enabled and self._path.is_file():
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self._mem[row["key"]] = row["reply"]

    def complete(self, system, user, **sampling) -> str:
        if not self.enabled:
            return self.inner.complete(system, user, **sampling)
        schema = sampling.get("response_format")
        key = cache_key(self.model, self.role, system, user,
                        {k: v for k, v in sampling.items() if k != "response_format"}, schema)
        if key in self._mem:
            return self._mem[key]
        reply = self.inner.complete(system, user, **sampling)
        self._mem[key] = reply
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "reply": reply}, ensure_ascii=False) + "\n")
        return reply
