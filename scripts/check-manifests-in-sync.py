#!/usr/bin/env python3
"""CI guard: .claude-plugin/plugin.json and .codex-plugin/plugin.json must
agree on every shared identity field. They're allowed to diverge only on
fields one platform doesn't use (there are none today, but this leaves
room for e.g. a future Codex-only "interface" block without breaking this
check -- SHARED_FIELDS below is the actual contract, not "every key must
match").
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED_FIELDS = ["name", "version", "description", "license", "keywords", "skills"]


def main() -> int:
    claude_path = REPO_ROOT / ".claude-plugin" / "plugin.json"
    codex_path = REPO_ROOT / ".codex-plugin" / "plugin.json"

    with open(claude_path, encoding="utf-8") as f:
        claude = json.load(f)
    with open(codex_path, encoding="utf-8") as f:
        codex = json.load(f)

    problems = []
    for field in SHARED_FIELDS:
        c_val = claude.get(field)
        x_val = codex.get(field)
        if c_val != x_val:
            problems.append(f"'{field}' differs: .claude-plugin={c_val!r} vs .codex-plugin={x_val!r}")

    if problems:
        print("Plugin manifests have drifted:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"OK: {len(SHARED_FIELDS)} shared fields match between .claude-plugin and .codex-plugin manifests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
