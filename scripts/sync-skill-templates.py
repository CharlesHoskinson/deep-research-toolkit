#!/usr/bin/env python3
"""Sync the canonical top-level skills/ tree into src/deep_research_toolkit/
skill_templates/, which is what actually ships inside a built wheel (package
data must live under the package's own source tree). The top-level skills/
stays the single source of truth -- this script is a one-way copy, never
edit skill_templates/ directly.

Run this after changing anything under skills/, and CI checks the two trees
match (see scripts/check-manifests-in-sync.py's sibling check).
"""
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "skills"
DEST = REPO_ROOT / "src" / "deep_research_toolkit" / "skill_templates"


def main() -> int:
    if not SOURCE.is_dir():
        print(f"no such directory: {SOURCE}", file=sys.stderr)
        return 1

    if DEST.exists():
        shutil.rmtree(DEST)
    shutil.copytree(SOURCE, DEST)

    file_count = sum(1 for _ in DEST.rglob("*") if _.is_file())
    print(f"synced {file_count} files from {SOURCE} -> {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
