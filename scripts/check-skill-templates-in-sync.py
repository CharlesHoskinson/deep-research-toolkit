#!/usr/bin/env python3
"""CI guard: src/deep_research_toolkit/skill_templates/ must be an exact
copy of the canonical skills/ tree. Run scripts/sync-skill-templates.py if
this fails.

Compares by content hash, not filecmp's default shallow (size + mtime) check:
a same-length edit to skills/ that forgot to re-run the sync script would slip
past a stat-based comparison but is caught here.
"""
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "skills"
DEST = REPO_ROOT / "src" / "deep_research_toolkit" / "skill_templates"


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*") if p.is_file()
    }


def main() -> int:
    if not DEST.is_dir():
        print(f"{DEST} does not exist -- run scripts/sync-skill-templates.py")
        return 1

    src, dst = _tree_hashes(SOURCE), _tree_hashes(DEST)
    problems = []
    for rel in sorted(set(src) - set(dst)):
        problems.append(f"only in skills/: {rel}")
    for rel in sorted(set(dst) - set(src)):
        problems.append(f"only in skill_templates/: {rel}")
    for rel in sorted(set(src) & set(dst)):
        if src[rel] != dst[rel]:
            problems.append(f"content differs: {rel}")

    if problems:
        print("skills/ and skill_templates/ have drifted:")
        for p in problems:
            print(f"  - {p}")
        print("\nRun: python scripts/sync-skill-templates.py")
        return 1

    print("OK: skill_templates/ matches skills/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
