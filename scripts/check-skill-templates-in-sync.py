#!/usr/bin/env python3
"""CI guard: src/deep_research_toolkit/skill_templates/ must be an exact
copy of the canonical skills/ tree. Run scripts/sync-skill-templates.py if
this fails.
"""
import filecmp
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "skills"
DEST = REPO_ROOT / "src" / "deep_research_toolkit" / "skill_templates"


def main() -> int:
    if not DEST.is_dir():
        print(f"{DEST} does not exist -- run scripts/sync-skill-templates.py")
        return 1

    cmp = filecmp.dircmp(SOURCE, DEST)
    problems = []

    def walk(c: filecmp.dircmp, rel: str = "") -> None:
        for name in c.left_only:
            problems.append(f"only in skills/: {rel}{name}")
        for name in c.right_only:
            problems.append(f"only in skill_templates/: {rel}{name}")
        for name in c.diff_files:
            problems.append(f"content differs: {rel}{name}")
        for name, sub in c.subdirs.items():
            walk(sub, f"{rel}{name}/")

    walk(cmp)

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
