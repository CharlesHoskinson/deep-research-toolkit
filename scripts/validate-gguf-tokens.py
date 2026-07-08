#!/usr/bin/env python3
"""CLI wrapper around `deep_research_toolkit.tunekit.gguf_check` (design doc
§6.2/§6.3): validates that a merged/exported GGUF's `<start_of_turn>`,
`<end_of_turn>`, BOS, and EOS tokens are token_type CONTROL, not NORMAL --
the unsloth#5070/#5386 corruption class that silently makes a served model
never stop generating. Report-only: prints a human-readable report and exits
0 (all control tokens correct) or 1 (at least one is wrong, or the file
isn't parseable) -- it never patches the GGUF itself.

Usage: python scripts/validate-gguf-tokens.py path/to/model.gguf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_research_toolkit.tunekit.gguf_check import (  # noqa: E402
    GGUFParseError,
    read_gguf_metadata,
    validate_control_tokens,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gguf_path", help="Path to the merged/exported .gguf file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    gguf_path = Path(args.gguf_path)
    try:
        with open(gguf_path, "rb") as f:
            data = f.read()
        metadata = read_gguf_metadata(data)
    except (GGUFParseError, OSError) as e:
        print(f"could not parse {gguf_path}: {e}", file=sys.stderr)
        return 1

    report = validate_control_tokens(metadata)

    if report["ok"]:
        print(f"OK: all control tokens correct in {gguf_path}")
        for name, ttype in report["checked"].items():
            print(f"  {name}: token_type={ttype}")
        return 0

    print(f"FAIL: control-token validation failed for {gguf_path}", file=sys.stderr)
    for error in report["errors"]:
        print(f"  - {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
