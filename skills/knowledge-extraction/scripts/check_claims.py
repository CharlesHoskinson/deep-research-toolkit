#!/usr/bin/env python3
"""Write-time verbatim gate for agent-authored claims.

Run after appending each extraction batch to claims.jsonl:

  python scripts/check_claims.py <run_dir>

Exit 0: every claim's every quote is a verbatim substring of its chunk.
Exit 1: failures listed as JSON on stdout -- fix or drop those claims
before extracting the next batch. Deterministic; no model, no network."""
import json
import sys

from deep_research_toolkit.common.claims_check import check_claims_file


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    report = check_claims_file(sys.argv[1])
    print(json.dumps(report, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
