#!/usr/bin/env python3
"""Optional: extract claims.jsonl from chunks.jsonl using the configured LLM
backend (only under llm.provider=local). Under the default agent provider,
do this by hand following SKILL.md instead.

python scripts/extract_claims.py <run_dir>
"""
import argparse
import sys
from pathlib import Path

from deep_research_toolkit.config import load_config
from deep_research_toolkit.llm.backend import LLMBackendNotConfigured, get_backend
from deep_research_toolkit.llm.extract import extract_claims_to_run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    args = parser.parse_args()
    config = load_config()
    try:
        result = extract_claims_to_run(Path(args.run_dir), "web", config, get_backend(config, role="extract"))
    except LLMBackendNotConfigured as e:
        sys.exit(str(e))
    print(result)


if __name__ == "__main__":
    sys.exit(main())
