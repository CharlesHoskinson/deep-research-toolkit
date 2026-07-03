#!/usr/bin/env python3
"""Thin CLI shim: fetch a URL via deep_research_toolkit.web.fetch and dump to stdout or a file."""
import argparse
import sys

from deep_research_toolkit.web.fetch import ScraplingNotInstalled, fetch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--mode", choices=["http", "stealth"], default="http")
    parser.add_argument("--css", help="CSS selector to extract instead of full page")
    parser.add_argument("--out", help="Write output here instead of stdout")
    args = parser.parse_args()

    try:
        result = fetch(args.url, mode=args.mode, css=args.css)
    except ScraplingNotInstalled as e:
        sys.exit(str(e))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print(result)


if __name__ == "__main__":
    main()
