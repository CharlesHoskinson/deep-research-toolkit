"""Scrapling wrapper: fetch a URL as text/markdown, http or stealth mode.

Lazy imports + a specific, actionable ImportError message -- this is the
pattern every skill script in this suite follows for its heavy extras
(see docs/environment.md): fail at the point of use, name the exact extra
needed, never a raw traceback.
"""
from __future__ import annotations


class ScraplingNotInstalled(RuntimeError):
    def __init__(self):
        super().__init__(
            "scrapling is not installed. This skill needs the 'web' extra: "
            'pip install "deep-research-toolkit[web]" && scrapling install'
        )


def fetch_http(url: str):
    try:
        from scrapling.fetchers import Fetcher
    except ImportError as e:
        raise ScraplingNotInstalled() from e
    return Fetcher.get(url)


def fetch_stealth(url: str):
    try:
        from scrapling.fetchers import StealthyFetcher
    except ImportError as e:
        raise ScraplingNotInstalled() from e
    return StealthyFetcher.fetch(url, headless=True, solve_cloudflare=True)


def fetch(url: str, mode: str = "http", css: str | None = None) -> str:
    """Fetch a URL and return either the full body or a CSS-selected subset."""
    page = fetch_stealth(url) if mode == "stealth" else fetch_http(url)
    if css:
        return "\n".join(page.css(css).getall())
    return page.body if hasattr(page, "body") else str(page)
