"""Every heavy-extra dependency must fail with a specific, actionable
message naming the exact extra to install -- never a raw traceback. This
uses the `sys.modules[name] = None` trick (a documented way to force the
next `import name` to raise ModuleNotFoundError) rather than actually
uninstalling packages, so it works regardless of what's installed in the
test environment."""
import sys

import pytest

from deep_research_toolkit.pdf.router import PdfDepsNotInstalled, _import_pypdf
from deep_research_toolkit.web.fetch import ScraplingNotInstalled, fetch_http


def _block_import(monkeypatch, module_name):
    monkeypatch.setitem(sys.modules, module_name, None)


def test_fetch_http_without_scrapling_raises_specific_error(monkeypatch):
    _block_import(monkeypatch, "scrapling")
    _block_import(monkeypatch, "scrapling.fetchers")
    with pytest.raises(ScraplingNotInstalled) as exc_info:
        fetch_http("https://example.com")
    assert "deep-research-toolkit[web]" in str(exc_info.value)
    assert "scrapling install" in str(exc_info.value)


def test_import_pypdf_without_pypdf_raises_specific_error(monkeypatch):
    _block_import(monkeypatch, "pypdf")
    with pytest.raises(PdfDepsNotInstalled) as exc_info:
        _import_pypdf()
    assert "deep-research-toolkit[pdf]" in str(exc_info.value)
