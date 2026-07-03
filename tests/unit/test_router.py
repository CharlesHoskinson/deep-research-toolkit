"""Table-driven tests for decide_route -- pure function over a signals dict,
no PDF/Docling/network needed. One case per route."""
import pytest

from deep_research_toolkit.pdf.router import decide_route, slugify

BASE_SIGNALS = {
    "avg_extractable_chars_per_page": 800.0,
    "has_acroform_fields": False,
    "image_only_page_ratio": 0.0,
    "table_like_page_ratio": 0.1,
    "detected_math_density": "low",
}


def _signals(**overrides):
    return {**BASE_SIGNALS, **overrides}


@pytest.mark.parametrize(
    "signals,expected_route",
    [
        (_signals(), "digital-text"),
        (_signals(has_acroform_fields=True), "form"),
        (_signals(image_only_page_ratio=0.9), "scanned"),
        (_signals(detected_math_density="high"), "scientific-math"),
        (_signals(table_like_page_ratio=0.7), "financial-legal"),
        (_signals(avg_extractable_chars_per_page=50.0), "slide-like"),
    ],
)
def test_decide_route(signals, expected_route):
    route, _note = decide_route(signals)
    assert route == expected_route


def test_form_wins_over_every_other_signal():
    # AcroForm check is first in priority order -- must win even if every
    # other signal also points somewhere else.
    signals = _signals(
        has_acroform_fields=True,
        image_only_page_ratio=0.9,
        detected_math_density="high",
        table_like_page_ratio=0.9,
    )
    route, _ = decide_route(signals)
    assert route == "form"


@pytest.mark.parametrize(
    "stem,expected",
    [
        ("Hydra Settlement Test Fixture", "hydra-settlement-test-fixture"),
        ("weird!!name__with--symbols", "weird-name-with-symbols"),
        ("", "document"),
        ("---", "document"),
    ],
)
def test_slugify(stem, expected):
    assert slugify(stem) == expected
