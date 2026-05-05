"""Tests for the shared CLI console helpers."""
from muse.cli_impl.console import (
    STATUS_STYLE,
    status_glyph,
    status_legend,
    truncate,
)


def test_status_style_covers_four_states():
    assert set(STATUS_STYLE) == {"enabled", "disabled", "recommended", "available"}


def test_status_glyph_returns_one_char():
    """Single-cell glyphs only; emoji-width breaks table alignment."""
    for status in STATUS_STYLE:
        text = status_glyph(status)
        assert len(text.plain) == 1, f"{status!r} glyph wider than one char"


def test_status_glyph_unknown_falls_back():
    """A typo'd status doesn't crash the renderer."""
    fallback = status_glyph("not-a-real-status")
    assert fallback.plain == "?"


def test_status_legend_contains_every_status():
    legend = status_legend().plain
    for status in STATUS_STYLE:
        assert status in legend


def test_truncate_no_op_when_fits():
    assert truncate("hello", 10) == "hello"
    assert truncate("hello", 5) == "hello"


def test_truncate_adds_ellipsis():
    assert truncate("hello world", 8) == "hello w…"
    assert truncate("hello world", 8).endswith("…")


def test_truncate_max_width_too_small():
    """When max_width is no larger than the ellipsis, return only the ellipsis."""
    assert truncate("hello", 1) == "…"
    assert truncate("hello", 0) == "…"
