"""Tests for the shared CLI console helpers."""
from muse.cli_impl.console import (
    STATUS_STYLE,
    status_glyph,
    status_legend,
    truncate,
)


def test_status_style_covers_five_states():
    """v0.40.0 lazy load split `enabled` into loaded / unloaded variants.

    The five states cover the full lifecycle a catalog row can be in
    visible to the operator: loaded (running on a worker), unloaded
    (catalog-enabled but not currently in the director's loaded set),
    disabled, recommended (curated; not pulled), available (bundled;
    not pulled).
    """
    assert set(STATUS_STYLE) == {
        "enabled_loaded",
        "enabled_unloaded",
        "disabled",
        "recommended",
        "available",
    }


def test_status_style_enabled_loaded_uses_filled_circle():
    """The fully-active state remains the bright filled circle."""
    glyph, _ = STATUS_STYLE["enabled_loaded"]
    assert glyph == "●"  # filled circle


def test_status_style_enabled_unloaded_uses_half_circle():
    """The catalog-enabled-but-not-loaded state is a dim half circle."""
    glyph, _ = STATUS_STYLE["enabled_unloaded"]
    assert glyph == "◐"  # half-filled circle


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
