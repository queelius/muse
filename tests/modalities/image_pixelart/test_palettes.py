"""Palette data, palette-spec grammar, OKLab and nearest-colour mapping."""
from __future__ import annotations

import re

import numpy as np
import pytest

from muse.modalities.image_pixelart.palettes import (
    NAMED_PALETTES,
    PaletteSpec,
    auto_palette,
    nearest_palette_index,
    parse_palette,
    srgb_to_oklab,
)

HEX = re.compile(r"^#[0-9a-f]{6}$")


def test_named_palettes_keys_exact():
    assert set(NAMED_PALETTES) == {
        "pico-8", "db16", "db32", "endesga-32", "nes", "gameboy",
    }


def test_named_palettes_sizes():
    assert len(NAMED_PALETTES["pico-8"]) == 16
    assert len(NAMED_PALETTES["db16"]) == 16
    assert len(NAMED_PALETTES["db32"]) == 32
    assert len(NAMED_PALETTES["endesga-32"]) == 32
    assert len(NAMED_PALETTES["gameboy"]) == 4
    assert len(NAMED_PALETTES["nes"]) >= 52


@pytest.mark.parametrize("name", sorted(NAMED_PALETTES))
def test_named_palette_values_are_lowercase_hex_and_unique(name):
    colors = NAMED_PALETTES[name]
    assert isinstance(colors, tuple)
    assert all(HEX.match(c) for c in colors), colors
    assert len(set(colors)) == len(colors)


def test_named_palette_spot_values():
    # Public values (Lospec / Lexaloffle), spot-checked.
    assert NAMED_PALETTES["pico-8"][0] == "#000000"
    assert NAMED_PALETTES["pico-8"][8] == "#ff004d"
    assert NAMED_PALETTES["pico-8"][15] == "#ffccaa"
    assert NAMED_PALETTES["db16"][0] == "#140c1c"
    assert NAMED_PALETTES["db16"][15] == "#deeed6"
    assert NAMED_PALETTES["db32"][5] == "#df7126"
    assert NAMED_PALETTES["db32"][31] == "#8a6f30"
    assert NAMED_PALETTES["endesga-32"][0] == "#be4a2f"
    assert NAMED_PALETTES["endesga-32"][26] == "#ff0044"
    assert NAMED_PALETTES["gameboy"] == ("#081820", "#346856", "#88c070", "#e0f8d0")
    assert "#0078f8" in NAMED_PALETTES["nes"]


def test_parse_named():
    spec = parse_palette("db32")
    assert spec == PaletteSpec(kind="named", colors=NAMED_PALETTES["db32"], count=32)
    assert parse_palette(" PICO-8 ").colors == NAMED_PALETTES["pico-8"]


def test_parse_auto_and_none():
    assert parse_palette("auto:16") == PaletteSpec(kind="auto", colors=(), count=16)
    assert parse_palette("auto").count == 32
    assert parse_palette("auto:2").count == 2
    assert parse_palette("auto:256").count == 256
    assert parse_palette("none") == PaletteSpec(kind="none", colors=(), count=None)


def test_parse_custom():
    spec = parse_palette("#FF0000, #00ff00,#0000ff")
    assert spec.kind == "custom"
    assert spec.colors == ("#ff0000", "#00ff00", "#0000ff")
    assert spec.count == 3


def test_palette_spec_is_frozen():
    spec = parse_palette("db16")
    with pytest.raises(Exception):
        spec.kind = "none"  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad",
    ["auto:1", "auto:300", "auto:x", "nope", "#12345g", "#12345", "", "#ff0000,,#00ff00",
     ",".join(["#000000"] * 2 + [f"#{i:06x}" for i in range(300)])],
)
def test_parse_palette_errors(bad):
    with pytest.raises(ValueError):
        parse_palette(bad)


def test_oklab_reference_values():
    lab = srgb_to_oklab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]]))
    assert lab.shape == (3, 3)
    np.testing.assert_allclose(lab[0], [1.0, 0.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(lab[1], [0.0, 0.0, 0.0], atol=1e-6)
    # Ottosson's reference value for sRGB red.
    np.testing.assert_allclose(lab[2], [0.62796, 0.22486, 0.12585], atol=1e-3)


def test_nearest_palette_index_exact_and_near():
    pal = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0]], dtype=np.uint8)
    rgb = np.array([[[0, 0, 0], [250, 250, 250]], [[255, 0, 0], [200, 10, 10]]], dtype=np.uint8)
    idx = nearest_palette_index(rgb, pal)
    assert idx.shape == (2, 2)
    assert idx.tolist() == [[0, 1], [2, 2]]


def test_auto_palette_uses_opaque_pixels_only():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:2] = [200, 30, 30]
    rgb[2:] = [10, 200, 10]
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2] = True
    pal = auto_palette(rgb, 4, mask)
    assert pal.tolist() == [[200, 30, 30]]


def test_auto_palette_caps_colour_count():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    pal = auto_palette(rgb, 8, None)
    assert pal.dtype == np.uint8 and pal.shape[1] == 3
    assert 1 <= len(pal) <= 8


def test_auto_palette_keeps_exact_colours_when_few():
    cols = np.array([[1, 2, 3], [200, 100, 50], [9, 9, 9]], dtype=np.uint8)
    rgb = cols[np.arange(30) % 3].reshape(5, 6, 3)
    pal = auto_palette(rgb, 16, None)
    assert {tuple(c) for c in pal.tolist()} == {tuple(c) for c in cols.tolist()}
