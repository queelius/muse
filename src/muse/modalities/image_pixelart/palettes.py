"""Palettes for pixel art: named retro palettes, the palette-spec grammar,
OKLab conversion, nearest-colour mapping and automatic (median cut) palettes.

Pillow + numpy only. Colours at the API boundary are lowercase ``#rrggbb``
strings; arrays are sRGB in 0..255 with the channel axis last.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

# Public palette values. Each entry was checked against Lospec's palette JSON
# (https://lospec.com/palette-list/<slug>.json) on 2026-10-28.
NAMED_PALETTES: dict[str, tuple[str, ...]] = {
    # PICO-8 by Lexaloffle Games (https://www.lexaloffle.com/pico-8.php),
    # https://lospec.com/palette-list/pico-8
    "pico-8": (
        "#000000", "#1d2b53", "#7e2553", "#008751", "#ab5236", "#5f574f",
        "#c2c3c7", "#fff1e8", "#ff004d", "#ffa300", "#ffec27", "#00e436",
        "#29adff", "#83769c", "#ff77a8", "#ffccaa",
    ),
    # DawnBringer 16 by DawnBringer (Richard Fhager),
    # https://lospec.com/palette-list/dawnbringer-16
    "db16": (
        "#140c1c", "#442434", "#30346d", "#4e4a4e", "#854c30", "#346524",
        "#d04648", "#757161", "#597dce", "#d27d2c", "#8595a1", "#6daa2c",
        "#d2aa99", "#6dc2ca", "#dad45e", "#deeed6",
    ),
    # DawnBringer 32 by DawnBringer (Richard Fhager),
    # https://lospec.com/palette-list/dawnbringer-32
    "db32": (
        "#000000", "#222034", "#45283c", "#663931", "#8f563b", "#df7126",
        "#d9a066", "#eec39a", "#fbf236", "#99e550", "#6abe30", "#37946e",
        "#4b692f", "#524b24", "#323c39", "#3f3f74", "#306082", "#5b6ee1",
        "#639bff", "#5fcde4", "#cbdbfc", "#ffffff", "#9badb7", "#847e87",
        "#696a6a", "#595652", "#76428a", "#ac3232", "#d95763", "#d77bba",
        "#8f974a", "#8a6f30",
    ),
    # Endesga 32 by ENDESGA, https://lospec.com/palette-list/endesga-32
    "endesga-32": (
        "#be4a2f", "#d77643", "#ead4aa", "#e4a672", "#b86f50", "#733e39",
        "#3e2731", "#a22633", "#e43b44", "#f77622", "#feae34", "#fee761",
        "#63c74d", "#3e8948", "#265c42", "#193c3e", "#124e89", "#0099db",
        "#2ce8f5", "#ffffff", "#c0cbdc", "#8b9bb4", "#5a6988", "#3a4466",
        "#262b44", "#181425", "#ff0044", "#68386c", "#b55088", "#f6757a",
        "#e8b796", "#c28569",
    ),
    # Nintendo Entertainment System (2C02 PPU, the 55 distinct entries of the
    # commonly used emulator table), https://lospec.com/palette-list/nintendo-entertainment-system
    "nes": (
        "#000000", "#fcfcfc", "#f8f8f8", "#bcbcbc", "#7c7c7c", "#a4e4fc",
        "#3cbcfc", "#0078f8", "#0000fc", "#b8b8f8", "#6888fc", "#0058f8",
        "#0000bc", "#d8b8f8", "#9878f8", "#6844fc", "#4428bc", "#f8b8f8",
        "#f878f8", "#d800cc", "#940084", "#f8a4c0", "#f85898", "#e40058",
        "#a80020", "#f0d0b0", "#f87858", "#f83800", "#a81000", "#fce0a8",
        "#fca044", "#e45c10", "#881400", "#f8d878", "#f8b800", "#ac7c00",
        "#503000", "#d8f878", "#b8f818", "#00b800", "#007800", "#b8f8b8",
        "#58d854", "#00a800", "#006800", "#b8f8d8", "#58f898", "#00a844",
        "#005800", "#00fcfc", "#00e8d8", "#008888", "#004058", "#f8d8f8",
        "#787878",
    ),
    # Nintendo Game Boy (DMG) greens as shipped by the bgb emulator,
    # https://lospec.com/palette-list/nintendo-gameboy-bgb (dark to light)
    "gameboy": ("#081820", "#346856", "#88c070", "#e0f8d0"),
}

AUTO_DEFAULT_COUNT = 32
MIN_COLORS = 2
MAX_COLORS = 256

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class PaletteSpec:
    """A parsed ``palette`` field.

    - ``named``: ``colors`` is the named palette, ``count`` its length.
    - ``auto``: ``colors`` is empty, ``count`` is N (median cut to N colours).
    - ``none``: no palette mapping; ``colors`` empty, ``count`` None.
    - ``custom``: ``colors`` is the user's list, ``count`` its length.
    """

    kind: Literal["named", "auto", "none", "custom"]
    colors: tuple[str, ...]
    count: int | None


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """``#rrggbb`` -> (r, g, b). Raises ValueError on anything else."""
    v = value.strip()
    if not _HEX_RE.match(v):
        raise ValueError(f"invalid colour {value!r}: expected #rrggbb")
    return int(v[1:3], 16), int(v[3:5], 16), int(v[5:7], 16)


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(c) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def palette_to_array(colors) -> np.ndarray:
    """A sequence of ``#rrggbb`` strings -> (K, 3) uint8."""
    return np.array([hex_to_rgb(c) for c in colors], dtype=np.uint8).reshape(-1, 3)


def parse_palette(spec: str) -> PaletteSpec:
    """Parse the ``palette`` grammar: a named palette, ``auto`` / ``auto:N``
    (2 <= N <= 256), ``none``, or a comma-separated ``#rrggbb`` list.

    Raises ValueError with a user-facing message on a bad spec.
    """
    if not isinstance(spec, str):
        raise ValueError("palette must be a string")
    s = spec.strip()
    low = s.lower()
    if not s:
        raise ValueError("palette must not be empty")
    if low in NAMED_PALETTES:
        colors = NAMED_PALETTES[low]
        return PaletteSpec(kind="named", colors=colors, count=len(colors))
    if low == "none":
        return PaletteSpec(kind="none", colors=(), count=None)
    if low == "auto":
        return PaletteSpec(kind="auto", colors=(), count=AUTO_DEFAULT_COUNT)
    if low.startswith("auto:"):
        raw = low[len("auto:"):].strip()
        if not raw.isdigit():
            raise ValueError(f"invalid palette {spec!r}: auto:N needs an integer N")
        n = int(raw)
        if not MIN_COLORS <= n <= MAX_COLORS:
            raise ValueError(
                f"invalid palette {spec!r}: N must be between {MIN_COLORS} and {MAX_COLORS}"
            )
        return PaletteSpec(kind="auto", colors=(), count=n)
    if s.startswith("#"):
        parts = [p.strip() for p in s.split(",")]
        out: list[str] = []
        for p in parts:
            if not _HEX_RE.match(p):
                raise ValueError(f"invalid palette colour {p!r}: expected #rrggbb")
            c = p.lower()
            if c not in out:
                out.append(c)
        if len(out) > MAX_COLORS:
            raise ValueError(f"a custom palette may list at most {MAX_COLORS} colours")
        return PaletteSpec(kind="custom", colors=tuple(out), count=len(out))
    names = ", ".join(NAMED_PALETTES)
    raise ValueError(
        f"unknown palette {spec!r}: use one of {names}, auto:N, none, "
        "or a comma-separated list of #rrggbb colours"
    )


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0..255, any numeric dtype, shape (..., 3)) -> OKLab float64 (..., 3).

    Björn Ottosson's OKLab (https://bottosson.github.io/posts/oklab/).
    """
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l_ = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack(
        [
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        ],
        axis=-1,
    )


def pack_rgb(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 -> (...) uint32 key ``r<<16 | g<<8 | b``."""
    a = np.asarray(rgb, dtype=np.uint32)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


def unpack_rgb(keys: np.ndarray) -> np.ndarray:
    k = np.asarray(keys, dtype=np.uint32)
    return np.stack([(k >> 16) & 255, (k >> 8) & 255, k & 255], axis=-1).astype(np.uint8)


_CHUNK_ELEMS = 1 << 20  # pixels x palette colours per distance block


def nearest_palette_index(rgb: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    """Index of the OKLab-nearest palette colour for every pixel.

    ``rgb`` is (..., 3) sRGB 0..255; ``palette_rgb`` is (K, 3). Returns an
    int array of shape ``rgb.shape[:-1]``. A pixel that is exactly a palette
    colour always maps to that colour (distance 0), so exact inputs never
    drift. Ties go to the lower palette index.
    """
    pal = np.asarray(palette_rgb).reshape(-1, 3)
    if len(pal) == 0:
        raise ValueError("palette is empty")
    flat = np.asarray(rgb).reshape(-1, 3)
    shape = np.asarray(rgb).shape[:-1]
    if flat.size == 0:
        return np.zeros(shape, dtype=np.intp)
    keys = pack_rgb(np.clip(np.rint(flat), 0, 255).astype(np.uint8))
    uniq, inverse = np.unique(keys, return_inverse=True)
    lab_u = srgb_to_oklab(unpack_rgb(uniq))
    lab_p = srgb_to_oklab(pal)
    best = np.empty(len(uniq), dtype=np.intp)
    chunk = max(256, _CHUNK_ELEMS // len(lab_p))
    for i in range(0, len(uniq), chunk):
        part = lab_u[i:i + chunk]
        d = ((part[:, None, :] - lab_p[None, :, :]) ** 2).sum(-1)
        best[i:i + chunk] = np.argmin(d, axis=1)
    return best[inverse.reshape(-1)].reshape(shape)


def quantize_labels(pixels: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Median-cut quantize a (N, 3) uint8 pixel list with Pillow.

    Returns ``(labels, palette)``: an (N,) label per pixel into ``palette``
    ((K, 3) uint8, K <= n). When the pixels hold at most ``n`` distinct
    colours they are returned exactly (no averaging).
    """
    px = np.asarray(pixels, dtype=np.uint8).reshape(-1, 3)
    if len(px) == 0:
        return np.zeros(0, dtype=np.intp), np.zeros((0, 3), dtype=np.uint8)
    keys = pack_rgb(px)
    uniq, inverse = np.unique(keys, return_inverse=True)
    if len(uniq) <= n:
        return inverse.reshape(-1).astype(np.intp), unpack_rgb(uniq)
    # Lay the pixel list out as a rectangle for Pillow; the padding repeats
    # pixels from the start (at most width-1 extra samples).
    width = min(len(px), 4096)
    rows = -(-len(px) // width)
    rect = np.resize(px, (rows * width, 3)).reshape(rows, width, 3)
    q = Image.fromarray(np.ascontiguousarray(rect)).quantize(
        colors=n, method=Image.Quantize.MEDIANCUT
    )
    idx = np.asarray(q).reshape(-1)[: len(px)].astype(np.intp)
    raw = np.array(q.getpalette() or [], dtype=np.uint8).reshape(-1, 3)
    used = np.unique(idx)
    remap = np.zeros(max(int(used.max()) + 1, 1), dtype=np.intp)
    remap[used] = np.arange(len(used))
    return remap[idx], raw[used]


def auto_palette(rgb: np.ndarray, n: int, mask: np.ndarray | None) -> np.ndarray:
    """An automatic palette of at most ``n`` colours, by Pillow median cut over
    the pixels where ``mask`` is True (all pixels when ``mask`` is None).

    Returns (K, 3) uint8 with K <= n (K = 0 when no pixel is selected).
    """
    arr = np.asarray(rgb)
    px = arr.reshape(-1, 3) if mask is None else arr[np.asarray(mask, dtype=bool)]
    _, pal = quantize_labels(px, int(n))
    return pal
