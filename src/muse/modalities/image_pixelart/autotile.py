"""Autotiles: RPG Maker MZ/MV A2 ground autotile blocks and the 47-tile blob set.

Pillow + numpy only. Deterministic code, no model: diffusion cannot produce
edge-consistent autotile sheets, so the edges are drawn from analytic masks.

**A2 layout.** An A2 block is 2T x 3T pixels, read as a 4 x 6 grid of
half-tile quadrants ``(qx, qy)``:

- ``qy`` 0-1, ``qx`` 0-1: the preview tile (an isolated patch; RPG Maker shows
  it in the palette);
- ``qy`` 0-1, ``qx`` 2-3: the inner-corner tile (terrain with a notch in each
  corner);
- ``qy`` 2-5: a 2T x 2T patch of terrain whose quadrants are the outer
  corners, the straight edges and the interior.

Each output tile is four quadrants (TL, TR, BL, BR) picked by
``FLOOR_AUTOTILE_TABLE`` (copied from RPG Maker MV). A quadrant's kind
depends only on its two sides and the corner between them, so each
(position, kind) pair has exactly one source quadrant in the block (see
``QUADRANT_SOURCES``).

**Seams.** ``build_a2`` makes every quadrant from a depth field in tile
coordinates: the distance into the terrain from the boundary of an inset
(``edge_width``) rounded rectangle with inner-corner notches. The boundary is
displaced by four seeded noise curves (top, bottom, left, right), each
T-periodic with equal first and last samples; the fill and base textures are
made seamless the same way (``pixelize(seamless=True)``: first row/column ==
last). Since every table entry takes a quadrant whose block position has the
same parity as its destination, the textures and curves line up in every
composed tile, so two tiles that may legally touch have identical touching
columns/rows. Rounding is limited (``corner_radius`` ceiling) so it never
reaches a quadrant border that a neighbour shares.

Never pass ``mode=`` to ``Image.fromarray`` here: Pillow 13 removed it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

from muse.modalities.image_pixelart.palettes import PaletteSpec, hex_to_rgb, parse_palette
from muse.modalities.image_pixelart.pixelize import PixelizeOptions, pixelize

# --------------------------------------------------------------------- the table

# Copied verbatim from RPG Maker MV corescript, ``Tilemap.FLOOR_AUTOTILE_TABLE``
# in js/rpg_core/Tilemap.js (concatenated into rpg_core.js), rpgtkoolmv/corescript
# commit 182e31449707ba7e406db0485c44c2a9d11e2dcd. MIT License,
# Copyright (c) 2015 KADOKAWA CORPORATION./YOJI OJIMA. (The spec cites the repo
# as rpgtkool/corescript; it lives at github.com/rpgtkoolmv/corescript.)
#
# Entry = autotile shape number 0..47; each entry lists the source quadrant
# (qx, qy), in half-tile units within the A2 block, for the destination
# quadrants TL, TR, BL, BR in that order.
FLOOR_AUTOTILE_TABLE: tuple[tuple[tuple[int, int], ...], ...] = tuple(
    tuple(tuple(q) for q in entry) for entry in [
        [[2, 4], [1, 4], [2, 3], [1, 3]], [[2, 0], [1, 4], [2, 3], [1, 3]],
        [[2, 4], [3, 0], [2, 3], [1, 3]], [[2, 0], [3, 0], [2, 3], [1, 3]],
        [[2, 4], [1, 4], [2, 3], [3, 1]], [[2, 0], [1, 4], [2, 3], [3, 1]],
        [[2, 4], [3, 0], [2, 3], [3, 1]], [[2, 0], [3, 0], [2, 3], [3, 1]],
        [[2, 4], [1, 4], [2, 1], [1, 3]], [[2, 0], [1, 4], [2, 1], [1, 3]],
        [[2, 4], [3, 0], [2, 1], [1, 3]], [[2, 0], [3, 0], [2, 1], [1, 3]],
        [[2, 4], [1, 4], [2, 1], [3, 1]], [[2, 0], [1, 4], [2, 1], [3, 1]],
        [[2, 4], [3, 0], [2, 1], [3, 1]], [[2, 0], [3, 0], [2, 1], [3, 1]],
        [[0, 4], [1, 4], [0, 3], [1, 3]], [[0, 4], [3, 0], [0, 3], [1, 3]],
        [[0, 4], [1, 4], [0, 3], [3, 1]], [[0, 4], [3, 0], [0, 3], [3, 1]],
        [[2, 2], [1, 2], [2, 3], [1, 3]], [[2, 2], [1, 2], [2, 3], [3, 1]],
        [[2, 2], [1, 2], [2, 1], [1, 3]], [[2, 2], [1, 2], [2, 1], [3, 1]],
        [[2, 4], [3, 4], [2, 3], [3, 3]], [[2, 4], [3, 4], [2, 1], [3, 3]],
        [[2, 0], [3, 4], [2, 3], [3, 3]], [[2, 0], [3, 4], [2, 1], [3, 3]],
        [[2, 4], [1, 4], [2, 5], [1, 5]], [[2, 0], [1, 4], [2, 5], [1, 5]],
        [[2, 4], [3, 0], [2, 5], [1, 5]], [[2, 0], [3, 0], [2, 5], [1, 5]],
        [[0, 4], [3, 4], [0, 3], [3, 3]], [[2, 2], [1, 2], [2, 5], [1, 5]],
        [[0, 2], [1, 2], [0, 3], [1, 3]], [[0, 2], [1, 2], [0, 3], [3, 1]],
        [[2, 2], [3, 2], [2, 3], [3, 3]], [[2, 2], [3, 2], [2, 1], [3, 3]],
        [[2, 4], [3, 4], [2, 5], [3, 5]], [[2, 0], [3, 4], [2, 5], [3, 5]],
        [[0, 4], [1, 4], [0, 5], [1, 5]], [[0, 4], [3, 0], [0, 5], [1, 5]],
        [[0, 2], [3, 2], [0, 3], [3, 3]], [[0, 2], [1, 2], [0, 5], [1, 5]],
        [[0, 4], [3, 4], [0, 5], [3, 5]], [[2, 2], [3, 2], [2, 5], [3, 5]],
        [[0, 2], [3, 2], [0, 5], [3, 5]], [[0, 0], [1, 0], [0, 1], [1, 1]],
    ]
)

# The duplicate: shapes 46 and 47 are both "isolated" (no neighbours). Shape 46
# is composed from the block's four outer corners, exactly like every other
# shape's edges; shape 47 is the hand-drawn top-left preview tile, which RPG
# Maker only shows in the editor palette. We keep 46 (so the isolated blob tile
# is built from the same quadrants as its neighbours' edges, and seams hold for
# an imported A2 whose preview tile differs) and drop 47. build_a2 draws the
# preview tile as the composition of shape 46, so for a generated block the two
# are identical anyway.
DROPPED_SHAPE = 47

# Blob sheet: 47 tiles in RPG Maker shape order (tile index == shape number),
# 8 columns x 6 rows; the 48th cell (column 7, row 5) is left empty.
SHEET_COLUMNS = 8
SHEET_ROWS = 6

# 8-neighbour bitmask, clockwise from north. A corner bit is only set when both
# adjacent sides are set (the blob reduction: 256 raw masks -> 47).
N, NE, E, SE, S, SW, W, NW = 1, 2, 4, 8, 16, 32, 64, 128
_CORNERS = ((NE, N, E), (SE, S, E), (SW, S, W), (NW, N, W))

# Destination quadrant positions: 0 TL, 1 TR, 2 BL, 3 BR, each with its
# vertical-side bit (N/S), horizontal-side bit (W/E) and the corner between.
_POS_BITS = ((N, W, NW), (N, E, NE), (S, W, SW), (S, E, SE))

# Quadrant kinds: "fill" (both sides and the corner connected), "inner" (both
# sides, not the corner: an inner-corner notch), "edge_v" (the N/S side open:
# a top/bottom edge), "edge_h" (the W/E side open: a left/right edge) and
# "outer" (both sides open: a rounded outer corner).
QUADRANT_SOURCES: dict[int, dict[str, tuple[int, int]]] = {
    0: {"fill": (2, 4), "inner": (2, 0), "edge_v": (2, 2), "edge_h": (0, 4), "outer": (0, 2)},
    1: {"fill": (1, 4), "inner": (3, 0), "edge_v": (1, 2), "edge_h": (3, 4), "outer": (3, 2)},
    2: {"fill": (2, 3), "inner": (2, 1), "edge_v": (2, 5), "edge_h": (0, 3), "outer": (0, 5)},
    3: {"fill": (1, 3), "inner": (3, 1), "edge_v": (1, 5), "edge_h": (3, 3), "outer": (3, 5)},
}

MIN_TILE_SIZE = 16
MAX_TILE_SIZE = 256
_BIG = 1.0e6  # depth of a quadrant with no boundary
_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_HIGHLIGHT_MIX = 0.35  # highlight = fill moved this far toward white
_DARK_SCALE = 0.4  # "dark" outline = mean fill colour times this


@dataclass(frozen=True)
class EdgeStyle:
    """How ``build_a2`` draws the terrain edge.

    - ``edge_width``: inset (px) of the terrain boundary from an open tile
      side; outside it the base (or transparency) shows. None -> T/8.
    - ``outline``: ``none``, ``dark`` (the mean fill colour darkened) or
      ``#rrggbb``: a 1 px line on the boundary's inner side.
    - ``highlight``: one lighter pixel line just inside the outline.
    - ``roughness``: 0..1; the boundary wanders by up to
      ``roughness * edge_width / 2`` px (rounded), seeded by ``seed``.
    - ``corner_radius``: outer-corner rounding (px). None -> T/4, lowered if
      needed to fit; an explicit value above the ceiling is an error.
    """

    edge_width: int | None = None
    outline: str = "dark"
    highlight: bool = True
    roughness: float = 0.3
    corner_radius: int | None = None
    seed: int = 0


@dataclass(frozen=True)
class BlobTile:
    """One blob tile: ``index`` (== RPG Maker shape number), ``x``/``y`` its
    column/row in the sheet (tile units), ``mask`` its reduced 8-neighbour
    bitmask (N=1, NE=2, E=4, SE=8, S=16, SW=32, W=64, NW=128)."""

    index: int
    x: int
    y: int
    mask: int


# ------------------------------------------------------------------ mask logic


def reduce_mask(mask: int) -> int:
    """Clear each corner bit whose two adjacent side bits are not both set."""
    for corner, a, b in _CORNERS:
        if not (mask & a and mask & b):
            mask &= ~corner
    return mask


def _quadrant_kind(mask: int, pos: int) -> str:
    v, h, c = _POS_BITS[pos]
    if mask & v and mask & h:
        return "fill" if mask & c else "inner"
    if mask & h:
        return "edge_v"
    if mask & v:
        return "edge_h"
    return "outer"


@lru_cache(maxsize=1)
def _shape_masks() -> tuple[int, ...]:
    """Reduced mask of each RPG Maker shape 0..46 (shape 47 dropped)."""
    by_entry = {entry: shape for shape, entry in enumerate(FLOOR_AUTOTILE_TABLE)
                if shape != DROPPED_SHAPE}
    out: dict[int, int] = {}
    for mask in sorted({reduce_mask(m) for m in range(256)}):
        entry = tuple(QUADRANT_SOURCES[p][_quadrant_kind(mask, p)] for p in range(4))
        shape = by_entry.get(entry)
        if shape is None or shape in out:  # the table and the semantics disagree
            raise RuntimeError(f"no unique autotile shape for mask {mask}")
        out[shape] = mask
    return tuple(out[s] for s in range(len(out)))


def blob_masks() -> tuple[int, ...]:
    """The 47 reduced masks, in sheet order (index == RPG Maker shape)."""
    return _shape_masks()


# ------------------------------------------------------------------ validation


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _check_tile_size(tile_size) -> int:
    if not _is_int(tile_size) or tile_size % 2 or not MIN_TILE_SIZE <= tile_size <= MAX_TILE_SIZE:
        raise ValueError(
            f"tile_size must be an even integer in {MIN_TILE_SIZE}..{MAX_TILE_SIZE} "
            "(a quadrant is half a tile)"
        )
    return int(tile_size)


def validate_a2(img: Image.Image, tile_size: int) -> None:
    """Raise ValueError unless ``img`` is exactly 2T x 3T."""
    t = _check_tile_size(tile_size)
    w, h = img.size
    if (w, h) != (2 * t, 3 * t):
        raise ValueError(
            f"an A2 autotile block for tile_size {t} must be exactly "
            f"{2 * t}x{3 * t} pixels (got {w}x{h})"
        )


@dataclass(frozen=True)
class _Geometry:
    t: int
    margin: int
    radius: float
    inner_radius: float
    outline: tuple[int, int, int] | str | None  # rgb, "dark" or None
    highlight: bool
    top: np.ndarray  # boundary displacement per X (top edge)
    bottom: np.ndarray  # per X
    left: np.ndarray  # per Y
    right: np.ndarray  # per Y


def _noise_curve(rng: np.random.Generator, t: int, amp: float) -> np.ndarray:
    """Integer displacement per pixel, T-periodic with curve[0] == curve[T-1]
    (so it lines up across a tile border exactly like the seamless fill)."""
    period = t - 1
    x = np.arange(period, dtype=np.float64)
    phases = rng.uniform(0.0, 2.0 * math.pi, 3)
    weights = rng.uniform(0.5, 1.0, 3) / np.array([1.0, 1.5, 2.0])
    s = sum(w * np.sin(2.0 * math.pi * (k + 1) * x / period + p)
            for k, (w, p) in enumerate(zip(weights, phases)))
    peak = float(np.abs(s).max()) or 1.0
    curve = np.rint(amp * s / peak).astype(np.int64)
    return np.concatenate([curve, curve[:1]])


def _geometry(style: EdgeStyle, t: int) -> _Geometry:
    if not isinstance(style, EdgeStyle):
        raise ValueError("style must be an EdgeStyle")
    half = t // 2
    if style.edge_width is None:
        margin = max(1, t // 8)
    else:
        if not _is_int(style.edge_width) or not 1 <= style.edge_width <= t // 4:
            raise ValueError(f"edge_width must be an integer in 1..{t // 4} for tile_size {t}")
        margin = int(style.edge_width)
    ol = style.outline
    if not isinstance(ol, str) or (ol not in ("none", "dark") and not _HEX_RE.match(ol)):
        raise ValueError("outline must be none, dark or #rrggbb")
    if not isinstance(style.highlight, bool):
        raise ValueError("highlight must be true or false")
    rough = style.roughness
    if (not isinstance(rough, (int, float, np.integer, np.floating)) or isinstance(rough, bool)
            or not math.isfinite(rough) or not 0.0 <= rough <= 1.0):
        raise ValueError("roughness must be a number in 0..1")
    if not _is_int(style.seed) or style.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    amp = float(rough) * margin / 2.0
    wander = math.ceil(amp)
    # Every quadrant border a neighbour shares must stay clear of the rounding
    # and of the outline/highlight rings: depth there >= the radius and >= 2.
    room = half - 1 - margin - wander
    if room < 2:
        raise ValueError(
            f"edge_width {margin} with roughness {rough} leaves no room in a {t} px tile"
        )
    if style.corner_radius is None:
        radius = min(t // 4, room)
    else:
        if not _is_int(style.corner_radius) or not 0 <= style.corner_radius <= room:
            raise ValueError(
                f"corner_radius must be an integer in 0..{room} for tile_size {t} "
                f"with edge_width {margin} and roughness {rough}"
            )
        radius = int(style.corner_radius)
    # Inner notches round only where both notch sides are well inside the
    # tile, so the notch still meets the neighbours' straight edges exactly.
    inner_radius = max(0, min(radius, margin - wander)) // 2

    rng = np.random.default_rng(int(style.seed))
    curves = [_noise_curve(rng, t, amp) for _ in range(4)]
    if amp == 0.0:
        curves = [np.zeros(t, dtype=np.int64) for _ in range(4)]
    outline: tuple[int, int, int] | str | None
    outline = None if ol == "none" else ("dark" if ol == "dark" else hex_to_rgb(ol))
    return _Geometry(t=t, margin=margin, radius=float(radius), inner_radius=float(inner_radius),
                     outline=outline, highlight=style.highlight, top=curves[0],
                     bottom=curves[1], left=curves[2], right=curves[3])


# --------------------------------------------------------------------- textures


def _hard_alpha(rgba: np.ndarray, threshold: int = 128) -> np.ndarray:
    out = rgba.copy()
    op = out[..., 3] >= threshold
    out[..., 3] = np.where(op, 255, 0).astype(np.uint8)
    out[~op] = 0
    return out


def _prepare_texture(img: Image.Image, t: int, what: str) -> np.ndarray:
    """Fit a texture to exactly T x T and make it seamless.

    The largest centred square that is a whole number of tiles is cropped and
    box-downscaled to T (a texture smaller than T is wrapped, never stretched,
    so its pixels stay whole), then ``pixelize`` with ``grid=1`` (no grid
    detection, nothing shrinks) and ``seamless=True`` makes its first and last
    rows and columns equal.
    """
    if not isinstance(img, Image.Image):
        raise ValueError(f"{what} must be an image")
    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    h, w = rgba.shape[:2]
    if w < 1 or h < 1:
        raise ValueError(f"{what} texture is empty")
    side = min(w, h)
    if side >= t:
        k = side // t
        crop = k * t
        y0, x0 = (h - crop) // 2, (w - crop) // 2
        rgba = rgba[y0:y0 + crop, x0:x0 + crop]
        if k > 1:
            rgba = np.array(Image.fromarray(np.ascontiguousarray(rgba)).resize(
                (t, t), Image.Resampling.BOX), dtype=np.uint8)
    else:
        rgba = rgba[np.ix_(np.arange(t) % h, np.arange(t) % w)]
    rgba = _hard_alpha(rgba)
    res = pixelize(Image.fromarray(np.ascontiguousarray(rgba)),
                   PixelizeOptions(grid=1, target_size=None, palette=parse_palette("none"),
                                   background="none", seamless=True))
    return np.array(res.image.convert("RGBA"), dtype=np.uint8)


def _flatten_base(base: np.ndarray) -> np.ndarray:
    """A ground base is opaque: transparent pixels take the mean base colour."""
    op = base[..., 3] == 255
    if not op.any():
        raise ValueError("base texture is fully transparent")
    out = base.copy()
    out[~op, :3] = np.rint(base[op, :3].astype(np.float64).mean(axis=0)).astype(np.uint8)
    out[..., 3] = 255
    return out


# ---------------------------------------------------------------------- drawing


def _round_and(a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    """Depth into the intersection of two half-planes (depths a, b), with the
    corner rounded to radius r (exact for straight sides)."""
    m = np.minimum(a, b)
    if r <= 0:
        return m
    near = (a < r) & (b < r)
    rounded = r - np.hypot(r - a, r - b)
    return np.where(near, rounded, m)


def _quadrant(pos: int, kind: str, g: _Geometry, fill: np.ndarray, base: np.ndarray | None,
              light: np.ndarray, dark: tuple[int, int, int]) -> np.ndarray:
    t, half, m = g.t, g.t // 2, g.margin
    xs = np.arange(half) + (pos % 2) * half
    ys = np.arange(half) + (pos // 2) * half
    X, Y = np.meshgrid(xs, ys)
    if pos < 2:
        dv = (Y - (m + g.top[X])).astype(np.float64)
    else:
        dv = ((t - 1 - Y) - (m + g.bottom[X])).astype(np.float64)
    if pos % 2 == 0:
        dh = (X - (m + g.left[Y])).astype(np.float64)
    else:
        dh = ((t - 1 - X) - (m + g.right[Y])).astype(np.float64)
    if kind == "fill":
        depth = np.full(X.shape, _BIG)
    elif kind == "edge_v":
        depth = dv
    elif kind == "edge_h":
        depth = dh
    elif kind == "outer":
        depth = _round_and(dv, dh, g.radius)
    else:  # inner: the notch is the (rounded) intersection of the outsides
        depth = -_round_and(-dv, -dh, g.inner_radius)

    out = np.zeros((half, half, 4), dtype=np.uint8)
    inside = depth >= 0
    under = base[Y, X] if base is not None else np.zeros((half, half, 4), dtype=np.uint8)
    out[~inside] = under[~inside]
    f = fill[Y, X]
    fill_op = f[..., 3] == 255
    out[inside & fill_op] = f[inside & fill_op]
    out[inside & ~fill_op] = under[inside & ~fill_op]
    ring = 0
    if g.outline is not None:
        sel = inside & (depth < 1)
        out[sel, :3] = dark if g.outline == "dark" else g.outline
        out[sel, 3] = 255
        ring = 1
    if g.highlight:
        sel = inside & fill_op & (depth >= ring) & (depth < ring + 1)
        out[sel] = light[Y, X][sel]
    return out


def _compose(a2: np.ndarray, entry, t: int) -> np.ndarray:
    half = t // 2
    tile = np.zeros((t, t, 4), dtype=np.uint8)
    for i, (qx, qy) in enumerate(entry):
        dx, dy = (i % 2) * half, (i // 2) * half
        tile[dy:dy + half, dx:dx + half] = a2[qy * half:(qy + 1) * half,
                                              qx * half:(qx + 1) * half]
    return tile


def build_a2(fill: Image.Image, base: Image.Image | None, tile_size: int, style: EdgeStyle,
             palette: PaletteSpec | None) -> Image.Image:
    """Draw an RPG Maker A2 autotile block (RGBA, exactly 2T x 3T).

    ``base`` None -> outside the terrain is transparent (a "decoration"
    autotile); a base -> opaque everywhere (a "ground" autotile). With a
    ``palette`` (other than ``none``) the block is palette-mapped by
    ``pixelize`` with ``grid=1`` and ``background="none"``. Alpha is only ever
    0 or 255. Raises ValueError for bad input.
    """
    t = _check_tile_size(tile_size)
    g = _geometry(style, t)
    if palette is not None and not isinstance(palette, PaletteSpec):
        raise ValueError("palette must be a PaletteSpec (see parse_palette) or None")
    f = _prepare_texture(fill, t, "fill")
    op = f[..., 3] == 255
    if not op.any():
        raise ValueError("fill texture is fully transparent")
    b = _flatten_base(_prepare_texture(base, t, "base")) if base is not None else None
    light = f.copy()
    light[..., :3] = np.rint(f[..., :3] + (255.0 - f[..., :3]) * _HIGHLIGHT_MIX).astype(np.uint8)
    mean = f[op, :3].astype(np.float64).mean(axis=0)
    dark = tuple(int(v) for v in np.rint(mean * _DARK_SCALE))

    half = t // 2
    a2 = np.zeros((3 * t, 2 * t, 4), dtype=np.uint8)
    for pos, kinds in QUADRANT_SOURCES.items():
        for kind, (qx, qy) in kinds.items():
            a2[qy * half:(qy + 1) * half, qx * half:(qx + 1) * half] = _quadrant(
                pos, kind, g, f, b, light, dark)
    # The preview tile is the isolated shape, composed like every other shape.
    a2[:t, :t] = _compose(a2, FLOOR_AUTOTILE_TABLE[46], t)

    img = Image.fromarray(a2)
    if palette is not None and palette.kind != "none":
        img = pixelize(img, PixelizeOptions(grid=1, target_size=None, palette=palette,
                                            background="none")).image
    return img.convert("RGBA")


# ------------------------------------------------------------------ blob-47


def expand_blob47(a2: Image.Image, tile_size: int) -> tuple[Image.Image, list[BlobTile]]:
    """Compose the 47 blob tiles from an A2 block (generated or imported).

    Returns the 8 x 6 tile sheet (RGBA; the last cell is empty) and the
    layout. Tile ``index`` is the RPG Maker shape number (0..46). An imported
    block's colours are used verbatim; its alpha is hardened at 128 so the
    sheet's alpha is only 0 or 255.
    """
    validate_a2(a2, tile_size)
    t = int(tile_size)
    src = _hard_alpha(np.array(a2.convert("RGBA"), dtype=np.uint8))
    sheet = np.zeros((SHEET_ROWS * t, SHEET_COLUMNS * t, 4), dtype=np.uint8)
    tiles: list[BlobTile] = []
    for shape, mask in enumerate(_shape_masks()):
        x, y = shape % SHEET_COLUMNS, shape // SHEET_COLUMNS
        sheet[y * t:(y + 1) * t, x * t:(x + 1) * t] = _compose(
            src, FLOOR_AUTOTILE_TABLE[shape], t)
        tiles.append(BlobTile(index=shape, x=x, y=y, mask=mask))
    return Image.fromarray(sheet), tiles
