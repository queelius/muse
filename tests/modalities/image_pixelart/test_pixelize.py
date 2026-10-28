"""Pixelize: grid snap, palette mapping, alpha, cleanup, outline, seamless, sheets.

All fixtures are synthetic and built here: random-colour sprites with a
transparent border, upscaled with nearest-neighbour and then softened (blur +
noise) to look like AI "fake pixels".
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from muse.modalities.image_pixelart.palettes import (
    NAMED_PALETTES,
    parse_palette,
    srgb_to_oklab,
)
from muse.modalities.image_pixelart.pixelize import (
    GridInfo,
    PixelizeOptions,
    PixelizeResult,
    detect_grid,
    pixelize,
)

PKG = Path(__file__).resolve().parents[3] / "src" / "muse" / "modalities" / "image_pixelart"


# --------------------------------------------------------------------- helpers


def _rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def make_sprite(w, h, colors, border=2, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    pal = np.array([_rgb(c) for c in colors], dtype=np.uint8)
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    idx = rng.integers(0, len(pal), size=(h, w))
    arr[..., :3] = pal[idx]
    arr[..., 3] = 255
    arr[:border] = 0
    arr[-border:] = 0
    arr[:, :border] = 0
    arr[:, -border:] = 0
    return Image.fromarray(arr)


def blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def jitter(img: Image.Image, noise: int, seed=1) -> Image.Image:
    rng = np.random.default_rng(seed)
    a = np.asarray(img.convert("RGBA")).astype(np.int16)
    a[..., :3] += rng.integers(-noise, noise + 1, size=a[..., :3].shape).astype(np.int16)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def norm(img: Image.Image) -> np.ndarray:
    """RGBA array with every transparent pixel normalised to (0,0,0,0)."""
    a = np.asarray(img.convert("RGBA")).copy()
    a[a[..., 3] == 0] = 0
    return a


def agreement(a: Image.Image, b: Image.Image) -> float:
    x, y = norm(a), norm(b)
    assert x.shape == y.shape, (x.shape, y.shape)
    return float(np.mean(np.all(x == y, axis=-1)))


def colours(img: Image.Image) -> set[str]:
    a = np.asarray(img.convert("RGBA"))
    op = a[a[..., 3] == 255][:, :3]
    return {"#%02x%02x%02x" % tuple(int(v) for v in c) for c in np.unique(op, axis=0)}


def opts(**kw) -> PixelizeOptions:
    base = dict(grid=None, target_size=None, palette=parse_palette("auto:32"),
                background="auto")
    base.update(kw)
    return PixelizeOptions(**base)


def orphans(a: np.ndarray, wrap: bool) -> np.ndarray:
    """Boolean map of opaque pixels whose 8 neighbours all share one other colour."""
    key = (a[..., 0].astype(np.uint32) << 24 | a[..., 1].astype(np.uint32) << 16
           | a[..., 2].astype(np.uint32) << 8 | a[..., 3].astype(np.uint32))
    key = np.where(a[..., 3] == 0, 0, key)
    if wrap:
        nb = [np.roll(np.roll(key, dy, 0), dx, 1)
              for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
        same = np.all([n == nb[0] for n in nb], axis=0)
        return (a[..., 3] == 255) & same & (nb[0] != key)
    out = np.zeros(key.shape, dtype=bool)
    h, w = key.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            win = key[y - 1:y + 2, x - 1:x + 2].ravel()
            nbs = np.delete(win, 4)
            if a[y, x, 3] == 255 and np.all(nbs == nbs[0]) and nbs[0] != key[y, x]:
                out[y, x] = True
    return out


DB12 = NAMED_PALETTES["db32"][:12]
SMALL = make_sprite(32, 32, colors=DB12)


# ------------------------------------------------------------------- grid snap


@pytest.mark.parametrize("k", [4, 6, 8])
def test_recovers_upscaled_sprite(k):
    small = SMALL
    big = small.resize((32 * k, 32 * k), Image.NEAREST)
    big = jitter(blur(big, 0.6), noise=6)
    r = pixelize(big, opts(palette=parse_palette("db32")))
    assert isinstance(r, PixelizeResult)
    assert r.grid.detected and r.grid.cell == k
    assert r.size == (32, 32)
    assert r.image.mode == "RGBA" and r.image.size == (32, 32)
    assert agreement(r.image, small) >= 0.98


def test_detect_grid_reports_offset_and_confidence():
    big = SMALL.resize((128, 128), Image.NEAREST).crop((3, 1, 126, 127))
    g = detect_grid(big)
    assert isinstance(g, GridInfo)
    assert g.detected and g.cell == 4
    # Original cell boundaries at multiples of 4 sit at x = 1, y = 3 in the crop.
    assert (g.offset_x, g.offset_y) == (1, 3)
    assert 0.5 <= g.confidence <= 1.0


def _kept_cells(start: int, end: int, k: int) -> tuple[int, int]:
    """Original art cells kept from source span [start, end): partial edge
    cells are kept when they cover at least half a cell."""
    half = math.ceil(k / 2)
    first = start // k
    if (first + 1) * k - start < half:
        first += 1
    last = (end - 1) // k
    if end - last * k < half:
        last -= 1
    return first, last + 1


@pytest.mark.parametrize("box", [(3, 1, 126, 127), (1, 2, 127, 125), (2, 3, 125, 128),
                                 (0, 0, 125, 126)])
@pytest.mark.parametrize("soft", [False, True])
def test_offset_crop_still_recovers(box, soft):
    k = 4
    big = SMALL.resize((128, 128), Image.NEAREST)
    if soft:
        big = jitter(blur(big, 0.6), noise=6)
    big = big.crop(box)
    r = pixelize(big, opts(palette=parse_palette("db32")))
    assert r.grid.detected and r.grid.cell == 4
    x0, x1 = _kept_cells(box[0], box[2], k)
    y0, y1 = _kept_cells(box[1], box[3], k)
    expected = SMALL.crop((x0, y0, x1, y1))
    assert r.size == expected.size
    assert agreement(r.image, expected) >= 0.98


def test_true_1to1_art_is_unchanged():
    small = SMALL
    r = pixelize(small, opts(palette=parse_palette("none")))
    assert r.grid.detected is False and r.grid.cell == 1
    assert r.size == small.size
    assert np.array_equal(np.asarray(r.image), np.asarray(small.convert("RGBA")))


@pytest.mark.parametrize("seed", range(6))
def test_1to1_sprites_are_not_detected(seed):
    small = make_sprite(48, 40, colors=NAMED_PALETTES["pico-8"], seed=seed)
    assert detect_grid(small).detected is False


def test_1to1_periodic_tile_is_not_detected():
    """A 1:1 brick tile (1 px mortar every 8 px) is periodic but not upscaled:
    snapping it to an 8 px grid would erase the mortar."""
    rng = np.random.default_rng(0)
    br = np.kron(rng.integers(60, 250, (8, 8, 3)), np.ones((8, 8, 1))).astype(np.uint8)
    br[::8, :] = 20
    br[:, ::8] = 20
    img = Image.fromarray(br)
    assert detect_grid(img).detected is False
    r = pixelize(img, opts(palette=parse_palette("none"), cleanup=False))
    assert r.size == (64, 64)
    assert np.array_equal(np.asarray(r.image)[..., :3], br)


def test_smooth_photo_is_not_detected():
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:200, 0:240]
    base = np.stack([128 + 90 * np.sin(xx / 23.0), 128 + 90 * np.cos(yy / 31.0),
                     128 + 60 * np.sin((xx + yy) / 17.0)], axis=-1)
    arr = np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)
    assert detect_grid(Image.fromarray(arr)).detected is False


def test_explicit_grid_one_means_no_downscale():
    big = SMALL.resize((128, 128), Image.NEAREST)
    r = pixelize(big, opts(grid=1, palette=parse_palette("none"), background="none"))
    assert r.size == (128, 128) and r.grid.cell == 1
    assert np.array_equal(norm(r.image), norm(big))


def test_explicit_grid_and_target_size():
    big = SMALL.resize((128, 128), Image.NEAREST)
    r = pixelize(big, opts(grid=4, palette=parse_palette("db32")))
    assert r.size == (32, 32) and r.grid.cell == 4
    assert agreement(r.image, SMALL) >= 0.99
    r2 = pixelize(big, opts(target_size=(32, 32), palette=parse_palette("db32")))
    assert r2.size == (32, 32)
    assert agreement(r2.image, SMALL) >= 0.99


def test_invalid_options_raise_value_error():
    for bad in (dict(grid=0), dict(grid=2, target_size=(8, 8)), dict(alpha_threshold=300),
                dict(outline="purple"), dict(background="sometimes"), dict(preview_scale=0),
                dict(preview_scale=17), dict(target_size=(999, 999)), dict(sheet=(0, 2))):
        with pytest.raises(ValueError):
            pixelize(SMALL, opts(**bad))


# --------------------------------------------------------------------- palette


def test_exact_palette_input_keeps_colours():
    small_in_db32 = make_sprite(24, 24, colors=NAMED_PALETTES["db32"], seed=7)
    r = pixelize(small_in_db32, opts(palette=parse_palette("db32")))
    assert colours(r.image) == colours(small_in_db32)
    assert np.array_equal(norm(r.image), norm(small_in_db32))
    big = small_in_db32.resize((96, 96), Image.NEAREST)
    r4 = pixelize(big, opts(palette=parse_palette("db32")))
    assert r4.grid.cell == 4
    assert colours(r4.image) == colours(small_in_db32)


def test_exact_auto_palette_input_keeps_colours():
    small = make_sprite(24, 24, colors=NAMED_PALETTES["endesga-32"][:10], seed=5)
    big = small.resize((72, 72), Image.NEAREST)
    r = pixelize(big, opts(palette=parse_palette("auto:16")))
    assert colours(r.image) == colours(small)


def test_only_palette_colours_and_binary_alpha():
    big = jitter(blur(SMALL.resize((128, 128), Image.NEAREST), 1.0), noise=10)
    r = pixelize(big, opts(palette=parse_palette("pico-8")))
    a = np.asarray(r.image)
    assert set(np.unique(a[..., 3]).tolist()) <= {0, 255}
    assert colours(r.image) <= set(NAMED_PALETTES["pico-8"])
    assert set(r.palette) == set(NAMED_PALETTES["pico-8"])


@pytest.mark.parametrize("spec", ["auto:8", "none", "#ff0000,#00ff00,#0000ff"])
def test_binary_alpha_for_every_palette_kind(spec):
    big = jitter(blur(SMALL.resize((96, 96), Image.NEAREST), 1.2), noise=8)
    r = pixelize(big, opts(palette=parse_palette(spec)))
    a = np.asarray(r.image)
    assert set(np.unique(a[..., 3]).tolist()) <= {0, 255}
    if r.metadata["palette_truncated"]:  # `none` lists the 256 most-used colours
        assert spec == "none" and len(r.palette) == 256
        assert set(r.palette) <= colours(r.image)
    else:
        assert colours(r.image) <= set(r.palette)
    if spec == "auto:8":
        assert len(r.palette) <= 8


# ------------------------------------------------------------------ background


def _sprite_on(bg: tuple[int, int, int], size=64, seed=2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.zeros((size, size, 4), dtype=np.uint8)
    a[..., :3] = bg
    a[..., 3] = 255
    pal = np.array([_rgb(c) for c in DB12], dtype=np.uint8)
    q = size // 4
    a[q:3 * q, q:3 * q, :3] = pal[rng.integers(0, len(pal), size=(2 * q, 2 * q))]
    return a


def test_background_auto_keys_uniform_magenta_border():
    a = _sprite_on((255, 0, 255))
    img = jitter(Image.fromarray(a), noise=3)
    r = pixelize(img, opts(grid=1, palette=parse_palette("none")))
    out = np.asarray(r.image)
    assert r.background_removed is True
    assert np.all(out[:16, :, 3] == 0) and np.all(out[:, :16, 3] == 0)
    assert np.all(out[16:48, 16:48, 3] == 255)


def test_background_auto_ignores_colour_touching_40pct_of_border():
    a = _sprite_on((255, 0, 255))
    # A green body touching 40% of the border ring (left column + most of top).
    green = (60, 180, 60)
    a[:, :2, :3] = green
    a[:2, : int(0.6 * 64), :3] = green
    ring = np.concatenate([a[0, :, :3], a[-1, :, :3], a[1:-1, 0, :3], a[1:-1, -1, :3]])
    frac = np.mean(np.all(ring == green, axis=-1))
    assert 0.35 <= frac <= 0.45
    r = pixelize(Image.fromarray(a), opts(grid=1, palette=parse_palette("none")))
    assert r.background_removed is False
    assert np.all(np.asarray(r.image)[..., 3] == 255)


def test_background_auto_preserves_existing_hard_alpha():
    a = _sprite_on((0, 0, 0))
    a[a[..., :3].sum(-1) == 0, 3] = 0  # transparent background, hard alpha
    a[:16, :, 3] = 0
    a[:, :16, 3] = 0
    a[48:, :, 3] = 0
    a[:, 48:, 3] = 0
    img = Image.fromarray(a)
    for bg in ("auto", "none", "transparent-keep"):
        r = pixelize(img, opts(grid=1, palette=parse_palette("none"), background=bg))
        out = np.asarray(r.image)
        assert np.array_equal(out[..., 3] == 0, a[..., 3] == 0), bg
        assert r.background_removed is False


def test_background_auto_does_not_key_interior_islands_of_border_colour():
    a = _sprite_on((255, 255, 255))
    a[30:34, 30:34, :3] = 255  # white "eye" inside the sprite
    r = pixelize(Image.fromarray(a), opts(grid=1, palette=parse_palette("none")))
    out = np.asarray(r.image)
    assert r.background_removed is True
    assert np.all(out[30:34, 30:34, 3] == 255)


def _serpentine_maze(n: int) -> np.ndarray:
    """Magenta corridors winding through dark walls, a fully magenta border
    and a single entrance: the worst case for row/column flood growth."""
    a = np.zeros((n, n, 4), dtype=np.uint8)
    a[..., :3] = (255, 0, 255)
    a[..., 3] = 255
    wall = (30, 30, 60)
    a[1, 1:n - 1, :3] = wall
    a[n - 2, 1:n - 1, :3] = wall
    a[1:n - 1, 1, :3] = wall
    a[1:n - 1, n - 2, :3] = wall
    a[1, 3, :3] = (255, 0, 255)  # the one entrance
    for i, y in enumerate(range(3, n - 3, 2)):
        a[y, 2:n - 2, :3] = wall
        gap = n - 4 if i % 2 == 0 else 2  # alternate ends: one long corridor
        a[y, gap, :3] = (255, 0, 255)
    return a


def test_background_flood_is_near_linear_on_a_maze():
    from muse.modalities.image_pixelart import pixelize as P

    a = _serpentine_maze(1024)
    corridor = np.all(a[..., :3] == (255, 0, 255), axis=-1)
    seed = np.zeros(corridor.shape, dtype=bool)
    seed[0] = seed[-1] = True
    seed[:, 0] = seed[:, -1] = True
    filled, rounds = P._flood_with_rounds(seed & corridor, corridor)
    assert np.array_equal(filled, corridor)  # the whole maze is reachable
    # Row/column growth needs ~n/2 rounds here; label propagation with
    # pointer jumping needs O(log n).
    assert rounds <= 40, rounds
    import time
    t = time.time()
    r = pixelize(Image.fromarray(a), opts(grid=1, palette=parse_palette("none"),
                                          cleanup=False))
    assert time.time() - t < 20  # the old flood took ~23 s here
    out = np.asarray(r.image)
    assert r.background_removed is True
    assert np.array_equal(out[..., 3] == 0, corridor)


def test_background_flood_matches_bfs_reference():
    from collections import deque

    from muse.modalities.image_pixelart import pixelize as P

    rng = np.random.default_rng(1)
    for _ in range(120):
        h, w = (int(v) for v in rng.integers(1, 30, 2))
        cand = rng.random((h, w)) < rng.uniform(0.3, 0.8)
        seed = rng.random((h, w)) < 0.03
        ref = np.zeros_like(cand)
        q = deque(zip(*np.nonzero(seed & cand)))
        for y, x in q:
            ref[y, x] = True
        while q:
            y, x = q.popleft()
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and cand[ny, nx] and not ref[ny, nx]:
                    ref[ny, nx] = True
                    q.append((ny, nx))
        assert np.array_equal(P._flood(seed, cand), ref)


def test_background_chroma_key_is_global():
    a = _sprite_on((0, 255, 0))
    a[30:34, 30:34, :3] = (0, 255, 0)
    r = pixelize(Image.fromarray(a), opts(grid=1, palette=parse_palette("none"),
                                          background="#00ff00"))
    out = np.asarray(r.image)
    assert r.background_removed is True
    assert np.all(out[30:34, 30:34, 3] == 0) and np.all(out[:16, :, 3] == 0)


def test_background_none_keeps_opaque_border():
    a = _sprite_on((255, 0, 255))
    r = pixelize(Image.fromarray(a), opts(grid=1, palette=parse_palette("none"),
                                          background="none"))
    assert r.background_removed is False
    assert np.all(np.asarray(r.image)[..., 3] == 255)


# ------------------------------------------------------- cleanup and outline


def test_cleanup_removes_orphans():
    a = np.zeros((9, 9, 4), dtype=np.uint8)
    a[..., :3] = (40, 120, 200)
    a[..., 3] = 255
    a[4, 4, :3] = (250, 20, 20)
    img = Image.fromarray(a)
    on = np.asarray(pixelize(img, opts(grid=1, palette=parse_palette("none"),
                                       background="none", cleanup=True)).image)
    off = np.asarray(pixelize(img, opts(grid=1, palette=parse_palette("none"),
                                        background="none", cleanup=False)).image)
    assert tuple(on[4, 4, :3]) == (40, 120, 200)
    assert tuple(off[4, 4, :3]) == (250, 20, 20)


def _sprite_with_eye() -> np.ndarray:
    a = np.zeros((16, 16, 4), dtype=np.uint8)
    a[3:13, 3:13] = (*_rgb("#eec39a"), 255)  # skin
    a[6, 6] = (*_rgb("#000000"), 255)  # a deliberate single-pixel eye
    return a


def test_default_cleanup_keeps_1to1_single_pixel_detail():
    a = _sprite_with_eye()
    r = pixelize(Image.fromarray(a), opts(palette=parse_palette("db32")))
    assert r.grid.cell == 1
    assert tuple(np.asarray(r.image)[6, 6, :3]) == (0, 0, 0)
    assert r.metadata["cleanup_applied"] is False
    forced = pixelize(Image.fromarray(a), opts(palette=parse_palette("db32"), cleanup=True))
    assert tuple(np.asarray(forced.image)[6, 6, :3]) == _rgb("#eec39a")
    assert forced.metadata["cleanup_applied"] is True


def test_default_cleanup_runs_when_the_grid_was_snapped():
    a = np.array(make_sprite(24, 24, colors=DB12, seed=3))
    a[6:9, 6:9] = (*_rgb("#639bff"), 255)
    a[7, 7] = (*_rgb("#ac3232"), 255)  # an orphan cell in AI output
    big = jitter(blur(Image.fromarray(a).resize((96, 96), Image.NEAREST), 0.6), noise=6)
    r = pixelize(big, opts(palette=parse_palette("db32")))
    assert r.grid.cell == 4
    assert r.metadata["cleanup_applied"] is True
    assert tuple(np.asarray(r.image)[7, 7, :3]) == _rgb("#639bff")
    kept = pixelize(big, opts(palette=parse_palette("db32"), cleanup=False))
    assert kept.metadata["cleanup_applied"] is False
    assert tuple(np.asarray(kept.image)[7, 7, :3]) == _rgb("#ac3232")


def test_cleanup_option_type_is_checked():
    with pytest.raises(ValueError):
        pixelize(SMALL, opts(cleanup="yes"))


def test_pixelize_options_cleanup_defaults_to_none():
    assert opts().cleanup is None


def test_outline_dark_adds_one_pixel_ring():
    a = np.zeros((12, 12, 4), dtype=np.uint8)
    a[4:8, 4:8] = (*_rgb("#d77643"), 255)
    img = Image.fromarray(a)
    r = pixelize(img, opts(grid=1, palette=parse_palette("endesga-32"), outline="dark"))
    out = np.asarray(r.image)
    sprite = a[..., 3] == 255
    ring = np.zeros_like(sprite)
    ring[3, 4:8] = ring[8, 4:8] = ring[4:8, 3] = ring[4:8, 8] = True
    assert np.all(out[ring, 3] == 255)
    assert np.all(out[~(ring | sprite), 3] == 0)  # diagonal corners stay empty
    ring_cols = {tuple(c) for c in out[ring, :3].tolist()}
    assert len(ring_cols) == 1
    (rc,) = ring_cols
    assert "#%02x%02x%02x" % rc in NAMED_PALETTES["endesga-32"]
    assert srgb_to_oklab(np.array(rc))[0] < srgb_to_oklab(np.array(_rgb("#d77643")))[0]


def test_outline_hex_colour():
    a = np.zeros((10, 10, 4), dtype=np.uint8)
    a[3:7, 3:7] = (200, 200, 200, 255)
    r = pixelize(Image.fromarray(a), opts(grid=1, palette=parse_palette("none"),
                                          outline="#102030"))
    out = np.asarray(r.image)
    assert tuple(out[2, 4]) == (16, 32, 48, 255)
    assert "#102030" in r.palette


def test_outline_dark_with_palette_none_on_many_colours_is_fast():
    rng = np.random.default_rng(9)
    a = np.zeros((200, 200, 4), dtype=np.uint8)
    a[20:180, 20:180, :3] = rng.integers(0, 256, size=(160, 160, 3))
    a[20:180, 20:180, 3] = 255
    import time
    t = time.time()
    r = pixelize(Image.fromarray(a), opts(grid=1, palette=parse_palette("none"),
                                          outline="dark", cleanup=False))
    assert time.time() - t < 10
    out = np.asarray(r.image)
    assert np.all(out[19, 20:180, 3] == 255) and out[19, 19, 3] == 0
    assert r.metadata["outline_pixels"] == 4 * 160


def test_alpha_threshold_zero_keeps_fully_transparent_pixels_transparent():
    a = np.zeros((8, 8, 4), dtype=np.uint8)
    a[2:6, 2:6] = (200, 100, 50, 10)
    r = pixelize(Image.fromarray(a), opts(grid=1, palette=parse_palette("none"),
                                          alpha_threshold=0, cleanup=False))
    out = np.asarray(r.image)
    assert np.all(out[2:6, 2:6, 3] == 255) and np.all(out[0, :, 3] == 0)


# --------------------------------------------------------------- seamless


@pytest.mark.parametrize("upscaled", [False, True])
def test_seamless_wraps(upscaled):
    rng = np.random.default_rng(11)
    pal = np.array([_rgb(c) for c in NAMED_PALETTES["db16"][:5]], dtype=np.uint8)
    small = np.zeros((24, 24, 4), dtype=np.uint8)
    small[..., :3] = pal[rng.integers(0, 5, size=(24, 24))]
    small[..., 3] = 255
    small[7:10, 7:10, :3] = pal[0]
    small[8, 8, :3] = pal[1]  # an orphan cleanup will remove
    img = Image.fromarray(small)
    grid = None
    if upscaled:
        img = jitter(blur(img.resize((96, 96), Image.NEAREST), 0.6), noise=5)
    else:
        grid = 1
    r = pixelize(img, opts(grid=grid, palette=parse_palette("db16"), background="none",
                           seamless=True))
    out = np.asarray(r.image)
    assert r.size == (24, 24)
    tiled = np.tile(out, (3, 3, 1))
    orph = orphans(tiled, wrap=False)[24:48, 24:48]
    band = np.zeros((24, 24), dtype=bool)
    band[:2] = band[-2:] = True
    band[:, :2] = band[:, -2:] = True
    assert not (orph & band).any()  # nothing breaks the cleanup rule at the seam
    if r.metadata["cleanup_applied"]:
        assert not orph.any()
    # Edges agree across the wrap.
    assert np.array_equal(out[:, 0], out[:, -1])
    assert np.array_equal(out[0], out[-1])


# ----------------------------------------------------------------- sheets


def test_sheet_shares_grid_and_palette():
    rng = np.random.default_rng(4)
    all_cols = [f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}"
                for c in rng.integers(0, 256, size=(24, 3))]
    cells = []
    for i in range(16):
        cs = [all_cols[(i * 3 + j) % 24] for j in range(6)]
        cells.append(make_sprite(16, 16, colors=cs, seed=100 + i))
    sheet = Image.new("RGBA", (64, 64))
    for i, c in enumerate(cells):
        sheet.paste(c, ((i % 4) * 16, (i // 4) * 16))
    big = jitter(blur(sheet.resize((256, 256), Image.NEAREST), 0.6), noise=4)
    r = pixelize(big, opts(palette=parse_palette("auto:16"), sheet=(4, 4)))
    assert isinstance(r.grid, GridInfo) and r.grid.cell == 4
    assert r.size == (64, 64)
    assert len(r.palette) <= 16
    pal = set(r.palette)
    for i in range(16):
        x, y = (i % 4) * 16, (i // 4) * 16
        assert colours(r.image.crop((x, y, x + 16, y + 16))) <= pal
    assert colours(r.image) <= pal
    assert r.metadata.get("sheet") == [4, 4]


def test_sheet_cells_do_not_bleed_outline_across_cells():
    sheet = np.zeros((8, 16, 4), dtype=np.uint8)
    sheet[2:6, 4:8] = (200, 50, 50, 255)  # touches the right edge of cell 0
    r = pixelize(Image.fromarray(sheet), opts(grid=1, palette=parse_palette("none"),
                                              outline="#000000", sheet=(2, 1)))
    out = np.asarray(r.image)
    assert np.all(out[:, 8, 3] == 0)  # cell 1's first column is untouched


# ---------------------------------------------------------------- preview


def test_preview_scale_is_nearest():
    big = jitter(blur(SMALL.resize((128, 128), Image.NEAREST), 0.6), noise=6)
    base = pixelize(big, opts(palette=parse_palette("db32")))
    r = pixelize(big, opts(palette=parse_palette("db32"), preview_scale=3))
    assert r.size == (32, 32)
    assert r.image.size == (96, 96)
    a = np.asarray(r.image)
    blocks = a.reshape(32, 3, 32, 3, 4)
    assert np.all(blocks == blocks[:, :1, :, :1, :])
    assert np.array_equal(a[::3, ::3], np.asarray(base.image))
    assert set(np.unique(a[..., 3]).tolist()) <= {0, 255}


# -------------------------------------------------------------- hygiene


def test_never_passes_mode_to_fromarray():
    for path in PKG.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "fromarray"):
                assert not any(kw.arg == "mode" for kw in node.keywords), path
                assert len(node.args) <= 1, path


def test_package_init_is_import_light_and_not_a_modality_yet():
    tree = ast.parse((PKG / "__init__.py").read_text())
    assert not [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
             if isinstance(t, ast.Name)}
    assert "MODALITY" not in names
