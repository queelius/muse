"""Autotile: RPG Maker A2 ground autotiles drawn from a fill (+ base) texture,
and their expansion to the 47-tile blob set.

All textures are synthetic and built here. The seam test enumerates every
geometrically legal neighbour pair (from a real 3x4 / 4x3 terrain
neighbourhood, so the shared corner bits agree too) and requires the two
touching pixel columns / rows to be identical over the whole edge.
"""
from __future__ import annotations

import ast
import inspect
import itertools
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from muse.modalities.image_pixelart import autotile
from muse.modalities.image_pixelart.autotile import (
    FLOOR_AUTOTILE_TABLE,
    SHEET_COLUMNS,
    SHEET_ROWS,
    BlobTile,
    EdgeStyle,
    blob_masks,
    build_a2,
    expand_blob47,
    validate_a2,
)
from muse.modalities.image_pixelart.palettes import NAMED_PALETTES, parse_palette

PKG = Path(__file__).resolve().parents[3] / "src" / "muse" / "modalities" / "image_pixelart"

N, NE, E, SE, S, SW, W, NW = 1, 2, 4, 8, 16, 32, 64, 128


# --------------------------------------------------------------------- helpers


def noise_tex(w, h, base_rgb, spread=60, seed=0) -> Image.Image:
    """Random-noise texture: deliberately NOT seamless (edges differ)."""
    rng = np.random.default_rng(seed)
    a = np.array(base_rgb, dtype=np.int16) + rng.integers(-spread, spread + 1, (h, w, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def fill_tex(T, seed=0) -> Image.Image:
    return noise_tex(T, T, (70, 150, 60), seed=seed)


def base_tex(T, seed=1) -> Image.Image:
    return noise_tex(T, T, (140, 100, 60), spread=30, seed=seed)


def arr(img) -> np.ndarray:
    return np.asarray(img.convert("RGBA"))


def tile_of(sheet: Image.Image, t: BlobTile, T: int) -> np.ndarray:
    return arr(sheet)[t.y * T:(t.y + 1) * T, t.x * T:(t.x + 1) * T]


def compose(a2: Image.Image, entry, T: int) -> np.ndarray:
    a, h = arr(a2), T // 2
    out = np.zeros((T, T, 4), dtype=np.uint8)
    for i, (qx, qy) in enumerate(entry):
        dx, dy = (i % 2) * h, (i // 2) * h
        out[dy:dy + h, dx:dx + h] = a[qy * h:(qy + 1) * h, qx * h:(qx + 1) * h]
    return out


def reduce_mask(m: int) -> int:
    for corner, a, b in ((NE, N, E), (SE, S, E), (SW, S, W), (NW, N, W)):
        if not (m & a and m & b):
            m &= ~corner
    return m


def mask_at(grid, r, c) -> int:
    """Raw 8-neighbour mask of cell (r, c) in a 0/1 grid (outside = 0), reduced."""
    def g(rr, cc):
        return 1 if 0 <= rr < len(grid) and 0 <= cc < len(grid[0]) and grid[rr][cc] else 0
    m = 0
    for bit, (dr, dc) in ((N, (-1, 0)), (NE, (-1, 1)), (E, (0, 1)), (SE, (1, 1)),
                          (S, (1, 0)), (SW, (1, -1)), (W, (0, -1)), (NW, (-1, -1))):
        if g(r + dr, c + dc):
            m |= bit
    return reduce_mask(m)


def legal_pairs(direction: str) -> set[tuple[int, int]]:
    """(mask_a, mask_b) for b directly east (or south) of a, both terrain,
    over every binary terrain neighbourhood of the pair."""
    pairs = set()
    if direction == "E":  # 3 rows x 4 cols, a at (1,1), b at (1,2)
        free = [(r, c) for r in range(3) for c in range(4) if (r, c) not in ((1, 1), (1, 2))]
        a_pos, b_pos, shape = (1, 1), (1, 2), (3, 4)
    else:  # 4 rows x 3 cols, a at (1,1), b at (2,1)
        free = [(r, c) for r in range(4) for c in range(3) if (r, c) not in ((1, 1), (2, 1))]
        a_pos, b_pos, shape = (1, 1), (2, 1), (4, 3)
    for bits in itertools.product((0, 1), repeat=len(free)):
        grid = [[0] * shape[1] for _ in range(shape[0])]
        grid[a_pos[0]][a_pos[1]] = grid[b_pos[0]][b_pos[1]] = 1
        for (r, c), v in zip(free, bits):
            grid[r][c] = v
        pairs.add((mask_at(grid, *a_pos), mask_at(grid, *b_pos)))
    return pairs


@pytest.fixture(scope="module")
def default_set():
    T = 16
    a2 = build_a2(fill_tex(T), None, T, EdgeStyle(), None)
    sheet, tiles = expand_blob47(a2, T)
    return T, a2, sheet, tiles


# ------------------------------------------------------------------- the table


def test_table_attribution_present():
    src = inspect.getsource(autotile)
    assert "rpg_core.js" in src
    assert "FLOOR_AUTOTILE_TABLE" in src and "MIT" in src


def test_table_shape():
    assert len(FLOOR_AUTOTILE_TABLE) == 48
    for entry in FLOOR_AUTOTILE_TABLE:
        assert len(entry) == 4
        for qx, qy in entry:
            assert 0 <= qx < 4 and 0 <= qy < 6
    # Spot checks against rpg_core.js: shape 0 (all interior), 46 (isolated,
    # from the block corners), 47 (the top-left preview tile).
    assert FLOOR_AUTOTILE_TABLE[0] == ((2, 4), (1, 4), (2, 3), (1, 3))
    assert FLOOR_AUTOTILE_TABLE[46] == ((0, 2), (3, 2), (0, 5), (3, 5))
    assert FLOOR_AUTOTILE_TABLE[47] == ((0, 0), (1, 0), (0, 1), (1, 1))


def test_blob_masks_are_the_47_reduced_masks():
    masks = blob_masks()
    assert len(masks) == 47 and len(set(masks)) == 47
    assert set(masks) == {reduce_mask(m) for m in range(256)}


def test_every_mask_maps_to_one_rpg_maker_shape(default_set):
    T, a2, sheet, tiles = default_set
    # Tile index == RPG Maker shape number; shape 47 (the preview duplicate)
    # is the dropped one, so indices are exactly 0..46.
    assert sorted(t.index for t in tiles) == list(range(47))
    for t in tiles:
        assert (t.x, t.y) == (t.index % SHEET_COLUMNS, t.index // SHEET_COLUMNS)
        np.testing.assert_array_equal(tile_of(sheet, t, T),
                                      compose(a2, FLOOR_AUTOTILE_TABLE[t.index], T))


# --------------------------------------------------------------------- build_a2


@pytest.mark.parametrize("T", [16, 24, 32, 48])
def test_build_a2_size_and_alpha(T):
    a2 = build_a2(fill_tex(T), None, T, EdgeStyle(), None)
    assert a2.mode == "RGBA"
    assert a2.size == (2 * T, 3 * T)
    assert set(np.unique(arr(a2)[..., 3]).tolist()) <= {0, 255}


def test_build_a2_without_base_has_transparent_outside():
    T = 16
    a = arr(build_a2(fill_tex(T), None, T, EdgeStyle(), None))
    assert a[0, 0, 3] == 0  # preview tile corner: outside the rounded inset
    assert a[T // 2, T // 2, 3] == 255  # preview tile centre: terrain
    assert (a[a[..., 3] == 0] == 0).all()  # transparent pixels are (0,0,0,0)


def test_build_a2_with_base_is_opaque():
    T = 24
    half = T // 2
    a = arr(build_a2(fill_tex(T), base_tex(T), T, EdgeStyle(), None))
    assert (a[..., 3] == 255).all()
    # The ground block is the decoration block (no base) composed over the
    # prepared base: wherever the decoration is transparent the base shows,
    # pixel for pixel, at the same tile-local position.
    deco = arr(build_a2(fill_tex(T), None, T, EdgeStyle(), None))
    under = autotile._flatten_base(autotile._prepare_texture(base_tex(T), T, "base"))
    preview, deco_preview = a[:T, :T], deco[:T, :T]
    expected = np.where(deco_preview[..., 3:] == 255, deco_preview, under)
    for qy in (0, 1):  # every corner quadrant of the isolated preview shape
        for qx in (0, 1):
            ys, xs = slice(qy * half, (qy + 1) * half), slice(qx * half, (qx + 1) * half)
            assert (preview[ys, xs, 3] == 255).all(), (qx, qy)
            assert np.array_equal(preview[ys, xs], expected[ys, xs]), (qx, qy)
            # Each corner really shows base outside the terrain.
            assert (deco_preview[ys, xs, 3] == 0).any(), (qx, qy)
    assert np.array_equal(a[0, 0], under[0, 0])


def test_build_a2_base_with_holes_is_still_opaque():
    T = 16
    b = np.array(base_tex(T).convert("RGBA"))
    b[::3, ::3, 3] = 0
    a = arr(build_a2(fill_tex(T), Image.fromarray(b), T, EdgeStyle(), None))
    assert (a[..., 3] == 255).all()


def test_build_a2_seeded_deterministic():
    T = 48  # roughness noise needs T >= 48 at the default 0.3 to move a pixel
    f, st = fill_tex(T), EdgeStyle(seed=7)
    one = build_a2(f, None, T, st, None).tobytes()
    assert one == build_a2(f, None, T, st, None).tobytes()
    assert one != build_a2(f, None, T, EdgeStyle(seed=8), None).tobytes()


def test_roughness_zero_ignores_seed():
    T = 48
    f = fill_tex(T)
    a = build_a2(f, None, T, EdgeStyle(roughness=0.0, seed=1), None).tobytes()
    b = build_a2(f, None, T, EdgeStyle(roughness=0.0, seed=2), None).tobytes()
    assert a == b


def test_build_a2_palette_mapped():
    T = 16
    pal = parse_palette("db32")
    a = arr(build_a2(fill_tex(T), base_tex(T), T, EdgeStyle(), pal))
    allowed = {tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in NAMED_PALETTES["db32"]}
    used = {tuple(int(v) for v in p) for p in a[..., :3].reshape(-1, 3)}
    assert used <= allowed
    assert set(np.unique(a[..., 3]).tolist()) == {255}


def test_palette_none_spec_is_accepted():
    T = 16
    f = fill_tex(T)
    a = build_a2(f, None, T, EdgeStyle(), parse_palette("none")).tobytes()
    assert a == build_a2(f, None, T, EdgeStyle(), None).tobytes()


def test_outline_and_highlight_rings():
    T = 32
    colour = (16, 32, 48)
    a2 = build_a2(Image.new("RGB", (T, T), (100, 160, 80)), None, T,
                  EdgeStyle(outline="#102030", highlight=True, roughness=0.0), None)
    a = arr(a2)
    m = T // 8  # default edge width: the margin of an open side
    # Top edge in the block's quadrant (qx=1, qy=2), which starts at y = T.
    col = a[T:T + T // 2, T // 2 + 1]
    assert (col[:m, 3] == 0).all()
    assert tuple(col[m, :3]) == colour
    assert tuple(col[m + 1, :3]) != colour and tuple(col[m + 1, :3]) != (100, 160, 80)
    assert tuple(col[m + 2, :3]) == (100, 160, 80)


def test_uniform_fill_interior_is_the_fill():
    T = 16
    sheet, tiles = expand_blob47(build_a2(Image.new("RGB", (T, T), (90, 140, 70)), None,
                                          T, EdgeStyle(), None), T)
    full = next(t for t in tiles if t.mask == 255)
    t = tile_of(sheet, full, T)
    assert (t[..., :3] == (90, 140, 70)).all() and (t[..., 3] == 255).all()


@pytest.mark.parametrize("size", [(7, 5), (16, 16), (100, 60), (64, 64)])
def test_fill_any_size_is_fitted(size):
    T = 16
    a2 = build_a2(noise_tex(*size, (60, 140, 60)), None, T, EdgeStyle(), None)
    assert a2.size == (2 * T, 3 * T)
    assert set(np.unique(arr(a2)[..., 3]).tolist()) <= {0, 255}


@pytest.mark.parametrize("kwargs", [
    {"outline": "black"},
    {"outline": "#12345"},
    {"roughness": 1.5},
    {"roughness": -0.1},
    {"edge_width": 0},
    {"edge_width": 5},  # > T/4 at T=16
    {"corner_radius": 9},
    {"corner_radius": -1},
    {"highlight": "yes"},
    {"seed": 1.5},
])
def test_bad_edge_style(kwargs):
    with pytest.raises(ValueError):
        build_a2(fill_tex(16), None, 16, EdgeStyle(**kwargs), None)


@pytest.mark.parametrize("T", [15, 6, 0, 1024])
def test_bad_tile_size(T):
    with pytest.raises(ValueError):
        build_a2(fill_tex(16), None, T, EdgeStyle(), None)


def test_transparent_fill_rejected():
    with pytest.raises(ValueError, match="transparent"):
        build_a2(Image.new("RGBA", (16, 16), (0, 0, 0, 0)), None, 16, EdgeStyle(), None)


# ------------------------------------------------------------------ validate_a2


def test_validate_a2_wrong_size():
    with pytest.raises(ValueError, match=r"96x144"):
        validate_a2(Image.new("RGBA", (96, 140)), 48)


def test_validate_a2_ok_and_bad_tile_size():
    validate_a2(Image.new("RGBA", (96, 144)), 48)
    with pytest.raises(ValueError):
        validate_a2(Image.new("RGBA", (30, 45)), 15)


# ----------------------------------------------------------------- expand_blob47


def test_blob47_has_47_distinct_masks(default_set):
    T, a2, sheet, tiles = default_set
    assert len(tiles) == 47
    assert len({t.mask for t in tiles}) == 47
    assert sheet.size == (8 * 16, 6 * 16) == (SHEET_COLUMNS * T, SHEET_ROWS * T)
    assert sheet.mode == "RGBA"
    assert set(np.unique(arr(sheet)[..., 3]).tolist()) <= {0, 255}
    # The 48th cell is empty.
    assert (arr(sheet)[5 * T:, 7 * T:] == 0).all()


def test_known_shapes(default_set):
    T, a2, sheet, tiles = default_set
    by_mask = {t.mask: t for t in tiles}
    full = tile_of(sheet, by_mask[255], T)
    np.testing.assert_array_equal(full, compose(a2, FLOOR_AUTOTILE_TABLE[0], T))
    assert (full[..., 3] == 255).all()
    iso = tile_of(sheet, by_mask[0], T)
    # Isolated: the outer edge on all four sides (outside is transparent).
    for edge in (iso[0], iso[-1], iso[:, 0], iso[:, -1]):
        assert (edge[..., 3] == 0).all()
    assert iso[T // 2, T // 2, 3] == 255
    # It is RPG Maker shape 46 (block corners); for a built A2 that equals the
    # A2's top-left preview tile, which build_a2 draws as that composition.
    assert by_mask[0].index == 46
    np.testing.assert_array_equal(iso, compose(a2, FLOOR_AUTOTILE_TABLE[46], T))
    np.testing.assert_array_equal(iso, arr(a2)[:T, :T])


def test_imported_a2_soft_alpha_is_binarized():
    T = 16
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (3 * T, 2 * T, 4)).astype(np.uint8)
    sheet, tiles = expand_blob47(Image.fromarray(a), T)
    s = arr(sheet)
    assert set(np.unique(s[..., 3]).tolist()) <= {0, 255}
    assert (s[s[..., 3] == 0] == 0).all()


def test_expand_rejects_wrong_size():
    with pytest.raises(ValueError, match=r"32x48"):
        expand_blob47(Image.new("RGBA", (32, 47)), 16)


SEAM_CASES = [
    dict(T=16, base=False, style=EdgeStyle(), palette=None),
    dict(T=24, base=True, style=EdgeStyle(seed=5), palette="db32"),
    dict(T=32, base=False, style=EdgeStyle(outline="#102030", highlight=False,
                                           corner_radius=0, roughness=0.6, seed=2),
         palette=None),
    dict(T=48, base=True, style=EdgeStyle(roughness=1.0, seed=3), palette=None),
    dict(T=48, base=False, style=EdgeStyle(outline="none", edge_width=12, roughness=1.0,
                                           corner_radius=4, seed=11), palette="auto:12"),
]


@pytest.mark.parametrize("case", SEAM_CASES, ids=lambda c: f"T{c['T']}-base{c['base']}")
def test_blob47_seams_match_for_every_legal_neighbour_pair(case):
    T = case["T"]
    pal = parse_palette(case["palette"]) if case["palette"] else None
    a2 = build_a2(fill_tex(T, seed=T), base_tex(T) if case["base"] else None, T,
                  case["style"], pal)
    sheet, tiles = expand_blob47(a2, T)
    by_mask = {t.mask: tile_of(sheet, t, T) for t in tiles}
    east, south = legal_pairs("E"), legal_pairs("S")
    assert len(east) > 47 and len(south) > 47
    for ma, mb in east:
        np.testing.assert_array_equal(by_mask[ma][:, -1], by_mask[mb][:, 0],
                                      err_msg=f"E seam {ma}->{mb}")
    for ma, mb in south:
        np.testing.assert_array_equal(by_mask[ma][-1], by_mask[mb][0],
                                      err_msg=f"S seam {ma}->{mb}")


def test_random_map_has_no_seams():
    """Paint a random 16x12 map with blob tiles; every terrain/terrain edge
    pair agrees (the same check as above, end to end)."""
    T = 16
    sheet, tiles = expand_blob47(build_a2(fill_tex(T), None, T, EdgeStyle(roughness=1.0),
                                          None), T)
    by_mask = {t.mask: tile_of(sheet, t, T) for t in tiles}
    rng = np.random.default_rng(0)
    grid = (rng.random((12, 16)) < 0.6).astype(int).tolist()
    for r in range(12):
        for c in range(16):
            if not grid[r][c]:
                continue
            a = by_mask[mask_at(grid, r, c)]
            if c + 1 < 16 and grid[r][c + 1]:
                np.testing.assert_array_equal(a[:, -1], by_mask[mask_at(grid, r, c + 1)][:, 0])
            if r + 1 < 12 and grid[r + 1][c]:
                np.testing.assert_array_equal(a[-1], by_mask[mask_at(grid, r + 1, c)][0])


# ---------------------------------------------------------------------- hygiene


@pytest.mark.parametrize("module", ["autotile.py", "exporters.py"])
def test_no_fromarray_mode_kwarg(module):
    tree = ast.parse((PKG / module).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "fromarray":
            assert not any(k.arg == "mode" for k in node.keywords), module
