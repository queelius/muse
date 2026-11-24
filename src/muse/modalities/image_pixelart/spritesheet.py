"""Sprite-sheet baking: slice, key background, clean stray components, fit & align,
lock a shared cross-frame palette, optionally mirror 5-to-8 directions, and stitch
an RGBA atlas.

Pillow + numpy only. Never pass ``mode=`` to ``Image.fromarray`` (Pillow 13
removed it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from muse.modalities.image_pixelart.palettes import (
    PaletteSpec,
    auto_palette,
    nearest_palette_index,
    pack_rgb,
    palette_to_array,
    parse_palette,
    rgb_to_hex,
    unpack_rgb,
)

DIRECTION_8_NAMES: tuple[str, ...] = (
    "up",
    "north_east",
    "side",
    "south_east",
    "down",
    "south_west",
    "left",
    "north_west",
)

BACKGROUND_MODES = ("auto", "white", "none")
MIRROR_MODES = ("none", "5to8")
ALIGN_MODES = ("bottom", "center")
RESAMPLE_MODES = ("nearest", "lanczos")
REPORTED_PALETTE_CAP = 256
_ENCLOSED_WHITE_MIN_PIXELS = 4
_BORDER_UNIFORM_FRACTION = 0.5


class SpritesheetTooLargeError(ValueError):
    """The output atlas would exceed ``max_output_side``."""

    def __init__(self, width: int, height: int, cap: int) -> None:
        self.width, self.height, self.cap = width, height, cap
        super().__init__(
            f"spritesheet atlas {width}x{height} exceeds max side {cap}"
        )


@dataclass(frozen=True)
class SpritesheetOptions:
    """Options for ``bake_spritesheet``."""

    grid: tuple[int, int] = (3, 2)  # (cols, rows)
    frame_size: tuple[int, int] = (48, 48)  # (width, height)
    fit_size: tuple[int, int] | None = None
    mirror_mode: str = "none"  # none | 5to8
    background: str = "auto"  # auto | white | none
    bg_threshold: int = 236
    fringe_threshold: int = 212
    color_tolerance: int = 24
    clear_enclosed_white: bool = True
    clean_stray: bool = True
    min_component_ratio: float = 0.03
    border_bleed_ratio: float = 0.15
    align: str = "bottom"  # bottom | center
    bottom_padding: int = 2
    resample: str = "nearest"  # nearest | lanczos
    hard_alpha: bool = True
    alpha_threshold: int = 128
    palette: PaletteSpec | str = "none"
    output_cols: int | None = None
    max_output_side: int | None = None


@dataclass(frozen=True)
class SpritesheetCell:
    """Metadata for one frame in the stitched atlas."""

    index: int
    name: str
    x: int
    y: int
    width: int
    height: int
    anchor_x: int
    anchor_y: int
    opaque_pixels: int


@dataclass
class SpritesheetResult:
    """Output of ``bake_spritesheet``: stitched RGBA atlas, per-frame images,
    cell layout metadata, and palette."""

    atlas: Image.Image
    cells: list[Image.Image]
    layout: list[SpritesheetCell]
    frame_size: tuple[int, int]
    atlas_size: tuple[int, int]
    output_grid: tuple[int, int]
    palette: list[str]
    metadata: dict = field(default_factory=dict)


# ------------------------------------------------------------------ validation


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _pair(v, name: str) -> tuple[int, int]:
    if not (isinstance(v, (tuple, list)) and len(v) == 2 and all(_is_int(x) for x in v)):
        raise ValueError(f"{name} must be a pair of positive integers")
    a, b = int(v[0]), int(v[1])
    if a < 1 or b < 1:
        raise ValueError(f"{name} values must be at least 1")
    return a, b


def _resolve_palette(palette: PaletteSpec | str | None) -> PaletteSpec:
    if palette is None:
        return PaletteSpec(kind="none", colors=(), count=None)
    if isinstance(palette, PaletteSpec):
        return palette
    return parse_palette(palette)


# ----------------------------------------------- vectorized run-graph labelling


def _row_runs(mask: np.ndarray) -> np.ndarray:
    """Label each horizontal run of True pixels 1..R (0 = False)."""
    h, w = mask.shape
    starts = mask.copy()
    starts[:, 1:] &= ~mask[:, :-1]
    ids = np.cumsum(starts.ravel(), dtype=np.int64).reshape(h, w)
    ids[~mask] = 0
    return ids


def _touching_pairs(top: np.ndarray, bot: np.ndarray,
                    both: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = both.copy()
    first[:, 1:] &= ~(both[:, :-1] & (top[:, 1:] == top[:, :-1])
                      & (bot[:, 1:] == bot[:, :-1]))
    return top[first], bot[first]


def _merge_runs(label: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hook-and-jump until every pair of touching runs shares its component root."""
    if len(a) == 0:
        return label
    while True:
        la, lb = label[a], label[b]
        diff = la != lb
        if not diff.any():
            break
        la, lb = la[diff], lb[diff]
        low = np.minimum(la, lb)
        np.minimum.at(label, la, low)
        np.minimum.at(label, lb, low)
        while True:
            nxt = label[label]
            if np.array_equal(nxt, label):
                break
            label = nxt
    return label


def _flood4(seed: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """4-connected flood of ``cand`` starting from ``seed``."""
    ids = _row_runs(cand)
    n_runs = int(ids.max())
    if n_runs == 0:
        return np.zeros(cand.shape, dtype=bool)
    label = np.arange(n_runs + 1, dtype=np.int64)
    seeded = np.unique(ids[seed & cand])
    if len(seeded) == 0:
        return np.zeros(cand.shape, dtype=bool)
    label[seeded] = 0
    if cand.shape[0] >= 2:
        a, b = _touching_pairs(ids[:-1], ids[1:], cand[:-1] & cand[1:])
        label = _merge_runs(label, a, b)
    return (label[ids] == 0) & cand


def _components4(mask: np.ndarray) -> np.ndarray:
    """4-connected components of ``mask`` (0 = False, >0 = component root id)."""
    ids = _row_runs(mask)
    n_runs = int(ids.max())
    if n_runs == 0:
        return np.zeros(mask.shape, dtype=np.int64)
    label = np.arange(n_runs + 1, dtype=np.int64)
    if mask.shape[0] >= 2:
        a, b = _touching_pairs(ids[:-1], ids[1:], mask[:-1] & mask[1:])
        label = _merge_runs(label, a, b)
    return label[ids]


def _components8(mask: np.ndarray) -> np.ndarray:
    """8-connected components of ``mask`` (0 = False, >0 = component root id)."""
    ids = _row_runs(mask)
    n_runs = int(ids.max())
    if n_runs == 0:
        return np.zeros(mask.shape, dtype=np.int64)
    label = np.arange(n_runs + 1, dtype=np.int64)
    h, w = mask.shape
    if h >= 2:
        a_parts: list[np.ndarray] = []
        b_parts: list[np.ndarray] = []
        # Vertical edges
        a, b = _touching_pairs(ids[:-1], ids[1:], mask[:-1] & mask[1:])
        if len(a):
            a_parts.append(a)
            b_parts.append(b)
        if w >= 2:
            # Down-right diagonal edges
            a, b = _touching_pairs(ids[:-1, :-1], ids[1:, 1:],
                                   mask[:-1, :-1] & mask[1:, 1:])
            if len(a):
                a_parts.append(a)
                b_parts.append(b)
            # Down-left diagonal edges
            a, b = _touching_pairs(ids[:-1, 1:], ids[1:, :-1],
                                   mask[:-1, 1:] & mask[1:, :-1])
            if len(a):
                a_parts.append(a)
                b_parts.append(b)
        if a_parts:
            label = _merge_runs(label, np.concatenate(a_parts), np.concatenate(b_parts))
    return label[ids]


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


# -------------------------------------------------------------- core stages


def slice_grid(image: Image.Image, cols: int, rows: int) -> list[Image.Image]:
    """Slice ``image`` into ``cols * rows`` RGBA cells in row-major order."""
    c, r = _pair((cols, rows), "grid")
    rgba = image.convert("RGBA")
    w, h = rgba.size
    if w < c or h < r:
        raise ValueError(
            f"image size {w}x{h} is smaller than grid {c}x{r}"
        )
    xs = [round(i * w / c) for i in range(c + 1)]
    ys = [round(j * h / r) for j in range(r + 1)]
    cells: list[Image.Image] = []
    for row in range(r):
        for col in range(c):
            cells.append(rgba.crop((xs[col], ys[row], xs[col + 1], ys[row + 1])))
    return cells


def flood_key_background(
    cell: Image.Image,
    mode: str = "auto",
    bg_threshold: int = 236,
    fringe_threshold: int = 212,
    color_tolerance: int = 24,
    clear_enclosed_white: bool = True,
) -> Image.Image:
    """Remove a solid or white sheet background via border flood fill,
    optional enclosed white pocket cleanup, and 1 px anti-aliased halo defringe.

    - ``auto``: leaves already-transparent RGBA borders intact; otherwise estimates
      the border colour (white or solid chroma) and flood-keys from the edges.
    - ``white``: flood-keys near-white low-saturation border regions (ideal for
      solid-white sprite sheets from image generators).
    - ``none``: returns an RGBA copy untouched.
    """
    if mode not in BACKGROUND_MODES:
        raise ValueError(f"background must be one of {', '.join(BACKGROUND_MODES)}")
    for name, val in (
        ("bg_threshold", bg_threshold),
        ("fringe_threshold", fringe_threshold),
        ("color_tolerance", color_tolerance),
    ):
        if not _is_int(val) or not 0 <= int(val) <= 255:
            raise ValueError(f"{name} must be an integer in 0..255")

    arr = np.array(cell.convert("RGBA"), dtype=np.uint8)
    h, w = arr.shape[:2]
    if mode == "none" or h == 0 or w == 0:
        return Image.fromarray(arr)

    bg_th = int(bg_threshold)
    fr_th = int(fringe_threshold)
    tol = int(color_tolerance)

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    rgb = arr[..., :3]
    alpha = arr[..., 3]
    rgb_min = rgb.min(axis=-1)
    rgb_max = rgb.max(axis=-1)
    chroma = rgb_max.astype(np.int16) - rgb_min.astype(np.int16)

    is_white_bg = mode == "white"
    ref_rgb: np.ndarray | None = None

    if mode == "auto":
        border_alpha = alpha[border]
        if float((border_alpha < 128).mean()) >= _BORDER_UNIFORM_FRACTION:
            # Already an RGBA sprite with a transparent border: keep alpha as-is.
            return Image.fromarray(arr)
        opaque_border_rgb = rgb[border & (alpha >= 128)]
        if len(opaque_border_rgb) == 0:
            return Image.fromarray(arr)
        ref_rgb = np.rint(np.median(opaque_border_rgb, axis=0)).astype(np.int16)
        border_diff = np.max(np.abs(opaque_border_rgb.astype(np.int16) - ref_rgb), axis=-1)
        if float((border_diff <= tol).mean()) < _BORDER_UNIFORM_FRACTION:
            # Non-uniform border (e.g. full-bleed textured tile): do not punch holes.
            return Image.fromarray(arr)
        if int(ref_rgb.min()) >= bg_th and int(ref_rgb.max() - ref_rgb.min()) <= tol:
            is_white_bg = True

    white_pred = (rgb_min >= bg_th) & (chroma <= tol)
    if is_white_bg:
        cand = (alpha == 0) | white_pred
    else:
        assert ref_rgb is not None
        diff = np.max(np.abs(rgb.astype(np.int16) - ref_rgb), axis=-1)
        cand = (alpha == 0) | (diff <= tol)

    bg_mask = _flood4(border, cand)
    arr[bg_mask] = 0

    if clear_enclosed_white and is_white_bg:
        rem_white = (arr[..., 3] > 0) & white_pred
        if rem_white.any():
            labels = _components4(rem_white)
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            clear_ids = counts >= _ENCLOSED_WHITE_MIN_PIXELS
            enclosed = clear_ids[labels]
            if enclosed.any():
                arr[enclosed] = 0
                bg_mask |= enclosed

    # 1 px anti-aliased halo defringe adjacent to keyed background
    if bg_mask.any():
        rim = _dilate4(bg_mask) & (arr[..., 3] > 0)
        if rim.any():
            if is_white_bg:
                fringe = rim & (rgb_min >= fr_th) & (chroma <= max(tol, 36))
            else:
                assert ref_rgb is not None
                fringe_tol = min(255, tol + max(0, bg_th - fr_th))
                diff = np.max(np.abs(rgb.astype(np.int16) - ref_rgb), axis=-1)
                fringe = rim & (diff <= fringe_tol)
            arr[fringe] = 0

    return Image.fromarray(arr)


def filter_stray_components(
    cell: Image.Image,
    min_component_ratio: float = 0.03,
    border_bleed_ratio: float = 0.15,
) -> Image.Image:
    """Discard tiny disconnected specks and neighbor-cell border bleed using
    8-connected components on opaque pixels.

    Always preserves the largest component (main body), plus any interior
    component of at least ``min_component_ratio * largest`` pixels and any
    border-touching component of at least ``border_bleed_ratio * largest`` pixels.
    """
    if not 0.0 <= float(min_component_ratio) <= 1.0:
        raise ValueError("min_component_ratio must be in [0, 1]")
    if not 0.0 <= float(border_bleed_ratio) <= 1.0:
        raise ValueError("border_bleed_ratio must be in [0, 1]")

    arr = np.array(cell.convert("RGBA"), dtype=np.uint8)
    opaque = arr[..., 3] > 0
    if not opaque.any():
        return Image.fromarray(arr)

    labels = _components8(opaque)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest = int(counts.max())
    if largest == 0:
        return Image.fromarray(arr)

    touches_border = np.zeros(len(counts), dtype=bool)
    touches_border[labels[0, :]] = True
    touches_border[labels[-1, :]] = True
    touches_border[labels[:, 0]] = True
    touches_border[labels[:, -1]] = True
    touches_border[0] = False

    min_interior = largest * float(min_component_ratio)
    min_border = largest * float(border_bleed_ratio)
    main_id = int(np.argmax(counts))

    keep = (counts >= min_interior) & (~touches_border | (counts >= min_border))
    keep[main_id] = True
    keep[0] = False

    keep_mask = keep[labels]
    arr[~keep_mask] = 0
    return Image.fromarray(arr)


def fit_and_align(
    cell: Image.Image,
    frame_size: tuple[int, int],
    fit_size: tuple[int, int] | None = None,
    align: str = "bottom",
    bottom_padding: int = 2,
    resample: str = "nearest",
    alpha_threshold: int = 128,
    hard_alpha: bool = True,
) -> Image.Image:
    """Crop a cell to its tight alpha bounding box, aspect-preserving scale into
    ``fit_size``, horizontally center, vertically anchor (``bottom`` or ``center``),
    and optionally binarize alpha."""
    fw, fh = _pair(frame_size, "frame_size")
    if align not in ALIGN_MODES:
        raise ValueError(f"align must be one of {', '.join(ALIGN_MODES)}")
    if resample not in RESAMPLE_MODES:
        raise ValueError(f"resample must be one of {', '.join(RESAMPLE_MODES)}")
    if not _is_int(alpha_threshold) or not 0 <= int(alpha_threshold) <= 255:
        raise ValueError("alpha_threshold must be an integer in 0..255")
    if not _is_int(bottom_padding) or int(bottom_padding) < 0:
        raise ValueError("bottom_padding must be a non-negative integer")

    pad = int(bottom_padding)
    if align == "bottom" and pad >= fh:
        raise ValueError(
            f"bottom_padding ({pad}) must be smaller than frame height ({fh})"
        )

    if fit_size is not None:
        fit_w, fit_h = _pair(fit_size, "fit_size")
        if fit_w > fw or fit_h > fh:
            raise ValueError(
                f"fit_size {fit_w}x{fit_h} exceeds frame_size {fw}x{fh}"
            )
    else:
        fit_w = max(1, fw - 2 * pad) if fw > 2 * pad else max(1, fw - pad)
        fit_h = max(1, fh - 2 * pad) if fh > 2 * pad else max(1, fh - pad)

    rgba = cell.convert("RGBA")
    arr = np.asarray(rgba)
    ath = int(alpha_threshold)
    mask = (arr[..., 3] >= ath) & (arr[..., 3] > 0) if hard_alpha else (arr[..., 3] > 0)
    if not mask.any():
        return Image.new("RGBA", (fw, fh), (0, 0, 0, 0))

    ys, xs = np.nonzero(mask)
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    cropped = rgba.crop((x0, y0, x1, y1))
    cw, ch = cropped.size

    scale = min(fit_w / cw, fit_h / ch)
    new_w = max(1, min(fit_w, int(round(cw * scale))))
    new_h = max(1, min(fit_h, int(round(ch * scale))))

    if (new_w, new_h) != (cw, ch):
        filt = (
            Image.Resampling.NEAREST
            if resample == "nearest"
            else Image.Resampling.LANCZOS
        )
        scaled = cropped.resize((new_w, new_h), filt)
    else:
        scaled = cropped

    if hard_alpha:
        sarr = np.array(scaled, dtype=np.uint8)
        op = (sarr[..., 3] >= ath) & (sarr[..., 3] > 0)
        sarr[op, 3] = 255
        sarr[~op] = 0
        scaled = Image.fromarray(sarr)

    canvas = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    paste_x = (fw - new_w) // 2
    if align == "bottom":
        paste_y = max(0, fh - pad - new_h)
    else:
        paste_y = (fh - new_h) // 2
    canvas.paste(scaled, (paste_x, paste_y))
    return canvas


def mirror_5to8(cells: list[Image.Image]) -> list[Image.Image]:
    """Expand the first 5 directional views ``(N/up, NE, E/side, SE, S/down)``
    into an 8-direction list ``[up, north_east, side, south_east, down,
    south_west, left, north_west]`` by horizontally mirroring ``SE -> SW``,
    ``E -> W``, and ``NE -> NW``."""
    if len(cells) < 5:
        raise ValueError(
            f"mirror_5to8 requires at least 5 cells, got {len(cells)}"
        )
    up, ne, side, se, down = cells[:5]
    sw = se.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    left = side.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    nw = ne.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return [up, ne, side, se, down, sw, left, nw]


# ----------------------------------------------------- palette locking & bake


def _colours_by_use(arrays: list[np.ndarray]) -> np.ndarray:
    parts = [a[a[..., 3] > 0, :3] for a in arrays]
    parts = [p for p in parts if len(p)]
    if not parts:
        return np.zeros(0, dtype=np.uint32)
    keys = pack_rgb(np.concatenate(parts, axis=0))
    uniq, counts = np.unique(keys, return_counts=True)
    return uniq[np.argsort(-counts, kind="stable")]


def _apply_shared_palette(
    frames: list[Image.Image],
    spec: PaletteSpec,
) -> tuple[list[Image.Image], list[str], int, bool]:
    """Lock all frames onto one shared OKLab palette when ``spec.kind != 'none'``."""
    arrays = [np.array(f.convert("RGBA"), dtype=np.uint8) for f in frames]
    if spec.kind == "none":
        used = _colours_by_use(arrays)
        truncated = len(used) > REPORTED_PALETTE_CAP
        palette = [rgb_to_hex(c) for c in unpack_rgb(used[:REPORTED_PALETTE_CAP])]
        return frames, palette, int(len(used)), truncated

    if spec.kind in ("named", "custom"):
        pal_rgb = palette_to_array(spec.colors)
        reported = list(spec.colors)
    else:
        opaque_parts = [a[a[..., 3] > 0, :3] for a in arrays]
        opaque_parts = [p for p in opaque_parts if len(p)]
        if opaque_parts:
            all_rgb = np.concatenate(opaque_parts, axis=0)
            pal_rgb = auto_palette(all_rgb, int(spec.count or 32), None)
        else:
            pal_rgb = np.zeros((0, 3), dtype=np.uint8)
        reported = [rgb_to_hex(c) for c in pal_rgb]

    if len(pal_rgb) > 0:
        mapped_frames: list[Image.Image] = []
        for a in arrays:
            op = a[..., 3] > 0
            if op.any():
                idx = nearest_palette_index(a[op, :3], pal_rgb)
                a[op, :3] = pal_rgb[idx]
            a[~op] = 0
            mapped_frames.append(Image.fromarray(a))
        frames = mapped_frames
        arrays = [np.asarray(f) for f in frames]

    used = _colours_by_use(arrays)
    return frames, reported, int(len(used)), False


def bake_spritesheet(image: Image.Image, opts: SpritesheetOptions) -> SpritesheetResult:
    """Run the full sprite-sheet baking pipeline on ``image``."""
    cols, rows = _pair(opts.grid, "grid")
    fw, fh = _pair(opts.frame_size, "frame_size")
    if opts.mirror_mode not in MIRROR_MODES:
        raise ValueError(f"mirror_mode must be one of {', '.join(MIRROR_MODES)}")
    if opts.mirror_mode == "5to8" and cols * rows < 5:
        raise ValueError(
            f"mirror_mode='5to8' requires a grid with at least 5 cells, got {cols}x{rows}"
        )
    if opts.background not in BACKGROUND_MODES:
        raise ValueError(f"background must be one of {', '.join(BACKGROUND_MODES)}")
    if opts.align not in ALIGN_MODES:
        raise ValueError(f"align must be one of {', '.join(ALIGN_MODES)}")
    if opts.resample not in RESAMPLE_MODES:
        raise ValueError(f"resample must be one of {', '.join(RESAMPLE_MODES)}")
    if opts.output_cols is not None:
        if not _is_int(opts.output_cols) or int(opts.output_cols) < 1:
            raise ValueError("output_cols must be a positive integer")
    if opts.max_output_side is not None:
        if not _is_int(opts.max_output_side) or int(opts.max_output_side) < 1:
            raise ValueError("max_output_side must be a positive integer")

    spec = _resolve_palette(opts.palette)

    n_out = 8 if opts.mirror_mode == "5to8" else cols * rows
    out_cols = (
        int(opts.output_cols)
        if opts.output_cols is not None
        else (4 if opts.mirror_mode == "5to8" else cols)
    )
    out_rows = math.ceil(n_out / out_cols)
    atlas_w, atlas_h = out_cols * fw, out_rows * fh
    if opts.max_output_side is not None and max(atlas_w, atlas_h) > int(opts.max_output_side):
        raise SpritesheetTooLargeError(atlas_w, atlas_h, int(opts.max_output_side))

    raw_cells = slice_grid(image, cols, rows)
    active_cells = raw_cells[:5] if opts.mirror_mode == "5to8" else raw_cells

    processed: list[Image.Image] = []
    for cell in active_cells:
        cleaned = flood_key_background(
            cell,
            mode=opts.background,
            bg_threshold=opts.bg_threshold,
            fringe_threshold=opts.fringe_threshold,
            color_tolerance=opts.color_tolerance,
            clear_enclosed_white=opts.clear_enclosed_white,
        )
        if opts.clean_stray:
            cleaned = filter_stray_components(
                cleaned,
                min_component_ratio=opts.min_component_ratio,
                border_bleed_ratio=opts.border_bleed_ratio,
            )
        fitted = fit_and_align(
            cleaned,
            frame_size=(fw, fh),
            fit_size=opts.fit_size,
            align=opts.align,
            bottom_padding=opts.bottom_padding,
            resample=opts.resample,
            alpha_threshold=opts.alpha_threshold,
            hard_alpha=opts.hard_alpha,
        )
        processed.append(fitted)

    # Lock shared palette across unique frames before mirroring so mirrored
    # counterparts stay byte-identical and don't double-weight auto:N median cut.
    processed, palette, colors_used, palette_truncated = _apply_shared_palette(
        processed, spec
    )

    if opts.mirror_mode == "5to8":
        frames = mirror_5to8(processed)
        names = list(DIRECTION_8_NAMES)
    else:
        frames = processed
        names = [f"cell_{i:02d}" for i in range(len(frames))]

    atlas = Image.new("RGBA", (atlas_w, atlas_h), (0, 0, 0, 0))
    anchor_x = fw // 2
    anchor_y = fh - int(opts.bottom_padding) if opts.align == "bottom" else fh // 2

    layout: list[SpritesheetCell] = []
    for i, (frame, name) in enumerate(zip(frames, names)):
        col = i % out_cols
        row = i // out_cols
        px, py = col * fw, row * fh
        atlas.paste(frame, (px, py))
        opaque_pixels = int((np.asarray(frame)[..., 3] > 0).sum())
        layout.append(
            SpritesheetCell(
                index=i,
                name=name,
                x=px,
                y=py,
                width=fw,
                height=fh,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                opaque_pixels=opaque_pixels,
            )
        )

    metadata = {
        "source_grid": [cols, rows],
        "output_grid": [out_cols, out_rows],
        "frame_count": len(frames),
        "mirror_mode": opts.mirror_mode,
        "background": opts.background,
        "align": opts.align,
        "bottom_padding": int(opts.bottom_padding),
        "resample": opts.resample,
        "hard_alpha": bool(opts.hard_alpha),
        "alpha_threshold": int(opts.alpha_threshold),
        "clean_stray": bool(opts.clean_stray),
        "clear_enclosed_white": bool(opts.clear_enclosed_white),
        "palette_kind": spec.kind,
        "colors_used": colors_used,
        "palette_truncated": palette_truncated,
    }
    return SpritesheetResult(
        atlas=atlas,
        cells=frames,
        layout=layout,
        frame_size=(fw, fh),
        atlas_size=(atlas_w, atlas_h),
        output_grid=(out_cols, out_rows),
        palette=palette,
        metadata=metadata,
    )
