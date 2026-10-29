"""Pixelize: turn soft, off-grid "pixel art" into real pixel art.

Pillow + numpy only. The fixed operation order is:

1. background removal (full resolution)
2. grid snap (detect or take the grid, majority-vote each cell)
3. palette mapping (nearest colour in OKLab)
4. alpha binarize (output alpha is only ever 0 or 255)
5. cleanup (orphan pixels; by default only when the grid was snapped)
6. outline
7. seamless wrap
8. nearest-neighbour preview scale

Sheets (``sheet=(C, R)``) share one grid and one palette; cleanup, outline and
seamless run per sheet cell, so nothing bleeds between cells.

Never pass ``mode=`` to ``Image.fromarray`` here: Pillow 13 removed it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from muse.modalities.image_pixelart.palettes import (
    PaletteSpec,
    auto_palette,
    hex_to_rgb,
    nearest_palette_index,
    pack_rgb,
    palette_to_array,
    quantize_labels,
    rgb_to_hex,
    srgb_to_oklab,
    unpack_rgb,
)

MIN_PERIOD = 2
MAX_PERIOD = 64
DETECT_THRESHOLD = 0.5
EDGE_SATURATION = 64.0  # premultiplied RGBA abs-difference units
BACKGROUND_UNIFORM_FRACTION = 0.9
BACKGROUND_DELTA_E = 6.0  # OKLab distance x 100
OUTLINE_HUE_DELTA = 0.25  # OKLab (a, b) distance
MAX_PREVIEW_SCALE = 16
REPORTED_PALETTE_CAP = 256
VOTE_LABELS = 256  # fine labels for the per-cell vote (palette auto / none)
VOTE_SAMPLE = 1 << 18  # pixels sampled to build those labels
LUT_BITS = 5  # nearest-colour lookup table resolution per channel
CORE_WEIGHT = 1.0  # vote weight of a cell's middle half
RING_WEIGHT = 0.05  # vote weight of its blurred outer quarters

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_BACKGROUND_WORDS = ("auto", "none", "transparent-keep")
_OUTLINE_WORDS = ("none", "dark")


@dataclass(frozen=True)
class GridInfo:
    """The art grid: ``cell`` source pixels per art pixel; the first full
    cell boundary sits at ``(offset_x, offset_y)`` (each in ``[0, cell)``)."""

    cell: int
    offset_x: int
    offset_y: int
    confidence: float
    detected: bool


@dataclass(frozen=True)
class PixelizeOptions:
    """Options for ``pixelize``.

    - ``grid``: None auto-detects; an integer forces that cell size; 1 means
      no grid snap and no downscale.
    - ``background``: ``auto`` keys a uniform border colour (flood from the
      border, only when >= 90% of border pixels are one colour); ``#rrggbb``
      is a global chroma key; ``none`` and ``transparent-keep`` are synonyms:
      nothing is keyed and the existing alpha is kept (binarized at
      ``alpha_threshold``).
    - ``cleanup``: None (default) cleans orphan pixels only when the grid was
      snapped (``grid.cell > 1``), so deliberate 1 px detail in true 1:1 art
      survives; True / False force it on / off.
    - ``max_preview_side``: when set, a preview whose long side (art size x
      ``preview_scale``) would exceed it raises ``PreviewTooLargeError``
      before the preview (or, when the art size is known up front, any
      other work) is allocated. The route sets it to
      ``limits.pixelart_max_input_side``.
    """

    grid: int | None  # None = auto-detect; 1 = no grid snap / downscale
    target_size: tuple[int, int] | None
    palette: PaletteSpec
    background: str  # auto / none / transparent-keep / #rrggbb
    alpha_threshold: int = 128
    cleanup: bool | None = None
    outline: str = "none"  # none / dark / #rrggbb
    seamless: bool = False
    sheet: tuple[int, int] | None = None  # (columns, rows)
    preview_scale: int = 1
    max_preview_side: int | None = None


class PreviewTooLargeError(ValueError):
    """The nearest-neighbour preview would exceed ``max_preview_side``."""

    def __init__(self, width: int, height: int, scale: int, cap: int) -> None:
        self.width, self.height, self.scale, self.cap = width, height, scale, cap
        super().__init__(
            f"preview {width * scale}x{height * scale} ({width}x{height} art at "
            f"preview_scale {scale}) exceeds max side {cap}"
        )


@dataclass
class PixelizeResult:
    image: Image.Image  # RGBA, preview-scaled
    grid: GridInfo
    size: tuple[int, int]  # art pixels (before preview scale)
    palette: list[str]
    background_removed: bool
    metadata: dict = field(default_factory=dict)


# ------------------------------------------------------------------ validation


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _pair(v, name: str) -> tuple[int, int]:
    if not (isinstance(v, (tuple, list)) and len(v) == 2 and all(_is_int(x) for x in v)):
        raise ValueError(f"{name} must be a pair of integers")
    a, b = int(v[0]), int(v[1])
    if a < 1 or b < 1:
        raise ValueError(f"{name} values must be at least 1")
    return a, b


def _validate(opts: PixelizeOptions, width: int, height: int) -> None:
    if width < 1 or height < 1:
        raise ValueError("image is empty")
    if not isinstance(opts.palette, PaletteSpec):
        raise ValueError("palette must be a PaletteSpec (see parse_palette)")
    if opts.grid is not None and (not _is_int(opts.grid) or opts.grid < 1):
        raise ValueError("grid must be 'auto' or an integer of at least 1")
    if opts.grid is not None and (opts.grid > width or opts.grid > height):
        raise ValueError(f"grid {int(opts.grid)} exceeds image size {width}x{height}")
    if opts.target_size is not None:
        tw, th = _pair(opts.target_size, "target_size")
        if opts.grid is not None:
            raise ValueError("grid and target_size are mutually exclusive")
        if tw > width or th > height:
            raise ValueError(
                f"target_size {tw}x{th} is larger than the image ({width}x{height})"
            )
    if opts.sheet is not None:
        c, r = _pair(opts.sheet, "sheet")
        if opts.target_size is not None:
            raise ValueError("sheet and target_size are mutually exclusive")
        if c > width or r > height:
            raise ValueError("sheet has more cells than the image has pixels")
    bg = opts.background
    if not isinstance(bg, str) or (bg not in _BACKGROUND_WORDS and not _HEX_RE.match(bg)):
        raise ValueError("background must be auto, none, transparent-keep or #rrggbb")
    if opts.cleanup is not None and not isinstance(opts.cleanup, bool):
        raise ValueError("cleanup must be true, false or unset")
    if not _is_int(opts.alpha_threshold) or not 0 <= opts.alpha_threshold <= 255:
        raise ValueError("alpha_threshold must be an integer in 0..255")
    ol = opts.outline
    if not isinstance(ol, str) or (ol not in _OUTLINE_WORDS and not _HEX_RE.match(ol)):
        raise ValueError("outline must be none, dark or #rrggbb")
    if not _is_int(opts.preview_scale) or not 1 <= opts.preview_scale <= MAX_PREVIEW_SCALE:
        raise ValueError(f"preview_scale must be an integer in 1..{MAX_PREVIEW_SCALE}")
    cap = opts.max_preview_side
    if cap is not None and (not _is_int(cap) or cap < 1):
        raise ValueError("max_preview_side must be a positive integer or unset")


def _check_preview(art_w: int, art_h: int, opts: PixelizeOptions) -> None:
    """Raise ``PreviewTooLargeError`` when the preview of ``art_w`` x ``art_h``
    art would exceed ``opts.max_preview_side`` on its long side."""
    cap, s = opts.max_preview_side, int(opts.preview_scale)
    if cap is not None and max(art_w, art_h) * s > cap:
        raise PreviewTooLargeError(int(art_w), int(art_h), s, int(cap))


# --------------------------------------------------------------- grid detection


def _opaque(rgba: np.ndarray, threshold: int) -> np.ndarray:
    """Pixels that count as opaque: alpha >= threshold. Alpha 0 is always
    transparent, even with threshold 0."""
    a = rgba[..., 3]
    return (a >= threshold) & (a > 0)


def _premultiplied_channels(rgba: np.ndarray):
    a = rgba[..., 3].astype(np.float32) / 255.0
    for c in range(3):
        yield rgba[..., c].astype(np.float32) * a
    yield rgba[..., 3].astype(np.float32)


def _edge_profiles(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column and row profiles of colour-change strength.

    Per pixel, the absolute premultiplied-RGBA difference to the next pixel;
    non-maximum suppression along the axis collapses a blurred edge onto its
    centre; each edge saturates at ``EDGE_SATURATION`` so a few strong
    silhouette or sheet-cell edges cannot outweigh the many cell edges; the
    result is summed across the other axis. ``px[j]`` measures the boundary
    between columns j and j+1.
    """
    return _axis_profile(rgba, axis=1), _axis_profile(rgba, axis=0)


def _axis_profile(rgba: np.ndarray, axis: int) -> np.ndarray:
    n = rgba.shape[axis]
    if n < 2:
        return np.zeros(0, dtype=np.float32)
    g = None
    for ch in _premultiplied_channels(rgba):
        d = np.abs(np.diff(ch, axis=axis))
        g = d if g is None else np.add(g, d, out=g)
    if axis == 0:
        g = g.T
    return np.minimum(_nms(g), EDGE_SATURATION).sum(axis=0)


def _nms(g: np.ndarray) -> np.ndarray:
    """Keep only local maxima along axis 1 (>= left, > right)."""
    if g.size == 0:
        return g
    left = np.empty_like(g)
    left[:, 0] = -1.0
    left[:, 1:] = g[:, :-1]
    right = np.empty_like(g)
    right[:, -1] = -1.0
    right[:, :-1] = g[:, 1:]
    return np.where((g >= left) & (g > right), g, 0.0)


def _autocorr(x: np.ndarray, maxlag: int) -> np.ndarray:
    """Pearson correlation of the profile with itself at lags 0..maxlag."""
    out = np.zeros(maxlag + 1)
    if maxlag < 1:
        return out
    x = x.astype(np.float64) - float(x.mean())
    for lag in range(1, maxlag + 1):
        u, v = x[:-lag], x[lag:]
        d = math.sqrt(float((u * u).sum()) * float((v * v).sum()))
        out[lag] = float((u * v).sum()) / d if d > 0 else 0.0
    return out


def _axis_maxlag(n: int) -> int:
    # A period must repeat at least three times along the axis.
    return min(MAX_PERIOD, n // 3)


def _period_scores(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Combined autocorrelation score per period (index = period)."""
    mx, my = _axis_maxlag(len(px)), _axis_maxlag(len(py))
    m = max(mx, my)
    s = np.full(m + 1, -1.0)
    if m < MIN_PERIOD:
        return s
    cx, cy = _autocorr(px, mx), _autocorr(py, my)
    for p in range(MIN_PERIOD, m + 1):
        vals = [c[p] for c, mm in ((cx, mx), (cy, my)) if p <= mm]
        s[p] = float(np.mean(vals))
    return s


def _comb(s: np.ndarray, p: int) -> float:
    """Mean score over the first multiples of p: a true period scores high at
    every multiple, a non-period alternates."""
    return float(np.mean(s[p::p][:8]))


def _phase(profile: np.ndarray, p: int) -> int:
    if p <= 1 or len(profile) < p:
        return 0
    r = np.arange(len(profile)) % p
    fold = np.bincount(r, weights=profile, minlength=p) / np.bincount(r, minlength=p)
    # Profile index j is the boundary between j and j+1: a cell starts at j+1.
    return (int(np.argmax(fold)) + 1) % p


def _uniformity(rgba: np.ndarray, k: int, ox: int, oy: int) -> float:
    """Fraction of pixels close to their cell's median colour, over sampled
    non-empty full cells. Upscaled art is near 1; 1:1 art with periodic
    detail (a 1 px mortar line every 8 px) is not."""
    h, w = rgba.shape[:2]
    gw, gh = (w - ox) // k, (h - oy) // k
    if gw < 1 or gh < 1:
        return 0.0
    rows = np.unique(np.linspace(0, gh - 1, min(gh, 48)).astype(int))
    cols = np.unique(np.linspace(0, gw - 1, min(gw, 48)).astype(int))
    block = rgba[oy:oy + gh * k, ox:ox + gw * k].reshape(gh, k, gw, k, 4)
    block = block[rows][:, :, cols].astype(np.float32)
    a = block[..., 3:4] / 255.0
    pm = np.concatenate([block[..., :3] * a, block[..., 3:4]], axis=-1)
    nonempty = pm[..., 3].max(axis=(1, 3)) > 0
    if not nonempty.any():
        return 0.0
    med = np.median(pm, axis=(1, 3), keepdims=True)
    dist = np.sqrt(((pm - med) ** 2).sum(-1))  # (r, k, c, k)
    close = (dist <= 48.0).mean(axis=(1, 3))
    return float(close[nonempty].mean())


def _confidence(s: np.ndarray, rgba: np.ndarray, p: int, ox: int, oy: int) -> float:
    if p >= len(s) or p < MIN_PERIOD:
        return 0.0
    comb = max(0.0, min(1.0, _comb(s, p)))
    u = _uniformity(rgba, p, ox, oy)
    return float(comb * max(0.0, min(1.0, (u - 0.7) / 0.2)))


def _detect(rgba: np.ndarray) -> GridInfo:
    """Grid detection.

    1. Column/row profiles of edge strength (see ``_edge_profiles``).
    2. Autocorrelation of each profile; the two axes are averaged per period.
    3. A true period scores high at all its multiples, so the cell is the
       smallest local-maximum period whose harmonic comb (``_comb``) is within
       75% of the best comb; offsets are the strongest phase of each profile.
    4. Confidence = comb x within-cell uniformity: snapping 1:1 art with
       periodic 1 px detail would destroy that detail, and such cells are not
       uniform. Below ``DETECT_THRESHOLD`` the image is treated as 1:1.
    """
    px, py = _edge_profiles(rgba)
    s = _period_scores(px, py)
    m = len(s) - 1
    if m < MIN_PERIOD:
        return GridInfo(cell=1, offset_x=0, offset_y=0, confidence=0.0, detected=False)
    combs = {p: _comb(s, p) for p in range(MIN_PERIOD, m + 1)}
    best = max(combs.values())
    chosen = max(combs, key=combs.get)
    if best > 0:
        for p in range(MIN_PERIOD, m + 1):
            lo = s[p - 1] if p - 1 >= MIN_PERIOD else -np.inf
            hi = s[p + 1] if p + 1 <= m else -np.inf
            if s[p] >= lo and s[p] >= hi and combs[p] >= 0.75 * best:
                chosen = p
                break
    ox, oy = _phase(px, chosen), _phase(py, chosen)
    conf = round(_confidence(s, rgba, chosen, ox, oy), 4)
    if conf >= DETECT_THRESHOLD:
        return GridInfo(cell=chosen, offset_x=ox, offset_y=oy, confidence=conf, detected=True)
    return GridInfo(cell=1, offset_x=0, offset_y=0, confidence=conf, detected=False)


def _explicit_grid(rgba: np.ndarray, k: int) -> GridInfo:
    if k == 1:
        return GridInfo(cell=1, offset_x=0, offset_y=0, confidence=1.0, detected=False)
    px, py = _edge_profiles(rgba)
    s = _period_scores(px, py)
    ox, oy = _phase(px, k), _phase(py, k)
    conf = round(_confidence(s, rgba, k, ox, oy), 4)
    return GridInfo(cell=k, offset_x=ox, offset_y=oy, confidence=conf, detected=False)


def detect_grid(img: Image.Image) -> GridInfo:
    """Detect the art grid of an image (square cells, 2..64 px).

    Returns ``cell=1, detected=False`` when no period is confident, so the
    image is treated as already 1:1.
    """
    return _detect(np.asarray(img.convert("RGBA")))


# ------------------------------------------------------------------ background


def _delta_e(lab: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.sqrt(((lab - ref) ** 2).sum(-1)) * 100.0


def _delta_e_to(rgb: np.ndarray, ref_lab: np.ndarray) -> np.ndarray:
    """OKLab distance x 100 from every pixel of ``rgb`` (..., 3) uint8 to one
    reference, computed once per distinct colour (keeps memory flat on big
    inputs)."""
    keys = pack_rgb(rgb)
    uniq, inv = np.unique(keys.ravel(), return_inverse=True)
    d = _delta_e(srgb_to_oklab(unpack_rgb(uniq)), ref_lab).astype(np.float32)
    return d[inv.reshape(-1)].reshape(keys.shape)


def _row_runs(cand: np.ndarray) -> np.ndarray:
    """Label each horizontal run of candidate pixels 1..R (0 = not a candidate)."""
    h, w = cand.shape
    starts = cand.copy()
    starts[:, 1:] &= ~cand[:, :-1]
    ids = np.cumsum(starts.ravel(), dtype=np.int64).reshape(h, w)
    ids[~cand] = 0
    return ids


def _run_edges(ids: np.ndarray, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of vertically touching runs, one pair per overlap segment."""
    both = cand[:-1] & cand[1:]
    top, bot = ids[:-1], ids[1:]
    first = both.copy()
    first[:, 1:] &= ~(both[:, :-1] & (top[:, 1:] == top[:, :-1])
                      & (bot[:, 1:] == bot[:, :-1]))
    return top[first], bot[first]


def _flood_with_rounds(seed: np.ndarray, cand: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected flood of ``cand`` from ``seed``; also returns the number of
    hooking rounds (a complexity proxy for tests).

    Connected-component labelling on the graph of horizontal runs: every
    round hooks each component root to the smallest neighbouring label, then
    pointer jumping flattens the forest. Seeded runs start at label 0, so a
    run is filled iff its component's minimum label is 0. Rounds are
    O(log runs) in practice (each round merges every non-minimal component),
    each O(runs + touching pairs), so a winding maze no longer costs one
    full-image pass per corridor turn.
    """
    ids = _row_runs(cand)
    n_runs = int(ids.max())
    if n_runs == 0:
        return np.zeros(cand.shape, dtype=bool), 0
    a, b = _run_edges(ids, cand)
    label = np.arange(n_runs + 1, dtype=np.int64)
    label[np.unique(ids[seed & cand])] = 0
    rounds = 0
    while True:
        la, lb = label[a], label[b]
        diff = la != lb
        if not diff.any():
            break
        rounds += 1
        la, lb = la[diff], lb[diff]
        low = np.minimum(la, lb)
        np.minimum.at(label, la, low)
        np.minimum.at(label, lb, low)
        while True:  # pointer jumping until every run points at its root
            nxt = label[label]
            if np.array_equal(nxt, label):
                break
            label = nxt
    filled = (label[ids] == 0) & cand
    return filled, rounds


def _flood(seed: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """4-connected flood of ``cand`` from ``seed`` (see ``_flood_with_rounds``)."""
    return _flood_with_rounds(seed, cand)[0]


def _ring_mask(h: int, w: int, regions) -> np.ndarray:
    ring = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in regions:
        ring[y0, x0:x1] = ring[y1 - 1, x0:x1] = True
        ring[y0:y1, x0] = ring[y0:y1, x1 - 1] = True
    return ring


def _remove_background(rgba: np.ndarray, regions, opts: PixelizeOptions) -> dict:
    """Key the background in place (alpha -> 0). Returns metadata."""
    bg = opts.background
    if bg in ("none", "transparent-keep"):
        return {"removed": False}
    opaque = _opaque(rgba, opts.alpha_threshold)
    if bg != "auto":
        ref = srgb_to_oklab(np.array(hex_to_rgb(bg)))
        mask = opaque & (_delta_e_to(rgba[..., :3], ref) < BACKGROUND_DELTA_E)
        rgba[mask] = 0
        return {"removed": bool(mask.any()), "color": bg.lower(), "mode": "chroma_key"}
    h, w = opaque.shape
    ring = _ring_mask(h, w, regions)
    n_ring = int(ring.sum())
    ring_op = ring & opaque
    if n_ring == 0 or ring_op.sum() < BACKGROUND_UNIFORM_FRACTION * n_ring:
        return {"removed": False}
    ring_lab = srgb_to_oklab(rgba[ring_op][:, :3])
    ref = np.median(ring_lab, axis=0)
    close = _delta_e(ring_lab, ref) < BACKGROUND_DELTA_E
    if close.any():
        ref = ring_lab[close].mean(axis=0)
        close = _delta_e(ring_lab, ref) < BACKGROUND_DELTA_E
    if close.sum() < BACKGROUND_UNIFORM_FRACTION * n_ring:
        return {"removed": False}
    inl = np.zeros_like(ring_op)
    inl[ring_op] = close
    cand = opaque & (_delta_e_to(rgba[..., :3], ref) < BACKGROUND_DELTA_E)
    mask = _flood(inl, cand)
    color = rgb_to_hex(np.rint(rgba[inl][:, :3].mean(axis=0)))
    rgba[mask] = 0
    return {"removed": bool(mask.any()), "mode": "border", "color": color}


# ------------------------------------------------------------------- grid snap


def _sheet_regions(width: int, height: int, sheet) -> tuple[list[tuple[int, int]],
                                                            list[tuple[int, int]]]:
    """Source spans of the sheet's columns and rows."""
    c, r = sheet if sheet is not None else (1, 1)
    xs = [(round(i * width / c), round((i + 1) * width / c)) for i in range(c)]
    ys = [(round(i * height / r), round((i + 1) * height / r)) for i in range(r)]
    return xs, ys


def _grid_spans(start: int, end: int, k: int, offset: int) -> list[tuple[int, int]]:
    """Cells of size k over [start, end) aligned to the global ``offset``.

    A partial edge cell is kept only when it covers at least half a cell.
    """
    if k == 1:
        return [(x, x + 1) for x in range(start, end)]
    half = math.ceil(k / 2)
    b0 = start + ((offset - start) % k)
    spans: list[tuple[int, int]] = []
    if b0 - start >= half:
        spans.append((start, b0))
    x = b0
    while x + k <= end:
        spans.append((x, x + k))
        x += k
    if end - x >= half:
        spans.append((x, end))
    return spans


def _target_spans(n: int, t: int) -> list[tuple[int, int]]:
    return [(i * n // t, (i + 1) * n // t) for i in range(t)]


def _axis_map(n: int, region_spans: list[list[tuple[int, int]]]) -> tuple[
        np.ndarray, np.ndarray, int, list[int]]:
    """Source index -> output index (-1 = unused), with every sheet region
    trimmed to the same art-pixel count.

    Returns (map, vote weight per source index, per-region count, the output
    index where each region starts). The weight is ``CORE_WEIGHT`` in the
    middle half of each span and ``RING_WEIGHT`` in its outer quarters, where
    blur from the neighbouring cell lives.
    """
    per = min(len(s) for s in region_spans)
    m = np.full(n, -1, dtype=np.int64)
    wt = np.zeros(n, dtype=np.float64)
    starts = []
    out = 0
    for spans in region_spans:
        starts.append(out)
        for a, b in spans[:per]:
            m[a:b] = out
            margin = (b - a) // 4
            wt[a:b] = RING_WEIGHT
            wt[a + margin:b - margin] = CORE_WEIGHT
            out += 1
    return m, wt, per, starts


def _mode_per_cell(cid: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                   n_cells: int, n_labels: int) -> np.ndarray:
    """Label with the largest vote weight per cell (ties -> lowest label);
    -1 for cells with no votes."""
    winner = np.full(n_cells, -1, dtype=np.int64)
    if cid.size == 0:
        return winner
    keys = cid.astype(np.int64) * n_labels + labels
    if n_cells * n_labels <= (1 << 24):
        votes = np.bincount(keys, weights=weights,
                            minlength=n_cells * n_labels).reshape(n_cells, n_labels)
        has = np.bincount(cid, minlength=n_cells) > 0
        winner[has] = votes[has].argmax(axis=1)
        return winner
    uniq, inv = np.unique(keys, return_inverse=True)
    votes = np.bincount(inv.reshape(-1), weights=weights)
    cells, labs = uniq // n_labels, uniq % n_labels
    order = np.lexsort((labs, -votes, cells))
    cells, labs = cells[order], labs[order]
    first = np.ones(len(cells), dtype=bool)
    first[1:] = cells[1:] != cells[:-1]
    winner[cells[first]] = labs[first]
    return winner


def _vote_labels(px: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """Approximate nearest-palette labels for the per-cell vote, in O(N).

    A pixel that is exactly a palette colour gets that colour's index; any
    other pixel gets the palette index OKLab-nearest to the centre of its
    32x32x32 RGB bin (a lookup table), which can differ from the pixel's own
    nearest colour near a bin edge. With a fixed palette the winning label
    IS the cell's colour (the LUT picks the voted cell colour; no exact
    re-mapping follows). With the fine median-cut labels of ``_fine_labels``
    the label only groups pixels; the cell colour is the mean of its winning
    pixels, later mapped exactly by ``nearest_palette_index``.
    """
    keys = pack_rgb(px)
    pal_keys = pack_rgb(pal)
    order = np.argsort(pal_keys, kind="stable")
    sorted_keys = pal_keys[order]
    pos = np.searchsorted(sorted_keys, keys)
    np.minimum(pos, len(sorted_keys) - 1, out=pos)
    exact = sorted_keys[pos] == keys
    del keys
    step = 1 << (8 - LUT_BITS)
    axis = np.arange(1 << LUT_BITS) * step + (step - 1) / 2.0
    centres = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    lut = nearest_palette_index(centres, pal).reshape(-1).astype(np.int16)
    shift = 8 - LUT_BITS
    bins = (px[:, 0] >> shift).astype(np.uint16) << (2 * LUT_BITS)
    bins |= (px[:, 1] >> shift).astype(np.uint16) << LUT_BITS
    bins |= (px[:, 2] >> shift).astype(np.uint16)
    labels = lut[bins]
    labels[exact] = order[pos[exact]]
    return labels


def _fine_labels(px: np.ndarray) -> np.ndarray:
    """Fine colour labels so near-identical colours vote together: exact
    colours when there are at most ``VOTE_LABELS`` of them, otherwise a
    median-cut palette of a strided sample."""
    sample = px[:: max(1, len(px) // VOTE_SAMPLE)]
    if len(np.unique(pack_rgb(sample))) <= VOTE_LABELS:
        uniq, inv = np.unique(pack_rgb(px), return_inverse=True)
        if len(uniq) <= VOTE_LABELS:
            return inv.reshape(-1)
    _, pal = quantize_labels(sample, VOTE_LABELS)
    return _vote_labels(px, pal)


def _snap(rgba: np.ndarray, xmap: np.ndarray, ymap: np.ndarray, xw: np.ndarray,
          yw: np.ndarray, gw: int, gh: int, opts: PixelizeOptions,
          pal: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Majority-vote each cell. Returns (cell RGB uint8 (gh, gw, 3), opaque mask).

    A cell is transparent when most of its pixels have alpha below the
    threshold. Colours vote on the quantized image (palette indices for a
    fixed palette, a fine median-cut labelling otherwise), so near-identical
    soft pixels vote together; the cell's core outvotes its blurred rim.
    """
    rgb = rgba[..., :3]
    opaque = _opaque(rgba, opts.alpha_threshold)
    xs, ys = np.flatnonzero(xmap >= 0), np.flatnonzero(ymap >= 0)
    if len(xs) == gw and len(ys) == gh:  # one source pixel per cell: gather
        return rgb[np.ix_(ys, xs)].copy(), opaque[np.ix_(ys, xs)].copy()
    n_cells = gw * gh
    valid = (ymap[:, None] >= 0) & (xmap[None, :] >= 0)
    cid_full = (ymap[:, None] * gw + xmap[None, :]).astype(np.int32 if n_cells < 2**31 else np.int64)
    total = np.bincount(cid_full[valid], minlength=n_cells)
    sel = valid & opaque
    cid = cid_full[sel]
    weights = (yw[:, None].astype(np.float32) * xw[None, :].astype(np.float32))[sel]
    n_op = np.bincount(cid, minlength=n_cells)
    cell_opaque = (n_op > 0) & (2 * n_op >= total)
    px = rgb[sel]
    out = np.zeros((n_cells, 3), dtype=np.uint8)
    if len(px):
        if pal is not None:
            labels = _vote_labels(px, pal)
            winner = _mode_per_cell(cid, labels, weights, n_cells, len(pal))
            ok = winner >= 0
            out[ok] = pal[winner[ok]]
        else:
            labels = _fine_labels(px)
            winner = _mode_per_cell(cid, labels, weights, n_cells, VOTE_LABELS)
            hit = labels == winner[cid]
            wh = weights[hit]
            sums = np.stack([np.bincount(cid[hit], weights=px[hit, c] * wh,
                                         minlength=n_cells) for c in range(3)], axis=-1)
            wsum = np.bincount(cid[hit], weights=wh, minlength=n_cells)
            ok = wsum > 0
            out[ok] = np.clip(np.rint(sums[ok] / wsum[ok, None]), 0, 255).astype(np.uint8)
    return out.reshape(gh, gw, 3), cell_opaque.reshape(gh, gw)


# ----------------------------------------------------- cleanup / outline / wrap


def _keys(img: np.ndarray) -> np.ndarray:
    """RGBA uint8 -> uint32 key; every transparent pixel is key 0."""
    a = img.astype(np.uint32)
    k = (a[..., 0] << 24) | (a[..., 1] << 16) | (a[..., 2] << 8) | a[..., 3]
    return np.where(img[..., 3] == 0, 0, k).astype(np.uint32)


_NEIGHBOURS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def _cleanup(img: np.ndarray, wrap: bool, only: np.ndarray | None = None) -> int:
    """In place: an opaque pixel whose 8 neighbours all share one other colour
    (transparent counts as a colour) takes that colour. Without ``wrap`` the
    image's edge pixels are left alone; with ``wrap``, ``only`` limits the
    pixels that may change. Returns the number changed."""
    h, w = img.shape[:2]
    if h < 3 or w < 3:
        return 0
    key = _keys(img)
    if wrap:
        nbs = [np.roll(key, (-dy, -dx), axis=(0, 1)) for dy, dx in _NEIGHBOURS]
        centre = key
        src = np.roll(img, (1, 0), axis=(0, 1))  # the (-1, 0) neighbour's pixel
    else:
        nbs = [key[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx] for dy, dx in _NEIGHBOURS]
        centre = key[1:-1, 1:-1]
        src = img[:-2, 1:-1]
    first = nbs[0]
    same = np.ones(first.shape, dtype=bool)
    for n in nbs[1:]:
        same &= n == first
    change = same & (centre != 0) & (first != centre)
    if wrap and only is not None:
        change &= only
    if wrap:
        img[change] = src[change]
    else:
        inner = img[1:-1, 1:-1]
        inner[change] = src[change]
    return int(change.sum())


def _dark_outline_colours(nb_rgb: np.ndarray, cands: np.ndarray) -> tuple[np.ndarray,
                                                                          np.ndarray]:
    """Outline colour (and its OKLab L) for each neighbour colour: the darkest
    candidate darker than the neighbour within ``OUTLINE_HUE_DELTA`` of its
    (a, b), else the darkest candidate overall."""
    lab_c = srgb_to_oklab(cands)
    darkest = int(np.argmin(lab_c[:, 0]))
    uniq, inv = np.unique(pack_rgb(nb_rgb), return_inverse=True)
    lab_u = srgb_to_oklab(unpack_rgb(uniq))
    pick = np.empty(len(uniq), dtype=np.intp)
    chunk = max(1, (1 << 20) // len(lab_c))
    for i in range(0, len(uniq), chunk):
        u = lab_u[i:i + chunk]
        ab = np.sqrt(((u[:, None, 1:] - lab_c[None, :, 1:]) ** 2).sum(-1))
        ok = (ab <= OUTLINE_HUE_DELTA) & (lab_c[None, :, 0] < u[:, None, 0])
        masked = np.where(ok, lab_c[None, :, 0], np.inf)
        best = np.argmin(masked, axis=1)
        pick[i:i + chunk] = np.where(ok.any(axis=1), best, darkest)
    chosen = pick[inv.reshape(-1)]
    return cands[chosen], lab_c[chosen, 0]


def _outline(img: np.ndarray, mode: str, cands: np.ndarray) -> int:
    """In place: every transparent pixel 4-adjacent to an opaque one becomes
    the outline colour (``#rrggbb``, or for ``dark`` the darkest candidate
    near the neighbour it borders; with several opaque neighbours the darkest
    result wins). Returns the number of outline pixels."""
    h, w = img.shape[:2]
    op = img[..., 3] == 255
    target = np.zeros((h, w), dtype=bool)
    target[1:] |= op[:-1]
    target[:-1] |= op[1:]
    target[:, 1:] |= op[:, :-1]
    target[:, :-1] |= op[:, 1:]
    target &= ~op
    if not target.any():
        return 0
    if mode != "dark":
        img[target] = (*hex_to_rgb(mode), 255)
        return int(target.sum())
    best_l = np.full((h, w), np.inf)
    best = np.zeros((h, w, 3), dtype=np.uint8)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = np.zeros_like(img)
        nb[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
            img[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
        m = target & (nb[..., 3] == 255)
        if not m.any():
            continue
        cols, ls = _dark_outline_colours(nb[m][:, :3], cands)
        yy, xx = np.nonzero(m)
        better = ls < best_l[yy, xx]
        best_l[yy[better], xx[better]] = ls[better]
        best[yy[better], xx[better]] = cols[better]
    img[target, :3] = best[target]
    img[target, 3] = 255
    return int(target.sum())


def _equalize_edges(img: np.ndarray) -> None:
    """Make column 0 == column W-1 and row 0 == row H-1, taking for each pair
    the colour that is more common in the two pixels either side of the seam
    (ties keep the first edge)."""
    for axis in (1, 0):
        a = img if axis == 1 else img.transpose(1, 0, 2)
        n = a.shape[1]
        if n < 2:
            continue
        k = _keys(a)
        first, last = k[:, 0], k[:, -1]
        near = [k[:, 0], k[:, -1]]
        if n >= 4:
            near += [k[:, 1], k[:, -2]]
        f_count = sum((x == first).astype(int) for x in near)
        l_count = sum((x == last).astype(int) for x in near)
        use_last = l_count > f_count
        pick = np.where(use_last[:, None], a[:, -1], a[:, 0])
        a[:, 0] = pick
        a[:, -1] = pick


def _seamless(img: np.ndarray) -> None:
    _equalize_edges(img)
    # With the edges equal, no edge pixel can be an orphan across the wrap.
    # Equalizing can only create orphans next to the edges, so one wrapped
    # cleanup pass over the 2 px seam band fixes them (and cannot create new
    # ones); the rest of the tile is left to the ``cleanup`` option.
    h, w = img.shape[:2]
    band = np.zeros((h, w), dtype=bool)
    band[:2] = band[-2:] = True
    band[:, :2] = band[:, -2:] = True
    _cleanup(img, wrap=True, only=band)


# ---------------------------------------------------------------------- driver


def _palette_array(spec: PaletteSpec) -> np.ndarray | None:
    if spec.kind in ("named", "custom"):
        return palette_to_array(spec.colors)
    return None


def _colours_by_use(img: np.ndarray) -> np.ndarray:
    """Packed RGB keys of the opaque colours, most used first (ties by value)."""
    keys = pack_rgb(img[img[..., 3] == 255][:, :3])
    if not len(keys):
        return keys
    uniq, counts = np.unique(keys, return_counts=True)
    return uniq[np.lexsort((uniq, -counts))]


def _hex_list(keys: np.ndarray) -> list[str]:
    return [rgb_to_hex(c) for c in unpack_rgb(keys)]


def pixelize(img: Image.Image, opts: PixelizeOptions) -> PixelizeResult:
    """Turn ``img`` into real pixel art (see the module docstring for the
    fixed order). Raises ValueError for invalid options."""
    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    h, w = rgba.shape[:2]
    _validate(opts, w, h)
    if opts.target_size is not None:  # the art size is known before any work
        _check_preview(*(int(v) for v in opts.target_size), opts)
    elif opts.grid == 1 and opts.sheet is None:
        _check_preview(w, h, opts)
    rgba[rgba[..., 3] == 0] = 0

    x_regions, y_regions = _sheet_regions(w, h, opts.sheet)
    regions = [(x0, y0, x1, y1) for (y0, y1) in y_regions for (x0, x1) in x_regions]

    # 1. background
    bg_meta = _remove_background(rgba, regions, opts)

    # 2. grid
    if opts.target_size is not None:
        tw, th = (int(v) for v in opts.target_size)
        grid = GridInfo(cell=max(1, round(max(w / tw, h / th))), offset_x=0, offset_y=0,
                        confidence=0.0, detected=False)
        x_spans, y_spans = [_target_spans(w, tw)], [_target_spans(h, th)]
        source = "target_size"
    else:
        if opts.grid is None:
            grid, source = _detect(rgba), "auto"
        else:
            grid, source = _explicit_grid(rgba, int(opts.grid)), "explicit"
        k = grid.cell
        x_spans = [_grid_spans(a, b, k, grid.offset_x) for a, b in x_regions]
        y_spans = [_grid_spans(a, b, k, grid.offset_y) for a, b in y_regions]
    if min(len(s) for s in x_spans) < 1 or min(len(s) for s in y_spans) < 1:
        raise ValueError("the grid leaves no whole art pixel in the image (or a sheet cell)")
    xmap, xw, cw, x_starts = _axis_map(w, x_spans)
    ymap, yw, ch, y_starts = _axis_map(h, y_spans)
    gw, gh = cw * len(x_spans), ch * len(y_spans)
    _check_preview(gw, gh, opts)  # before the vote and the preview allocate

    # 2 + 3. vote per cell, then map to the palette (nearest in OKLab)
    spec = opts.palette
    pal = _palette_array(spec)
    cell_rgb, cell_op = _snap(rgba, xmap, ymap, xw, yw, gw, gh, opts, pal)
    if spec.kind == "auto":
        pal = auto_palette(cell_rgb, int(spec.count or 32), cell_op)
    if pal is not None and len(pal) and cell_op.any():
        cell_rgb[cell_op] = pal[nearest_palette_index(cell_rgb[cell_op], pal)]

    # 4. alpha binarize
    art = np.zeros((gh, gw, 4), dtype=np.uint8)
    art[cell_op, :3] = cell_rgb[cell_op]
    art[cell_op, 3] = 255

    # Outline colour: snapped into a fixed palette so sheets never drift.
    outline = opts.outline
    if outline not in _OUTLINE_WORDS and spec.kind in ("named", "custom"):
        o = np.array(hex_to_rgb(outline), dtype=np.uint8)
        outline = rgb_to_hex(pal[nearest_palette_index(o[None], pal)[0]])

    # 5-7 per sheet cell
    changed = outlined = 0
    if pal is not None and len(pal):
        outline_cands = pal
    else:  # palette none: the most used colours stand in for the palette
        outline_cands = unpack_rgb(_colours_by_use(art)[:REPORTED_PALETTE_CAP])
    do_cleanup = opts.cleanup if opts.cleanup is not None else grid.cell > 1
    for y0 in y_starts:
        for x0 in x_starts:
            cell = art[y0:y0 + ch, x0:x0 + cw]
            if do_cleanup:
                changed += _cleanup(cell, wrap=False)
            if outline != "none":
                outlined += _outline(cell, outline, outline_cands)
            if opts.seamless:
                _seamless(cell)

    # Reported palette.
    used = _colours_by_use(art)
    truncated = False
    if spec.kind in ("named", "custom"):
        palette = list(spec.colors)
    elif spec.kind == "auto":
        palette = [rgb_to_hex(c) for c in pal]
        extra = used[~np.isin(used, pack_rgb(pal))] if len(pal) else used
        palette += _hex_list(extra)  # a custom outline colour
    else:
        truncated = len(used) > REPORTED_PALETTE_CAP
        palette = _hex_list(used[:REPORTED_PALETTE_CAP])

    # 8. preview (nearest-neighbour only)
    s = int(opts.preview_scale)
    out = art if s == 1 else np.repeat(np.repeat(art, s, axis=0), s, axis=1)
    image = Image.fromarray(np.ascontiguousarray(out))

    metadata = {
        "grid_source": source,
        "palette_kind": spec.kind,
        "colors_used": int(len(used)),
        "palette_truncated": truncated,
        "background": opts.background,
        "alpha_threshold": int(opts.alpha_threshold),
        "cleanup_applied": bool(do_cleanup),
        "cleanup_changed": changed,
        "outline": outline,
        "outline_pixels": outlined,
        "seamless": bool(opts.seamless),
        "preview_scale": s,
        "output_size": [int(out.shape[1]), int(out.shape[0])],
        "sheet": list(opts.sheet) if opts.sheet is not None else None,
        "sheet_cell_size": [cw, ch] if opts.sheet is not None else None,
    }
    if bg_meta.get("removed"):
        metadata["background_color"] = bg_meta.get("color")
    return PixelizeResult(
        image=image,
        grid=grid,
        size=(gw, gh),
        palette=palette,
        background_removed=bool(bg_meta.get("removed")),
        metadata=metadata,
    )
