"""``SpriteMatte``: deterministic, weightless CPU foreground matting runtime.

Pure Pillow + numpy (lazy-imported inside ``matte`` so package and script
discovery work on hosts without Pillow or numpy). Designed for extracting clean
RGBA sprites, props, and cutouts from opaque generated images with solid,
white, or uniform-border backgrounds.

Pipeline:
1. Background sampling & 4-connected border flood-fill keying (``auto``,
   ``white``, or ``corners``).
2. Optional enclosed neutral-white pocket removal (``clear_enclosed_white``),
   ignoring tiny 1-3 px specular highlights.
3. Optional 1px anti-aliased boundary halo defringing & color decontamination
   (``defringe``, controlled by ``fringe_threshold``).
4. Optional binary alpha snapping (``hard_alpha``, ``alpha_threshold``).
5. Optional tight alpha bounding-box crop (``crop``).

Never pass ``mode=`` to ``Image.fromarray``: Pillow 13 removed it.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.image_matting.protocol import (
    VALID_BACKGROUNDS,
    MattingOptions,
    MattingResult,
)

logger = logging.getLogger(__name__)

# Minimum connected-component size (pixels) cleared by clear_enclosed_white;
# preserves tiny 1-3 px eye glints and specular sparkles inside the subject.
_ENCLOSED_MIN_PIXELS = 4


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_options(opts: MattingOptions) -> None:
    if opts.background not in VALID_BACKGROUNDS:
        raise ValueError(
            f"background must be one of {list(VALID_BACKGROUNDS)}; got {opts.background!r}"
        )
    for name in ("bg_threshold", "color_tolerance", "alpha_threshold", "fringe_threshold"):
        val = getattr(opts, name)
        if not _is_int(val) or not 0 <= val <= 255:
            raise ValueError(f"{name} must be an integer in [0, 255]")
    for name in ("hard_alpha", "defringe", "clear_enclosed_white", "crop"):
        if not isinstance(getattr(opts, name), bool):
            raise ValueError(f"{name} must be a boolean")


def _row_runs(cand: Any) -> Any:
    """Label each horizontal run of candidate pixels 1..R (0 = not candidate)."""
    import numpy as np

    h, w = cand.shape
    starts = cand.copy()
    starts[:, 1:] &= ~cand[:, :-1]
    ids = np.cumsum(starts.ravel(), dtype=np.int64).reshape(h, w)
    ids[~cand] = 0
    return ids


def _run_edges(ids: Any, cand: Any) -> tuple[Any, Any]:
    """Return pairs of vertically touching horizontal runs."""
    both = cand[:-1] & cand[1:]
    top, bot = ids[:-1], ids[1:]
    first = both.copy()
    first[:, 1:] &= ~(
        both[:, :-1]
        & (top[:, 1:] == top[:, :-1])
        & (bot[:, 1:] == bot[:, :-1])
    )
    return top[first], bot[first]


def _merge_runs(label: Any, a: Any, b: Any) -> Any:
    """Hook-and-jump union-find over horizontal runs."""
    import numpy as np

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


def _flood_4(seed: Any, cand: Any) -> Any:
    """4-connected flood-fill of ``cand`` starting from ``seed``."""
    import numpy as np

    ids = _row_runs(cand)
    n_runs = int(ids.max())
    if n_runs == 0:
        return np.zeros(cand.shape, dtype=bool)
    label = np.arange(n_runs + 1, dtype=np.int64)
    seeded = np.unique(ids[seed & cand])
    if seeded.size == 0:
        return np.zeros(cand.shape, dtype=bool)
    label[seeded] = 0
    label = _merge_runs(label, *_run_edges(ids, cand))
    return (label[ids] == 0) & cand


def _components_4(cand: Any) -> Any:
    """4-connected component labeling of ``cand`` (0 = background)."""
    import numpy as np

    ids = _row_runs(cand)
    n_runs = int(ids.max())
    if n_runs == 0:
        return np.zeros(cand.shape, dtype=np.int64)
    label = _merge_runs(
        np.arange(n_runs + 1, dtype=np.int64),
        *_run_edges(ids, cand),
    )
    return label[ids]


def _dominant_color(pixels_rgb: Any) -> Any:
    """Find the dominant RGB color among an ``(N, 3)`` array via 8-level
    quantized mode followed by median refinement."""
    import numpy as np

    q = pixels_rgb.astype(np.int32) >> 3
    keys = (q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    best_key = uniq[int(np.argmax(counts))]
    inliers = pixels_rgb[keys == best_key]
    return np.rint(np.median(inliers, axis=0)).astype(np.int16)


def _defringe_1px(
    rgba: Any,
    bg_rgb: Any | None,
    *,
    is_white_bg: bool,
    bg_threshold: int,
    color_tolerance: int,
    fringe_threshold: int,
) -> int:
    """Clean 1px anti-aliased background halos along the transparent edge.

    Identifies currently non-transparent pixels adjacent (4-connected) to at
    least one transparent pixel. For pixels blended with the background color
    or bright neutral fringe above ``fringe_threshold``, attenuates alpha and
    decontaminates RGB from opaque interior neighbors so no bright halo remains.
    Returns the count of modified boundary pixels.
    """
    import numpy as np

    h, w = rgba.shape[:2]
    if h < 2 and w < 2:
        return 0

    alpha = rgba[..., 3]
    trans = alpha == 0
    if not trans.any() or trans.all():
        return 0

    # 4-connected adjacency to transparent pixels
    adj_trans = np.zeros((h, w), dtype=bool)
    if h > 1:
        adj_trans[1:, :] |= trans[:-1, :]
        adj_trans[:-1, :] |= trans[1:, :]
    if w > 1:
        adj_trans[:, 1:] |= trans[:, :-1]
        adj_trans[:, :-1] |= trans[:, 1:]

    boundary = (~trans) & adj_trans
    if not boundary.any():
        return 0

    rgb = rgba[..., :3].astype(np.float32)
    rgb_i16 = rgba[..., :3].astype(np.int16)
    chroma = rgb_i16.max(axis=-1) - rgb_i16.min(axis=-1)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

    # Identify fringe pixels on the 1px boundary and compute attenuation in [0, 1]
    # where 0.0 -> fully transparent and 1.0 -> keep original alpha.
    scale = np.ones((h, w), dtype=np.float32)
    affected = np.zeros((h, w), dtype=bool)

    if is_white_bg or bg_rgb is None:
        # Neutral / bright halo blended against a light/white background.
        max_chroma = max(int(color_tolerance) * 2, 48)
        fringe = boundary & (luma >= float(fringe_threshold)) & (chroma <= max_chroma)
        if fringe.any():
            span = max(1.0, float(bg_threshold) - float(fringe_threshold))
            # Pixels at or above bg_threshold drop to 0; between fringe_threshold
            # and bg_threshold ramp smoothly down toward 0.
            att = np.clip((float(bg_threshold) - luma) / span, 0.0, 1.0)
            if fringe_threshold >= bg_threshold:
                att = np.zeros_like(luma)
            scale = np.where(fringe, att, scale)
            affected |= fringe
    else:
        # Colored backdrop: attenuate boundary pixels lying in the halo band
        # just outside color_tolerance toward bg_rgb.
        dist = np.abs(rgb_i16 - bg_rgb).max(axis=-1).astype(np.float32)
        halo_margin = max(16.0, float(255 - fringe_threshold))
        outer = float(color_tolerance) + halo_margin
        fringe = boundary & (dist <= outer)
        if fringe.any():
            att = np.clip((dist - float(color_tolerance)) / halo_margin, 0.0, 1.0)
            scale = np.where(fringe, att, scale)
            affected |= fringe

    if not affected.any():
        return 0

    # Decontaminate RGB of remaining semi-transparent fringe pixels using the
    # average color of adjacent non-boundary opaque interior neighbors.
    interior = (~trans) & (~boundary)
    if interior.any():
        sum_rgb = np.zeros((h, w, 3), dtype=np.float32)
        cnt = np.zeros((h, w), dtype=np.float32)
        int_f = interior.astype(np.float32)
        int_rgb = rgb * int_f[..., None]
        if h > 1:
            sum_rgb[1:, :] += int_rgb[:-1, :]
            cnt[1:, :] += int_f[:-1, :]
            sum_rgb[:-1, :] += int_rgb[1:, :]
            cnt[:-1, :] += int_f[1:, :]
        if w > 1:
            sum_rgb[:, 1:] += int_rgb[:, :-1]
            cnt[:, 1:] += int_f[:, :-1]
            sum_rgb[:, :-1] += int_rgb[:, 1:]
            cnt[:, :-1] += int_f[:, 1:]
        has_int = affected & (cnt > 0)
        if has_int.any():
            clean_rgb = np.rint(sum_rgb[has_int] / cnt[has_int, None]).clip(0, 255)
            rgba[has_int, :3] = clean_rgb.astype(np.uint8)

    new_alpha = np.rint(alpha.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)
    rgba[affected, 3] = new_alpha[affected]
    rgba[rgba[..., 3] == 0] = 0
    return int(affected.sum())


def _alpha_fractions(alpha: Any) -> tuple[float, float]:
    total = int(alpha.size)
    if total == 0:
        return 0.0, 0.0
    transparent = float((alpha == 0).sum() / total)
    opaque = float((alpha == 255).sum() / total)
    return transparent, opaque


class SpriteMatte:
    """Deterministic CPU sprite matting backend (no weights)."""

    def __init__(
        self,
        *,
        model_id: str = "sprite-matte",
        hf_repo: str = "",
        local_dir: str | None = None,
        device: str = "cpu",
        **_: Any,
    ) -> None:
        self.model_id = model_id
        self.hf_repo = hf_repo
        self.local_dir = local_dir
        self.device = "cpu"

    def matte(
        self,
        image: Any,
        opts: MattingOptions | None = None,
        **_: Any,
    ) -> MattingResult:
        """Extract an RGBA foreground cutout from ``image``."""
        import numpy as np
        from PIL import Image

        opts = opts if opts is not None else MattingOptions()
        _validate_options(opts)

        rgba = np.array(image.convert("RGBA"), dtype=np.uint8, copy=True)
        h, w = rgba.shape[:2]
        if h < 1 or w < 1:
            raise ValueError("image is empty")

        rgb = rgba[..., :3].astype(np.int16)
        alpha = rgba[..., 3]

        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        near_white = (
            (rgb.min(axis=-1) >= opts.bg_threshold)
            & ((rgb.max(axis=-1) - rgb.min(axis=-1)) <= opts.color_tolerance)
        )

        bg_rgb: Any | None = None
        is_white_bg = False

        if opts.background == "white":
            bg_rgb = np.array([255, 255, 255], dtype=np.int16)
            is_white_bg = True
            cand = (alpha == 0) | near_white
            seed = border & cand
        elif opts.background == "corners":
            ps_h = max(1, min(4, h // 8 if h >= 8 else 1))
            ps_w = max(1, min(4, w // 8 if w >= 8 else 1))
            corner_mask = np.zeros((h, w), dtype=bool)
            corner_mask[:ps_h, :ps_w] = True
            corner_mask[:ps_h, w - ps_w :] = True
            corner_mask[h - ps_h :, :ps_w] = True
            corner_mask[h - ps_h :, w - ps_w :] = True

            opaque_corners = corner_mask & (alpha > 0)
            if opaque_corners.any():
                bg_rgb = _dominant_color(rgb[opaque_corners])
                is_white_bg = bool(
                    bg_rgb.min() >= opts.bg_threshold
                    and (bg_rgb.max() - bg_rgb.min()) <= opts.color_tolerance
                )
                dist = np.abs(rgb - bg_rgb).max(axis=-1)
                cand = (alpha == 0) | (dist <= opts.color_tolerance)
                if is_white_bg:
                    cand |= near_white
                seed = corner_mask & cand
            else:
                cand = alpha == 0
                seed = corner_mask & cand
        else:  # "auto"
            opaque_border = border & (alpha > 0)
            if opaque_border.any():
                bg_rgb = _dominant_color(rgb[opaque_border])
                is_white_bg = bool(
                    bg_rgb.min() >= opts.bg_threshold
                    and (bg_rgb.max() - bg_rgb.min()) <= opts.color_tolerance
                )
                dist = np.abs(rgb - bg_rgb).max(axis=-1)
                cand = (alpha == 0) | (dist <= opts.color_tolerance)
                if is_white_bg:
                    cand |= near_white
                seed = border & cand
            else:
                cand = alpha == 0
                seed = border & cand

        bg_mask = _flood_4(seed, cand)
        rgba[bg_mask] = 0

        enclosed_cleared = 0
        if opts.clear_enclosed_white:
            pockets = (rgba[..., 3] > 0) & near_white
            if pockets.any():
                labels = _components_4(pockets)
                sizes = np.bincount(labels.ravel())
                sizes[0] = 0
                big = sizes >= _ENCLOSED_MIN_PIXELS
                big[0] = False
                if big.any():
                    rgba[big[labels]] = 0
                    enclosed_cleared = int(big.sum())

        defringed_pixels = 0
        if opts.defringe:
            defringed_pixels = _defringe_1px(
                rgba,
                bg_rgb,
                is_white_bg=is_white_bg,
                bg_threshold=opts.bg_threshold,
                color_tolerance=opts.color_tolerance,
                fringe_threshold=opts.fringe_threshold,
            )

        if opts.hard_alpha:
            keep = (rgba[..., 3] >= opts.alpha_threshold) & (rgba[..., 3] > 0)
            rgba[..., 3] = np.where(keep, 255, 0).astype(np.uint8)
            rgba[~keep] = 0

        cropped = False
        if opts.crop:
            ys, xs = np.where(rgba[..., 3] > 0)
            if ys.size > 0:
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                if (y1 - y0, x1 - x0) != (h, w):
                    rgba = rgba[y0:y1, x0:x1].copy()
                    cropped = True

        out_h, out_w = rgba.shape[:2]
        transparent_fraction, opaque_fraction = _alpha_fractions(rgba[..., 3])
        out_img = Image.fromarray(rgba)
        metadata: dict[str, Any] = {
            "model": self.model_id,
            "background_mode": opts.background,
            "background_rgb": bg_rgb.tolist() if bg_rgb is not None else None,
            "enclosed_pockets_cleared": enclosed_cleared,
            "defringed_pixels": defringed_pixels,
            "cropped": cropped,
        }
        return MattingResult(
            image=out_img,
            width=out_w,
            height=out_h,
            transparent_fraction=transparent_fraction,
            opaque_fraction=opaque_fraction,
            metadata=metadata,
        )
