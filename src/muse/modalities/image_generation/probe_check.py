"""Degenerate-output check for the image/generation probe.

A pull probe that only proves a model loads and returns *an* image passes a
model that draws noise (an overdriven LoRA) or a flat frame. This measures
the probe image on its luminance, downsampled with a box filter to at most
256 px on the long side:

- std: standard deviation of the luminance (0-255). Below 2.0 the frame is
  flat.
- autocorr: lag-1 spatial autocorrelation, the mean of the horizontal and
  vertical neighbour Pearson correlations. Natural and generated images read
  above 0.85 and i.i.d. noise about 0; below 0.5 the frame is noise-like.

Images whose source short side is under 128 px get the flatness check only: 1x
pixel art reads 0.44-0.48 at 16 px. Nearest-upsampling them first would not
help, since k-fold nearest noise reads (k-1)/k and would pass. So noise in a
tiny image is not caught; probe images are the model's default size
(512 px and up in practice).

Calibration (2026-10-28): synthetic gaussian, uniform and near-black
(mean 20, std 15) noise at 512 and 1024 px read autocorr -0.007..0.001
(flagged); flat colours read std 0 (flagged); 57 real images (photos and
wallpapers, muse example outputs, raw 512 px klein sprites and walk sheets,
pixelized 64 px sprites, 128 px walk sheets, tilesets) all pass, with
std >= 8.2 and autocorr >= 0.62 (raw generations >= 0.84; the 64 px sprites
read 0.71-0.92 but are under the autocorr cut-off).

numpy and PIL are imported inside the functions: the probe runs in the
model venv, and nothing here loads at `muse --help` time.
"""
from __future__ import annotations

from typing import Any

STD_FLOOR = 2.0
AUTOCORR_FLOOR = 0.5
MAX_SIDE = 256
MIN_AUTOCORR_SIDE = 128


class DegenerateImageError(RuntimeError):
    """A probe image that is flat or noise-like."""


def _lag1(a, b) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float((a * a).sum() * (b * b).sum()) ** 0.5
    # A constant slice has no correlation to measure; flatness is the std
    # check's job, so it does not count as noise here.
    return 1.0 if denom == 0.0 else float((a * b).sum()) / denom


def image_stats(image: Any) -> tuple[float, float | None]:
    """(std, autocorr) of `image`'s luminance; autocorr is None when the
    source image's short side is under MIN_AUTOCORR_SIDE px."""
    import numpy as np
    from PIL import Image

    from muse.modalities.image_generation.codec import to_pil

    im = to_pil(image).convert("L")
    w, h = im.size
    # The skip is decided on the source size: a 1024x256 frame downsamples
    # to 256x64 but is no tiny image.
    skip_autocorr = min(w, h) < MIN_AUTOCORR_SIDE
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BOX)
    a = np.asarray(im, dtype=np.float64)
    std = float(a.std())
    if skip_autocorr or min(a.shape) < 2:
        return std, None
    horizontal = _lag1(a[:, :-1], a[:, 1:])
    vertical = _lag1(a[:-1, :], a[1:, :])
    return std, (horizontal + vertical) / 2


def check_not_degenerate(image: Any, *, step: str = "generate") -> None:
    """Raise DegenerateImageError when `image` is flat or noise-like.

    `step` names the probe call that produced it (generate, reference edit).
    """
    std, ac = image_stats(image)
    if std < STD_FLOOR or (ac is not None and ac < AUTOCORR_FLOOR):
        shown = "n/a" if ac is None else f"{ac:.3f}"
        raise DegenerateImageError(
            f"probe output looks degenerate: std={std:.2f}, autocorr={shown} "
            f"({step}; flat below std {STD_FLOOR}, noise-like below "
            f"autocorr {AUTOCORR_FLOOR}): the model loads but draws no image"
        )
