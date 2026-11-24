"""Muse image/matting modality protocol.

``ImageMattingModel`` is the backend contract: ``matte`` takes an opaque or
semi-transparent PIL image plus ``MattingOptions`` and returns a
``MattingResult`` carrying an RGBA foreground image and alpha statistics.

Import-light: images are typed ``Any``, so importing this module loads
neither PIL nor numpy nor torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

VALID_BACKGROUNDS = ("auto", "white", "corners")


@dataclass(frozen=True)
class MattingOptions:
    """Parameters controlling foreground extraction / background removal.

    - ``background``: ``"auto"`` samples the dominant border color, ``"white"``
      keys near-white borders, ``"corners"`` samples and seeds from image
      corners (used by deterministic keying runtimes).
    - ``bg_threshold``: minimum RGB channel intensity ``[0, 255]`` treated as
      near-white background.
    - ``color_tolerance``: per-channel RGB tolerance ``[0, 255]`` for matching
      the sampled background color (and neutral chroma bound for near-white).
    - ``hard_alpha``: when True, binarize alpha at ``alpha_threshold`` so
      output alpha is strictly 0 or 255.
    - ``alpha_threshold``: cutoff ``[0, 255]`` used when ``hard_alpha=True``.
    - ``defringe``: clean 1px anti-aliased background halos along the matte
      boundary and zero RGB on fully transparent pixels.
    - ``fringe_threshold``: luminance / edge cutoff ``[0, 255]`` controlling
      how aggressively 1px boundary fringe pixels are attenuated.
    - ``clear_enclosed_white``: also remove enclosed neutral-white pockets
      (e.g. gaps between roots, arches, or limbs) unreachable from the border.
    - ``crop``: tightly crop the output to the non-transparent alpha bounding
      box (no-op if the output is completely transparent).
    """

    background: str = "auto"
    bg_threshold: int = 236
    color_tolerance: int = 24
    hard_alpha: bool = False
    alpha_threshold: int = 128
    defringe: bool = True
    fringe_threshold: int = 212
    clear_enclosed_white: bool = False
    crop: bool = False


@dataclass
class MattingResult:
    """RGBA foreground extraction result."""

    image: Any
    width: int
    height: int
    transparent_fraction: float
    opaque_fraction: float
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ImageMattingModel(Protocol):
    """Protocol for image/matting backends."""

    @property
    def model_id(self) -> str: ...

    def matte(
        self,
        image: Any,
        opts: MattingOptions | None = None,
        **kwargs: Any,
    ) -> MattingResult: ...
