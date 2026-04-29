"""Muse image/segmentation modality protocol.

Defines ImageSegmentationModel (backend contract), MaskRecord (one
mask plus provenance), and SegmentationResult (synthesis return).
Backends produce a list of masks per call, sorted by score descending,
truncated to max_masks.

Axis-order convention split (documented here so callers don't trip):

  - ``PIL.Image.size``       -> ``(W, H)``        used by ``image_size``
  - numpy 2D arrays          -> ``[H, W]``        used by ``mask``
  - COCO bbox                -> ``[x, y, w, h]``  used by ``bbox``
  - COCO RLE size field      -> ``[H, W]``        used at the wire layer

Each field follows the convention of its native ecosystem; clients
following the convention of the field they read get consistent
results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class MaskRecord:
    """One segmentation mask plus provenance.

    ``mask`` is a 2D numpy bool / uint8 array shaped ``(H, W)``. The
    codec converts it to either a base64-encoded PNG or a COCO-style
    RLE dict at the wire layer.

    ``score`` is the model's confidence in this mask in ``[0, 1]``.
    ``bbox`` is ``(x, y, w, h)`` per COCO bbox convention.
    ``area`` is ``int(mask.sum())``.
    """

    mask: Any
    score: float
    bbox: tuple[int, int, int, int]
    area: int


@dataclass
class SegmentationResult:
    """A list of masks plus image-level metadata.

    ``image_size`` is ``(W, H)`` per PIL convention.
    ``mode`` is the dispatch mode actually used by the runtime
    (``"auto"``, ``"points"``, ``"boxes"``, ``"text"``).
    ``seed`` is ``-1`` when no seed was supplied.
    """

    masks: list[MaskRecord]
    image_size: tuple[int, int]
    mode: str
    seed: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ImageSegmentationModel(Protocol):
    """Protocol for promptable segmentation backends."""

    @property
    def model_id(self) -> str: ...

    def segment(
        self,
        image: Any,
        *,
        mode: str = "auto",
        prompt: str | None = None,
        points: list[list[int]] | None = None,
        boxes: list[list[int]] | None = None,
        max_masks: int | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> SegmentationResult: ...
