"""Muse image/upscale modality protocol.

Defines ImageUpscaleModel (backend contract) and UpscaleResult
(synthesis return). Backends produce a single upscaled image per
call; n is enforced at the route layer by issuing n calls.

UpscaleResult records both the original (input) and upscaled (output)
dimensions so callers can verify the achieved scale factor without
re-decoding the image bytes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class UpscaleResult:
    """A single upscaled image plus provenance metadata.

    `image` is typed as Any so backends can return PIL.Image, numpy
    arrays, or torch tensors. The codec normalizes to PIL before
    encoding.
    """
    image: Any
    original_width: int
    original_height: int
    upscaled_width: int
    upscaled_height: int
    scale: int
    seed: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ImageUpscaleModel(Protocol):
    """Protocol for image-to-image super-resolution backends."""

    @property
    def model_id(self) -> str: ...

    @property
    def supported_scales(self) -> list[int]: ...

    def upscale(
        self,
        image: Any,
        *,
        scale: int | None = None,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> UpscaleResult: ...
