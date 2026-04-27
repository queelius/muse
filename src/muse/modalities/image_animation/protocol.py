"""Modality protocol for image/animation.

AnimationResult holds a list of PIL.Image frames + timing metadata. The
codec layer transforms the list into webp/gif/mp4/frames_b64 bytes for
the HTTP response.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class AnimationResult:
    """One generated animation: ordered frames + timing + provenance.

    frames: list[Any] (typed loosely to avoid forcing PIL on the protocol
    boundary; codec normalizes to PIL before encoding).
    """
    frames: list[Any]
    fps: int
    width: int
    height: int
    seed: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class AnimationModel(Protocol):
    """Protocol for animation backends."""

    @property
    def model_id(self) -> str: ...

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        frames: int | None = None,
        fps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        init_image: Any = None,
        strength: float | None = None,
        **kwargs,
    ) -> AnimationResult: ...
