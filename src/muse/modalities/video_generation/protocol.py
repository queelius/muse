"""Modality protocol for video/generation.

VideoResult holds a list of PIL.Image frames + timing metadata. The
codec layer transforms the list into mp4/webm/frames_b64 bytes for
the HTTP response.

video/generation is muse's narrative-clip sibling to image/animation:
longer durations, single play, mp4 default, transformer-based
backbones (Wan, CogVideoX) instead of UNet+motion-adapter pairs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class VideoResult:
    """One generated video clip plus timing + provenance.

    `frames` is loosely typed (Any) at the protocol boundary so backends
    can return PIL.Image, numpy arrays, or torch tensors. The codec
    layer normalizes to PIL.Image before encoding.

    `duration_seconds` is the actual clip duration (frames / fps), not
    the requested duration; runtimes that align to model-native frame
    counts may return slightly different values than requested.
    """
    frames: list[Any]
    fps: int
    width: int
    height: int
    duration_seconds: float
    seed: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class VideoGenerationModel(Protocol):
    """Protocol for video generation backends."""

    @property
    def model_id(self) -> str: ...

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        duration_seconds: float | None = None,
        fps: int | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> VideoResult: ...
