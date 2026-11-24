"""Modality protocol for ``audio/process``.

Defines :class:`AudioProcessOptions`, :class:`AudioProcessResult`, and the
structural :class:`AudioProcessModel` protocol. Keeps imports stdlib-only so
importing the modality package stays light.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class AudioProcessOptions:
    """Parameters controlling the deterministic audio processing pipeline."""

    trim_silence: bool = True
    trim_threshold_db: float = -42.0
    pad_lead_ms: float = 20.0
    pad_tail_ms: float = 65.0
    fade_in_ms: float = 5.0
    fade_out_ms: float = 15.0
    loop_fade_s: float = 0.0
    target_peak_dbfs: float | None = -3.0
    mono: bool = False
    sample_rate: int | None = None
    remove_dc: bool = True


@dataclass
class AudioProcessResult:
    """One processed audio clip plus mastering telemetry."""

    audio: Any
    sample_rate: int
    channels: int
    metrics: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class AudioProcessModel(Protocol):
    """Structural protocol satisfied by any ``audio/process`` backend."""

    @property
    def model_id(self) -> str: ...

    def process(
        self,
        audio: Any,
        sample_rate: int,
        options: AudioProcessOptions | None = None,
    ) -> AudioProcessResult: ...
