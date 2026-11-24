"""Modality protocol and core score data structures for ``music/midi``.

Defines the structural ``MidiGenerationModel`` protocol and the
``MidiNote`` / ``MidiTrack`` / ``MidiScore`` / ``MidiGenerationResult``
dataclasses shared by procedural and neural MIDI runtimes, codecs, and
route handlers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class MidiNote:
    """One note event expressed in musical beats (quarter-note units)."""

    start_beat: float
    duration_beats: float
    pitch: int
    velocity: int = 80

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_beat": round(float(self.start_beat), 6),
            "duration_beats": round(float(self.duration_beats), 6),
            "pitch": int(self.pitch),
            "velocity": int(self.velocity),
        }


@dataclass
class MidiTrack:
    """One instrument track (potentially belonging to an adaptive layer)."""

    name: str
    instrument: str
    gm_program: int = 0
    channel: int = 0
    layer: str = "base"
    pan: int = 64
    volume: int = 100
    notes: list[MidiNote] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer,
            "instrument": self.instrument,
            "channel": int(self.channel),
            "gm_program": int(self.gm_program),
            "notes": [n.to_dict() for n in self.notes],
        }


@dataclass
class MidiScore:
    """Multi-track symbolic score backing an SMF Type 1 composition."""

    bpm: int = 120
    time_signature: tuple[int, int] = (4, 4)
    key: str = "C"
    mode: str = "major"
    bars: int = 16
    ppqn: int = 480
    tracks: list[MidiTrack] = field(default_factory=list)

    @property
    def beats_per_bar(self) -> float:
        num, den = self.time_signature
        return float(num) * (4.0 / float(den))

    @property
    def total_beats(self) -> float:
        return self.bars * self.beats_per_bar

    def to_dict(self) -> dict[str, Any]:
        num, den = self.time_signature
        return {
            "bpm": int(self.bpm),
            "time_signature": [int(num), int(den)],
            "key": self.key,
            "mode": self.mode,
            "bars": int(self.bars),
            "ppqn": int(self.ppqn),
            "tracks": [t.to_dict() for t in self.tracks],
        }


@dataclass
class MidiGenerationResult:
    """Generated symbolic score together with its encoded SMF Type 1 bytes."""

    score: MidiScore
    midi_bytes: bytes
    seed: int | None = None
    model_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MidiGenerationModel(Protocol):
    """Structural protocol every ``music/midi`` backend satisfies."""

    @property
    def model_id(self) -> str: ...

    def generate(
        self,
        prompt: str,
        *,
        key: str | None = None,
        mode: str | None = None,
        bpm: int | None = None,
        time_signature: str = "4/4",
        bars: int = 16,
        instruments: list[str] | None = None,
        layers: list[str] | None = None,
        seed: int | None = None,
        temperature: float = 0.9,
        **kwargs: Any,
    ) -> MidiGenerationResult: ...
