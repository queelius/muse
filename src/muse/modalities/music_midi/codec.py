"""Wire-format encoding helpers for ``music/midi`` responses."""
from __future__ import annotations

import base64
import uuid
from typing import Any

from muse.modalities.music_midi.protocol import MidiGenerationResult


MIDI_MEDIA_TYPE = "audio/midi"
MIDI_CONTENT_DISPOSITION = 'attachment; filename="composition.mid"'


def encode_midi_b64(midi_bytes: bytes) -> str:
    """Base64-encode raw SMF bytes to an ASCII string."""
    return base64.b64encode(midi_bytes).decode("ascii")


def decode_midi_b64(midi_b64: str) -> bytes:
    """Decode a base64-encoded SMF payload back to raw bytes."""
    return base64.b64decode(midi_b64.encode("ascii"))


def encode_midi_json(
    result: MidiGenerationResult,
    *,
    model_id: str | None = None,
    response_id: str | None = None,
) -> dict[str, Any]:
    """Build the JSON response envelope for ``POST /v1/music/midi``."""
    return {
        "id": response_id or f"midi-{uuid.uuid4().hex[:16]}",
        "object": "music.midi",
        "model": model_id or result.model_id,
        "midi_b64": encode_midi_b64(result.midi_bytes),
        "score": result.score.to_dict(),
        "metadata": dict(result.metadata),
    }
