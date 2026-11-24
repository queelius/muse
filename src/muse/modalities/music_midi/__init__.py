"""``music/midi`` modality: symbolic multi-track SMF Type 1 MIDI + score generation.

Wire contract:
  - ``POST /v1/music/midi`` (``application/json``):
    - ``response_format="json"`` (default) -> JSON envelope with ``midi_b64``,
      structured multi-track ``score`` (including adaptive ``layer`` tags),
      and generation ``metadata``.
    - ``response_format="midi"`` -> raw ``audio/midi`` Standard MIDI File
      Type 1 bytes.

Bundled weightless CPU model: ``midi-composer`` (``ProceduralMidiComposer``).
Neural symbolic HF checkpoints (ABC causal LMs) resolve via ``hf.py`` to
``AbcMidiLMRuntime``.
"""
from muse.modalities.music_midi.client import MidiClient
from muse.modalities.music_midi.protocol import (
    MidiGenerationModel,
    MidiGenerationResult,
    MidiNote,
    MidiScore,
    MidiTrack,
)
from muse.modalities.music_midi.routes import build_router


MODALITY = "music/midi"

# One weightless bundled model can serve requests that omit `model`;
# AST-read by muse.core.discovery, so keep this a literal tuple.
MODEL_OPTIONAL_PATHS = ("/v1/music/midi",)

DISCOVERY = {
    "summary": (
        "Symbolic multi-track MIDI composition. Generates Standard MIDI File "
        "(SMF Type 1) bytes and a structured note-level score with General MIDI "
        "instruments, modal harmony, and optional harmonically locked adaptive "
        "layers (base, combat, storm, night, altitude) for Web Audio engines "
        "and DAWs."
    ),
    "use_for": [
        "symbolic multi-track MIDI compositions and structured musical scores",
        "adaptive layered game music (base, combat, storm, night, altitude)",
        "editable notation/MIDI sketches ready for DAWs or Web Audio synths",
    ],
    "not_for": [
        "rendered audio waveforms or vocals (use audio/generation)",
        "spoken narration (use audio/speech)",
    ],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/music/midi",
            "request_content_type": "application/json",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_generate_midi",
            "file_fields": [],
            "media_fields": ["midi_b64"],
            "example": {
                "model": "<model>",
                "prompt": "pastoral medieval village theme with lute and flute",
                "bars": 16,
            },
        },
    ],
}


def _probe_call(model: MidiGenerationModel) -> dict:
    """Generate a short 4-bar piece and verify its SMF Type 1 header."""
    result = model.generate(
        "gentle medieval lute and flute melody",
        bars=4,
        seed=0,
    )
    assert isinstance(result.midi_bytes, (bytes, bytearray)), "midi_bytes must be bytes"
    assert result.midi_bytes[:4] == b"MThd", (
        f"probe expected SMF header b'MThd', got {result.midi_bytes[:4]!r}"
    )
    assert len(result.score.tracks) >= 1, "probe expected at least one MIDI track"
    return {
        "bytes": len(result.midi_bytes),
        "tracks": len(result.score.tracks),
        "bars": result.score.bars,
    }


PROBE_DEFAULTS = {
    "shape": "4-bar multi-track SMF Type 1 composition",
    "call": _probe_call,
}


__all__ = [
    "MODALITY",
    "MODEL_OPTIONAL_PATHS",
    "DISCOVERY",
    "PROBE_DEFAULTS",
    "build_router",
    "MidiClient",
    "MidiGenerationModel",
    "MidiGenerationResult",
    "MidiNote",
    "MidiScore",
    "MidiTrack",
]
