"""midi-composer: deterministic weightless CPU MIDI composer for ``music/midi``.

Composes multi-track Standard MIDI File (SMF Type 1) scores with modal chord
progressions, A/B melodic form, General MIDI instrument presets, and
harmonically locked adaptive layers (``base``, ``combat``, ``storm``,
``night``, ``altitude``). Pure Python standard library on CPU: no weights and
no external ML packages required.
"""
from muse.modalities.music_midi.runtimes.procedural import (
    ProceduralMidiComposer as Model,
)


MANIFEST = {
    "model_id": "midi-composer",
    "modality": "music/midi",
    "weights": "none",
    "description": (
        "Deterministic procedural multi-track MIDI composer (SMF Type 1 + "
        "structured score) with modal harmony, A/B form, GM presets, and "
        "harmonically locked adaptive layers. Pure Python CPU, no weights, MIT."
    ),
    "summary": (
        "Composes deterministic multi-track SMF Type 1 MIDI and adaptive game "
        "music layers on CPU with zero weights"
    ),
    "strengths": [
        "Instant deterministic CPU composition seeded per request, zero weights",
        "Harmonically locked adaptive layers (base, combat, storm, night, altitude)",
        "Valid SMF Type 1 output plus note-level JSON score across 7 modes and 4/4, 3/4, 6/8",
    ],
    "limits": [
        "Rule-based symbolic composition rather than learned neural phrasing",
        "Outputs symbolic MIDI/score only; pair with a synth or Web Audio engine for sound",
    ],
    "quality_tier": "good",
    "license": "MIT",
    "pip_extras": (),
    "system_packages": (),
    "capabilities": {
        "device": "cpu",
        "memory_gb": 0.25,
        "supports_layers": True,
        "supported_modes": [
            "major",
            "minor",
            "dorian",
            "mixolydian",
            "lydian",
            "phrygian",
            "aeolian",
        ],
        "supported_time_signatures": ["4/4", "3/4", "6/8"],
        "max_bars": 64,
    },
}
