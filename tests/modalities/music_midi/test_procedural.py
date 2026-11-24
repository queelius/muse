"""Tests for ``ProceduralMidiComposer`` (deterministic CPU symbolic composer)."""
from __future__ import annotations

import pytest

from muse.modalities.music_midi.protocol import MidiGenerationModel
from muse.modalities.music_midi.runtimes.procedural import ProceduralMidiComposer
from muse.modalities.music_midi.smf import parse_smf


def test_procedural_satisfies_protocol():
    composer = ProceduralMidiComposer()
    assert isinstance(composer, MidiGenerationModel)
    assert composer.model_id == "midi-composer"


def test_determinism_same_seed_produces_identical_midi_and_score():
    composer = ProceduralMidiComposer()
    r1 = composer.generate(
        "pastoral medieval village theme with lute and flute",
        key="D",
        mode="dorian",
        bpm=110,
        bars=16,
        seed=42,
    )
    r2 = composer.generate(
        "pastoral medieval village theme with lute and flute",
        key="D",
        mode="dorian",
        bpm=110,
        bars=16,
        seed=42,
    )
    r3 = composer.generate(
        "pastoral medieval village theme with lute and flute",
        key="D",
        mode="dorian",
        bpm=110,
        bars=16,
        seed=99,
    )
    assert r1.midi_bytes == r2.midi_bytes
    assert r1.score.to_dict() == r2.score.to_dict()
    assert r1.midi_bytes != r3.midi_bytes


def test_prompt_mood_inference_varies_tempo_mode_and_instruments():
    composer = ProceduralMidiComposer()
    combat = composer.generate("intense epic boss battle clash", seed=7)
    lullaby = composer.generate("quiet moonlit night lullaby", seed=7)

    assert combat.score.bpm > lullaby.score.bpm
    assert combat.score.mode == "minor"
    assert lullaby.score.mode == "aeolian"
    combat_insts = {t.instrument for t in combat.score.tracks}
    lullaby_insts = {t.instrument for t in lullaby.score.tracks}
    assert "drums" in combat_insts
    assert "celesta" in lullaby_insts


def test_prompt_key_and_instrument_extraction():
    composer = ProceduralMidiComposer()
    res = composer.generate(
        "gentle forest melody in F lydian with harp and bell",
        bars=8,
        seed=12,
    )
    assert res.score.key == "F"
    assert res.score.mode == "lydian"
    insts = [t.instrument for t in res.score.tracks]
    assert "harp" in insts
    assert "bell" in insts


@pytest.mark.parametrize("ts,expected", [("4/4", (4, 4)), ("3/4", (3, 4)), ("6/8", (6, 8))])
def test_supported_time_signatures_round_trip(ts: str, expected: tuple[int, int]):
    composer = ProceduralMidiComposer()
    res = composer.generate(
        "tavern folk dance",
        time_signature=ts,
        bars=8,
        instruments=["lute", "flute", "drums"],
        seed=5,
    )
    assert res.score.time_signature == expected
    parsed = parse_smf(res.midi_bytes)
    assert parsed.time_signature == expected
    assert parsed.bars == 8
    assert len(parsed.tracks) == 3
    drum_tracks = [t for t in parsed.tracks if t.channel == 9]
    assert len(drum_tracks) == 1
    assert len(drum_tracks[0].notes) > 0


@pytest.mark.parametrize(
    "mode",
    ["major", "minor", "dorian", "mixolydian", "lydian", "phrygian", "aeolian"],
)
def test_all_seven_modes_compose_valid_smf(mode: str):
    composer = ProceduralMidiComposer()
    res = composer.generate("exploration theme", key="E", mode=mode, bars=8, seed=3)
    assert res.score.key == "E"
    assert res.score.mode == mode
    assert res.midi_bytes[:4] == b"MThd"
    parsed = parse_smf(res.midi_bytes)
    assert parsed.key == "E"
    assert parsed.mode == mode
    assert all(len(t.notes) > 0 for t in parsed.tracks)


def test_adaptive_layers_harmonically_locked_and_tagged():
    composer = ProceduralMidiComposer()
    res = composer.generate(
        "overworld medieval meadow",
        key="D",
        mode="dorian",
        bpm=108,
        bars=16,
        instruments=["lute", "flute"],
        layers=["base", "combat", "storm", "night", "altitude"],
        seed=21,
    )
    found_layers = {t.layer for t in res.score.tracks}
    assert found_layers == {"base", "combat", "storm", "night", "altitude"}
    # Every track stays within the 16-bar loop boundary
    max_beat = res.score.total_beats
    for tr in res.score.tracks:
        assert len(tr.notes) > 0
        for n in tr.notes:
            assert 0.0 <= n.start_beat < max_beat
            assert 0 <= n.pitch <= 127
            assert 1 <= n.velocity <= 127

    # Round-trips cleanly with layer metadata intact
    parsed = parse_smf(res.midi_bytes)
    assert {t.layer for t in parsed.tracks} == found_layers
    assert parsed.bars == 16
    assert parsed.bpm == 108


def test_invalid_inputs_raise_value_error():
    composer = ProceduralMidiComposer()
    with pytest.raises(ValueError):
        composer.generate("   ")
    with pytest.raises(ValueError):
        composer.generate("theme", key="H#")
    with pytest.raises(ValueError):
        composer.generate("theme", mode="pentatonic")
    with pytest.raises(ValueError):
        composer.generate("theme", time_signature="5/4")
