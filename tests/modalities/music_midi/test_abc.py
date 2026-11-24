"""Tests for the pure-Python ABC notation -> ``MidiScore`` parser (``abc.py``)."""
from __future__ import annotations

import pytest

from muse.modalities.music_midi.abc import extract_abc_block, parse_abc, parse_abc_key
from muse.modalities.music_midi.smf import encode_smf, parse_smf


def test_parse_abc_key_modes_and_signatures():
    tonic, mode, sig = parse_abc_key("G")
    assert (tonic, mode) == ("G", "major")
    assert sig == {"F": 1}

    tonic, mode, sig = parse_abc_key("Dm")
    assert (tonic, mode) == ("D", "minor")
    assert sig == {"B": -1}

    tonic, mode, sig = parse_abc_key("D dor")
    assert (tonic, mode) == ("D", "dorian")
    assert sig == {}  # D dorian has same key signature as C major

    tonic, mode, sig = parse_abc_key("A mix")
    assert (tonic, mode) == ("A", "mixolydian")
    assert sig == {"F": 1, "C": 1}


def test_parse_abc_headers_accidentals_octaves_durations_rests_and_chords():
    abc = """
    X:1
    T:Village Reel
    M:4/4
    L:1/8
    Q:1/4=116
    K:G
    | G2 ^F G | =F z/2 [GBd]3/2 c' | C,2 _B, A, |
    """
    score = parse_abc(abc)
    assert score.bpm == 116
    assert score.time_signature == (4, 4)
    assert score.key == "G"
    assert score.mode == "major"
    assert len(score.tracks) == 1

    notes = score.tracks[0].notes
    # Bar 1: G2 (67, 1.0b), ^F (66, 0.5b), G (67, 0.5b)
    assert notes[0].pitch == 67 and pytest.approx(notes[0].duration_beats) == 1.0
    assert notes[1].pitch == 66 and pytest.approx(notes[1].duration_beats) == 0.5
    assert notes[2].pitch == 67 and pytest.approx(notes[2].start_beat) == 1.5

    # Bar 2: =F overrides key sig F# -> F natural (65, 0.5b)
    assert notes[3].pitch == 65
    # Followed by z/2 (0.25b rest), then chord [GBd]3/2 (0.75b) at beat 2.75
    chord_notes = [n for n in notes if pytest.approx(n.start_beat) == 2.75]
    assert sorted(n.pitch for n in chord_notes) == [67, 71, 74]
    assert all(pytest.approx(n.duration_beats) == 0.75 for n in chord_notes)

    # c' = C6 (84), C, = C3 (48), _B, = Bb3 (58)
    pitches = [n.pitch for n in notes]
    assert 84 in pitches
    assert 48 in pitches
    assert 58 in pitches

    # Encodes cleanly to SMF Type 1 and round-trips
    smf = encode_smf(score)
    assert smf[:4] == b"MThd"
    rt = parse_smf(smf)
    assert len(rt.tracks[0].notes) == len(notes)


def test_parse_abc_measure_accidental_resets_at_barline():
    abc = """
    X:1
    M:4/4
    L:1/4
    K:C
    ^F F | F2 |
    """
    score = parse_abc(abc)
    pitches = [n.pitch for n in score.tracks[0].notes]
    # First F is F#, second F in same bar stays F#, third F after barline returns to F natural
    assert pitches == [66, 66, 65]


def test_parse_abc_multi_voice_and_midi_program():
    abc = """
    X:1
    T:Duet
    M:3/4
    L:1/4
    Q:1/4=100
    K:D dor
    V:1 name="Flute" program=73
    V:2 name="Lute" program=24
    [V:1] D F A | G F E |
    [V:2] D,2 A, | C,2 G, |
    """
    score = parse_abc(abc)
    assert score.time_signature == (3, 4)
    assert score.key == "D"
    assert score.mode == "dorian"
    assert len(score.tracks) == 2
    v1, v2 = score.tracks
    assert v1.name == "Flute"
    assert v1.gm_program == 73
    assert v2.name == "Lute"
    assert v2.gm_program == 24
    assert len(v1.notes) == 6
    assert len(v2.notes) == 4


def test_parse_abc_tuplets_ties_and_broken_rhythm():
    abc = """
    X:1
    M:4/4
    L:1/8
    K:C
    (3CDE F-F G>A |
    """
    score = parse_abc(abc)
    notes = score.tracks[0].notes
    # Triplet (3CDE -> 3 notes of duration (0.5 * 2/3) = 1/3 beat each
    assert pytest.approx(notes[0].duration_beats, abs=1e-4) == 1.0 / 3.0
    assert pytest.approx(notes[1].duration_beats, abs=1e-4) == 1.0 / 3.0
    assert pytest.approx(notes[2].duration_beats, abs=1e-4) == 1.0 / 3.0
    # F-F tied into single quarter note (1.0 beat)
    assert notes[3].pitch == 65
    assert pytest.approx(notes[3].duration_beats, abs=1e-4) == 1.0
    # G>A dotted eighth (0.75) + sixteenth (0.25)
    assert notes[4].pitch == 67 and pytest.approx(notes[4].duration_beats, abs=1e-4) == 0.75
    assert notes[5].pitch == 69 and pytest.approx(notes[5].duration_beats, abs=1e-4) == 0.25


def test_extract_abc_block_strips_markdown_and_preamble():
    raw = (
        "Here is your composition:\n"
        "```abc\n"
        "X:1\nM:4/4\nL:1/4\nK:C\nC D E F |\n"
        "```\nEnjoy!"
    )
    block = extract_abc_block(raw)
    assert block.startswith("X:1")
    score = parse_abc(block)
    assert [n.pitch for n in score.tracks[0].notes] == [60, 62, 64, 65]


def test_parse_abc_rejects_empty_or_noteless():
    with pytest.raises(ValueError):
        parse_abc("")
    with pytest.raises(ValueError):
        parse_abc("X:1\nT:Silent\nK:C\n| z4 |")
