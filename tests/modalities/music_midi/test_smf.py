"""Tests for pure-Python SMF Type 1 encoder and decoder (``smf.py``)."""
from __future__ import annotations

import struct

import pytest

from muse.modalities.music_midi.protocol import MidiNote, MidiScore, MidiTrack
from muse.modalities.music_midi.smf import (
    decode_vlq,
    encode_smf,
    encode_vlq,
    parse_smf,
    resolve_gm_instrument,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, b"\x00"),
        (0x40, b"\x40"),
        (0x7F, b"\x7F"),
        (0x80, b"\x81\x00"),
        (0x2000, b"\xC0\x00"),
        (0x3FFF, b"\xFF\x7F"),
        (0x4000, b"\x81\x80\x00"),
        (0x100000, b"\xC0\x80\x00"),
        (0x0FFFFFFF, b"\xFF\xFF\xFF\x7F"),
    ],
)
def test_vlq_round_trip(value: int, expected: bytes):
    encoded = encode_vlq(value)
    assert encoded == expected
    decoded, next_pos = decode_vlq(encoded, 0)
    assert decoded == value
    assert next_pos == len(encoded)


def test_vlq_rejects_negative_and_truncated():
    with pytest.raises(ValueError):
        encode_vlq(-1)
    with pytest.raises(ValueError):
        decode_vlq(b"\x81", 0)
    with pytest.raises(ValueError):
        decode_vlq(b"\x81\x81\x81\x81\x00", 0)


def test_encode_and_parse_smf_type1_round_trip():
    score = MidiScore(
        bpm=112,
        time_signature=(3, 4),
        key="D",
        mode="dorian",
        bars=4,
        ppqn=480,
        tracks=[
            MidiTrack(
                name="Lute",
                instrument="lute",
                gm_program=24,
                channel=0,
                layer="base",
                pan=48,
                volume=95,
                notes=[
                    MidiNote(0.0, 1.0, 62, 88),
                    MidiNote(1.0, 0.5, 65, 80),
                    MidiNote(1.5, 1.5, 69, 92),
                    # Adjacent note on identical pitch tests note-off before note-on
                    MidiNote(3.0, 1.0, 69, 85),
                ],
            ),
            MidiTrack(
                name="War Drums",
                instrument="drums",
                gm_program=0,
                channel=9,
                layer="combat",
                pan=64,
                volume=110,
                notes=[
                    MidiNote(0.0, 0.25, 36, 100),
                    MidiNote(1.0, 0.25, 38, 90),
                    MidiNote(2.0, 0.25, 42, 70),
                ],
            ),
        ],
    )

    raw = encode_smf(score)
    assert raw[:4] == b"MThd"
    hdr_len, fmt, n_tracks, ppqn = struct.unpack(">IHHH", raw[4:14])
    assert hdr_len == 6
    assert fmt == 1
    assert n_tracks == 3  # conductor + 2 instrument tracks
    assert ppqn == 480

    parsed = parse_smf(raw)
    assert parsed.bpm == 112
    assert parsed.time_signature == (3, 4)
    assert parsed.key == "D"
    assert parsed.mode == "dorian"
    assert parsed.bars == 4
    assert parsed.ppqn == 480
    assert len(parsed.tracks) == 2

    lute, drums = parsed.tracks
    assert lute.name == "Lute"
    assert lute.instrument == "lute"
    assert lute.gm_program == 24
    assert lute.channel == 0
    assert lute.layer == "base"
    assert lute.pan == 48
    assert lute.volume == 95
    assert len(lute.notes) == 4
    assert lute.notes[0].pitch == 62
    assert lute.notes[0].velocity == 88
    assert pytest.approx(lute.notes[2].duration_beats, abs=1e-3) == 1.5
    assert pytest.approx(lute.notes[3].start_beat, abs=1e-3) == 3.0

    assert drums.name == "War Drums"
    assert drums.instrument == "drums"
    assert drums.channel == 9
    assert drums.layer == "combat"
    assert [n.pitch for n in drums.notes] == [36, 38, 42]


def test_parse_smf_handles_running_status_and_vel0_note_off():
    """Manually crafted Type 0 track using running status and vel=0 note-off."""
    # Delta 0: Note On ch0 C4 (60) vel 90 -> 00 90 3C 5A
    # Delta 480 (0x83 0x60): Running status Note On C4 vel 0 (note-off) -> 83 60 3C 00
    # Delta 0: End of Track -> 00 FF 2F 00
    track_body = bytes([
        0x00, 0x90, 0x3C, 0x5A,
        0x83, 0x60, 0x3C, 0x00,
        0x00, 0xFF, 0x2F, 0x00,
    ])
    smf = (
        b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480)
        + b"MTrk" + struct.pack(">I", len(track_body)) + track_body
    )
    score = parse_smf(smf)
    assert len(score.tracks) == 1
    assert len(score.tracks[0].notes) == 1
    note = score.tracks[0].notes[0]
    assert note.pitch == 60
    assert note.velocity == 90
    assert pytest.approx(note.start_beat, abs=1e-4) == 0.0
    assert pytest.approx(note.duration_beats, abs=1e-4) == 1.0


def test_parse_smf_rejects_corrupt_header():
    with pytest.raises(ValueError, match="MThd"):
        parse_smf(b"RIFF0000WAVE")
    # SMPTE negative division rejected
    bad_div = b"MThd" + struct.pack(">IHHH", 6, 1, 0, 0xE728)
    with pytest.raises(ValueError, match="SMPTE"):
        parse_smf(bad_div)


def test_resolve_gm_instrument_presets_and_percussion():
    assert resolve_gm_instrument("lute") == (24, False)
    assert resolve_gm_instrument("flute") == (73, False)
    assert resolve_gm_instrument("fiddle") == (110, False)
    assert resolve_gm_instrument("celesta") == (8, False)
    assert resolve_gm_instrument("accordion") == (21, False)
    assert resolve_gm_instrument("harp") == (46, False)
    assert resolve_gm_instrument("strings") == (48, False)
    assert resolve_gm_instrument("pad") == (89, False)
    assert resolve_gm_instrument("bell") == (14, False)
    assert resolve_gm_instrument("drums") == (0, True)
    assert resolve_gm_instrument("gm:42") == (42, False)
