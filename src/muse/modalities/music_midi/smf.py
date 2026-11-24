"""Pure-Python Standard MIDI File (SMF Type 1) encoder and decoder.

Produces spec-compliant SMF Type 1 files (``MThd`` format 1, default
``ppqn=480``) with a conductor meta track (tempo, time signature, key
signature, loop boundary EOT) followed by one ``MTrk`` chunk per
instrument track (program change, volume CC 7, pan CC 10, sorted
note-on/note-off with VLQ delta times, GM percussion on channel 9).

Also provides ``parse_smf(bytes) -> MidiScore`` for verification,
testing, and round-trip inspection.
"""
from __future__ import annotations

import math
import struct
from collections import defaultdict, deque

from muse.modalities.music_midi.protocol import MidiNote, MidiScore, MidiTrack


# General MIDI 1 instrument name -> program number (0-indexed, 0..127).
# Percussion ("drums", "percussion", "kit") maps to channel 9, program 0.
GM_INSTRUMENTS: dict[str, int] = {
    # Keyboards & pitched percussion
    "piano": 0,
    "acoustic_piano": 0,
    "bright_piano": 1,
    "electric_piano": 4,
    "harpsichord": 6,
    "clavinet": 7,
    "celesta": 8,
    "glockenspiel": 9,
    "music_box": 10,
    "vibraphone": 11,
    "marimba": 12,
    "xylophone": 13,
    "bell": 14,
    "bells": 14,
    "tubular_bells": 14,
    "dulcimer": 15,
    "organ": 19,
    "church_organ": 19,
    "reed_organ": 20,
    "accordion": 21,
    "harmonica": 22,
    # Plucked & guitars
    "lute": 24,
    "guitar": 24,
    "acoustic_guitar": 24,
    "steel_guitar": 25,
    "electric_guitar": 27,
    "overdriven_guitar": 29,
    "distortion_guitar": 30,
    # Bass
    "bass": 32,
    "acoustic_bass": 32,
    "electric_bass": 33,
    "finger_bass": 33,
    "pick_bass": 34,
    "synth_bass": 38,
    # Strings & ensemble
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "contrabass": 43,
    "tremolo_strings": 44,
    "pizzicato": 45,
    "harp": 46,
    "timpani": 47,
    "strings": 48,
    "slow_strings": 49,
    "synth_strings": 50,
    "choir": 52,
    "voice": 53,
    # Brass
    "trumpet": 56,
    "trombone": 57,
    "tuba": 58,
    "french_horn": 60,
    "horn": 60,
    "brass": 61,
    # Woodwinds
    "saxophone": 65,
    "oboe": 68,
    "english_horn": 69,
    "bassoon": 70,
    "clarinet": 71,
    "piccolo": 72,
    "flute": 73,
    "recorder": 74,
    "pan_flute": 75,
    "ocarina": 79,
    # Synth leads & pads
    "lead": 80,
    "synth_lead": 80,
    "pad": 89,
    "warm_pad": 89,
    "new_age_pad": 88,
    "polysynth": 90,
    "choir_pad": 91,
    "bowed_pad": 92,
    "atmosphere": 99,
    # Ethnic / folk
    "sitar": 104,
    "banjo": 105,
    "shamisen": 106,
    "koto": 107,
    "kalimba": 108,
    "bagpipe": 109,
    "fiddle": 110,
    # Percussion (handled on channel 9)
    "drums": 0,
    "drum": 0,
    "percussion": 0,
    "kit": 0,
}

PERCUSSION_NAMES: frozenset[str] = frozenset({
    "drums", "drum", "percussion", "kit", "drum_kit", "drumkit",
})

# Canonical reverse map for common programs when decoding SMF without
# an explicit Instrument Name meta event.
_PROGRAM_TO_NAME: dict[int, str] = {
    0: "piano",
    6: "harpsichord",
    8: "celesta",
    9: "glockenspiel",
    11: "vibraphone",
    12: "marimba",
    14: "bell",
    19: "organ",
    21: "accordion",
    24: "lute",
    32: "bass",
    40: "violin",
    42: "cello",
    46: "harp",
    47: "timpani",
    48: "strings",
    52: "choir",
    56: "trumpet",
    60: "horn",
    61: "brass",
    68: "oboe",
    71: "clarinet",
    73: "flute",
    79: "ocarina",
    80: "synth_lead",
    89: "pad",
    110: "fiddle",
}

# Circle of fifths: tonic pitch name -> fifths index for major key.
_MAJOR_KEY_SF: dict[str, int] = {
    "Cb": -7, "Gb": -6, "Db": -5, "Ab": -4, "Eb": -3, "Bb": -2, "F": -1,
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
    # Enharmonic aliases
    "G#": -4, "D#": -3, "A#": -2, "E#": -1, "B#": 0, "Fb": 4,
}

# Mode offset relative to parallel major on the circle of fifths.
_MODE_SF_OFFSET: dict[str, int] = {
    "lydian": 1,
    "major": 0,
    "mixolydian": -1,
    "dorian": -2,
    "minor": -3,
    "aeolian": -3,
    "phrygian": -4,
}

_MINOR_MODES: frozenset[str] = frozenset({"minor", "aeolian", "dorian", "phrygian"})

_SF_TO_MAJOR_KEY: dict[int, str] = {
    -7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
    0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
}

_SF_TO_MINOR_KEY: dict[int, str] = {
    -7: "Ab", -6: "Eb", -5: "Bb", -4: "F", -3: "C", -2: "G", -1: "D",
    0: "A", 1: "E", 2: "B", 3: "F#", 4: "C#", 5: "G#", 6: "D#", 7: "A#",
}


def resolve_gm_instrument(name: str) -> tuple[int, bool]:
    """Return ``(gm_program, is_percussion)`` for an instrument preset or GM name.

    Accepts preset names (``"lute"``, ``"flute"``, ``"drums"``), spaces
    or hyphens (``"acoustic guitar"``), or explicit ``"gm:46"`` numbers.
    Unknown names fall back cleanly to program 0 (piano).
    """
    cleaned = name.strip().lower().replace("-", "_").replace(" ", "_")
    if cleaned in PERCUSSION_NAMES:
        return 0, True
    if cleaned.startswith("gm:"):
        try:
            prog = int(cleaned.split(":", 1)[1])
            return max(0, min(127, prog)), False
        except ValueError:
            return 0, False
    if cleaned.isdigit():
        return max(0, min(127, int(cleaned))), False
    return GM_INSTRUMENTS.get(cleaned, 0), False


def encode_vlq(value: int) -> bytes:
    """Encode a non-negative integer as a MIDI Variable-Length Quantity."""
    if value < 0:
        raise ValueError(f"VLQ value must be non-negative, got {value}")
    if value == 0:
        return b"\x00"
    chunks: list[int] = []
    v = int(value)
    chunks.append(v & 0x7F)
    v >>= 7
    while v > 0:
        chunks.append((v & 0x7F) | 0x80)
        v >>= 7
    chunks.reverse()
    return bytes(chunks)


def decode_vlq(data: bytes, offset: int) -> tuple[int, int]:
    """Decode a MIDI Variable-Length Quantity starting at ``offset``.

    Returns ``(value, next_offset)``.
    """
    value = 0
    length = len(data)
    for count in range(4):
        if offset >= length:
            raise ValueError("truncated VLQ in MIDI stream")
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            return value, offset
        if count == 3:
            raise ValueError("VLQ exceeds 4 bytes in MIDI stream")
    raise ValueError("invalid VLQ in MIDI stream")  # pragma: no cover


def _key_signature_bytes(key: str, mode: str) -> bytes:
    """Build the 2-byte ``(sf, mi)`` payload for Meta Event 0x59."""
    canonical_key = _normalize_key_name(key)
    mode_l = (mode or "major").lower()
    base_sf = _MAJOR_KEY_SF.get(canonical_key, 0)
    sf = base_sf + _MODE_SF_OFFSET.get(mode_l, 0)
    # Wrap enharmonically into [-7, 7]
    while sf > 7:
        sf -= 12
    while sf < -7:
        sf += 12
    mi = 1 if mode_l in _MINOR_MODES else 0
    return struct.pack(">bB", sf, mi)


def _normalize_key_name(key: str) -> str:
    k = (key or "C").strip()
    if not k:
        return "C"
    letter = k[0].upper()
    acc = k[1:].replace("♯", "#").replace("♭", "b")
    if acc not in ("", "#", "b"):
        acc = ""
    return letter + acc


def _meta_event(meta_type: int, payload: bytes) -> bytes:
    return bytes([0xFF, meta_type]) + encode_vlq(len(payload)) + payload


def _wrap_track_chunk(body: bytes) -> bytes:
    return b"MTrk" + struct.pack(">I", len(body)) + body


def _encode_conductor_track(score: MidiScore) -> bytes:
    """Build Track 0: tempo, time signature, key signature, and loop EOT."""
    ppqn = max(1, int(score.ppqn))
    bpm = max(1, int(score.bpm))
    us_per_quarter = max(1, round(60_000_000 / bpm))
    num, den = score.time_signature
    den_pow = int(round(math.log2(max(1, den))))
    # Standard MIDI clocks per metronome click (36 for compound /8 meters).
    clocks_per_click = 36 if (den == 8 and num % 3 == 0) else 24

    events: list[bytes] = []
    # Track Name
    events.append(b"\x00" + _meta_event(0x03, b"Conductor"))
    # Key/Mode hint in standard SMF text meta so round-trip preserves exact mode
    meta_tag = f"muse:key={_normalize_key_name(score.key)};mode={score.mode};bars={score.bars}"
    events.append(b"\x00" + _meta_event(0x01, meta_tag.encode("utf-8")))
    # Set Tempo (FF 51 03 tt tt tt)
    tempo_bytes = us_per_quarter.to_bytes(3, "big")
    events.append(b"\x00" + _meta_event(0x51, tempo_bytes))
    # Time Signature (FF 58 04 nn dd cc bb)
    ts_bytes = bytes([num & 0xFF, den_pow & 0xFF, clocks_per_click, 8])
    events.append(b"\x00" + _meta_event(0x58, ts_bytes))
    # Key Signature (FF 59 02 sf mi)
    events.append(b"\x00" + _meta_event(0x59, _key_signature_bytes(score.key, score.mode)))

    # End of Track placed at exact loop boundary (bars * beats_per_bar * ppqn)
    total_ticks = max(0, round(score.total_beats * ppqn))
    events.append(encode_vlq(total_ticks) + _meta_event(0x2F, b""))
    return _wrap_track_chunk(b"".join(events))


def _encode_instrument_track(track: MidiTrack, ppqn: int) -> bytes:
    """Build one instrument ``MTrk`` chunk."""
    ch = int(track.channel) & 0x0F
    program = max(0, min(127, int(track.gm_program)))
    volume = max(0, min(127, int(track.volume)))
    pan = max(0, min(127, int(track.pan)))

    header_events: list[bytes] = []
    header_events.append(b"\x00" + _meta_event(0x03, track.name.encode("utf-8")))
    header_events.append(b"\x00" + _meta_event(0x04, track.instrument.encode("utf-8")))
    if track.layer:
        header_events.append(
            b"\x00" + _meta_event(0x01, f"layer:{track.layer}".encode("utf-8"))
        )
    # Program Change (skip bank; standard GM program select)
    header_events.append(b"\x00" + bytes([0xC0 | ch, program]))
    # CC 7 (Channel Volume) and CC 10 (Pan)
    header_events.append(b"\x00" + bytes([0xB0 | ch, 7, volume]))
    header_events.append(b"\x00" + bytes([0xB0 | ch, 10, pan]))

    # Schedule note-on and note-off events sorted cleanly by tick.
    # Order 0 = note-off, Order 1 = note-on so adjacent notes on the same
    # pitch cleanly release before re-attacking at the identical tick.
    scheduled: list[tuple[int, int, int, bytes]] = []
    for note in track.notes:
        start_tick = max(0, round(float(note.start_beat) * ppqn))
        dur_ticks = max(1, round(float(note.duration_beats) * ppqn))
        end_tick = start_tick + dur_ticks
        pitch = max(0, min(127, int(note.pitch)))
        vel = max(1, min(127, int(note.velocity)))
        scheduled.append((start_tick, 1, pitch, bytes([0x90 | ch, pitch, vel])))
        scheduled.append((end_tick, 0, pitch, bytes([0x80 | ch, pitch, 0])))

    scheduled.sort(key=lambda item: (item[0], item[1], item[2]))

    body_parts = list(header_events)
    prev_tick = 0
    for tick, _order, _pitch, msg in scheduled:
        delta = tick - prev_tick
        prev_tick = tick
        body_parts.append(encode_vlq(delta) + msg)

    # End of Track
    body_parts.append(b"\x00" + _meta_event(0x2F, b""))
    return _wrap_track_chunk(b"".join(body_parts))


def encode_smf(score: MidiScore) -> bytes:
    """Encode a ``MidiScore`` into Standard MIDI File Type 1 bytes."""
    ppqn = max(1, int(score.ppqn))
    num_tracks = 1 + len(score.tracks)
    header = b"MThd" + struct.pack(">IHHH", 6, 1, num_tracks, ppqn)
    chunks = [header, _encode_conductor_track(score)]
    for track in score.tracks:
        chunks.append(_encode_instrument_track(track, ppqn))
    return b"".join(chunks)


def _parse_muse_meta(text: str) -> dict[str, str]:
    """Parse ``muse:key=D;mode=dorian;bars=16`` text meta tag if present."""
    if not text.startswith("muse:"):
        return {}
    out: dict[str, str] = {}
    for pair in text[5:].split(";"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_smf(data: bytes) -> MidiScore:
    """Parse Standard MIDI File (Type 0 or Type 1) bytes into a ``MidiScore``.

    Intended for round-trip verification, inspection, and downstream
    symbolic tooling. Handles running status, note-on with velocity 0 as
    note-off, overlapping notes, and meta events.
    """
    if len(data) < 14 or data[:4] != b"MThd":
        raise ValueError("not a valid Standard MIDI File: missing MThd header")
    header_len = struct.unpack(">I", data[4:8])[0]
    if header_len < 6 or 8 + header_len > len(data):
        raise ValueError(f"invalid MThd header length: {header_len}")
    fmt, num_tracks, division = struct.unpack(">HHH", data[8:14])
    if fmt not in (0, 1):
        raise ValueError(f"unsupported SMF format: {fmt} (expected 0 or 1)")
    if division & 0x8000:
        raise ValueError("SMPTE time division is not supported")
    ppqn = division if division > 0 else 480

    offset = 8 + header_len
    bpm = 120
    time_signature = (4, 4)
    key = "C"
    mode = "major"
    explicit_bars: int | None = None
    muse_key_override: str | None = None
    muse_mode_override: str | None = None
    max_tick = 0
    tracks: list[MidiTrack] = []

    for track_idx in range(num_tracks):
        if offset + 8 > len(data):
            raise ValueError(f"truncated SMF before track {track_idx}")
        chunk_id = data[offset : offset + 4]
        chunk_len = struct.unpack(">I", data[offset + 4 : offset + 8])[0]
        offset += 8
        chunk_end = offset + chunk_len
        if chunk_end > len(data):
            raise ValueError(f"truncated chunk {chunk_id!r} at track {track_idx}")
        if chunk_id != b"MTrk":
            offset = chunk_end
            continue

        pos = offset
        tick = 0
        running_status = 0
        track_name = ""
        instrument_name = ""
        layer_name = "base"
        program_by_ch: dict[int, int] = {}
        volume_by_ch: dict[int, int] = {}
        pan_by_ch: dict[int, int] = {}
        active_notes: dict[tuple[int, int], deque[tuple[int, int]]] = defaultdict(deque)
        notes_by_ch: dict[int, list[MidiNote]] = defaultdict(list)

        while pos < chunk_end:
            delta, pos = decode_vlq(data, pos)
            tick += delta
            if pos >= chunk_end:
                break
            status = data[pos]
            if status >= 0x80:
                pos += 1
                if status < 0xF0:
                    running_status = status
            else:
                if running_status == 0:
                    raise ValueError(f"data byte without running status at offset {pos}")
                status = running_status

            if status == 0xFF:
                if pos >= chunk_end:
                    raise ValueError("truncated meta event")
                meta_type = data[pos]
                pos += 1
                meta_len, pos = decode_vlq(data, pos)
                if pos + meta_len > chunk_end:
                    raise ValueError("truncated meta event payload")
                payload = data[pos : pos + meta_len]
                pos += meta_len

                if meta_type == 0x51 and len(payload) == 3:
                    us_per_q = int.from_bytes(payload, "big")
                    if us_per_q > 0:
                        bpm = round(60_000_000 / us_per_q)
                elif meta_type == 0x58 and len(payload) >= 2:
                    ts_num = payload[0]
                    ts_den = 1 << payload[1]
                    if ts_num > 0 and ts_den > 0:
                        time_signature = (ts_num, ts_den)
                elif meta_type == 0x59 and len(payload) == 2:
                    sf, mi = struct.unpack(">bB", payload)
                    if mi == 1:
                        key = _SF_TO_MINOR_KEY.get(sf, "A")
                        mode = "minor"
                    else:
                        key = _SF_TO_MAJOR_KEY.get(sf, "C")
                        mode = "major"
                elif meta_type == 0x03:
                    track_name = payload.decode("utf-8", errors="replace")
                elif meta_type == 0x04:
                    instrument_name = payload.decode("utf-8", errors="replace")
                elif meta_type == 0x01:
                    text = payload.decode("utf-8", errors="replace")
                    if text.startswith("layer:"):
                        layer_name = text[6:].strip() or "base"
                    elif text.startswith("muse:"):
                        parsed = _parse_muse_meta(text)
                        if "key" in parsed:
                            muse_key_override = parsed["key"]
                        if "mode" in parsed:
                            muse_mode_override = parsed["mode"]
                        if "bars" in parsed and parsed["bars"].isdigit():
                            explicit_bars = int(parsed["bars"])
                elif meta_type == 0x2F:
                    if tick > max_tick:
                        max_tick = tick
                    break
            elif status in (0xF0, 0xF7):
                sysex_len, pos = decode_vlq(data, pos)
                pos += sysex_len
            else:
                msg_type = status & 0xF0
                ch = status & 0x0F
                if msg_type in (0xC0, 0xD0):
                    if pos >= chunk_end:
                        raise ValueError("truncated 1-byte channel message")
                    d1 = data[pos]
                    pos += 1
                    if msg_type == 0xC0:
                        program_by_ch[ch] = d1 & 0x7F
                elif msg_type in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                    if pos + 2 > chunk_end:
                        raise ValueError("truncated 2-byte channel message")
                    d1 = data[pos] & 0x7F
                    d2 = data[pos + 1] & 0x7F
                    pos += 2
                    if msg_type == 0x90 and d2 > 0:
                        active_notes[(ch, d1)].append((tick, d2))
                    elif msg_type == 0x80 or (msg_type == 0x90 and d2 == 0):
                        q = active_notes.get((ch, d1))
                        if q:
                            start_t, vel = q.popleft()
                            dur_t = max(1, tick - start_t)
                            notes_by_ch[ch].append(
                                MidiNote(
                                    start_beat=start_t / float(ppqn),
                                    duration_beats=dur_t / float(ppqn),
                                    pitch=d1,
                                    velocity=vel,
                                )
                            )
                    elif msg_type == 0xB0:
                        if d1 == 7:
                            volume_by_ch[ch] = d2
                        elif d1 == 10:
                            pan_by_ch[ch] = d2
                else:
                    raise ValueError(f"unsupported MIDI status byte: 0x{status:02X}")

        if tick > max_tick:
            max_tick = tick

        # Close any dangling notes cleanly at track end
        for (ch, pitch), q in active_notes.items():
            while q:
                start_t, vel = q.popleft()
                dur_t = max(1, tick - start_t)
                notes_by_ch[ch].append(
                    MidiNote(
                        start_beat=start_t / float(ppqn),
                        duration_beats=dur_t / float(ppqn),
                        pitch=pitch,
                        velocity=vel,
                    )
                )

        # Skip pure conductor track (no notes and no channel programs/CCs)
        channels = sorted(set(notes_by_ch) | set(program_by_ch))
        if not channels:
            offset = chunk_end
            continue

        for ch in channels:
            ch_notes = sorted(
                notes_by_ch.get(ch, []),
                key=lambda n: (n.start_beat, n.pitch),
            )
            prog = program_by_ch.get(ch, 0)
            if ch == 9:
                inst = instrument_name or "drums"
            else:
                inst = instrument_name or _PROGRAM_TO_NAME.get(prog, f"gm:{prog}")
            name = track_name or inst
            tracks.append(
                MidiTrack(
                    name=name,
                    instrument=inst,
                    gm_program=prog,
                    channel=ch,
                    layer=layer_name,
                    pan=pan_by_ch.get(ch, 64),
                    volume=volume_by_ch.get(ch, 100),
                    notes=ch_notes,
                )
            )
        offset = chunk_end

    if muse_key_override:
        key = muse_key_override
    if muse_mode_override:
        mode = muse_mode_override

    beats_per_bar = time_signature[0] * (4.0 / time_signature[1])
    if explicit_bars is not None and explicit_bars > 0:
        bars = explicit_bars
    else:
        max_beat = max_tick / float(ppqn)
        for tr in tracks:
            for n in tr.notes:
                end_b = n.start_beat + n.duration_beats
                if end_b > max_beat:
                    max_beat = end_b
        bars = max(1, math.ceil(max_beat / beats_per_bar)) if beats_per_bar > 0 else 1

    return MidiScore(
        bpm=bpm,
        time_signature=time_signature,
        key=key,
        mode=mode,
        bars=bars,
        ppqn=ppqn,
        tracks=tracks,
    )
