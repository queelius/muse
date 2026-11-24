"""Pure-Python ABC notation -> ``MidiScore`` parser.

Parses standard ABC 2.1 headers (``X:``, ``T:``, ``M:``, ``L:``, ``Q:``,
``K:``, ``V:``) and body syntax (accidentals ``^``/``_``/``=``, octaves
``'``/``,``, durations ``2``/``/2``/``3/2``, broken rhythm ``>``/``<``,
rests ``z``/``x``/``Z``, chords ``[CEG]``, tuplets ``(3``, ties ``-``,
bar lines ``|``, inline fields ``[K:...]``/``[V:...]``, and
``%%MIDI program`` directives) into a ``MidiScore``.

Used by ``AbcMidiLMRuntime`` to convert neural symbolic LM output into
SMF Type 1 MIDI files and structured scores.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from muse.modalities.music_midi.protocol import MidiNote, MidiScore, MidiTrack
from muse.modalities.music_midi.smf import resolve_gm_instrument


# Pitch-class semitones for natural note letters (C4 = 60).
_LETTER_SEMITONE: dict[str, int] = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}

# Sharp and flat accumulation order on the circle of fifths.
_SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
_FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")

# Circle-of-fifths index for major keys.
_MAJOR_FIFTHS: dict[str, int] = {
    "Cb": -7, "Gb": -6, "Db": -5, "Ab": -4, "Eb": -3, "Bb": -2, "F": -1,
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
    "G#": -4, "D#": -3, "A#": -2,
}

# Mode aliases -> (canonical_mode, circle_of_fifths_offset)
_MODE_ALIASES: dict[str, tuple[str, int]] = {
    "": ("major", 0),
    "maj": ("major", 0),
    "major": ("major", 0),
    "ion": ("major", 0),
    "ionian": ("major", 0),
    "m": ("minor", -3),
    "min": ("minor", -3),
    "minor": ("minor", -3),
    "aeo": ("aeolian", -3),
    "aeolian": ("aeolian", -3),
    "dor": ("dorian", -2),
    "dorian": ("dorian", -2),
    "mix": ("mixolydian", -1),
    "mixolydian": ("mixolydian", -1),
    "lyd": ("lydian", 1),
    "lydian": ("lydian", 1),
    "phr": ("phrygian", -4),
    "phrygian": ("phrygian", -4),
    "loc": ("phrygian", -5),
    "locrian": ("phrygian", -5),
}

_KEY_RE = re.compile(r"^([A-Ga-g])([#b]?)\s*([A-Za-z]*)")
_FRAC_RE = re.compile(r"^(\d+)\s*/\s*(\d+)$")
_TEMPO_RE = re.compile(r"(?:(\d+)\s*/\s*(\d+)\s*=\s*)?(\d+)")
_DUR_RE = re.compile(r"^(\d*)(/*)(\d*)")
_NOTE_RE = re.compile(r"^(\^{1,2}|_{1,2}|=)?([A-Ga-g])([',]*)")
_MIDI_PROG_RE = re.compile(r"^%%MIDI\s+program\s+(?:(\d+)\s+)?(\d+)", re.IGNORECASE)
_MIDI_CHAN_RE = re.compile(r"^%%MIDI\s+channel\s+(\d+)", re.IGNORECASE)
_VOICE_RE = re.compile(r"^(\S+)(.*)$")


def parse_abc_key(spec: str) -> tuple[str, str, dict[str, int]]:
    """Parse an ABC ``K:`` specification into ``(tonic, mode, key_sig_map)``."""
    cleaned = spec.strip()
    if not cleaned or cleaned.lower() in ("none", "hp", "perc"):
        return "C", "major", {}
    m = _KEY_RE.match(cleaned)
    if not m:
        return "C", "major", {}
    letter = m.group(1).upper()
    acc = m.group(2)
    mode_token = m.group(3).lower()
    tonic = letter + acc

    # Match full or 3-char prefix (per ABC 2.1 spec: "dor", "mix", "min", etc.)
    if mode_token in _MODE_ALIASES:
        canonical_mode, offset = _MODE_ALIASES[mode_token]
    elif len(mode_token) >= 3 and mode_token[:3] in _MODE_ALIASES:
        canonical_mode, offset = _MODE_ALIASES[mode_token[:3]]
    else:
        canonical_mode, offset = ("major", 0)

    fifths = _MAJOR_FIFTHS.get(tonic, 0) + offset
    while fifths > 7:
        fifths -= 12
    while fifths < -7:
        fifths += 12

    sig: dict[str, int] = {}
    if fifths > 0:
        for note_l in _SHARP_ORDER[:fifths]:
            sig[note_l] = 1
    elif fifths < 0:
        for note_l in _FLAT_ORDER[: -fifths]:
            sig[note_l] = -1
    return tonic, canonical_mode, sig


def _parse_meter(spec: str) -> tuple[int, int]:
    s = spec.strip()
    if s == "C":
        return (4, 4)
    if s == "C|":
        return (2, 2)
    m = _FRAC_RE.match(s)
    if m:
        num = max(1, int(m.group(1)))
        den = max(1, int(m.group(2)))
        return (num, den)
    return (4, 4)


def _parse_unit_length(spec: str) -> float:
    """Return unit length in quarter-note beats (e.g. ``1/8`` -> ``0.5``)."""
    m = _FRAC_RE.match(spec.strip())
    if m:
        num = max(1, int(m.group(1)))
        den = max(1, int(m.group(2)))
        return (float(num) / float(den)) * 4.0
    return 0.5


def _parse_tempo(spec: str) -> int:
    """Return quarter-note BPM from an ABC ``Q:`` field."""
    # Strip optional quoted tempo text like "Allegro"
    s = re.sub(r'"[^"]*"', "", spec).strip()
    m = _TEMPO_RE.search(s)
    if not m:
        return 120
    num_s, den_s, val_s = m.groups()
    val = int(val_s)
    if num_s and den_s:
        beat_unit = (float(num_s) / float(den_s)) * 4.0
        bpm = round(val * beat_unit)
    else:
        bpm = val
    return max(20, min(300, bpm))


def _parse_duration_multiplier(text: str, pos: int) -> tuple[float, int]:
    """Parse an ABC duration suffix (e.g. ``2``, ``/2``, ``3/2``, ``//``)."""
    m = _DUR_RE.match(text[pos:])
    if not m or m.end() == 0:
        return 1.0, pos
    num_s, slashes, den_s = m.groups()
    end = pos + m.end()
    num = int(num_s) if num_s else 1
    if not slashes:
        return float(num), end
    if den_s:
        den = max(1, int(den_s))
    else:
        den = 1 << len(slashes)
    return float(num) / float(den), end


def _parse_voice_attrs(rest: str) -> dict[str, str]:
    """Parse key=value attributes on a ``V:`` directive."""
    attrs: dict[str, str] = {}
    for m in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|(\S+))', rest):
        attrs[m.group(1).lower()] = m.group(2) if m.group(2) is not None else m.group(3)
    return attrs


@dataclass
class _VoiceState:
    voice_id: str
    name: str
    instrument: str = "piano"
    gm_program: int = 0
    channel: int = 0
    beat: float = 0.0
    notes: list[MidiNote] = field(default_factory=list)
    bar_accidentals: dict[tuple[str, int], int] = field(default_factory=dict)
    pending_tie: dict[int, int] = field(default_factory=dict)  # pitch -> note idx
    pending_broken: float = 1.0


def _note_to_midi_pitch(
    acc_str: str | None,
    letter: str,
    octave_marks: str,
    key_sig: dict[str, int],
    bar_accidentals: dict[tuple[str, int], int],
) -> int:
    is_lower = letter.islower()
    upper = letter.upper()
    octave = 5 if is_lower else 4
    octave += octave_marks.count("'") - octave_marks.count(",")

    if acc_str:
        if acc_str == "=":
            acc = 0
        elif acc_str.startswith("^"):
            acc = len(acc_str)
        else:
            acc = -len(acc_str)
        bar_accidentals[(upper, octave)] = acc
    elif (upper, octave) in bar_accidentals:
        acc = bar_accidentals[(upper, octave)]
    else:
        acc = key_sig.get(upper, 0)

    pitch = (octave + 1) * 12 + _LETTER_SEMITONE[upper] + acc
    return max(0, min(127, pitch))


def _tuplet_factor(p: int) -> float:
    """Standard ABC tuplet duration multiplier for ``(p``."""
    table = {
        2: 3.0 / 2.0,
        3: 2.0 / 3.0,
        4: 3.0 / 4.0,
        5: 4.0 / 5.0,
        6: 4.0 / 6.0,
        7: 4.0 / 7.0,
        8: 6.0 / 8.0,
        9: 8.0 / 9.0,
    }
    return table.get(p, 2.0 / max(1, p))


def extract_abc_block(text: str) -> str:
    """Extract the most plausible ABC tune block from raw LLM text output.

    Strips markdown code fences and any conversational preamble before
    the first ABC header line (``X:``, ``M:``, ``L:``, ``Q:``, ``K:``).
    """
    cleaned = text.strip()
    fence = re.search(r"```(?:abc)?\s*\n(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()

    lines = cleaned.splitlines()
    start_idx = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^[XTMQLKV]:", stripped):
            start_idx = idx
            break
    return "\n".join(lines[start_idx:]).strip()


def parse_abc(
    abc_text: str,
    *,
    default_bpm: int = 120,
    default_time_signature: tuple[int, int] = (4, 4),
    default_key: str = "C",
    default_mode: str = "major",
    ppqn: int = 480,
) -> MidiScore:
    """Parse an ABC notation string into a multi-track ``MidiScore``.

    Raises ``ValueError`` if the input contains no playable notes.
    """
    if not abc_text or not abc_text.strip():
        raise ValueError("empty ABC notation")

    time_sig = default_time_signature
    meter_explicit = False
    unit_beats: float | None = None
    bpm = int(default_bpm)
    key = default_key
    mode = default_mode
    key_sig: dict[str, int] = {}
    if default_key:
        _, _, key_sig = parse_abc_key(f"{default_key} {default_mode}")

    voices: dict[str, _VoiceState] = {}
    voice_order: list[str] = []
    current_voice_id = "1"
    in_body = False

    def _get_voice(vid: str, attrs: dict[str, str] | None = None) -> _VoiceState:
        nonlocal voice_order
        if vid not in voices:
            ch = len(voice_order)
            if ch >= 9:
                ch += 1  # skip GM percussion channel unless percussion
            ch = ch % 16
            name = (attrs or {}).get("name") or (attrs or {}).get("nm") or f"Voice {vid}"
            inst = name.lower() if (attrs and ("name" in attrs or "nm" in attrs)) else "piano"
            prog, is_perc = resolve_gm_instrument(inst)
            if attrs and "program" in attrs and attrs["program"].isdigit():
                prog = max(0, min(127, int(attrs["program"])))
            if is_perc:
                ch = 9
            voices[vid] = _VoiceState(
                voice_id=vid,
                name=name,
                instrument=inst if inst != f"voice {vid}" else "piano",
                gm_program=prog,
                channel=ch,
            )
            voice_order.append(vid)
        elif attrs:
            v = voices[vid]
            if "name" in attrs or "nm" in attrs:
                v.name = attrs.get("name") or attrs.get("nm") or v.name
                prog, is_perc = resolve_gm_instrument(v.name)
                if prog != 0 or is_perc:
                    v.instrument = v.name.lower()
                    v.gm_program = prog
                    if is_perc:
                        v.channel = 9
            if "program" in attrs and attrs["program"].isdigit():
                v.gm_program = max(0, min(127, int(attrs["program"])))
        return voices[vid]

    def _effective_unit_beats() -> float:
        if unit_beats is not None:
            return unit_beats
        num, den = time_sig
        return 0.25 if (num / float(den)) < 0.75 else 0.5

    def _apply_header(tag: str, val: str) -> None:
        nonlocal time_sig, meter_explicit, unit_beats, bpm, key, mode, key_sig
        nonlocal current_voice_id, in_body
        if tag == "M":
            time_sig = _parse_meter(val)
            meter_explicit = True
        elif tag == "L":
            unit_beats = _parse_unit_length(val)
        elif tag == "Q":
            bpm = _parse_tempo(val)
        elif tag == "K":
            key, mode, key_sig = parse_abc_key(val)
            in_body = True
        elif tag == "V":
            m = _VOICE_RE.match(val.strip())
            if m:
                vid = m.group(1)
                attrs = _parse_voice_attrs(m.group(2))
                _get_voice(vid, attrs)
                current_voice_id = vid

    def _parse_music_line(line: str) -> None:
        nonlocal current_voice_id
        voice = _get_voice(current_voice_id)
        pos = 0
        n = len(line)
        tuplet_remaining = 0
        tuplet_scale = 1.0
        last_chord_note_indices: list[int] = []

        while pos < n:
            ch = line[pos]

            # Whitespace & continuations
            if ch in " \t\r\n\\":
                pos += 1
                continue

            # Grace notes {abc} -> skip cleanly
            if ch == "{":
                close = line.find("}", pos + 1)
                pos = close + 1 if close != -1 else n
                continue

            # Guitar chord symbols "Am7" -> skip cleanly
            if ch == '"':
                close = line.find('"', pos + 1)
                pos = close + 1 if close != -1 else n
                continue

            # Decorations !fermata! or +trill+
            if ch in "!+":
                close = line.find(ch, pos + 1)
                pos = close + 1 if close != -1 else pos + 1
                continue

            # Bar lines reset measure accidentals
            if ch in "|:":
                while pos < n and line[pos] in "|:[]1234567890":
                    if line[pos] == "[" and pos + 1 < n and line[pos + 1] not in "|:1234567890":
                        break
                    pos += 1
                voice.bar_accidentals.clear()
                continue

            # Broken rhythm markers > and <
            if ch in "><":
                count = 0
                while pos < n and line[pos] == ch:
                    count += 1
                    pos += 1
                # One '>' => prev *= 1.5, next *= 0.5; '<' => prev *= 0.5, next *= 1.5
                factor = 1.0 - (0.5 ** count)
                if ch == ">":
                    prev_mult = 1.0 + factor
                    next_mult = 1.0 - factor
                else:
                    prev_mult = 1.0 - factor
                    next_mult = 1.0 + factor
                if last_chord_note_indices:
                    old_dur = voice.notes[last_chord_note_indices[0]].duration_beats
                    new_dur = max(0.03125, old_dur * prev_mult)
                    delta = new_dur - old_dur
                    for idx in last_chord_note_indices:
                        voice.notes[idx].duration_beats = new_dur
                    voice.beat = max(0.0, voice.beat + delta)
                voice.pending_broken = next_mult
                continue

            # Tuplets (3 or (3:2:3
            if ch == "(":
                if pos + 1 < n and line[pos + 1].isdigit():
                    m_tup = re.match(r"^\((\d)(?::(\d)(?::(\d))?)?", line[pos:])
                    if m_tup:
                        p = int(m_tup.group(1))
                        q = int(m_tup.group(2)) if m_tup.group(2) else None
                        r = int(m_tup.group(3)) if m_tup.group(3) else p
                        tuplet_scale = (q / float(p)) if q else _tuplet_factor(p)
                        tuplet_remaining = r
                        pos += m_tup.end()
                        continue
                # Slur open
                pos += 1
                continue

            if ch == ")":
                pos += 1
                continue

            # Tie marker
            if ch == "-":
                for idx in last_chord_note_indices:
                    voice.pending_tie[voice.notes[idx].pitch] = idx
                pos += 1
                continue

            # Inline field [K:...] or chord [CEG]2
            if ch == "[":
                if pos + 2 < n and line[pos + 1] in "MLQKV" and line[pos + 2] == ":":
                    close = line.find("]", pos + 3)
                    if close != -1:
                        tag = line[pos + 1]
                        val = line[pos + 3 : close].strip()
                        _apply_header(tag, val)
                        voice = _get_voice(current_voice_id)
                        pos = close + 1
                        continue
                # Chord [CEG]
                close = line.find("]", pos + 1)
                if close != -1:
                    inner = line[pos + 1 : close]
                    pos = close + 1
                    outer_mult, pos = _parse_duration_multiplier(line, pos)
                    scale = voice.pending_broken
                    voice.pending_broken = 1.0
                    if tuplet_remaining > 0:
                        scale *= tuplet_scale
                        tuplet_remaining -= 1
                    u_beats = _effective_unit_beats()
                    chord_indices: list[int] = []
                    chord_dur: float | None = None
                    ipos = 0
                    while ipos < len(inner):
                        m_note = _NOTE_RE.match(inner[ipos:])
                        if not m_note:
                            ipos += 1
                            continue
                        acc_s, letter, oct_s = m_note.groups()
                        ipos += m_note.end()
                        in_mult, ipos = _parse_duration_multiplier(inner, ipos)
                        pitch = _note_to_midi_pitch(
                            acc_s, letter, oct_s, key_sig, voice.bar_accidentals,
                        )
                        dur = max(0.03125, u_beats * in_mult * outer_mult * scale)
                        if chord_dur is None:
                            chord_dur = dur
                        if pitch in voice.pending_tie:
                            prev_idx = voice.pending_tie.pop(pitch)
                            voice.notes[prev_idx].duration_beats += dur
                            chord_indices.append(prev_idx)
                        else:
                            voice.notes.append(
                                MidiNote(
                                    start_beat=voice.beat,
                                    duration_beats=dur,
                                    pitch=pitch,
                                    velocity=82,
                                )
                            )
                            chord_indices.append(len(voice.notes) - 1)
                    if chord_dur is not None:
                        voice.beat += chord_dur
                        last_chord_note_indices = chord_indices
                    continue
                pos += 1
                continue

            # Rests: z, x (unit rest) and Z (multi-measure rest)
            if ch in "zx":
                pos += 1
                mult, pos = _parse_duration_multiplier(line, pos)
                scale = voice.pending_broken
                voice.pending_broken = 1.0
                if tuplet_remaining > 0:
                    scale *= tuplet_scale
                    tuplet_remaining -= 1
                voice.beat += _effective_unit_beats() * mult * scale
                last_chord_note_indices = []
                continue

            if ch == "Z":
                pos += 1
                m_bars = re.match(r"^(\d*)", line[pos:])
                n_bars = int(m_bars.group(1)) if (m_bars and m_bars.group(1)) else 1
                if m_bars:
                    pos += m_bars.end()
                beats_per_bar = time_sig[0] * (4.0 / time_sig[1])
                voice.beat += n_bars * beats_per_bar
                last_chord_note_indices = []
                continue

            # Single note
            m_note = _NOTE_RE.match(line[pos:])
            if m_note:
                acc_s, letter, oct_s = m_note.groups()
                pos += m_note.end()
                mult, pos = _parse_duration_multiplier(line, pos)
                scale = voice.pending_broken
                voice.pending_broken = 1.0
                if tuplet_remaining > 0:
                    scale *= tuplet_scale
                    tuplet_remaining -= 1
                dur = max(0.03125, _effective_unit_beats() * mult * scale)
                pitch = _note_to_midi_pitch(
                    acc_s, letter, oct_s, key_sig, voice.bar_accidentals,
                )
                if pitch in voice.pending_tie:
                    prev_idx = voice.pending_tie.pop(pitch)
                    voice.notes[prev_idx].duration_beats += dur
                    last_chord_note_indices = [prev_idx]
                else:
                    voice.notes.append(
                        MidiNote(
                            start_beat=voice.beat,
                            duration_beats=dur,
                            pitch=pitch,
                            velocity=84,
                        )
                    )
                    last_chord_note_indices = [len(voice.notes) - 1]
                voice.beat += dur
                continue

            # Skip unknown decorations/articulation characters (., ~, H, u, v, etc.)
            pos += 1

    for raw_line in abc_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        # Handle %%MIDI directives before comment stripping
        if stripped.startswith("%%"):
            m_prog = _MIDI_PROG_RE.match(stripped)
            if m_prog:
                prog = max(0, min(127, int(m_prog.group(2))))
                v = _get_voice(current_voice_id)
                v.gm_program = prog
                continue
            m_chan = _MIDI_CHAN_RE.match(stripped)
            if m_chan:
                ch_1 = int(m_chan.group(1))
                v = _get_voice(current_voice_id)
                v.channel = max(0, min(15, ch_1 - 1))
                continue
            continue

        # Strip inline % comments
        comment_idx = stripped.find("%")
        if comment_idx != -1:
            stripped = stripped[:comment_idx].strip()
        if not stripped:
            continue

        # Full-line header / field directive (e.g. X:1, T:Title, M:4/4, K:D dor, V:1)
        if len(stripped) >= 2 and stripped[0].isalpha() and stripped[1] == ":":
            tag = stripped[0]
            val = stripped[2:].strip()
            if tag in ("X", "T", "C", "R", "S", "N", "O", "Z", "B", "D", "F", "G", "H", "I", "P", "w", "W"):
                continue
            if tag in ("M", "L", "Q", "K", "V"):
                _apply_header(tag, val)
                continue

        _parse_music_line(stripped)

    tracks: list[MidiTrack] = []
    max_beat = 0.0
    for vid in voice_order:
        v = voices[vid]
        if not v.notes:
            continue
        for note in v.notes:
            end_b = note.start_beat + note.duration_beats
            if end_b > max_beat:
                max_beat = end_b
        tracks.append(
            MidiTrack(
                name=v.name,
                instrument=v.instrument,
                gm_program=v.gm_program,
                channel=v.channel,
                layer="base",
                notes=v.notes,
            )
        )

    if not tracks:
        raise ValueError("ABC notation contained no playable notes")

    beats_per_bar = time_sig[0] * (4.0 / time_sig[1])
    bars = max(1, math.ceil(max_beat / beats_per_bar)) if beats_per_bar > 0 else 1

    return MidiScore(
        bpm=bpm,
        time_signature=time_sig if meter_explicit else default_time_signature,
        key=key,
        mode=mode,
        bars=bars,
        ppqn=ppqn,
        tracks=tracks,
    )
