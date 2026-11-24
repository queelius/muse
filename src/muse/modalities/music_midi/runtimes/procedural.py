"""``ProceduralMidiComposer``: deterministic weightless CPU MIDI composer.

Pure Python standard library (zero external ML dependencies, zero weights).
Infers mood, key, mode, tempo, and instrumentation from a free-text prompt
when callers leave those fields unspecified, then composes a multi-track
Standard MIDI File (Type 1) with:

- A / B / A' musical form with cadential resolution at the loop boundary
- Mode-idiomatic chord progressions across 7 modes
- Contour-shaped melodic motifs with phrase variation and temperature control
- Role-aware tracks (lead melody, plucked/arpeggio, bass, sustained pad, bell,
  and GM channel 9 percussion) tailored to 4/4, 3/4, and 6/8 meters
- Harmonically locked adaptive layers (``base``, ``combat``, ``storm``,
  ``night``, ``altitude``, or custom layer names) sharing identical bar-by-bar
  harmony for seamless crossfading in adaptive Web Audio MIDI engines.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
from typing import Any

from muse.modalities.music_midi.protocol import (
    MidiGenerationResult,
    MidiNote,
    MidiScore,
    MidiTrack,
)
from muse.modalities.music_midi.smf import (
    GM_INSTRUMENTS,
    PERCUSSION_NAMES,
    encode_smf,
    resolve_gm_instrument,
)


# Pitch class offsets (C = 0 .. B = 11).
KEY_TO_SEMITONE: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

SUPPORTED_MODES: tuple[str, ...] = (
    "major", "minor", "dorian", "mixolydian", "lydian", "phrygian", "aeolian",
)

MODE_INTERVALS: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
}

# Idiomatic 4-bar chord degree progressions (0-indexed scale degrees)
# for Section A (home) and Section B (contrasting bridge).
_PROGRESSIONS: dict[str, dict[str, list[tuple[int, int, int, int]]]] = {
    "major": {
        "A": [(0, 4, 5, 3), (0, 3, 4, 0), (0, 5, 3, 4)],
        "B": [(5, 3, 1, 4), (3, 4, 2, 5)],
    },
    "minor": {
        "A": [(0, 5, 2, 6), (0, 3, 5, 4), (0, 6, 5, 4)],
        "B": [(3, 5, 6, 4), (5, 2, 3, 4)],
    },
    "aeolian": {
        "A": [(0, 5, 2, 6), (0, 6, 5, 6), (0, 3, 6, 0)],
        "B": [(3, 5, 6, 0), (5, 6, 3, 4)],
    },
    "dorian": {
        "A": [(0, 3, 0, 6), (0, 6, 3, 0), (0, 2, 3, 6)],
        "B": [(3, 4, 6, 0), (2, 3, 5, 6)],
    },
    "mixolydian": {
        "A": [(0, 6, 3, 0), (0, 3, 6, 3), (0, 6, 4, 0)],
        "B": [(3, 6, 4, 0), (6, 3, 1, 4)],
    },
    "lydian": {
        "A": [(0, 1, 0, 4), (0, 1, 5, 4), (0, 4, 1, 0)],
        "B": [(5, 1, 4, 0), (1, 5, 2, 4)],
    },
    "phrygian": {
        "A": [(0, 1, 0, 6), (0, 1, 6, 0), (0, 5, 1, 0)],
        "B": [(3, 1, 6, 0), (5, 1, 3, 1)],
    },
}

# Prompt mood profiles: (keywords, default_mode, bpm_range, keys, instruments, energy)
_MOOD_PROFILES: list[tuple[tuple[str, ...], str, tuple[int, int], tuple[str, ...], list[str], float]] = [
    (
        ("battle", "combat", "boss", "epic", "war", "action", "chase", "fight", "clash"),
        "minor",
        (132, 150),
        ("D", "E", "A", "C"),
        ["strings", "fiddle", "brass", "drums"],
        0.92,
    ),
    (
        ("tavern", "village", "medieval", "folk", "town", "market", "celtic", "inn", "bard", "pastoral"),
        "dorian",
        (102, 118),
        ("D", "G", "A", "E"),
        ["lute", "flute", "fiddle", "drums"],
        0.65,
    ),
    (
        ("dungeon", "cave", "dark", "eerie", "haunted", "crypt", "shadow", "ominous", "mystery", "horror"),
        "phrygian",
        (74, 92),
        ("E", "D", "A", "C"),
        ["strings", "bell", "pad", "bass"],
        0.45,
    ),
    (
        ("storm", "thunder", "tempest", "ocean", "sea", "wind", "rain", "gale"),
        "minor",
        (114, 130),
        ("C", "D", "G", "E"),
        ["strings", "accordion", "lute", "drums"],
        0.82,
    ),
    (
        ("night", "moon", "lullaby", "dream", "sleep", "star", "twilight", "nocturne", "quiet"),
        "aeolian",
        (68, 84),
        ("A", "E", "D", "F"),
        ["celesta", "harp", "pad"],
        0.32,
    ),
    (
        ("forest", "meadow", "nature", "peaceful", "calm", "ambient", "serene", "gentle", "garden"),
        "lydian",
        (78, 96),
        ("F", "C", "G", "D"),
        ["harp", "flute", "celesta", "pad"],
        0.42,
    ),
    (
        ("sad", "melancholy", "somber", "tragic", "lonely", "lament", "grief", "ruins"),
        "minor",
        (66, 80),
        ("A", "D", "E", "G"),
        ["cello", "harp", "flute", "pad"],
        0.35,
    ),
    (
        ("heroic", "adventure", "quest", "journey", "bright", "happy", "joyful", "festival", "overworld"),
        "mixolydian",
        (114, 128),
        ("G", "D", "C", "A"),
        ["flute", "lute", "strings", "drums"],
        0.74,
    ),
    (
        ("waltz", "ballroom", "dance", "court", "royal", "castle"),
        "major",
        (108, 124),
        ("C", "G", "F", "D"),
        ["accordion", "harp", "strings", "flute"],
        0.62,
    ),
]

# Instrument role classification for multi-track arrangement.
_LEAD_INSTRUMENTS: frozenset[str] = frozenset({
    "flute", "fiddle", "violin", "oboe", "clarinet", "piccolo", "ocarina",
    "pan_flute", "recorder", "trumpet", "synth_lead", "lead", "saxophone",
    "bagpipe", "harmonica",
})
_PLUCK_INSTRUMENTS: frozenset[str] = frozenset({
    "lute", "harp", "guitar", "acoustic_guitar", "steel_guitar",
    "electric_guitar", "banjo", "koto", "shamisen", "sitar", "kalimba",
    "marimba", "harpsichord", "piano", "acoustic_piano", "bright_piano",
    "electric_piano", "accordion", "dulcimer", "pizzicato",
})
_BASS_INSTRUMENTS: frozenset[str] = frozenset({
    "bass", "acoustic_bass", "electric_bass", "finger_bass", "pick_bass",
    "synth_bass", "cello", "contrabass", "bassoon", "tuba", "trombone",
    "timpani",
})
_PAD_INSTRUMENTS: frozenset[str] = frozenset({
    "pad", "warm_pad", "new_age_pad", "choir_pad", "bowed_pad", "atmosphere",
    "strings", "slow_strings", "synth_strings", "tremolo_strings", "choir",
    "voice", "organ", "church_organ", "reed_organ", "brass", "horn",
    "french_horn", "viola",
})
_BELL_INSTRUMENTS: frozenset[str] = frozenset({
    "celesta", "bell", "bells", "tubular_bells", "glockenspiel",
    "music_box", "vibraphone", "xylophone",
})

#GM drum note numbers (channel 9)
_DRUM_KICK = 36
_DRUM_SNARE = 38
_DRUM_CL_HAT = 42
_DRUM_LOW_TOM = 45
_DRUM_OP_HAT = 46
_DRUM_CRASH = 49
_DRUM_TAMB = 54
_DRUM_SHAKER = 70

_KEY_HINT_RE = re.compile(
    r"\b(?:in\s+)?([A-G](?:#|b)?)\s+"
    r"(major|minor|dorian|mixolydian|lydian|phrygian|aeolian)\b",
    re.IGNORECASE,
)


def parse_time_signature(spec: str) -> tuple[int, int]:
    """Parse ``"4/4"``, ``"3/4"``, or ``"6/8"`` into ``(numerator, denominator)``."""
    cleaned = (spec or "4/4").strip()
    if cleaned == "4/4":
        return (4, 4)
    if cleaned == "3/4":
        return (3, 4)
    if cleaned == "6/8":
        return (6, 8)
    raise ValueError(
        f"unsupported time_signature {spec!r}; expected one of '4/4', '3/4', '6/8'"
    )


def normalize_key(key: str) -> str:
    """Validate and canonicalize a tonic key name (e.g. ``'d'`` -> ``'D'``, ``'bb'`` -> ``'Bb'``)."""
    k = (key or "").strip()
    if not k:
        raise ValueError("key must not be empty")
    canon = k[0].upper() + k[1:].replace("♯", "#").replace("♭", "b")
    if canon not in KEY_TO_SEMITONE:
        raise ValueError(
            f"unsupported key {key!r}; expected a pitch name like 'C', 'D', 'F#', 'Bb'"
        )
    return canon


def _scale_pitch(tonic_semi: int, intervals: tuple[int, ...], degree: int, octave: int) -> int:
    """Convert a scale degree (may be negative or >= 7) + base octave to MIDI pitch."""
    n = len(intervals)
    oct_shift, deg_mod = divmod(degree, n)
    midi = (octave + 1 + oct_shift) * 12 + tonic_semi + intervals[deg_mod]
    return max(24, min(108, midi))


def _chord_pitches(
    tonic_semi: int,
    intervals: tuple[int, ...],
    chord_degree: int,
    octave: int,
) -> list[int]:
    """Return triad pitches (root, 3rd, 5th) built diatonically on ``chord_degree``."""
    return [
        _scale_pitch(tonic_semi, intervals, chord_degree + offset, octave)
        for offset in (0, 2, 4)
    ]


def _extract_prompt_instruments(prompt_lower: str) -> list[str]:
    """Find GM instrument presets explicitly named in the prompt."""
    found: list[str] = []
    # Check multi-word names first, then single words, preserving order of mention.
    matches: list[tuple[int, str]] = []
    candidates = sorted(
        set(GM_INSTRUMENTS.keys()) | set(PERCUSSION_NAMES),
        key=len,
        reverse=True,
    )
    occupied: list[tuple[int, int]] = []
    for name in candidates:
        pattern_text = name.replace("_", r"[\s_-]+")
        m = re.search(rf"\b{pattern_text}\b", prompt_lower)
        if not m:
            continue
        span = m.span()
        if any(not (span[1] <= s or span[0] >= e) for s, e in occupied):
            continue
        occupied.append(span)
        matches.append((span[0], name))
    matches.sort(key=lambda pair: pair[0])
    for _, name in matches:
        canon = "drums" if name in PERCUSSION_NAMES else name
        if canon not in found:
            found.append(canon)
    return found


def _infer_prompt_defaults(
    prompt: str,
    rng: random.Random,
) -> tuple[str, str, int, list[str], float]:
    """Infer ``(key, mode, bpm, instruments, energy)`` from ``prompt``."""
    lower = prompt.lower()

    # Check explicit key+mode mention in prompt text (e.g. "in D dorian")
    hint_key: str | None = None
    hint_mode: str | None = None
    m_key = _KEY_HINT_RE.search(prompt)
    if m_key:
        raw_k = m_key.group(1)
        cand = raw_k[0].upper() + raw_k[1:]
        if cand in KEY_TO_SEMITONE:
            hint_key = cand
        hint_mode = m_key.group(2).lower()

    # Score mood profiles by keyword hits
    best_profile = None
    best_hits = 0
    for profile in _MOOD_PROFILES:
        keywords = profile[0]
        hits = sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}", lower))
        if hits > best_hits:
            best_hits = hits
            best_profile = profile

    if best_profile is not None:
        _, prof_mode, (lo_bpm, hi_bpm), keys, default_insts, energy = best_profile
    else:
        prof_mode = rng.choice(["major", "dorian", "minor", "mixolydian"])
        lo_bpm, hi_bpm = (96, 120)
        keys = ("C", "G", "D", "A", "F")
        default_insts = ["lute", "flute", "strings", "drums"]
        energy = 0.58

    # Tempo modifier adjectives in prompt
    if any(w in lower for w in ("fast", "frantic", "rapid", "upbeat", "energetic", "intense")):
        lo_bpm = min(200, lo_bpm + 16)
        hi_bpm = min(215, hi_bpm + 18)
        energy = min(1.0, energy + 0.15)
    elif any(w in lower for w in ("slow", "relaxed", "still", "meditative", "soft")):
        lo_bpm = max(50, lo_bpm - 16)
        hi_bpm = max(62, hi_bpm - 14)
        energy = max(0.2, energy - 0.15)

    inferred_mode = hint_mode or prof_mode
    inferred_key = hint_key or rng.choice(keys)
    inferred_bpm = rng.randint(lo_bpm, hi_bpm)

    mentioned_insts = _extract_prompt_instruments(lower)
    if mentioned_insts:
        insts = list(mentioned_insts)
        # Ensure at least two complementary voices when user named only one
        if len(insts) == 1:
            sole = insts[0]
            if sole in PERCUSSION_NAMES:
                insts = ["lute", "flute", sole]
            elif sole in _LEAD_INSTRUMENTS:
                insts.append("lute")
            else:
                insts.append("flute")
    else:
        insts = list(default_insts)

    return inferred_key, inferred_mode, inferred_bpm, insts, energy


def _classify_role(instrument: str) -> str:
    clean = instrument.strip().lower().replace("-", "_").replace(" ", "_")
    if clean in PERCUSSION_NAMES:
        return "drums"
    if clean in _BASS_INSTRUMENTS:
        return "bass"
    if clean in _BELL_INSTRUMENTS:
        return "bell"
    if clean in _PAD_INSTRUMENTS:
        return "pad"
    if clean in _PLUCK_INSTRUMENTS:
        return "pluck"
    if clean in _LEAD_INSTRUMENTS:
        return "lead"
    return "lead"


def _build_section_plan(bars: int) -> list[str]:
    """Return a per-bar section label (``'A'``, ``'A2'``, ``'B'``, ``'A3'``)."""
    sections: list[str] = []
    phrase_labels = ("A", "A2", "B", "A3")
    if bars <= 4:
        return ["A"] * bars
    if bars <= 8:
        half = bars // 2
        return ["A"] * half + ["B"] * (bars - half)
    for bar_idx in range(bars):
        phrase_idx = (bar_idx // 4) % len(phrase_labels)
        sections.append(phrase_labels[phrase_idx])
    return sections


def _build_chord_progression(
    mode: str,
    bars: int,
    sections: list[str],
    rng: random.Random,
) -> list[int]:
    """Build a bar-by-bar scale-degree chord progression matching A/B form."""
    bank = _PROGRESSIONS.get(mode, _PROGRESSIONS["major"])
    prog_a = rng.choice(bank["A"])
    prog_b = rng.choice(bank["B"])

    chords: list[int] = []
    for bar_idx in range(bars):
        sec = sections[bar_idx]
        pos = bar_idx % 4
        if sec == "B":
            deg = prog_b[pos]
        else:
            deg = prog_a[pos]
        # Cadential pull on the final two bars so loops resolve smoothly
        if bars >= 4 and bar_idx == bars - 2:
            deg = 4 if mode in ("major", "minor", "lydian") else 6
        elif bar_idx == bars - 1:
            deg = 0
        chords.append(deg)
    return chords


def _meter_rhythm_grid(ts: tuple[int, int]) -> tuple[float, list[float], list[float]]:
    """Return ``(beats_per_bar, strong_beats, sub_grid)`` for a time signature."""
    if ts == (3, 4):
        return 3.0, [0.0, 1.0, 2.0], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    if ts == (6, 8):
        # 6 eighth notes = 3.0 quarter-note beats; compound pulse at 0.0 and 1.5
        return 3.0, [0.0, 1.5], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    return 4.0, [0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]


def _generate_motif(
    ts: tuple[int, int],
    rng: random.Random,
    temperature: float,
    energy: float,
) -> list[tuple[float, float, int]]:
    """Generate a 2-bar rhythmic + relative-degree motif: ``[(beat_in_2bar, dur, rel_deg)]``."""
    bpb, _, _ = _meter_rhythm_grid(ts)
    step_choices = [0.5, 1.0, 1.5] if ts != (6, 8) else [0.5, 1.0, 1.5]
    if energy > 0.72 and temperature >= 0.5:
        step_choices = [0.5, 0.5, 1.0, 1.5]
    elif energy < 0.42:
        step_choices = [1.0, 1.5, 2.0]

    motif: list[tuple[float, float, int]] = []
    # Build over 2 bars (total_len = 2 * bpb)
    total_len = 2.0 * bpb
    cur = 0.0
    rel_deg = rng.choice([0, 2, 4])
    rest_prob = max(0.05, min(0.28, 0.22 - 0.14 * energy + 0.06 * (1.0 - min(1.0, temperature))))

    while cur < total_len - 1e-6:
        remaining = total_len - cur
        # Leave breathing room at the end of a 2-bar motif
        if cur >= total_len - 0.5 and rng.random() < 0.45:
            break
        dur = rng.choice(step_choices)
        if dur > remaining:
            dur = remaining
        # Keep notes aligned to bar boundaries where cleaner
        bar_end = (int(cur // bpb) + 1) * bpb
        if cur + dur > bar_end and (bar_end - cur) >= 0.5:
            dur = bar_end - cur

        if cur > 0.0 and rng.random() < rest_prob:
            cur += dur
            continue

        motif.append((cur, dur, rel_deg))
        # Stepwise contour with occasional leaps scaled by temperature
        leap_prob = min(0.55, 0.15 + 0.22 * temperature)
        if rng.random() < leap_prob:
            step = rng.choice([-3, -2, 2, 3])
        else:
            step = rng.choice([-1, -1, 0, 1, 1])
        rel_deg = max(-3, min(7, rel_deg + step))
        cur += dur

    if not motif:
        motif.append((0.0, min(1.0, bpb), 0))
    return motif


def _render_lead_notes(
    tonic_semi: int,
    intervals: tuple[int, ...],
    chords: list[int],
    sections: list[str],
    ts: tuple[int, int],
    rng: random.Random,
    temperature: float,
    energy: float,
    base_octave: int = 5,
    vel_base: int = 86,
) -> list[MidiNote]:
    bpb, strong_beats, _ = _meter_rhythm_grid(ts)
    motif_a = _generate_motif(ts, rng, temperature, energy)
    motif_b = _generate_motif(ts, rng, min(2.0, temperature + 0.15), min(1.0, energy + 0.1))

    notes: list[MidiNote] = []
    bars = len(chords)

    for pair_start in range(0, bars, 2):
        sec = sections[pair_start]
        is_b = sec == "B"
        motif = motif_b if is_b else motif_a
        transpose = 2 if is_b else (1 if sec == "A2" and temperature > 0.6 else 0)
        invert = sec == "A3" and temperature > 0.8 and rng.random() < 0.4

        for offset_beat, dur, rel_deg in motif:
            bar_offset = int(offset_beat // bpb)
            bar_idx = pair_start + bar_offset
            if bar_idx >= bars:
                continue
            beat_in_bar = offset_beat - bar_offset * bpb
            chord_deg = chords[bar_idx]
            eff_rel = (-rel_deg if invert else rel_deg) + transpose

            # Anchor strong beats toward chord tones (0, 2, 4) for musicality
            is_strong = any(abs(beat_in_bar - sb) < 1e-6 for sb in strong_beats)
            if is_strong and rng.random() > 0.28 * min(1.5, temperature):
                chord_tones = (0, 2, 4, 7)
                eff_rel = min(chord_tones, key=lambda ct: abs(ct - eff_rel))

            # Resolve final bar cleanly to tonic
            if bar_idx == bars - 1 and beat_in_bar >= bpb - 1.5:
                eff_rel = 0

            pitch = _scale_pitch(tonic_semi, intervals, chord_deg + eff_rel, base_octave)
            # Keep comfortable melodic range (C4..C6)
            while pitch > 86:
                pitch -= 12
            while pitch < 57:
                pitch += 12

            # Subtle phrase dynamics
            accent = 9 if abs(beat_in_bar) < 1e-6 else (4 if is_strong else -4)
            jitter = rng.randint(-4, 4) if temperature > 0.0 else 0
            vel = max(35, min(120, vel_base + accent + jitter))
            abs_start = bar_idx * bpb + beat_in_bar
            max_dur = (bars * bpb) - abs_start
            eff_dur = max(0.125, min(dur * 0.94, max_dur))
            notes.append(MidiNote(abs_start, eff_dur, pitch, vel))

    return notes


def _render_pluck_notes(
    tonic_semi: int,
    intervals: tuple[int, ...],
    chords: list[int],
    ts: tuple[int, int],
    rng: random.Random,
    energy: float,
    base_octave: int = 3,
    vel_base: int = 72,
) -> list[MidiNote]:
    bpb, _, sub_grid = _meter_rhythm_grid(ts)
    notes: list[MidiNote] = []
    arp_patterns = (
        [0, 1, 2, 1, 0, 2, 1, 2],
        [0, 2, 1, 2, 0, 1, 2, 1],
        [0, 1, 2, 3, 2, 1, 0, 2],
    )
    pattern = rng.choice(arp_patterns)
    step_dur = 0.5 if energy >= 0.45 else 1.0
    grid = sub_grid if step_dur == 0.5 else [b for b in sub_grid if abs(b - round(b)) < 1e-6]

    for bar_idx, chord_deg in enumerate(chords):
        triad = _chord_pitches(tonic_semi, intervals, chord_deg, base_octave)
        ext = triad + [triad[0] + 12]
        bar_start = bar_idx * bpb
        for step_idx, beat_in_bar in enumerate(grid):
            idx = pattern[step_idx % len(pattern)] % len(ext)
            pitch = ext[idx]
            accent = 8 if step_idx == 0 else (-3 if step_idx % 2 == 1 else 2)
            vel = max(30, min(115, vel_base + accent + rng.randint(-3, 3)))
            notes.append(
                MidiNote(
                    start_beat=bar_start + beat_in_bar,
                    duration_beats=step_dur * 0.9,
                    pitch=pitch,
                    velocity=vel,
                )
            )
    return notes


def _render_bass_notes(
    tonic_semi: int,
    intervals: tuple[int, ...],
    chords: list[int],
    ts: tuple[int, int],
    rng: random.Random,
    energy: float,
    vel_base: int = 84,
) -> list[MidiNote]:
    bpb, strong_beats, _ = _meter_rhythm_grid(ts)
    notes: list[MidiNote] = []
    for bar_idx, chord_deg in enumerate(chords):
        root = _scale_pitch(tonic_semi, intervals, chord_deg, 2)
        fifth = _scale_pitch(tonic_semi, intervals, chord_deg + 4, 2)
        bar_start = bar_idx * bpb

        if ts == (4, 4):
            if energy >= 0.7:
                # Driving root-root-fifth-root pulse
                hits = [(0.0, 0.9, root, 6), (1.0, 0.9, root, -2),
                        (2.0, 0.9, fifth, 2), (3.0, 0.9, root, -4)]
            else:
                hits = [(0.0, 1.9, root, 6), (2.0, 1.9, fifth, -2)]
        elif ts == (3, 4):
            hits = [(0.0, 1.9, root, 6), (2.0, 0.95, fifth, -4)]
        else:  # 6/8
            hits = [(0.0, 1.4, root, 6), (1.5, 1.4, fifth, -2)]

        for offset, dur, pitch, acc in hits:
            vel = max(35, min(118, vel_base + acc + rng.randint(-3, 3)))
            notes.append(MidiNote(bar_start + offset, dur, pitch, vel))
    return notes


def _render_pad_notes(
    tonic_semi: int,
    intervals: tuple[int, ...],
    chords: list[int],
    ts: tuple[int, int],
    base_octave: int = 4,
    vel_base: int = 58,
) -> list[MidiNote]:
    bpb, _, _ = _meter_rhythm_grid(ts)
    notes: list[MidiNote] = []
    for bar_idx, chord_deg in enumerate(chords):
        triad = _chord_pitches(tonic_semi, intervals, chord_deg, base_octave)
        bar_start = bar_idx * bpb
        for voice_idx, pitch in enumerate(triad):
            vel = max(28, min(100, vel_base - voice_idx * 3))
            notes.append(MidiNote(bar_start, bpb * 0.98, pitch, vel))
    return notes


def _render_bell_notes(
    tonic_semi: int,
    intervals: tuple[int, ...],
    chords: list[int],
    ts: tuple[int, int],
    rng: random.Random,
    base_octave: int = 5,
    vel_base: int = 64,
) -> list[MidiNote]:
    bpb, _, _ = _meter_rhythm_grid(ts)
    notes: list[MidiNote] = []
    for bar_idx, chord_deg in enumerate(chords):
        # Play resonant accents on phrase starts and every 2nd bar
        if bar_idx % 2 == 0 or bar_idx == len(chords) - 1:
            triad = _chord_pitches(tonic_semi, intervals, chord_deg, base_octave)
            pitch = triad[0] if bar_idx % 4 == 0 else triad[2]
            bar_start = bar_idx * bpb
            notes.append(MidiNote(bar_start, min(2.0, bpb * 0.9), pitch, vel_base))
            if bpb >= 3.0 and rng.random() < 0.55:
                notes.append(
                    MidiNote(bar_start + (bpb / 2.0), 1.0, triad[1], max(32, vel_base - 10))
                )
    return notes


def _render_drum_notes(
    bars: int,
    ts: tuple[int, int],
    rng: random.Random,
    energy: float,
    martial: bool = False,
) -> list[MidiNote]:
    bpb, _, sub_grid = _meter_rhythm_grid(ts)
    notes: list[MidiNote] = []
    base_vel = 96 if martial else int(62 + 34 * energy)

    for bar_idx in range(bars):
        bar_start = bar_idx * bpb
        is_phrase_start = bar_idx % 4 == 0
        is_fill_bar = (bar_idx + 1) % 4 == 0

        if is_phrase_start and (martial or energy > 0.72):
            notes.append(MidiNote(bar_start, 0.5, _DRUM_CRASH, min(118, base_vel + 8)))

        if ts == (4, 4):
            notes.append(MidiNote(bar_start + 0.0, 0.25, _DRUM_KICK, base_vel))
            if energy >= 0.55 or martial:
                notes.append(MidiNote(bar_start + 2.0, 0.25, _DRUM_KICK, base_vel - 6))
            notes.append(MidiNote(bar_start + 1.0, 0.25, _DRUM_SNARE, base_vel - 4))
            notes.append(MidiNote(bar_start + 3.0, 0.25, _DRUM_SNARE, base_vel - 2))
            if martial:
                notes.append(MidiNote(bar_start + 2.5, 0.25, _DRUM_KICK, base_vel - 8))
                notes.append(MidiNote(bar_start + 3.5, 0.25, _DRUM_LOW_TOM, base_vel - 4))
            hat_note = _DRUM_CL_HAT if (energy >= 0.55 or martial) else _DRUM_TAMB
            for beat in sub_grid:
                if energy < 0.45 and abs(beat - round(beat)) > 1e-6:
                    continue
                vel = max(32, base_vel - (18 if abs(beat - round(beat)) < 1e-6 else 26))
                notes.append(MidiNote(bar_start + beat, 0.2, hat_note, vel))
            if is_fill_bar and (energy >= 0.6 or martial):
                notes.append(MidiNote(bar_start + 3.5, 0.25, _DRUM_SNARE, min(120, base_vel)))
        elif ts == (3, 4):
            notes.append(MidiNote(bar_start + 0.0, 0.25, _DRUM_KICK, base_vel))
            notes.append(MidiNote(bar_start + 1.0, 0.25, _DRUM_TAMB, base_vel - 14))
            notes.append(MidiNote(bar_start + 2.0, 0.25, _DRUM_SNARE if martial else _DRUM_SHAKER, base_vel - 10))
            if martial:
                notes.append(MidiNote(bar_start + 1.5, 0.25, _DRUM_LOW_TOM, base_vel - 6))
        else:  # 6/8 compound feel (pulses at 0.0 and 1.5)
            notes.append(MidiNote(bar_start + 0.0, 0.25, _DRUM_KICK, base_vel))
            notes.append(MidiNote(bar_start + 1.5, 0.25, _DRUM_SNARE, base_vel - 4))
            if martial or energy >= 0.65:
                notes.append(MidiNote(bar_start + 2.5, 0.25, _DRUM_KICK, base_vel - 10))
            for beat in sub_grid:
                vel = max(32, base_vel - (16 if beat in (0.0, 1.5) else 28))
                notes.append(MidiNote(bar_start + beat, 0.2, _DRUM_SHAKER if not martial else _DRUM_CL_HAT, vel))

    return notes


class _ChannelAllocator:
    """Allocates distinct MIDI channels 0..15, reserving channel 9 for drums."""

    def __init__(self) -> None:
        self._next_melodic = 0

    def alloc(self, is_percussion: bool) -> int:
        if is_percussion:
            return 9
        ch = self._next_melodic
        if ch == 9:
            ch = 10
        self._next_melodic = (ch + 1) % 16
        if self._next_melodic == 9:
            self._next_melodic = 10
        return ch


def _build_adaptive_layer_tracks(
    layer: str,
    tonic_semi: int,
    intervals: tuple[int, ...],
    chords: list[int],
    sections: list[str],
    ts: tuple[int, int],
    rng: random.Random,
    temperature: float,
    channels: _ChannelAllocator,
) -> list[MidiTrack]:
    """Build harmonically locked adaptive tracks for a named non-base layer."""
    layer_l = layer.strip().lower()
    bpb, _, sub_grid = _meter_rhythm_grid(ts)
    tracks: list[MidiTrack] = []

    if layer_l == "combat":
        # Driving 8th-note ostinato strings + war percussion
        ostinato_notes: list[MidiNote] = []
        for bar_idx, chord_deg in enumerate(chords):
            triad = _chord_pitches(tonic_semi, intervals, chord_deg, 3)
            bar_start = bar_idx * bpb
            for idx, beat in enumerate(sub_grid):
                pitch = triad[0] if idx % 2 == 0 else triad[2]
                vel = 102 if idx == 0 else (88 if idx % 2 == 0 else 76)
                ostinato_notes.append(MidiNote(bar_start + beat, 0.4, pitch, vel))
        prog, _ = resolve_gm_instrument("strings")
        tracks.append(
            MidiTrack(
                name="Combat Ostinato",
                instrument="strings",
                gm_program=prog,
                channel=channels.alloc(False),
                layer=layer,
                pan=54,
                volume=108,
                notes=ostinato_notes,
            )
        )
        drum_notes = _render_drum_notes(len(chords), ts, rng, energy=0.95, martial=True)
        tracks.append(
            MidiTrack(
                name="Combat Percussion",
                instrument="drums",
                gm_program=0,
                channel=channels.alloc(True),
                layer=layer,
                pan=64,
                volume=112,
                notes=drum_notes,
            )
        )
    elif layer_l == "storm":
        # Swelling tremolo low strings + wind-like rapid chordal runs
        trem_notes: list[MidiNote] = []
        for bar_idx, chord_deg in enumerate(chords):
            triad = _chord_pitches(tonic_semi, intervals, chord_deg, 3)
            bar_start = bar_idx * bpb
            for idx, beat in enumerate(sub_grid):
                pitch = triad[idx % len(triad)]
                vel = 70 + int(22 * (idx / max(1, len(sub_grid) - 1)))
                trem_notes.append(MidiNote(bar_start + beat, 0.45, pitch, min(115, vel)))
        prog_s, _ = resolve_gm_instrument("tremolo_strings")
        tracks.append(
            MidiTrack(
                name="Storm Tremolo",
                instrument="tremolo_strings",
                gm_program=prog_s,
                channel=channels.alloc(False),
                layer=layer,
                pan=40,
                volume=98,
                notes=trem_notes,
            )
        )
        prog_p, _ = resolve_gm_instrument("pad")
        tracks.append(
            MidiTrack(
                name="Storm Swell",
                instrument="pad",
                gm_program=prog_p,
                channel=channels.alloc(False),
                layer=layer,
                pan=88,
                volume=86,
                notes=_render_pad_notes(tonic_semi, intervals, chords, ts, base_octave=3, vel_base=74),
            )
        )
    elif layer_l == "night":
        # High celesta shimmer + quiet nocturnal harp arpeggios
        prog_c, _ = resolve_gm_instrument("celesta")
        tracks.append(
            MidiTrack(
                name="Night Celesta",
                instrument="celesta",
                gm_program=prog_c,
                channel=channels.alloc(False),
                layer=layer,
                pan=78,
                volume=82,
                notes=_render_bell_notes(tonic_semi, intervals, chords, ts, rng, base_octave=6, vel_base=54),
            )
        )
        prog_h, _ = resolve_gm_instrument("harp")
        tracks.append(
            MidiTrack(
                name="Night Harp",
                instrument="harp",
                gm_program=prog_h,
                channel=channels.alloc(False),
                layer=layer,
                pan=50,
                volume=76,
                notes=_render_pluck_notes(tonic_semi, intervals, chords, ts, rng, energy=0.3, base_octave=4, vel_base=52),
            )
        )
    elif layer_l == "altitude":
        # Airy high-register pad harmonics + sparse crystalline bells
        prog_p, _ = resolve_gm_instrument("pad")
        tracks.append(
            MidiTrack(
                name="Altitude Air",
                instrument="pad",
                gm_program=prog_p,
                channel=channels.alloc(False),
                layer=layer,
                pan=64,
                volume=80,
                notes=_render_pad_notes(tonic_semi, intervals, chords, ts, base_octave=5, vel_base=52),
            )
        )
        prog_b, _ = resolve_gm_instrument("bell")
        tracks.append(
            MidiTrack(
                name="Altitude Chimes",
                instrument="bell",
                gm_program=prog_b,
                channel=channels.alloc(False),
                layer=layer,
                pan=82,
                volume=74,
                notes=_render_bell_notes(tonic_semi, intervals, chords, ts, rng, base_octave=6, vel_base=50),
            )
        )
    else:
        # Generic custom adaptive layer: harmonically locked counter-melody
        prog, _ = resolve_gm_instrument("flute")
        tracks.append(
            MidiTrack(
                name=f"{layer.title()} Countermelody",
                instrument="flute",
                gm_program=prog,
                channel=channels.alloc(False),
                layer=layer,
                pan=74,
                volume=88,
                notes=_render_lead_notes(
                    tonic_semi, intervals, chords, sections, ts, rng,
                    temperature=temperature, energy=0.55, base_octave=5, vel_base=72,
                ),
            )
        )
    return tracks


class ProceduralMidiComposer:
    """Deterministic weightless CPU MIDI composer (pure stdlib)."""

    def __init__(
        self,
        *,
        model_id: str = "midi-composer",
        hf_repo: str = "",
        local_dir: str | None = None,
        device: str = "cpu",
        **_: Any,
    ) -> None:
        self.model_id = model_id
        self.hf_repo = hf_repo
        self.local_dir = local_dir
        self.device = "cpu"

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
        **_: Any,
    ) -> MidiGenerationResult:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")
        ts = parse_time_signature(time_signature)
        eff_bars = max(4, min(64, int(bars)))
        eff_temp = max(0.0, min(2.0, float(temperature)))

        if seed is None:
            eff_seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
        else:
            eff_seed = int(seed)

        # Combine prompt hash + seed so distinct prompts diverge even under a
        # fixed seed, while identical (prompt, seed, params) are 100% deterministic.
        digest = hashlib.sha256(f"{eff_seed}:{prompt.strip().lower()}".encode("utf-8")).digest()
        derived_seed = int.from_bytes(digest[:8], "big") ^ eff_seed
        rng = random.Random(derived_seed)

        inf_key, inf_mode, inf_bpm, inf_insts, energy = _infer_prompt_defaults(prompt, rng)

        eff_key = normalize_key(key) if key is not None else inf_key
        if mode is not None:
            eff_mode = mode.strip().lower()
            if eff_mode not in MODE_INTERVALS:
                raise ValueError(
                    f"unsupported mode {mode!r}; expected one of {SUPPORTED_MODES}"
                )
        else:
            eff_mode = inf_mode

        eff_bpm = max(40, min(220, int(bpm))) if bpm is not None else inf_bpm
        eff_instruments = (
            [i.strip() for i in instruments if i and i.strip()]
            if instruments
            else inf_insts
        )
        if not eff_instruments:
            eff_instruments = ["lute", "flute"]

        if layers is not None:
            seen_layers: list[str] = []
            for lyr in layers:
                cleaned = lyr.strip()
                if cleaned and cleaned not in seen_layers:
                    seen_layers.append(cleaned)
            eff_layers = seen_layers or ["base"]
        else:
            eff_layers = ["base"]

        tonic_semi = KEY_TO_SEMITONE[eff_key]
        intervals = MODE_INTERVALS[eff_mode]
        sections = _build_section_plan(eff_bars)
        chords = _build_chord_progression(eff_mode, eff_bars, sections, rng)

        channels = _ChannelAllocator()
        tracks: list[MidiTrack] = []

        # Base layer tracks (whenever "base" is included or as primary foundation)
        include_base = "base" in [l.lower() for l in eff_layers]
        if include_base:
            pan_spread = [64, 44, 84, 32, 96, 52, 76]
            has_lead = False
            has_bass = False
            for idx, inst_name in enumerate(eff_instruments):
                prog, is_perc = resolve_gm_instrument(inst_name)
                role = _classify_role(inst_name)
                # First non-drum instrument defaults to lead if no explicit lead yet
                if not is_perc and not has_lead and idx == 0 and role not in ("pluck", "bass", "pad"):
                    role = "lead"
                ch = channels.alloc(is_perc)
                pan = 64 if is_perc else pan_spread[idx % len(pan_spread)]

                if role == "drums":
                    notes = _render_drum_notes(eff_bars, ts, rng, energy=energy)
                    vol = 96
                elif role == "lead":
                    has_lead = True
                    notes = _render_lead_notes(
                        tonic_semi, intervals, chords, sections, ts, rng,
                        temperature=eff_temp, energy=energy, base_octave=5, vel_base=88,
                    )
                    vol = 104
                elif role == "pluck":
                    notes = _render_pluck_notes(
                        tonic_semi, intervals, chords, ts, rng,
                        energy=energy, base_octave=3, vel_base=74,
                    )
                    vol = 92
                elif role == "bass":
                    has_bass = True
                    notes = _render_bass_notes(
                        tonic_semi, intervals, chords, ts, rng,
                        energy=energy, vel_base=86,
                    )
                    vol = 98
                elif role == "bell":
                    notes = _render_bell_notes(
                        tonic_semi, intervals, chords, ts, rng,
                        base_octave=5, vel_base=66,
                    )
                    vol = 84
                else:  # pad / strings
                    notes = _render_pad_notes(
                        tonic_semi, intervals, chords, ts,
                        base_octave=4, vel_base=60,
                    )
                    vol = 82

                tracks.append(
                    MidiTrack(
                        name=inst_name.replace("_", " ").title(),
                        instrument=inst_name,
                        gm_program=prog,
                        channel=ch,
                        layer="base",
                        pan=pan,
                        volume=vol,
                        notes=notes,
                    )
                )

            # Ensure at least one bass anchor when arrangement has multiple pitched
            # voices and none is explicitly a bass instrument (subtle harmonic depth).
            if not has_bass and len(eff_instruments) >= 3 and instruments is None:
                prog_b, _ = resolve_gm_instrument("bass")
                tracks.append(
                    MidiTrack(
                        name="Bass",
                        instrument="bass",
                        gm_program=prog_b,
                        channel=channels.alloc(False),
                        layer="base",
                        pan=64,
                        volume=90,
                        notes=_render_bass_notes(
                            tonic_semi, intervals, chords, ts, rng,
                            energy=energy, vel_base=80,
                        ),
                    )
                )

        # Harmonically locked adaptive layers
        for lyr in eff_layers:
            if lyr.lower() == "base":
                continue
            tracks.extend(
                _build_adaptive_layer_tracks(
                    lyr, tonic_semi, intervals, chords, sections, ts, rng,
                    temperature=eff_temp, channels=channels,
                )
            )

        score = MidiScore(
            bpm=eff_bpm,
            time_signature=ts,
            key=eff_key,
            mode=eff_mode,
            bars=eff_bars,
            ppqn=480,
            tracks=tracks,
        )
        midi_bytes = encode_smf(score)

        return MidiGenerationResult(
            score=score,
            midi_bytes=midi_bytes,
            seed=eff_seed,
            model_id=self.model_id,
            metadata={
                "prompt": prompt,
                "seed": eff_seed,
                "temperature": eff_temp,
                "layers": eff_layers,
                "form": "".join(sections[i][0] for i in range(0, eff_bars, 4)),
                "chord_degrees": chords,
                "runtime": "procedural",
            },
        )
