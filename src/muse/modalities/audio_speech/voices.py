"""Saved reference voices for audio/speech: saving and synthesis reads.

Layout per voice: ``<root>/<name>/reference.wav`` + ``<root>/<name>/voice.json``.
Writes build the directory under ``<root>/.tmp-<random>`` and ``os.replace``
it into place, under the voice root's exclusive lock (``VoiceCatalog.lock``,
shared with every other process), so readers never see a half-written voice.
Listing, details and metadata edits live in the stdlib ``voice_catalog``;
this module adds the numpy side (validating and writing clips, reading the
waveform for synthesis). No heavy imports: stdlib ``wave`` + numpy only.
"""
from __future__ import annotations

import datetime
import hashlib
import io
import json
import os
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from muse.core import config
from muse.modalities.audio_speech.protocol import (
    VoiceError,
    VoiceUnreadableError,
)
from muse.modalities.audio_speech.voice_catalog import (  # noqa: F401  (re-exported)
    NAME_PATTERN,
    VoiceCatalog,
    normalize_metadata,
    read_voice_file,
    voice_store_root,
)

VOICE_SAMPLE_RATE = 24000
MIN_SECONDS = 1.0
MAX_SECONDS = 30.0
MAX_TRANSCRIPT_CHARS = 1000

_WAV = "reference.wav"
_JSON = "voice.json"


def _muse_home() -> Path:
    """Return the muse config directory, creating it if needed."""
    home = Path(config.get("paths.home")).expanduser()
    home.mkdir(parents=True, exist_ok=True)
    return home


def voices_dir(model_id: str) -> Path:
    """Return the voice storage directory for a model, creating it if needed."""
    d = _muse_home() / "voices" / model_id
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass(frozen=True)
class StoredVoice:
    name: str
    audio_path: Path
    transcript: str
    sample_rate: int
    duration_seconds: float
    sha256: str
    read_only: bool
    identity: tuple
    meta_path: Path | None = None


_read_file = read_voice_file

def _load(name: str, wav_path: Path, json_path: Path, *, read_only: bool) -> StoredVoice:
    try:
        json_bytes, json_id = _read_file(json_path)
        wav_bytes, wav_id = _read_file(wav_path)
        meta = json.loads(json_bytes.decode("utf-8"))
        transcript = str(meta["transcript"])
        sample_rate = int(meta["sample_rate"])
        duration = float(meta["duration_seconds"])
    except FileNotFoundError as exc:
        raise VoiceUnreadableError(f"voice {name!r} is incomplete") from exc
    except OSError as exc:
        raise VoiceUnreadableError(f"voice {name!r} cannot be read") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise VoiceUnreadableError(f"voice {name!r} metadata is unreadable: {exc}") from exc
    return StoredVoice(
        name=name,
        audio_path=wav_path,
        transcript=transcript,
        sample_rate=sample_rate,
        duration_seconds=duration,
        sha256=hashlib.sha256(wav_bytes).hexdigest(),
        read_only=read_only,
        identity=(wav_id, json_id),
        meta_path=json_path,
    )


def load_bundled_voice(name: str, wav_path: Path, json_path: Path) -> StoredVoice:
    """Load a read-only voice shipped alongside a model script."""
    return _load(name, Path(wav_path), Path(json_path), read_only=True)


def read_voice_audio(stored: StoredVoice) -> np.ndarray:
    """Return a stored voice's reference audio as 1-D float32 in [-1, 1].

    Opens ``audio_path`` without following symlinks and refuses a file whose
    identity no longer matches ``stored.identity`` (replaced since ``get``),
    so the waveform always pairs with the transcript it was loaded with.
    """
    try:
        data, wav_id = _read_file(stored.audio_path)
    except OSError as exc:  # FileNotFoundError included: replaced or deleted
        raise VoiceUnreadableError(f"voice {stored.name!r} cannot be read") from exc
    if wav_id != stored.identity[0]:
        raise VoiceUnreadableError(f"voice {stored.name!r} changed while loading")
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            if (w.getnchannels() != 1 or w.getsampwidth() != 2
                    or w.getframerate() != stored.sample_rate):
                raise VoiceUnreadableError(
                    f"voice {stored.name!r} is not mono 16-bit "
                    f"{stored.sample_rate} Hz audio")
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as exc:
        raise VoiceUnreadableError(f"voice {stored.name!r} audio is unreadable") from exc
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if audio.size == 0:
        raise VoiceUnreadableError(f"voice {stored.name!r} audio is empty")
    return audio


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class VoiceStore:
    def __init__(self, root: Path, *, bundled: Mapping[str, StoredVoice] | None = None,
                 max_voices: int | None = None):
        self.root = Path(root)
        self._bundled = dict(bundled or {})
        self._max = max_voices
        self.catalog = VoiceCatalog.for_root(self.root, bundled={
            n: (sv.audio_path, sv.meta_path or Path(sv.audio_path).with_suffix(".json"))
            for n, sv in self._bundled.items()})
        # The catalog's in-process lock: holding it blocks every reader and
        # writer of this root in this process (other processes wait on the
        # file lock that VoiceCatalog.lock takes on first acquisition).
        self._lock = self.catalog._rlock

    # -- reading -----------------------------------------------------------

    def names(self) -> list[str]:
        return self.catalog.names()

    def get(self, name: str) -> StoredVoice:
        if name in self._bundled:
            return self._bundled[name]
        # Under the shared lock so a save can't expose a missing directory
        # or pair one version's transcript with another's audio.
        with self.catalog.lock(exclusive=False):
            d = self.catalog.saved_dir_unlocked(name)
            return _load(name, d / _WAV, d / _JSON, read_only=False)

    def get_with_audio(self, name: str) -> tuple[StoredVoice, np.ndarray]:
        """``get`` plus the reference waveform, read under the shared lock so
        a save or delete in any process cannot land between the two reads."""
        with self.catalog.lock(exclusive=False):
            stored = self.get(name)
            try:
                return stored, read_voice_audio(stored)
            except VoiceUnreadableError:
                # A bundled voice's package files can be rewritten under a
                # live worker (`muse models refresh`); reload it once from
                # package data. Saved voices keep failing closed.
                if not (stored.read_only and name in self._bundled
                        and stored.meta_path is not None):
                    raise
                fresh = _load(name, stored.audio_path, stored.meta_path,
                              read_only=True)
                audio = read_voice_audio(fresh)
                self._bundled[name] = fresh
                return fresh, audio

    # -- writing -----------------------------------------------------------

    @staticmethod
    def _validate(name, audio, sample_rate, transcript) -> tuple[np.ndarray, str]:
        if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
            raise VoiceError(
                "invalid_voice_name",
                "voice name must match [a-z0-9][a-z0-9_-]{0,63}")
        audio = np.asarray(audio)
        if (audio.ndim != 1 or audio.size == 0
                or not np.issubdtype(audio.dtype, np.floating)
                or not np.all(np.isfinite(audio))):
            raise VoiceError(
                "invalid_reference_audio",
                "reference audio must be a non-empty 1-D finite float array")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise VoiceError(
                "invalid_reference_audio", "sample_rate must be a positive integer")
        seconds = audio.size / sample_rate
        if seconds < MIN_SECONDS or seconds > MAX_SECONDS:
            raise VoiceError(
                "invalid_reference_audio",
                f"reference audio must be {MIN_SECONDS:g}-{MAX_SECONDS:g} seconds "
                f"(got {seconds:.2f})")
        text = (transcript or "").strip()
        if not text or len(text) > MAX_TRANSCRIPT_CHARS:
            raise VoiceError(
                "invalid_transcript",
                f"transcript must be 1-{MAX_TRANSCRIPT_CHARS} characters")
        return np.clip(audio.astype("float32"), -1.0, 1.0), text

    def save(self, name: str, audio: np.ndarray, sample_rate: int, transcript: str, *,
             overwrite: bool = False, source_filename: str | None = None,
             metadata: Mapping[str, Any] | None = None) -> StoredVoice:
        audio, text = self._validate(name, audio, sample_rate, transcript)
        fields = normalize_metadata(metadata or {}, partial=False)
        if name in self._bundled:
            raise VoiceError("voice_read_only", f"voice {name!r} is bundled and read-only")
        wav = _wav_bytes(audio, sample_rate)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta = {
            "name": name,
            "transcript": text,
            "sample_rate": sample_rate,
            "duration_seconds": audio.size / sample_rate,
            "sha256": hashlib.sha256(wav).hexdigest(),
            "created_at": now,
            "updated_at": now,
            "source_filename": source_filename,
            **fields,
        }
        # The exclusive lock creates the root (mode 0700) on first save.
        with self.catalog.lock(exclusive=True):
            final = self.root / name
            exists = final.exists() or final.is_symlink()
            if exists and not overwrite:
                raise VoiceError("voice_exists", f"voice {name!r} already exists", 409)
            if (not exists and self._max is not None
                    and len(self.catalog.saved_names_unlocked()) >= self._max):
                raise VoiceError(
                    "voice_limit_reached", f"voice limit of {self._max} reached", 409)
            tmp = self.root / f".tmp-{os.urandom(6).hex()}"
            old = None
            try:
                tmp.mkdir(mode=0o700)
                (tmp / _WAV).write_bytes(wav)
                (tmp / _JSON).write_text(json.dumps(meta, indent=2), encoding="utf-8")
                if exists:
                    old = self.root / f".old-{os.urandom(6).hex()}"
                    os.replace(final, old)
                os.replace(tmp, final)
            except BaseException:
                if old is not None and not final.exists():
                    os.replace(old, final)
                    old = None
                shutil.rmtree(tmp, ignore_errors=True)
                raise
            finally:
                if old is not None:
                    shutil.rmtree(old, ignore_errors=True)
        return self.get(name)

    def delete(self, name: str) -> None:
        """Delete a saved voice (409 ``voice_read_only`` for a bundled one)."""
        self.catalog.delete(name)
