"""Saved reference voices for audio/speech.

Layout per voice: ``<root>/<name>/reference.wav`` + ``<root>/<name>/voice.json``.
Writes build the directory under ``<root>/.tmp-<random>`` and ``os.replace``
it into place, so readers never see a half-written voice. No heavy imports:
stdlib ``wave`` + numpy only.
"""
from __future__ import annotations

import datetime
import hashlib
import io
import json
import os
import re
import shutil
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from muse.core import securefs
from muse.modalities.audio_speech.protocol import (
    UnknownVoiceError,
    VoiceError,
    VoiceUnreadableError,
)

VOICE_SAMPLE_RATE = 24000
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MIN_SECONDS = 1.0
MAX_SECONDS = 30.0
MAX_TRANSCRIPT_CHARS = 1000

_WAV = "reference.wav"
_JSON = "voice.json"


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


def _read_file(path: Path) -> tuple[bytes, tuple]:
    """Read one regular file without following symlinks; return bytes and
    its (mtime_ns, size, ino) taken from the same descriptor."""
    fd = securefs.open_regular_file(
        path, os.O_RDONLY, error=VoiceUnreadableError, label="voice file")
    with os.fdopen(fd, "rb") as fh:
        st = os.fstat(fh.fileno())
        return fh.read(), (st.st_mtime_ns, st.st_size, st.st_ino)


def _load(name: str, wav_path: Path, json_path: Path, *, read_only: bool) -> StoredVoice:
    try:
        json_bytes, json_id = _read_file(json_path)
        wav_bytes, wav_id = _read_file(wav_path)
        meta = json.loads(json_bytes.decode("utf-8"))
        transcript = str(meta["transcript"])
        sample_rate = int(meta["sample_rate"])
        duration = float(meta["duration_seconds"])
    except FileNotFoundError as exc:
        raise VoiceUnreadableError(f"voice {name!r} is incomplete: {exc}") from exc
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
    )


def load_bundled_voice(name: str, wav_path: Path, json_path: Path) -> StoredVoice:
    """Load a read-only voice shipped alongside a model script."""
    return _load(name, Path(wav_path), Path(json_path), read_only=True)


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
        self._lock = threading.Lock()

    # -- reading -----------------------------------------------------------

    def _saved_names(self) -> list[str]:
        try:
            entries = sorted(self.root.iterdir())
        except (FileNotFoundError, NotADirectoryError):
            return []
        names = []
        for d in entries:
            if not NAME_PATTERN.match(d.name) or d.is_symlink() or not d.is_dir():
                continue
            if not (d / _WAV).is_file():
                continue
            try:
                json.loads(_read_file(d / _JSON)[0].decode("utf-8"))
            except (VoiceUnreadableError, FileNotFoundError, ValueError):
                continue
            names.append(d.name)
        return names

    def names(self) -> list[str]:
        return sorted(self._bundled) + [
            n for n in self._saved_names() if n not in self._bundled]

    def _saved_dir(self, name: str) -> Path:
        if not isinstance(name, str) or not NAME_PATTERN.match(name):
            raise UnknownVoiceError(name, self.names())
        d = self.root / name
        if d.is_symlink() or not d.is_dir():
            raise UnknownVoiceError(name, self.names())
        return d

    def get(self, name: str) -> StoredVoice:
        if name in self._bundled:
            return self._bundled[name]
        d = self._saved_dir(name)
        return _load(name, d / _WAV, d / _JSON, read_only=False)

    # -- writing -----------------------------------------------------------

    @staticmethod
    def _validate(name, audio, sample_rate, transcript) -> tuple[np.ndarray, str]:
        if not isinstance(name, str) or not NAME_PATTERN.match(name):
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
             overwrite: bool = False, source_filename: str | None = None) -> StoredVoice:
        audio, text = self._validate(name, audio, sample_rate, transcript)
        if name in self._bundled:
            raise VoiceError("voice_read_only", f"voice {name!r} is bundled and read-only")
        wav = _wav_bytes(audio, sample_rate)
        meta = {
            "name": name,
            "transcript": text,
            "sample_rate": sample_rate,
            "duration_seconds": audio.size / sample_rate,
            "sha256": hashlib.sha256(wav).hexdigest(),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_filename": source_filename,
        }
        with self._lock:
            securefs.ensure_directory(
                self.root, error=VoiceUnreadableError, label="voices directory")
            final = self.root / name
            exists = final.exists() or final.is_symlink()
            if exists and not overwrite:
                raise VoiceError("voice_exists", f"voice {name!r} already exists", 409)
            if (not exists and self._max is not None
                    and len(self._saved_names()) >= self._max):
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
        if name in self._bundled:
            raise VoiceError("voice_read_only", f"voice {name!r} is bundled and read-only")
        with self._lock:
            d = self._saved_dir(name)
            gone = self.root / f".old-{os.urandom(6).hex()}"
            os.replace(d, gone)
            shutil.rmtree(gone, ignore_errors=True)
