"""The saved-voice catalog for audio/speech: metadata rules, reads, edits.

Stdlib only (plus ``muse.core``), so the gateway can serve voice listings,
details, reference audio and metadata edits from disk without loading a
speech model. ``voices.VoiceStore`` (numpy) builds on this module for saves.

Metadata fields a voice may carry beside its clip and transcript:

- ``category``: slash path, 1-4 segments of ``[a-z0-9][a-z0-9_-]{0,31}``
- ``tags``: at most 16 labels of the same shape, deduplicated and sorted
- ``description``: what it sounds like (at most 500 characters)
- ``notes``: what it is for and how to use it (at most 2000 characters)
- ``design``: ``{model, instructions}`` when the clip came from a designed take

Input is normalized the way a person types it: lowercased, trimmed, outer
slashes stripped, a comma string accepted for tags, and an empty string or
null clearing a field.
"""
from __future__ import annotations

import contextlib
import datetime
import json
import os
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import quote

from muse.core import config, securefs
from muse.modalities.audio_speech.voice_errors import (
    UnknownVoiceError,
    VoiceError,
    VoiceUnreadableError,
)

METADATA_KEYS = ("category", "tags", "description", "notes", "design")
SEGMENT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
MAX_CATEGORY_DEPTH = 4
MAX_TAGS = 16
MAX_DESCRIPTION_CHARS = 500
MAX_NOTES_CHARS = 2000
MAX_DESIGN_MODEL_CHARS = 256
MAX_DESIGN_INSTRUCTIONS_CHARS = 4096


def _invalid(field: str, rule: str) -> VoiceError:
    return VoiceError("invalid_voice_metadata", f"{field}: {rule}", 400)


def _category(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid("category", "must be a string or null")
    path = value.strip().lower().strip("/")
    if not path:
        return None
    segments = path.split("/")
    if len(segments) > MAX_CATEGORY_DEPTH:
        raise _invalid(
            "category", f"at most {MAX_CATEGORY_DEPTH} segments (got {len(segments)})")
    for seg in segments:
        if not SEGMENT_PATTERN.fullmatch(seg):
            raise _invalid(
                "category",
                f"segment {seg!r} must match [a-z0-9][a-z0-9_-]{{0,31}}")
    return path


def _tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.split(",")
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raise _invalid("tags", "must be a list of strings or a comma-separated string")
    out: set[str] = set()
    for tag in raw:
        if not isinstance(tag, str):
            raise _invalid("tags", "every tag must be a string")
        tag = tag.strip().lower()
        if not tag:
            continue
        if not SEGMENT_PATTERN.fullmatch(tag):
            raise _invalid("tags", f"tag {tag!r} must match [a-z0-9][a-z0-9_-]{{0,31}}")
        out.add(tag)
    if len(out) > MAX_TAGS:
        raise _invalid("tags", f"at most {MAX_TAGS} tags (got {len(out)})")
    return sorted(out)


def _text(field: str, value: Any, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid(field, "must be a string or null")
    text = value.strip()
    if not text:
        return None
    if len(text) > limit:
        raise _invalid(field, f"at most {limit} characters (got {len(text)})")
    return text


def _design(value: Any) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise _invalid("design", "must be an object {model, instructions} or null")
    extra = set(value) - {"model", "instructions"}
    if extra:
        raise _invalid("design", f"unknown keys {sorted(extra)}; allowed: model, instructions")
    model = _text("design.model", value.get("model"), MAX_DESIGN_MODEL_CHARS)
    instructions = _text(
        "design.instructions", value.get("instructions"), MAX_DESIGN_INSTRUCTIONS_CHARS)
    if model is None and instructions is None:
        raise _invalid("design", "needs a non-empty model or instructions")
    return {"model": model, "instructions": instructions}


_NORMALIZERS = {
    "category": _category,
    "tags": _tags,
    "description": lambda v: _text("description", v, MAX_DESCRIPTION_CHARS),
    "notes": lambda v: _text("notes", v, MAX_NOTES_CHARS),
    "design": _design,
}


def normalize_metadata(fields: Mapping[str, Any], *, partial: bool) -> dict:
    """Validate and normalize voice metadata.

    Returns only the known keys present in ``fields`` (unknown keys are
    ignored); with ``partial=False`` the absent ones are filled with null
    (``[]`` for tags). Raises ``VoiceError("invalid_voice_metadata", ...,
    400)`` naming the field and the rule.
    """
    out: dict[str, Any] = {}
    for key in METADATA_KEYS:
        if key in fields:
            out[key] = _NORMALIZERS[key](fields[key])
        elif not partial:
            out[key] = [] if key == "tags" else None
    return out


@dataclass(frozen=True)
class VoiceEntry:
    """One voice as the catalog describes it (no audio)."""

    name: str
    transcript: str
    sample_rate: int
    duration_seconds: float
    sha256: str | None
    read_only: bool
    category: str | None
    tags: tuple[str, ...]
    description: str | None
    notes: str | None
    design: dict | None
    created_at: str | None
    updated_at: str | None

    def to_dict(self, model_id: str) -> dict:
        """The wire shape of one voice."""
        return {
            "object": "audio.voice",
            "model": model_id,
            "name": self.name,
            "category": self.category,
            "tags": list(self.tags),
            "description": self.description,
            "notes": self.notes,
            "design": dict(self.design) if self.design else None,
            "transcript": self.transcript,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "sha256": self.sha256,
            "read_only": self.read_only,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "audio_url": (f"/v1/audio/speech/voices/{quote(self.name, safe='')}"
                          f"/audio?model={quote(model_id, safe='')}"),
        }


def _lenient(normalize, value: Any, default: Any) -> Any:
    try:
        out = normalize(value)
    except VoiceError:
        return default
    return default if out is None else out


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def entry_from_meta(name: str, meta: Mapping[str, Any], *, read_only: bool) -> VoiceEntry:
    """A VoiceEntry from a ``voice.json`` mapping.

    Lenient about the optional metadata (a malformed field reads as null or
    empty, so one hand-edited file never breaks a listing); strict about the
    fields synthesis needs, which raise ``VoiceUnreadableError``.
    """
    try:
        transcript = str(meta["transcript"])
        sample_rate = int(meta["sample_rate"])
        duration = float(meta["duration_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VoiceUnreadableError(f"voice {name!r} metadata is unreadable: {exc}") from exc
    return VoiceEntry(
        name=name,
        transcript=transcript,
        sample_rate=sample_rate,
        duration_seconds=duration,
        sha256=_opt_str(meta.get("sha256")),
        read_only=read_only,
        category=_lenient(_category, meta.get("category"), None),
        tags=tuple(_lenient(_tags, meta.get("tags") if isinstance(
            meta.get("tags"), (list, tuple)) else None, [])),
        description=_lenient(
            lambda v: _text("description", v, MAX_DESCRIPTION_CHARS),
            meta.get("description"), None),
        notes=_lenient(lambda v: _text("notes", v, MAX_NOTES_CHARS), meta.get("notes"), None),
        design=_lenient(_design, meta.get("design"), None),
        created_at=_opt_str(meta.get("created_at")),
        updated_at=_opt_str(meta.get("updated_at")),
    )


def filter_entries(
    entries: Sequence[VoiceEntry],
    *,
    category: str | None,
    tags: Sequence[str],
    q: str | None,
) -> list[VoiceEntry]:
    """Entries under ``category`` (the path or anything below it), carrying
    every tag in ``tags``, whose text contains ``q`` (case-insensitive)."""
    path = (category or "").strip().lower().strip("/")
    wanted = {t.strip().lower() for t in tags if t and t.strip()}
    needle = (q or "").strip().lower()
    out = []
    for e in entries:
        if path and not (e.category == path or (e.category or "").startswith(path + "/")):
            continue
        if wanted and not wanted <= set(e.tags):
            continue
        if needle:
            hay = " ".join(filter(None, (
                e.name, e.category, e.description, e.notes, " ".join(e.tags)))).lower()
            if needle not in hay:
                continue
        out.append(e)
    return out


def category_counts(entries: Sequence[VoiceEntry]) -> list[dict]:
    """``[{path, count}]`` for every category path and its ancestors, where
    ``count`` is the number of voices at or below that path."""
    counts: dict[str, int] = {}
    for e in entries:
        if not e.category:
            continue
        parts = e.category.split("/")
        for i in range(1, len(parts) + 1):
            prefix = "/".join(parts[:i])
            counts[prefix] = counts.get(prefix, 0) + 1
    return [{"path": p, "count": counts[p]} for p in sorted(counts)]


# --- storage ---------------------------------------------------------------
#
# Layout per saved voice: ``<root>/<name>/reference.wav`` + ``voice.json``.
# ``<root>/.lock`` is the advisory lock every process (gateway and worker)
# takes: shared for reads, exclusive for writes, so a reader never sees a
# directory mid-swap or a transcript paired with another version's clip.

NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
DATA_DIR = Path(__file__).resolve().parent / "data"
BUNDLED_VOICES: dict[str, tuple[Path, Path]] = {
    "narrator": (DATA_DIR / "narrator.wav", DATA_DIR / "narrator.json"),
}
_WAV = "reference.wav"
_JSON = "voice.json"
_LOCK = ".lock"


def voice_store_root(model_id: str) -> Path:
    """Where a model's saved voices live (``<paths.home>/voices/<model_id>``).

    Not created here: an exclusive lock creates it mode 0700 on first write.
    """
    return Path(config.get("paths.home")).expanduser() / "voices" / model_id


def read_voice_file(path: Path) -> tuple[bytes, tuple]:
    """Read one regular file without following symlinks; return its bytes
    and (mtime_ns, size, ino) taken from the same descriptor.

    ``FileNotFoundError`` propagates; other failures raise
    ``VoiceUnreadableError``.
    """
    fd = securefs.open_regular_file(
        path, os.O_RDONLY, error=VoiceUnreadableError, label="voice file")
    with os.fdopen(fd, "rb") as fh:
        st = os.fstat(fh.fileno())
        return fh.read(), (st.st_mtime_ns, st.st_size, st.st_ino)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


_INSTANCES: dict[tuple, "VoiceCatalog"] = {}
_INSTANCES_LOCK = threading.Lock()


class VoiceCatalog:
    """The saved and bundled voices of one voice root.

    One instance per (root, bundled voices) per process (``for_root``), so
    every caller in a process shares its in-process lock; the file lock
    covers other processes.
    """

    def __init__(self, root: Path, *, bundled: Mapping[str, tuple[Path, Path]]):
        self.root = Path(root)
        self.bundled = {n: (Path(w), Path(j)) for n, (w, j) in bundled.items()}
        self._rlock = threading.RLock()
        self._depth = 0
        self._exclusive = False
        self._handle: Any = None
        self._memo: dict[str, tuple[tuple, VoiceEntry]] = {}

    @classmethod
    def for_root(
        cls, root: Path, *, bundled: Mapping[str, tuple[Path, Path]] = BUNDLED_VOICES,
    ) -> "VoiceCatalog":
        resolved = Path(root).expanduser().resolve()
        key = (str(resolved), tuple(sorted(
            (n, str(Path(w)), str(Path(j))) for n, (w, j) in bundled.items())))
        with _INSTANCES_LOCK:
            inst = _INSTANCES.get(key)
            if inst is None:
                inst = _INSTANCES[key] = cls(resolved, bundled=bundled)
            return inst

    @classmethod
    def for_model(cls, model_id: str) -> "VoiceCatalog":
        return cls.for_root(voice_store_root(model_id))

    # -- locking -----------------------------------------------------------

    @contextlib.contextmanager
    def lock(self, *, exclusive: bool) -> Iterator[None]:
        """Hold the voice root's lock (re-entrant within this process).

        The first acquisition takes the file lock; nested ones reuse it. A
        nested exclusive request under a shared hold raises RuntimeError
        (upgrading would deadlock against another shared holder). A shared
        lock on a root that does not exist yet holds no file lock and
        creates nothing.
        """
        with self._rlock:
            if self._depth == 0:
                handle = None
                if exclusive:
                    securefs.ensure_directory(
                        self.root, error=VoiceUnreadableError, label="voices directory")
                if exclusive or self.root.is_dir():
                    handle = securefs.open_lock_file(
                        self.root / _LOCK, error=VoiceUnreadableError,
                        label="voices lock")
                    try:
                        securefs.lock_file(handle, exclusive=exclusive)
                    except (OSError, ImportError) as exc:
                        handle.close()
                        raise VoiceUnreadableError(
                            f"cannot lock voices directory {self.root}: {exc}") from exc
                self._handle = handle
                self._exclusive = exclusive
            elif exclusive and not self._exclusive:
                raise RuntimeError("voice lock upgrade from shared to exclusive")
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0 and self._handle is not None:
                    handle, self._handle = self._handle, None
                    try:
                        securefs.unlock_file(handle)
                    finally:
                        handle.close()

    # -- reading (callers hold the lock for the *_unlocked helpers) ---------

    def _parse(self, name: str, path: Path, *, read_only: bool) -> VoiceEntry:
        """The entry in ``path`` (a voice.json), memoized on file identity.

        ``FileNotFoundError`` propagates; anything else unreadable raises
        ``VoiceUnreadableError``.
        """
        try:
            info = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise VoiceUnreadableError(f"voice {name!r} cannot be read") from exc
        identity = securefs.file_identity(path, info)
        hit = self._memo.get(name)
        if hit is not None and hit[0] == identity:
            return hit[1]
        data, _ = read_voice_file(path)
        try:
            meta = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise VoiceUnreadableError(f"voice {name!r} metadata is unreadable: {exc}") from exc
        if not isinstance(meta, dict):
            raise VoiceUnreadableError(f"voice {name!r} metadata is not an object")
        entry = entry_from_meta(name, meta, read_only=read_only)
        self._memo[name] = (identity, entry)
        return entry

    def _bundled_entry(self, name: str) -> VoiceEntry:
        try:
            return self._parse(name, self.bundled[name][1], read_only=True)
        except FileNotFoundError as exc:
            raise VoiceUnreadableError(f"bundled voice {name!r} is missing") from exc

    def saved_names_unlocked(self) -> list[str]:
        """Saved voice names (sorted), skipping anything incomplete or corrupt."""
        try:
            dirs = sorted(self.root.iterdir())
        except (FileNotFoundError, NotADirectoryError):
            return []
        except OSError as exc:
            raise VoiceUnreadableError("voices directory cannot be read") from exc
        names = []
        for d in dirs:
            try:
                if not NAME_PATTERN.fullmatch(d.name) or d.is_symlink() or not d.is_dir():
                    continue
                if not (d / _WAV).is_file():
                    continue
                self._parse(d.name, d / _JSON, read_only=False)
            except (VoiceUnreadableError, FileNotFoundError):
                continue
            except OSError as exc:
                raise VoiceUnreadableError("voices directory cannot be read") from exc
            names.append(d.name)
        return names

    def _names_unlocked(self) -> list[str]:
        return sorted(self.bundled) + [
            n for n in self.saved_names_unlocked() if n not in self.bundled]

    def saved_dir_unlocked(self, name: str) -> Path:
        """The directory of saved voice ``name``; UnknownVoiceError otherwise."""
        if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
            raise UnknownVoiceError(name, self._names_unlocked())
        d = self.root / name
        try:
            missing = d.is_symlink() or not d.is_dir()
        except OSError as exc:
            raise VoiceUnreadableError("voices directory cannot be read") from exc
        if missing:
            raise UnknownVoiceError(name, self._names_unlocked())
        return d

    def names(self) -> list[str]:
        """Bundled voices first, then saved ones (sorted)."""
        with self.lock(exclusive=False):
            return self._names_unlocked()

    def entries(self) -> list[VoiceEntry]:
        """Every readable voice, in ``names()`` order."""
        with self.lock(exclusive=False):
            out = [self._bundled_entry(n) for n in sorted(self.bundled)]
            for n in self.saved_names_unlocked():
                if n not in self.bundled:
                    out.append(self._parse(n, self.root / n / _JSON, read_only=False))
            return out

    def entry(self, name: str) -> VoiceEntry:
        if name in self.bundled:
            return self._bundled_entry(name)
        with self.lock(exclusive=False):
            d = self.saved_dir_unlocked(name)
            try:
                return self._parse(name, d / _JSON, read_only=False)
            except FileNotFoundError as exc:
                raise VoiceUnreadableError(f"voice {name!r} is incomplete") from exc

    def audio_bytes(self, name: str) -> bytes:
        """The voice's reference clip (WAV bytes)."""
        if name in self.bundled:
            try:
                return read_voice_file(self.bundled[name][0])[0]
            except FileNotFoundError as exc:
                raise VoiceUnreadableError(f"bundled voice {name!r} is missing") from exc
        with self.lock(exclusive=False):
            d = self.saved_dir_unlocked(name)
            try:
                return read_voice_file(d / _WAV)[0]
            except FileNotFoundError as exc:
                raise VoiceUnreadableError(f"voice {name!r} is incomplete") from exc

    # -- writing -------------------------------------------------------------

    def _read_only(self, name: str) -> VoiceError:
        return VoiceError("voice_read_only", f"voice {name!r} is bundled and read-only", 409)

    def update(self, name: str, patch: Mapping[str, Any]) -> VoiceEntry:
        """Merge metadata into a saved voice's voice.json (never its audio,
        transcript or created_at) and stamp ``updated_at``."""
        if name in self.bundled:
            raise self._read_only(name)
        fields = normalize_metadata(patch, partial=True)
        if not self.root.is_dir():
            raise UnknownVoiceError(name, sorted(self.bundled))
        with self.lock(exclusive=True):
            d = self.saved_dir_unlocked(name)
            try:
                meta = json.loads(read_voice_file(d / _JSON)[0].decode("utf-8"))
            except FileNotFoundError as exc:
                raise VoiceUnreadableError(f"voice {name!r} is incomplete") from exc
            except (UnicodeDecodeError, ValueError) as exc:
                raise VoiceUnreadableError(
                    f"voice {name!r} metadata is unreadable: {exc}") from exc
            if not isinstance(meta, dict):
                raise VoiceUnreadableError(f"voice {name!r} metadata is not an object")
            meta.update(fields)
            meta["updated_at"] = _now()
            securefs.atomic_write_text(
                d / _JSON, json.dumps(meta, indent=2), error=VoiceUnreadableError)
            return self._parse(name, d / _JSON, read_only=False)

    def delete(self, name: str) -> None:
        if name in self.bundled:
            raise self._read_only(name)
        if not self.root.is_dir():
            raise UnknownVoiceError(name, sorted(self.bundled))
        with self.lock(exclusive=True):
            d = self.saved_dir_unlocked(name)
            gone = self.root / f".old-{os.urandom(6).hex()}"
            os.replace(d, gone)
            self._memo.pop(name, None)
            shutil.rmtree(gone, ignore_errors=True)
