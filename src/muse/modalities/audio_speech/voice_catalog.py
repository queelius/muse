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

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from muse.modalities.audio_speech.voice_errors import (
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
