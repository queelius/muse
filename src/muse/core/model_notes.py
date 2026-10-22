"""Agent-facing model notes: what a model is for, its strengths and limits.

An agent choosing between two models for one pipeline step reads these.
Four hand-written fields:

  summary       one line on what the model is for
  strengths     short list
  limits        short list (non-commercial licenses always appear here)
  quality_tier  "best" | "good" | "draft", an absolute judgment within the
                model's modality across every bundled and curated model

Sources, first match wins (the whole record comes from one source):

  1. the manifest's top-level fields (bundled scripts' MANIFEST);
  2. the curated.yaml entry with the same id (curated URI models, whose
     resolver-synthesized manifests carry no notes);
  3. `ModelNotes(manifest["description"], (), (), None)`.

Stdlib-only; `muse.core.curated` is imported lazily because it imports
`validate_note_fields` from here.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)

QUALITY_TIERS = frozenset({"best", "good", "draft"})
NOTE_FIELDS = ("summary", "strengths", "limits", "quality_tier")


@dataclass(frozen=True)
class ModelNotes:
    summary: str | None
    strengths: tuple[str, ...]
    limits: tuple[str, ...]
    quality_tier: str | None


_EMPTY = ModelNotes(None, (), (), None)


def validate_note_fields(d: Mapping[str, Any], owner: str) -> ModelNotes:
    """Strictly validate the note fields of `d`; raise ValueError naming `owner`.

    Absent fields default to None / (). Present fields must be: `summary`
    a non-empty string, `strengths` / `limits` lists of non-empty strings,
    `quality_tier` one of QUALITY_TIERS.
    """
    summary = d.get("summary")
    if summary is not None and (not isinstance(summary, str) or not summary.strip()):
        raise ValueError(f"entry {owner!r}: 'summary' must be a non-empty string")
    lists: dict[str, tuple[str, ...]] = {}
    for name in ("strengths", "limits"):
        value = d.get(name, ())
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ValueError(
                f"entry {owner!r}: {name!r} must be a list of non-empty strings"
            )
        lists[name] = tuple(value)
    tier = d.get("quality_tier")
    if tier is not None and (not isinstance(tier, str) or tier not in QUALITY_TIERS):
        raise ValueError(
            f"entry {owner!r}: 'quality_tier' must be one of "
            f"{sorted(QUALITY_TIERS)}, got {tier!r}"
        )
    return ModelNotes(summary, lists["strengths"], lists["limits"], tier)


def _has_notes(d: Mapping[str, Any]) -> bool:
    return any(d.get(name) for name in NOTE_FIELDS)


def model_notes(model_id: str, manifest: Mapping[str, Any]) -> ModelNotes:
    """Return the notes for one model (see module docstring for precedence).

    Lenient: a malformed manifest note (for example a hand-edited catalog
    manifest) logs a warning and degrades field by field instead of
    raising, because discovery must not fail on one bad entry.
    """
    description = manifest.get("description")
    notes = _EMPTY
    if _has_notes(manifest):
        notes = _lenient(manifest, model_id)
    else:
        from muse.core.curated import find_curated

        curated = find_curated(model_id)
        if curated is not None:
            notes = ModelNotes(
                curated.summary, curated.strengths, curated.limits, curated.quality_tier,
            )
    if not notes.summary and isinstance(description, str) and description:
        notes = ModelNotes(description, notes.strengths, notes.limits, notes.quality_tier)
    return notes


def _lenient(d: Mapping[str, Any], model_id: str) -> ModelNotes:
    try:
        return validate_note_fields(d, model_id)
    except ValueError as exc:
        logger.warning("ignoring malformed model notes: %s", exc)
    summary = d.get("summary")
    tier = d.get("quality_tier")

    def _strs(value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(v for v in value if isinstance(v, str) and v.strip())

    return ModelNotes(
        summary if isinstance(summary, str) and summary.strip() else None,
        _strs(d.get("strengths")),
        _strs(d.get("limits")),
        tier if isinstance(tier, str) and tier in QUALITY_TIERS else None,
    )
