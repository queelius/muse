"""Model-free audio/speech routes: the saved-voice catalog (muse.core.light).

Served in workers (over their registry) and in the gateway (from disk,
ahead of its proxy), so listing, inspecting, auditioning, annotating and
deleting voices never loads a speech model. Saving a new voice
(``POST /v1/audio/speech/voices``) decodes audio and stays in ``routes``.

For a model with ``supports_saved_voices`` every route here reads or edits
``VoiceCatalog.for_model(model)``. For other models, ``GET /voices`` lists
the backend's preset voices (in a worker) or forwards to the worker (in the
gateway); the other routes answer 400 ``voice_saving_not_supported``.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from fastapi import APIRouter, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from muse.core.errors import error_response
from muse.core.light import LightCatalog, LightModel, resolve_light
from muse.modalities.audio_speech.voice_catalog import (
    VoiceCatalog,
    VoiceEntry,
    category_counts,
    filter_entries,
)
from muse.modalities.audio_speech.voice_errors import VoiceError, VoiceUnreadableError

logger = logging.getLogger(__name__)

MODALITY = "audio/speech"
_MODEL_HELP = (
    "The speech model whose voices to use, e.g. qwen3-tts-0.6b-base (saved, "
    "cloneable voices) or kokoro-82m (preset voices).")


class VoicePatch(BaseModel):
    """Metadata changes for one saved voice. Only the fields you send change;
    a field sent as null is cleared. Audio and transcript are not editable
    (save again with ``overwrite`` instead)."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(None, description=_MODEL_HELP)
    category: str | None = Field(None, description=(
        "Slash path of 1-4 segments of [a-z0-9][a-z0-9_-]{0,31}, e.g. "
        "narrator/epic or npc/human/guard. Lowercased and trimmed."))
    tags: list[str] | str | None = Field(None, description=(
        "Labels as a list or a comma-separated string (at most 16, each "
        "[a-z0-9][a-z0-9_-]{0,31}); replaces the voice's tags."))
    description: str | None = Field(None, description=(
        "What the voice sounds like (at most 500 characters)."))
    notes: str | None = Field(None, description=(
        "What the voice is for and how to use it (at most 2000 characters)."))
    design: dict[str, str | None] | None = Field(None, description=(
        "How the reference take was made: {model, instructions} "
        "(instructions at most 4096 characters)."))


def _saved(m: LightModel) -> bool:
    return bool(m.capabilities.get("supports_saved_voices"))


def _not_supported() -> Response:
    return error_response(
        400, "voice_saving_not_supported", "This model does not support saved voices")


def _voice_error(exc: Exception) -> Response:
    if isinstance(exc, VoiceError):  # UnknownVoiceError included (404)
        return error_response(exc.status, exc.code, str(exc))
    logger.error("stored voice unreadable: %s", exc, exc_info=exc)
    return error_response(
        500, "voice_unreadable", "the requested voice could not be read; see server logs")


async def _catalog_call(m: LightModel, fn: Callable[[VoiceCatalog], Any]) -> Any:
    """Run ``fn`` on the model's voice catalog off the event loop."""
    return await asyncio.to_thread(lambda: fn(VoiceCatalog.for_model(m.model_id)))


def _preset_entry(name: str) -> VoiceEntry:
    return VoiceEntry(
        name=name, transcript="", sample_rate=0, duration_seconds=0.0, sha256=None,
        read_only=True, category=None, tags=(), description=None, notes=None,
        design=None, created_at=None, updated_at=None)


def build_light_router(catalog: LightCatalog) -> APIRouter:
    router = APIRouter(prefix="/v1/audio", tags=["audio/speech"])

    async def _resolve(model: str | None) -> LightModel | Response:
        return await resolve_light(catalog, MODALITY, model)

    @router.get("/speech/voices")
    async def list_voices(
        request: Request,
        model: str | None = Query(None, description=_MODEL_HELP),
        category: str | None = Query(None, description=(
            "Only voices in this category or below it (narrator matches "
            "narrator and narrator/epic).")),
        tag: list[str] = Query([], description=(
            "Only voices carrying this tag; repeat for several (all must match).")),
        q: str | None = Query(None, description=(
            "Only voices whose name, category, description, notes or tags "
            "contain this text (case-insensitive).")),
    ):
        m = await _resolve(model)
        if not isinstance(m, LightModel):
            return m
        if _saved(m):
            try:
                entries = await _catalog_call(m, lambda c: c.entries())
            except (VoiceError, VoiceUnreadableError) as exc:
                return _voice_error(exc)
            categories = category_counts(entries)

            def render(e: VoiceEntry) -> dict:
                return e.to_dict(m.model_id)
        elif m.backend is not None:
            try:
                names = await asyncio.to_thread(
                    lambda: list(getattr(m.backend, "voices", None) or []))
            except VoiceUnreadableError as exc:
                return _voice_error(exc)
            entries = [_preset_entry(str(n)) for n in names]
            categories = []

            def render(e: VoiceEntry) -> dict:
                return {"object": "audio.voice", "model": m.model_id,
                        "name": e.name, "read_only": True}
        else:
            return await catalog.forward(request, m.model_id)
        shown = filter_entries(entries, category=category, tags=tag, q=q)
        return {
            "object": "list",
            "model": m.model_id,
            "voices": [e.name for e in shown],
            "data": [render(e) for e in shown],
            "categories": categories,
        }

    @router.get("/speech/voices/{name}")
    async def get_voice(name: str, model: str | None = Query(None, description=_MODEL_HELP)):
        m = await _resolve(model)
        if not isinstance(m, LightModel):
            return m
        if not _saved(m):
            return _not_supported()
        try:
            entry = await _catalog_call(m, lambda c: c.entry(name))
        except (VoiceError, VoiceUnreadableError) as exc:
            return _voice_error(exc)
        return entry.to_dict(m.model_id)

    @router.get(
        "/speech/voices/{name}/audio",
        response_class=Response,
        responses={200: {"content": {"audio/wav": {}},
                         "description": "The voice's reference clip (WAV)."}},
    )
    async def get_voice_audio(
        name: str, model: str | None = Query(None, description=_MODEL_HELP),
    ):
        m = await _resolve(model)
        if not isinstance(m, LightModel):
            return m
        if not _saved(m):
            return _not_supported()
        try:
            data = await _catalog_call(m, lambda c: c.audio_bytes(name))
        except (VoiceError, VoiceUnreadableError) as exc:
            return _voice_error(exc)
        # `name` passed the catalog's [a-z0-9_-] name check, so it is header-safe.
        return Response(content=data, media_type="audio/wav", headers={
            "Content-Disposition": f'inline; filename="{name}.wav"'})

    @router.patch("/speech/voices/{name}")
    async def update_voice(name: str, body: VoicePatch):
        m = await _resolve(body.model)
        if not isinstance(m, LightModel):
            return m
        if not _saved(m):
            return _not_supported()
        fields = {k: getattr(body, k) for k in body.model_fields_set if k != "model"}
        try:
            entry = await _catalog_call(m, lambda c: c.update(name, fields))
        except (VoiceError, VoiceUnreadableError) as exc:
            return _voice_error(exc)
        return entry.to_dict(m.model_id)

    @router.delete("/speech/voices/{name}")
    async def delete_voice(name: str, model: str | None = Query(None, description=_MODEL_HELP)):
        m = await _resolve(model)
        if not isinstance(m, LightModel):
            return m
        if not _saved(m):
            return _not_supported()
        try:
            await _catalog_call(m, lambda c: c.delete(name))
        except (VoiceError, VoiceUnreadableError) as exc:
            return _voice_error(exc)
        except Exception:  # noqa: BLE001
            logger.exception("deleting voice failed")
            return error_response(
                500, "internal_error", "deleting the voice failed; see server logs")
        return {"object": "audio.voice.deleted", "model": m.model_id,
                "name": name, "deleted": True}

    return router

