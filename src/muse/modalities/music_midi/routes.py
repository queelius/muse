"""FastAPI route for ``POST /v1/music/midi``."""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.inference import run_locked
from muse.modalities.music_midi.codec import (
    MIDI_CONTENT_DISPOSITION,
    MIDI_MEDIA_TYPE,
    encode_midi_json,
)


MODALITY = "music/midi"

logger = logging.getLogger(__name__)

_VALID_KEYS: frozenset[str] = frozenset({
    "C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb",
    "G", "G#", "Ab", "A", "A#", "Bb", "B",
})


class MidiGenerationRequest(BaseModel):
    """JSON request body for ``POST /v1/music/midi``."""

    model: str | None = None
    prompt: str = Field(..., min_length=1, max_length=4000)
    key: str | None = None
    mode: Literal[
        "major", "minor", "dorian", "mixolydian", "lydian", "phrygian", "aeolian",
    ] | None = None
    bpm: int | None = Field(default=None, ge=40, le=220)
    time_signature: Literal["4/4", "3/4", "6/8"] = "4/4"
    bars: int = Field(default=16, ge=4, le=64)
    instruments: list[str] | None = None
    layers: list[str] | None = None
    seed: int | None = Field(default=None, ge=0)
    temperature: float = Field(default=0.9, ge=0.0, le=2.0)
    response_format: Literal["json", "midi"] = "json"


def _validate_key(key: str | None) -> str | None:
    if key is None:
        return None
    cleaned = key.strip()
    if not cleaned:
        return None
    canon = cleaned[0].upper() + cleaned[1:].replace("♯", "#").replace("♭", "b")
    if canon not in _VALID_KEYS:
        raise ValueError(
            f"unsupported key {key!r}; expected a pitch name like 'C', 'D', 'F#', 'Bb'"
        )
    return canon


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/music", tags=[MODALITY])

    @router.post("/midi")
    async def generate_midi(req: MidiGenerationRequest):
        try:
            canonical_key = _validate_key(req.key)
        except ValueError as exc:
            return error_response(400, "invalid_parameter", str(exc))

        backend = registry.get_or_404(MODALITY, req.model)

        try:
            result = await run_locked(
                backend,
                lambda: backend.generate(
                    req.prompt,
                    key=canonical_key,
                    mode=req.mode,
                    bpm=req.bpm,
                    time_signature=req.time_signature,
                    bars=req.bars,
                    instruments=req.instruments,
                    layers=req.layers,
                    seed=req.seed,
                    temperature=req.temperature,
                ),
                label="music/midi generate call",
                message="music/midi backend failed; see server logs",
                passthrough=(ValueError,),
            )
        except ValueError as exc:
            return error_response(400, "invalid_parameter", str(exc))

        if isinstance(result, JSONResponse):
            return result

        if req.response_format == "midi":
            return Response(
                content=result.midi_bytes,
                media_type=MIDI_MEDIA_TYPE,
                headers={
                    "Content-Disposition": MIDI_CONTENT_DISPOSITION,
                    "X-Muse-Model": backend.model_id,
                },
            )

        return JSONResponse(
            content=encode_midi_json(result, model_id=backend.model_id),
        )

    return router
