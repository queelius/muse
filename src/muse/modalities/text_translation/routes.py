"""FastAPI routes for /v1/translate, /translate (alias), and /languages.

LibreTranslate-compat shape:
  request:  {"q": str|list[str], "source": str, "target": str,
             "format"?: "text", "model"?: str}
  response: {"translatedText": str|list[str]}

  GET /languages -> [{"code", "name", "targets"}, ...]

Error envelopes follow muse's OpenAI-compat convention (see
muse.core.errors): 404 raises ModelNotFoundError; every other error path
returns error_response() so the bare {"error": ...} envelope reaches the
client without FastAPI's {"detail": ...} wrap. `q` type/presence errors
are left to pydantic (422): `q: str | list[str]` with no default means a
missing field or a wrong-shaped value (e.g. an int, or a list containing
a non-str item) is rejected before the handler body ever runs.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from muse.core import config
from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.text_translation.codec import (
    languages_payload,
    normalize_q,
    shape_response,
)
from muse.modalities.text_translation.protocol import UnsupportedLanguageError


# MODALITY defined locally to avoid the __init__ circular import;
# sibling modalities all do this.
MODALITY = "text/translation"


logger = logging.getLogger(__name__)


class _TranslateRequest(BaseModel):
    q: str | list[str]
    source: str
    target: str
    format: str = "text"
    model: str | None = None


def _invalid_language_message(bad_code: str, valid_codes) -> str:
    """Build the 400 invalid_language message: names the offending code,
    lists the supported codes (or a summary, if there are more than 20 --
    m2m100/NLLB support ~100 to 200 languages, and dumping the full list
    into every error message is not useful)."""
    codes = sorted(valid_codes)
    if len(codes) > 20:
        listing = f"{len(codes)} supported codes, e.g. {', '.join(codes[:10])}, ..."
    else:
        listing = ", ".join(codes) if codes else "(none)"
    return f"unsupported language {bad_code!r}; supported: {listing}"


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    async def _translate(req: _TranslateRequest):
        if req.source == "auto":
            return error_response(
                400, "source_detection_not_supported",
                "explicit source language required; detection is planned",
            )
        if req.format != "text":
            return error_response(
                400, "unsupported_format",
                f"format {req.format!r} is not supported; only 'text' is",
            )

        try:
            texts, scalar = normalize_q(req.q)
        except ValueError as e:
            return error_response(400, "invalid_parameter", str(e))

        # Read per-request (not cached at import time): an operator can
        # change MUSE_TRANSLATE_MAX_CHARS / config.yaml without restarting
        # the worker.
        max_chars = config.get("limits.translate_max_chars")
        total_chars = sum(len(t) for t in texts)
        if total_chars > max_chars:
            return error_response(
                400, "input_too_long",
                f"q totals {total_chars} chars, exceeding "
                f"limits.translate_max_chars={max_chars}",
            )

        try:
            backend = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )

        # Pre-dispatch pair validation against supported_languages() (T2
        # review note): checks membership, never materializes the full
        # dict into the response, so this stays cheap even for m2m100's
        # ~100-language table.
        supported = backend.supported_languages()
        if req.source not in supported:
            return error_response(
                400, "invalid_language",
                _invalid_language_message(req.source, supported.keys()),
            )
        # source == target (identity round-trip) is always allowed per the
        # wire contract, even if the backend's supported_languages() target
        # list for `source` does not literally include itself (many-to-many
        # families like m2m100 list every OTHER language as a target).
        if req.target != req.source:
            targets = supported.get(req.source, [])
            if req.target not in targets:
                return error_response(
                    400, "invalid_language",
                    _invalid_language_message(req.target, targets),
                )

        # backend.translate is sync (transformers generate); offload so a
        # slow inference doesn't block sibling /health, /v1/models, or
        # other in-flight requests on the same worker. Per-backend lock
        # serializes calls into one model without blocking siblings on
        # the same worker (mirrors text_summarization / text_classification).
        def _translate_call():
            with backend._inference_lock:
                return backend.translate(texts, source=req.source, target=req.target)

        try:
            result = await asyncio.to_thread(_translate_call)
        except UnsupportedLanguageError as e:
            # Belt and suspenders: the pre-dispatch check above catches
            # the common case, but a runtime (e.g. opus-mt's fixed pair,
            # or NLLB's ISO->FLORES gap) may reject a pair the pre-check
            # let through.
            valid = e.supported.keys() if isinstance(e.supported, dict) else e.supported
            return error_response(
                400, "invalid_language",
                _invalid_language_message(e.code, valid),
            )
        except Exception:  # noqa: BLE001
            # Log the real exception server-side (torch/CUDA/path detail
            # is operationally useful) but never leak it to the client:
            # str(e) can carry internal filesystem paths, CUDA driver
            # text, or other backend-implementation detail that has no
            # business in an external-facing error body.
            logger.exception("translate failed")
            return error_response(
                500, "internal_error",
                "translation backend failed; see server logs",
            )

        body = shape_response(result.texts, scalar=scalar)
        return JSONResponse(content=body)

    async def _translate_entry(request: Request):
        """Content-type-aware entry: real LibreTranslate clients POST
        application/x-www-form-urlencoded (the reference LT server accepts
        both form and JSON), so a JSON-only pydantic binding 422s an
        unmodified client (live-validation finding, 2026-07-09). Form
        bodies are mapped onto the same _TranslateRequest model: repeated
        `q` keys become a batch, and LT's `api_key` field (sent
        unconditionally by some clients) is ignored -- muse has no LT API
        keys. Validation failures return 422 in FastAPI's {detail: ...}
        shape either way, so both parse paths degrade identically.
        """
        content_type = request.headers.get("content-type", "")
        if ("application/x-www-form-urlencoded" in content_type
                or "multipart/form-data" in content_type):
            form = await request.form()
            data: dict = {}
            qs = form.getlist("q")
            if qs:
                data["q"] = qs[0] if len(qs) == 1 else list(qs)
            for key in ("source", "target", "format", "model"):
                value = form.get(key)
                if value is not None:
                    data[key] = value
        else:
            try:
                data = await request.json()
            except Exception:  # noqa: BLE001
                return JSONResponse(
                    status_code=422,
                    content={"detail": "request body is not valid JSON"},
                )
            if not isinstance(data, dict):
                return JSONResponse(
                    status_code=422,
                    content={"detail": "request body must be a JSON object"},
                )
        try:
            req = _TranslateRequest(
                **{k: v for k, v in data.items() if k != "api_key"}
            )
        except ValidationError as e:
            # Mirror FastAPI's default 422 envelope so existing clients
            # and tests that match on {"detail": [...]} keep working.
            return JSONResponse(status_code=422, content={"detail": e.errors(
                include_url=False, include_input=False,
            )})
        return await _translate(req)

    router.add_api_route("/v1/translate", _translate_entry, methods=["POST"])
    router.add_api_route("/translate", _translate_entry, methods=["POST"])

    @router.get("/languages")
    async def languages(model: str | None = Query(default=None)):
        try:
            backend = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=model or "<default>", modality=MODALITY,
            )
        supported = backend.supported_languages()
        return JSONResponse(content=languages_payload(supported))

    return router
