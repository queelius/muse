"""FastAPI route for ``POST /v1/audio/process`` (multipart/form-data).

Accepts an audio upload plus deterministic mastering parameters and returns
either a JSON envelope (``object: "audio.process"`` with ``b64_audio`` and
``metrics``) or raw encoded audio bytes (``response_format="audio"``).

Decoding, DSP processing, and encoding run off the event loop under the
backend's ``_inference_lock``. Heavy DSP dependencies (numpy, scipy) are
imported lazily inside the handler so importing the modality stays light.
"""
from __future__ import annotations

import logging
import math

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from muse.core import config
from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.audio_decode import AudioDecodeError
from muse.modalities._common.inference import offload_call
from muse.modalities._common.uploads import read_upload
from muse.modalities.audio_process.protocol import AudioProcessOptions

logger = logging.getLogger(__name__)

MODALITY = "audio/process"
_ALLOWED_OUTPUT_FORMATS = ("wav", "flac", "opus")
_ALLOWED_RESPONSE_FORMATS = ("json", "audio")


def _max_upload_bytes() -> int:
    return int(config.get_positive("server.max_request_body_mb")) * 1024 * 1024


def _validate_params(
    *,
    trim_threshold_db: float,
    pad_lead_ms: float,
    pad_tail_ms: float,
    fade_in_ms: float,
    fade_out_ms: float,
    loop_fade_s: float,
    target_peak_dbfs: float | None,
    sample_rate: int | None,
    output_format: str,
    response_format: str,
) -> str | None:
    if output_format not in _ALLOWED_OUTPUT_FORMATS:
        return f"output_format must be one of {list(_ALLOWED_OUTPUT_FORMATS)}"
    if response_format not in _ALLOWED_RESPONSE_FORMATS:
        return f"response_format must be one of {list(_ALLOWED_RESPONSE_FORMATS)}"
    if not math.isfinite(trim_threshold_db) or not (-120.0 <= trim_threshold_db <= 0.0):
        return "trim_threshold_db must be in [-120.0, 0.0]"
    if not math.isfinite(pad_lead_ms) or pad_lead_ms < 0.0:
        return "pad_lead_ms must be >= 0"
    if not math.isfinite(pad_tail_ms) or pad_tail_ms < 0.0:
        return "pad_tail_ms must be >= 0"
    if not math.isfinite(fade_in_ms) or fade_in_ms < 0.0:
        return "fade_in_ms must be >= 0"
    if not math.isfinite(fade_out_ms) or fade_out_ms < 0.0:
        return "fade_out_ms must be >= 0"
    if not math.isfinite(loop_fade_s) or not (0.0 <= loop_fade_s <= 30.0):
        return "loop_fade_s must be in [0.0, 30.0]"
    if target_peak_dbfs is not None:
        if not math.isfinite(target_peak_dbfs) or not (-48.0 <= target_peak_dbfs <= 0.0):
            return "target_peak_dbfs must be in [-48.0, 0.0]"
    if sample_rate is not None and not (8000 <= sample_rate <= 96000):
        return "sample_rate must be in [8000, 96000]"
    return None


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/audio", tags=[MODALITY])

    @router.post("/process")
    async def process_audio_route(
        file: UploadFile = File(...),
        model: str | None = Form(None),
        trim_silence: bool = Form(True),
        trim_threshold_db: float = Form(-42.0),
        pad_lead_ms: float = Form(20.0),
        pad_tail_ms: float = Form(65.0),
        fade_in_ms: float = Form(5.0),
        fade_out_ms: float = Form(15.0),
        loop_fade_s: float = Form(0.0),
        target_peak_dbfs: float | None = Form(-3.0),
        mono: bool = Form(False),
        sample_rate: int | None = Form(None),
        output_format: str = Form("wav"),
        response_format: str = Form("json"),
    ):
        err = _validate_params(
            trim_threshold_db=trim_threshold_db,
            pad_lead_ms=pad_lead_ms,
            pad_tail_ms=pad_tail_ms,
            fade_in_ms=fade_in_ms,
            fade_out_ms=fade_out_ms,
            loop_fade_s=loop_fade_s,
            target_peak_dbfs=target_peak_dbfs,
            sample_rate=sample_rate,
            output_format=output_format,
            response_format=response_format,
        )
        if err is not None:
            return error_response(400, "invalid_parameter", err)

        backend = registry.get_or_404(MODALITY, model)

        max_bytes = _max_upload_bytes()
        max_mb = max_bytes // (1024 * 1024)
        raw = await read_upload(
            file,
            max_bytes=max_bytes,
            limit=f"MUSE_MAX_REQUEST_BODY_MB={max_mb}",
        )
        if isinstance(raw, JSONResponse):
            return raw

        from muse.modalities.audio_process.codec import (
            UnsupportedFormatError,
            content_type_for,
            decode_audio_bytes,
            encode_audio,
            encode_json_response,
        )

        options = AudioProcessOptions(
            trim_silence=trim_silence,
            trim_threshold_db=trim_threshold_db,
            pad_lead_ms=pad_lead_ms,
            pad_tail_ms=pad_tail_ms,
            fade_in_ms=fade_in_ms,
            fade_out_ms=fade_out_ms,
            loop_fade_s=loop_fade_s,
            target_peak_dbfs=target_peak_dbfs,
            mono=mono,
            sample_rate=sample_rate,
        )

        def _work():
            audio, src_sr = decode_audio_bytes(raw)
            result = backend.process(audio, src_sr, options)
            encoded = encode_audio(
                result.audio, result.sample_rate, result.channels, output_format,
            )
            if response_format == "audio":
                return encoded
            return encode_json_response(
                result,
                model_id=backend.model_id,
                output_format=output_format,
                audio_bytes=encoded,
            )

        try:
            out = await offload_call(
                _work, lock=getattr(backend, "_inference_lock", None),
            )
        except AudioDecodeError as exc:
            return error_response(415, "unsupported_media_type", str(exc))
        except UnsupportedFormatError as exc:
            return error_response(400, "invalid_parameter", str(exc))
        except ValueError as exc:
            return error_response(400, "invalid_parameter", str(exc))
        except Exception:  # noqa: BLE001
            logger.exception("audio/process failed")
            return error_response(
                500, "internal_error", "audio process backend failed; see server logs",
            )

        if response_format == "audio":
            return Response(content=out, media_type=content_type_for(output_format))
        return JSONResponse(content=out)

    return router
