"""POST /v1/systemone: typed Choice / Score / Noul decisions.

The body is read with its own 128 KiB bound and parsed strictly (see
validation.py) before the model is looked up, on the worker as on the
gateway. Execution goes through one DecisionExecutor per model.

Errors use the OpenAI envelope: 400 invalid request or input-token
limit, 413 body/state size, 404 unknown model, 503 queue full/timeout or
an unusable backend, 500 invalid backend output, and 499 (never seen by
the departed client) when it disconnected and the batch was stopped. No
response carries user text, local paths or native exception text.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from muse.cli_impl.queueing import QueueFull, QueueTimeout
from muse.core import config
from muse.core.disconnect import ClientDisconnected, unless_disconnected
from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.decision_evaluation.codec import encode_response
from muse.modalities.decision_evaluation.execution import DecisionExecutor
from muse.modalities.decision_evaluation.protocol import (
    DecisionBackendError,
    DecisionInputError,
)
from muse.modalities.decision_evaluation.validation import MAX_BODY_BYTES, parse_request


# MODALITY defined locally to avoid the __init__ circular import;
# sibling modalities all do this.
MODALITY = "decision/evaluation"
MAX_CONCURRENCY = 1
MAX_QUEUE_DEPTH = 32

logger = logging.getLogger(__name__)

_UNAVAILABLE = {"backend_unusable", "shutting_down"}


async def _read_body(request: Request) -> bytes:
    """The raw body, refusing more than MAX_BODY_BYTES even without a
    Content-Length (the stream is counted as it arrives)."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise DecisionInputError(
            "body_too_large", f"request body exceeds {MAX_BODY_BYTES} bytes", status=413,
        )
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            raise DecisionInputError(
                "body_too_large", f"request body exceeds {MAX_BODY_BYTES} bytes", status=413,
            )
        chunks.append(chunk)
    return b"".join(chunks)


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()
    executors: dict[str, DecisionExecutor] = {}

    def _executor(model_id: str) -> DecisionExecutor:
        executor = executors.get(model_id)
        if executor is None:
            executor = DecisionExecutor(
                model_id, max_concurrency=MAX_CONCURRENCY, max_queue_depth=MAX_QUEUE_DEPTH,
            )
            executors[model_id] = executor
        return executor

    async def _close_executors() -> None:
        for executor in list(executors.values()):
            await executor.close()

    router.on_shutdown.append(_close_executors)

    @router.post("/v1/systemone")
    async def systemone(request: Request):
        try:
            req = parse_request(await _read_body(request))
        except DecisionInputError as exc:
            return error_response(exc.status, exc.code, exc.message)

        try:
            backend = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(model_id=req.model, modality=MODALITY)

        budget = config.get("server.queue_timeout_seconds") or 0.0
        deadline = time.monotonic() + max(0.0, float(budget))
        try:
            # A client that gives up stops the batch before its next question.
            batch = await unless_disconnected(
                request,
                _executor(backend.model_id).execute(backend, req, deadline=deadline),
            )
            body = encode_response(req, batch, model_id=backend.model_id)
        except ClientDisconnected:
            return error_response(499, "client_disconnected", "the client disconnected")
        except DecisionInputError as exc:
            return error_response(exc.status, exc.code, exc.message)
        except QueueFull:
            return error_response(
                503, "queue_full", f"queue for model {req.model!r} is full",
            )
        except QueueTimeout:
            return error_response(
                503, "queue_timeout", f"timed out waiting for model {req.model!r}",
            )
        except DecisionBackendError as exc:
            if exc.code in _UNAVAILABLE:
                return error_response(503, "model_unusable", exc.message)
            logger.error("decision backend %s failed: %s", backend.model_id, exc.code)
            return error_response(
                500, "decision_backend_error",
                "the decision backend produced an invalid result",
            )
        except Exception as exc:  # noqa: BLE001
            # Type only: native messages can carry paths or user text.
            logger.error(
                "decision backend %s raised %s", backend.model_id, type(exc).__name__,
            )
            return error_response(
                500, "decision_backend_error", "the decision backend failed",
            )
        return JSONResponse(body)

    return router
