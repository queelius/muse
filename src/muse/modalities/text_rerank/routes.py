"""FastAPI routes for /v1/rerank.

Cohere-compat shape:
  request:  {"query": str, "documents": list[str],
             "top_n"?: int, "return_documents"?: bool, "model"?: str}
  response: {"id", "model", "results": [{"index", "relevance_score",
             "document"?: {"text": str}}], "meta": {"billed_units": ...}}

Error envelopes follow muse's OpenAI-compat convention. 404 raises
ModelNotFoundError; 400 returns error_response() so the bare
{"error": ...} envelope reaches the client without FastAPI's
{"detail": ...} wrap.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from muse.core import config
from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.text_rerank.codec import encode_rerank_response


# MODALITY defined locally to avoid the __init__ circular import;
# sibling modalities all do this.
MODALITY = "text/rerank"


logger = logging.getLogger(__name__)


# Defaults are conservative. A request with documents=10000 can OOM the
# worker by trying to materialize a giant batch of pairs. Caps are
# tunable via env so power users with big GPUs can lift them.
#
# Read per-request via muse.core.config so changes take effect without
# a restart (matches the text_classification / text_translation pattern).
def _max_documents() -> int:
    return config.get("limits.rerank_max_documents")


def _max_query_chars() -> int:
    return config.get("limits.rerank_max_query_chars")


def _max_doc_chars() -> int:
    return config.get("limits.rerank_max_doc_chars")


class _RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_n: int | None = Field(default=None, ge=1, le=1000)
    model: str | None = None
    return_documents: bool = False


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/rerank")
    async def rerank(req: _RerankRequest):
        if not req.query:
            return error_response(
                400, "invalid_parameter", "query must not be empty",
            )
        max_query_chars = _max_query_chars()
        if len(req.query) > max_query_chars:
            return error_response(
                400, "invalid_parameter",
                f"query exceeds MUSE_RERANK_MAX_QUERY_CHARS={max_query_chars}",
            )
        if not req.documents:
            return error_response(
                400, "invalid_parameter", "documents must be a non-empty list",
            )
        max_documents = _max_documents()
        if len(req.documents) > max_documents:
            return error_response(
                400, "invalid_parameter",
                f"documents batch size {len(req.documents)} exceeds "
                f"MUSE_RERANK_MAX_DOCUMENTS={max_documents}",
            )
        empty_idx = next(
            (i for i, s in enumerate(req.documents) if not s), None,
        )
        if empty_idx is not None:
            return error_response(
                400, "invalid_parameter",
                f"documents[{empty_idx}] must not be empty",
            )
        max_doc_chars = _max_doc_chars()
        too_long = next(
            (i for i, s in enumerate(req.documents) if len(s) > max_doc_chars),
            None,
        )
        if too_long is not None:
            return error_response(
                400, "invalid_parameter",
                f"documents[{too_long}] exceeds "
                f"MUSE_RERANK_MAX_DOC_CHARS={max_doc_chars}",
            )

        try:
            backend = registry.get(MODALITY, req.model)
        except KeyError:
            raise ModelNotFoundError(
                model_id=req.model or "<default>", modality=MODALITY,
            )

        effective_id = backend.model_id

        # backend.rerank is sync (cross-encoder predict); offload so a
        # slow inference doesn't block sibling /health, /v1/models, or
        # other in-flight requests on the same worker. Per-backend lock
        # serializes calls into one model without blocking siblings on
        # the same worker (H2 fix, v0.45.7).
        def _rerank():
            with backend._inference_lock:
                return backend.rerank(req.query, req.documents, req.top_n)

        try:
            results = await asyncio.to_thread(_rerank)
        except Exception:  # noqa: BLE001
            # Log the real exception server-side but never leak it to the
            # client: str(e) can carry internal filesystem paths, CUDA
            # driver text, or other backend-implementation detail.
            logger.exception("rerank failed")
            return error_response(
                500, "internal_error",
                "rerank backend failed; see server logs",
            )
        body = encode_rerank_response(
            results,
            model_id=effective_id,
            return_documents=req.return_documents,
        )
        return JSONResponse(content=body)

    return router
