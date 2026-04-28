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
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
_MAX_DOCUMENTS = int(os.environ.get("MUSE_RERANK_MAX_DOCUMENTS", "1000"))
_MAX_QUERY_CHARS = int(os.environ.get("MUSE_RERANK_MAX_QUERY_CHARS", "4000"))
_MAX_DOC_CHARS = int(os.environ.get("MUSE_RERANK_MAX_DOC_CHARS", "100000"))


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
        if len(req.query) > _MAX_QUERY_CHARS:
            return error_response(
                400, "invalid_parameter",
                f"query exceeds MUSE_RERANK_MAX_QUERY_CHARS={_MAX_QUERY_CHARS}",
            )
        if not req.documents:
            return error_response(
                400, "invalid_parameter", "documents must be a non-empty list",
            )
        if len(req.documents) > _MAX_DOCUMENTS:
            return error_response(
                400, "invalid_parameter",
                f"documents batch size {len(req.documents)} exceeds "
                f"MUSE_RERANK_MAX_DOCUMENTS={_MAX_DOCUMENTS}",
            )
        empty_idx = next(
            (i for i, s in enumerate(req.documents) if not s), None,
        )
        if empty_idx is not None:
            return error_response(
                400, "invalid_parameter",
                f"documents[{empty_idx}] must not be empty",
            )
        too_long = next(
            (i for i, s in enumerate(req.documents) if len(s) > _MAX_DOC_CHARS),
            None,
        )
        if too_long is not None:
            return error_response(
                400, "invalid_parameter",
                f"documents[{too_long}] exceeds "
                f"MUSE_RERANK_MAX_DOC_CHARS={_MAX_DOC_CHARS}",
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
        # other in-flight requests on the same worker.
        results = await asyncio.to_thread(
            backend.rerank, req.query, req.documents, req.top_n,
        )
        body = encode_rerank_response(
            results,
            model_id=effective_id,
            return_documents=req.return_documents,
        )
        return JSONResponse(content=body)

    return router
