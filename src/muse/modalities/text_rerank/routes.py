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

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.inference import cap, check_text_batch, run_locked
from muse.modalities.text_rerank.codec import encode_rerank_response


# MODALITY defined locally to avoid the __init__ circular import;
# sibling modalities all do this.
MODALITY = "text/rerank"


logger = logging.getLogger(__name__)


# Finite caps: a request with documents=10000 could OOM the worker
# materializing pairs. Tunable via env / config.yaml; read per request with
# config.get_positive.
_MAX_DOCUMENTS_KEY = "limits.rerank_max_documents"
_MAX_QUERY_CHARS_KEY = "limits.rerank_max_query_chars"
_MAX_DOC_CHARS_KEY = "limits.rerank_max_doc_chars"


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
        max_query_chars = cap(_MAX_QUERY_CHARS_KEY)
        if len(req.query) > max_query_chars:
            return error_response(
                400, "invalid_parameter",
                f"query exceeds MUSE_RERANK_MAX_QUERY_CHARS={max_query_chars}",
            )
        if not req.documents:
            return error_response(
                400, "invalid_parameter", "documents must be a non-empty list",
            )
        empty_idx = next(
            (i for i, s in enumerate(req.documents) if not s), None,
        )
        if empty_idx is not None:
            return error_response(
                400, "invalid_parameter",
                f"documents[{empty_idx}] must not be empty",
            )
        invalid = check_text_batch(
            req.documents, field="documents",
            max_batch_key=_MAX_DOCUMENTS_KEY, max_chars_key=_MAX_DOC_CHARS_KEY,
        )
        if invalid is not None:
            return invalid

        backend = registry.get_or_404(MODALITY, req.model)

        effective_id = backend.model_id

        results = await run_locked(
            backend, lambda: backend.rerank(req.query, req.documents, req.top_n),
            label="rerank", message="rerank backend failed; see server logs",
        )
        if isinstance(results, JSONResponse):
            return results
        body = encode_rerank_response(
            results,
            model_id=effective_id,
            return_documents=req.return_documents,
        )
        return JSONResponse(content=body)

    return router
