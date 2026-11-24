"""decision/evaluation modality.

Public surface:
  - MODALITY: str (MIME-shaped tag; used by discover_modalities)
  - build_router(registry) -> APIRouter (mounts POST /v1/systemone and
    POST /v1/systemone/batch)
  - PROBE_DEFAULTS: the three-type probe run by `muse models probe` and
    after `muse pull`
  - CONCURRENCY_CEILING: hard per-model limits the gateway applies to
    every decision model, whatever its manifest says
  - STRICT_REQUEST_PARSERS: the gateway parses /v1/systemone and
    /v1/systemone/batch bodies with the strict parser before selecting a model
  - CANCEL_ON_DISCONNECT_PATHS: the gateway drops its worker connection
    when a client disconnects, so the worker stops the batch
  - DecisionClient (HTTP client returning the full envelope)
  - the protocol records and errors

Wire contract: a text-only subset of TypeSafe's System One API (Choice,
Score, Noul) with an explicit Muse catalog `model`. See
docs/superpowers/specs/2026-10-18-decision-api-design.md.
"""
from fastapi import APIRouter

from muse.modalities.decision_evaluation.client import DecisionClient
from muse.modalities.decision_evaluation.probe import PROBE_SHAPE, probe_backend
from muse.modalities.decision_evaluation.protocol import (
    DecisionBackendError,
    DecisionBatch,
    DecisionBatchItem,
    DecisionBatchRequest,
    DecisionCancelled,
    DecisionInputError,
    DecisionModel,
    DecisionQuestion,
    DecisionRequest,
    QuestionDistribution,
)
from muse.modalities.decision_evaluation.routes import build_router as _build_router


MODALITY = "decision/evaluation"

DISCOVERY = {
    "summary": (
        "Typed decisions. Answers Choice, Score and Noul questions about a "
        "state with complete probability distributions instead of generated "
        "text. Text only; the model must be named."
    ),
    "use_for": [
        (
            "classifying or rating a piece of text against explicit criteria with "
            "calibrated probabilities"
        ),
        "routing or triage where the score distribution matters",
    ],
    "not_for": ["free-form generation (use chat/completion)", "images or audio inputs"],
    "routes": [
        {
            "method": "POST",
            "path": "/v1/systemone",
            "request_content_type": "application/json",
            "response_content_type": "application/json",
            "compat": "typesafe:systemone",
            "mcp_tool": "muse_decide",
            "file_fields": [],
            "media_fields": [],
            "example": {
                "model": "<model>",
                "state": "Customer message: I was charged twice this month.",
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "Which team should handle this?",
                        "criteria": {
                            "billing": "Charges and refunds",
                            "technical": "Software faults",
                        },
                    },
                },
            },
        },
        {
            "method": "POST",
            "path": "/v1/systemone/batch",
            "request_content_type": "application/json",
            "response_content_type": "application/json",
            "compat": None,
            "mcp_tool": "muse_decide_batch",
            "file_fields": [],
            "media_fields": [],
            "example": {
                "model": "<model>",
                "items": [
                    {
                        "id": "npc-1",
                        "state": "Guard sees an intruder near the gate.",
                        "questions": {
                            "action": {
                                "type": "choice",
                                "instructions": "How should the guard respond?",
                                "criteria": {
                                    "alert": "Sound the alarm",
                                    "ignore": "Continue patrolling",
                                },
                            },
                        },
                    },
                ],
            },
        },
    ],
}

# Literals: read by muse.core.discovery via AST, without importing.
CONCURRENCY_CEILING = {"max_concurrency": 1, "max_queue_depth": 32}
STRICT_REQUEST_PARSERS = {
    "/v1/systemone": "muse.modalities.decision_evaluation.validation:parse_request",
    "/v1/systemone/batch": "muse.modalities.decision_evaluation.validation:parse_batch_request",
}
CANCEL_ON_DISCONNECT_PATHS = ("/v1/systemone", "/v1/systemone/batch")

# Per-modality probe defaults read by `muse models probe`.
PROBE_DEFAULTS = {
    "shape": PROBE_SHAPE,
    "call": probe_backend,
}


def build_router(registry) -> APIRouter:
    """Mount POST /v1/systemone and POST /v1/systemone/batch."""
    return _build_router(registry)


__all__ = [
    "MODALITY",
    "DISCOVERY",
    "CONCURRENCY_CEILING",
    "STRICT_REQUEST_PARSERS",
    "CANCEL_ON_DISCONNECT_PATHS",
    "PROBE_DEFAULTS",
    "build_router",
    "DecisionClient",
    "DecisionBackendError",
    "DecisionBatch",
    "DecisionBatchItem",
    "DecisionBatchRequest",
    "DecisionCancelled",
    "DecisionInputError",
    "DecisionModel",
    "DecisionQuestion",
    "DecisionRequest",
    "QuestionDistribution",
]
