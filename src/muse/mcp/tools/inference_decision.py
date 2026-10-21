"""Inference decision tool.

One tool wrapping the decision modality:
  - muse_decide  /v1/systemone

It returns the server's full envelope (model, every answer with its
probabilities, usage) as one TextContent block. An HTTP error is left to
``MCPServer.call_handler``, which reports the status and the server's
error code/message (both safe by construction), never an answer;
transport errors are reduced to their type so local details do not leak.
"""
from __future__ import annotations

import httpx
from mcp.types import Tool

from muse.mcp.client import MuseClient
from muse.mcp.tools import INFERENCE_TOOLS, ToolEntry
from muse.mcp.tools._common import json_block
from muse.modalities.decision_evaluation.protocol import DecisionInputError


_ARGS = frozenset({"model", "state", "questions"})


class DecisionToolError(RuntimeError):
    """A sanitized muse_decide failure."""


def _handle_decide(client: MuseClient, args: dict) -> list[dict]:
    extra = set(args) - _ARGS
    if extra:
        raise DecisionToolError("muse_decide accepts only model, state and questions")
    if not isinstance(args.get("model"), str) or not args["model"]:
        raise DecisionToolError("muse_decide requires an explicit catalog model id")
    try:
        out = client.decide(
            model=args["model"], state=args.get("state"), questions=args.get("questions"),
        )
    except DecisionInputError as exc:
        raise DecisionToolError(f"invalid request ({exc.code}): {exc.message}") from None
    except httpx.RequestError as exc:
        raise DecisionToolError(f"could not reach muse ({type(exc).__name__})") from None
    return [json_block(out, ensure_ascii=False)]


_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["choice", "score", "noul"]},
        "instructions": {"type": "string"},
        "criteria": {
            "description": (
                "choice: object of option key -> description (string or null), "
                "2-24 options; score: array of 2-10 ordered level descriptions; "
                "noul: optional object with 'true' and/or 'false' descriptions"
            ),
        },
    },
    "required": ["type", "instructions"],
    "additionalProperties": False,
}


INFERENCE_TOOLS.extend([
    ToolEntry(
        tool=Tool(
            name="muse_decide",
            description=(
                "Ask a local decision model typed questions about one shared "
                "state: choice (pick one option), score (an ordered level) or "
                "noul (probability a statement is true). Answers come from "
                "complete probability distributions, not generated text; "
                "confidence measures concentration, not correctness, and the "
                "models are not safety gates. Returns {model, answers, usage}."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": (
                            "Required catalog id of a decision/evaluation model "
                            "(e.g. 'decision-0.8b-q8' or 'decima-base')."
                        ),
                    },
                    "state": {
                        "description": "Shared context: a nonempty string, object or array.",
                    },
                    "questions": {
                        "type": "object",
                        "description": "1-16 questions keyed by caller-chosen id.",
                        "additionalProperties": _QUESTION_SCHEMA,
                    },
                },
                "required": ["model", "state", "questions"],
                "additionalProperties": False,
            },
        ),
        handler=_handle_decide,
    ),
])
