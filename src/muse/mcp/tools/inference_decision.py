"""Inference decision tool.

One tool wrapping the decision modality:
  - muse_decide  /v1/systemone

It returns the server's full envelope (model, every answer with its
probabilities, usage) as one TextContent block. An HTTP error is raised
as a short error naming the status and the server's error code/message
(both safe by construction), never turned into an answer; transport
errors are reduced to their type so local details do not leak.
"""
from __future__ import annotations

import json
from typing import Any

import httpx
from mcp.types import Tool

from muse.mcp.client import MuseClient
from muse.mcp.tools import INFERENCE_TOOLS, ToolEntry
from muse.modalities.decision_evaluation.protocol import DecisionInputError


_ARGS = frozenset({"model", "state", "questions"})
_MAX_ERROR_CHARS = 300


class DecisionToolError(RuntimeError):
    """A sanitized muse_decide failure."""


def _json_block(payload: Any) -> dict:
    return {"type": "text", "text": json.dumps(payload, indent=2, ensure_ascii=False)}


def _upstream_error(response: httpx.Response) -> str:
    try:
        error = response.json().get("error") or {}
        code, message = str(error.get("code", "")), str(error.get("message", ""))
    except Exception:  # noqa: BLE001
        code, message = "", ""
    text = f"muse returned HTTP {response.status_code}"
    if code:
        text += f" ({code})"
    if message:
        text += f": {message[:_MAX_ERROR_CHARS]}"
    return text


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
    except httpx.HTTPStatusError as exc:
        raise DecisionToolError(_upstream_error(exc.response)) from None
    except httpx.HTTPError as exc:
        raise DecisionToolError(f"could not reach muse ({type(exc).__name__})") from None
    return [_json_block(out)]


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
