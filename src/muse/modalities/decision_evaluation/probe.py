"""Representative inference for `muse models probe` and probe-on-pull.

One short batch with a Choice, a Score and a Noul question, validated
end to end through the strict parser and the codec. Any missing,
incomplete or invalid distribution raises, which fails the probe and
therefore the pull. This checks that the model loads and answers every
question type; it says nothing about answer quality.
"""
from __future__ import annotations

import json

from muse.modalities.decision_evaluation.codec import encode_response
from muse.modalities.decision_evaluation.protocol import DecisionModel, JsonValue
from muse.modalities.decision_evaluation.validation import parse_request


PROBE_SHAPE = "1 short state; choice (2 options), score (3 levels), noul"

_QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this message?",
        "criteria": {"billing": "Charges and refunds", "technical": "Software faults"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this message?",
        "criteria": ["Routine", "Needs a prompt reply", "Immediate action"],
    },
    "duplicate": {
        "type": "noul",
        "instructions": "The customer reports a duplicate charge.",
    },
}
_STATE = "Customer message: I was charged twice for my subscription this month."
_EXPECTED_TYPES = {qid: q["type"] for qid, q in _QUESTIONS.items()}


def probe_backend(backend: DecisionModel) -> dict[str, JsonValue]:
    """Run the three-type batch on a loaded backend; return the response."""
    body = json.dumps(
        {"model": backend.model_id, "state": _STATE, "questions": _QUESTIONS},
    ).encode("utf-8")
    request = parse_request(body)
    batch = backend.evaluate(backend.prepare(request))
    response = encode_response(request, batch, model_id=backend.model_id)
    answers = response["answers"]
    got = {qid: answer.get("type") for qid, answer in answers.items()}
    if got != _EXPECTED_TYPES:
        raise RuntimeError("decision probe returned the wrong answer types")
    if response["usage"]["input_tokens"] <= 0:
        raise RuntimeError("decision probe reported no input tokens")
    return response
