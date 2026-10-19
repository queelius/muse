"""Shared fixtures for decision/evaluation tests."""
from __future__ import annotations

import copy
import json

import pytest


_MIXED = {
    "model": "decision-0.8b-q8",
    "state": {"message": "I was charged twice and need a refund."},
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Charges and refunds",
                "technical": "Software faults",
            },
        },
        "duplicate": {
            "type": "noul",
            "instructions": "The customer reports a duplicate charge.",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is the request?",
            "criteria": [
                "Routine",
                "Needs prompt attention",
                "Immediate outage response",
            ],
        },
    },
}


@pytest.fixture
def mixed_payload() -> dict:
    """The spec's ordered team/duplicate/urgency request, as a fresh dict."""
    return copy.deepcopy(_MIXED)


@pytest.fixture
def encode_body():
    """Serialize a payload the way a UTF-8 JSON client would."""

    def _encode(payload) -> bytes:
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    return _encode


@pytest.fixture
def mixed_body(mixed_payload, encode_body) -> bytes:
    return encode_body(mixed_payload)
