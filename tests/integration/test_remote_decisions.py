"""Integration tests for POST /v1/systemone against a live muse server.

Opt-in via MUSE_REMOTE_SERVER. Each test runs once per decision model id
in MUSE_DECISION_MODEL_IDS (comma-separated; default
"decision-0.8b-q8,decima-base") and skips a model the server does not
serve.

These check the contract (complete, finite, normalized distributions;
order and ID preservation; limits), not answer quality.
"""
from __future__ import annotations

import math
import os

import pytest
import requests

from muse.modalities.decision_evaluation import DecisionClient


MODEL_IDS = [
    m.strip() for m in os.environ.get(
        "MUSE_DECISION_MODEL_IDS", "decision-0.8b-q8,decima-base",
    ).split(",") if m.strip()
]

STATE = {"message": "I was charged twice for my subscription and need a refund."}
QUESTIONS = {
    "équipe": {
        "type": "choice",
        "instructions": "Which team should handle this message?",
        "criteria": {"technical": "Software faults", "billing": "Charges and refunds",
                     "sales": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this message?",
        "criteria": ["Routine", "Needs a prompt reply", "Immediate action"],
    },
    "refund": {
        "type": "noul",
        "instructions": "The customer asks for money back.",
        "criteria": {"true": "They want a refund", "false": "Anything else"},
    },
}


@pytest.fixture(params=MODEL_IDS)
def decision_model(request, remote_health):
    model_id = request.param
    if model_id not in (remote_health.get("models") or []):
        pytest.skip(f"muse server doesn't serve {model_id!r}")
    return model_id


def _assert_distribution(probs: dict, keys: list[str]) -> None:
    assert list(probs) == keys
    assert all(isinstance(v, float) and math.isfinite(v) and v >= 0 for v in probs.values())
    assert math.fsum(probs.values()) == pytest.approx(1.0, abs=1e-6)


def test_protocol_mixed_batch(remote_url, decision_model):
    out = DecisionClient(remote_url).decide(
        model=decision_model, state=STATE, questions=QUESTIONS,
    )
    assert out["model"] == decision_model
    assert list(out["answers"]) == ["équipe", "urgency", "refund"]
    choice = out["answers"]["équipe"]
    _assert_distribution(choice["probabilities"], ["technical", "billing", "sales"])
    assert choice["choice"] == max(choice["probabilities"], key=choice["probabilities"].get)
    assert 0.0 <= choice["confidence"] <= 1.0
    score = out["answers"]["urgency"]
    _assert_distribution(score["probabilities"], ["0", "1", "2"])
    assert 0.0 <= score["score"] <= 2.0
    assert 0.0 <= out["answers"]["refund"]["noul"] <= 1.0
    assert "confidence" not in out["answers"]["refund"]
    assert out["usage"]["input_tokens"] > 0 and out["usage"]["output_tokens"] == 0


def test_protocol_reordering_options_moves_keys_with_them(remote_url, decision_model):
    reordered = dict(QUESTIONS)
    criteria = QUESTIONS["équipe"]["criteria"]
    reordered["équipe"] = {**QUESTIONS["équipe"],
                           "criteria": dict(reversed(list(criteria.items())))}
    out = DecisionClient(remote_url).decide(
        model=decision_model, state=STATE, questions=reordered,
    )
    _assert_distribution(out["answers"]["équipe"]["probabilities"],
                         ["sales", "billing", "technical"])


def test_protocol_too_long_input_is_rejected_not_truncated(remote_url, decision_model):
    long_state = "word " * 3000  # over both models' token budgets, under 64 KiB
    r = requests.post(
        f"{remote_url}/v1/systemone",
        json={"model": decision_model, "state": long_state, "questions": QUESTIONS},
        timeout=120,
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "input_too_long"


def test_protocol_duplicate_model_key_is_rejected(remote_url, decision_model):
    raw = ('{"model":"x","model":"%s","state":"s",'
           '"questions":{"q":{"type":"noul","instructions":"i"}}}' % decision_model)
    r = requests.post(f"{remote_url}/v1/systemone", data=raw.encode(),
                      headers={"content-type": "application/json"}, timeout=30)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "duplicate_key"
