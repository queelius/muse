"""Answer encoding and concentration statistics for decision/evaluation."""
from __future__ import annotations

import math

import pytest

from muse.modalities.decision_evaluation.codec import (
    NORMALIZATION_TOLERANCE,
    choice_confidence,
    encode_response,
    score_confidence,
    score_expectation,
)
from muse.modalities.decision_evaluation.protocol import (
    DecisionBackendError,
    DecisionBatch,
    QuestionDistribution,
)
from muse.modalities.decision_evaluation.validation import parse_request


def test_reference_values():
    assert choice_confidence((0.7, 0.2, 0.1)) == pytest.approx(0.55)
    assert score_confidence((0.7, 0.2, 0.1)) == pytest.approx(0.4)
    assert choice_confidence((0.5, 0.5)) == 0.0
    assert score_expectation((0.7, 0.2, 0.1)) == pytest.approx(0.4)


def test_confidence_bounds():
    assert choice_confidence((1.0, 0.0, 0.0)) == 1.0
    assert choice_confidence((1 / 3, 1 / 3, 1 / 3)) == pytest.approx(0.0, abs=1e-12)
    assert choice_confidence((1 / 3, 1 / 3, 1 / 3)) >= 0.0
    assert score_confidence((0.0, 1.0, 0.0)) == 1.0


def test_score_confidence_uses_the_first_mode_and_floors_at_zero():
    # Bimodal: m = 0 (first mode), spread 1.0 exceeds U = 2/3.
    assert score_confidence((0.5, 0.0, 0.5)) == 0.0
    # Same peak height, but Choice ignores order and so differs.
    assert choice_confidence((0.5, 0.0, 0.5)) == pytest.approx(0.25)
    # K = 4 gives U = 1. A tie between levels 0 and 1 measures spread from
    # level 0 (0.65 -> 0.35), not level 1 (0.55 -> 0.45).
    assert score_confidence((0.45, 0.45, 0.1, 0.0)) == pytest.approx(0.35)


def test_score_confidence_two_levels():
    # K = 2: U = 0.5; m = 0; spread = p1.
    assert score_confidence((0.8, 0.2)) == pytest.approx(1 - 0.2 / 0.5)


# --- encode_response ------------------------------------------------------


def _batch(*pairs, input_tokens=42):
    return DecisionBatch(
        answers=tuple(QuestionDistribution(qid, probs) for qid, probs in pairs),
        input_tokens=input_tokens,
    )


_MIXED_RESULT = (
    ("team", {"billing": 0.9, "technical": 0.1}),
    ("duplicate", {"true": 0.8, "false": 0.2}),
    ("urgency", {"0": 0.7, "1": 0.2, "2": 0.1}),
)


def test_mixed_response_shape(mixed_body):
    req = parse_request(mixed_body)
    out = encode_response(req, _batch(*_MIXED_RESULT), model_id="decision-0.8b-q8")

    assert list(out) == ["model", "answers", "usage"]
    assert out["model"] == "decision-0.8b-q8"
    assert list(out["answers"]) == ["team", "duplicate", "urgency"]

    team = out["answers"]["team"]
    assert team == {
        "type": "choice",
        "choice": "billing",
        "probabilities": {"billing": 0.9, "technical": 0.1},
        "confidence": pytest.approx(0.8),
    }

    duplicate = out["answers"]["duplicate"]
    assert duplicate == {"type": "noul", "noul": 0.8}
    assert "confidence" not in duplicate

    urgency = out["answers"]["urgency"]
    assert urgency["type"] == "score"
    assert urgency["score"] == pytest.approx(0.4)
    assert urgency["legend"] == {
        "0": "Routine", "1": "Needs prompt attention", "2": "Immediate outage response",
    }
    assert urgency["probabilities"] == {"0": 0.7, "1": 0.2, "2": 0.1}
    assert list(urgency["probabilities"]) == ["0", "1", "2"]
    assert urgency["confidence"] == pytest.approx(0.4)

    assert out["usage"] == {"input_tokens": 42, "output_tokens": 0}


def test_answers_follow_request_order_not_backend_order(mixed_body):
    req = parse_request(mixed_body)
    out = encode_response(req, _batch(*reversed(_MIXED_RESULT)), model_id="m")
    assert list(out["answers"]) == ["team", "duplicate", "urgency"]


def test_probability_keys_follow_request_order(mixed_body):
    req = parse_request(mixed_body)
    shuffled = (
        ("team", {"technical": 0.1, "billing": 0.9}),
        ("duplicate", {"false": 0.2, "true": 0.8}),
        ("urgency", {"2": 0.1, "0": 0.7, "1": 0.2}),
    )
    out = encode_response(req, _batch(*shuffled), model_id="m")
    assert list(out["answers"]["team"]["probabilities"]) == ["billing", "technical"]
    assert list(out["answers"]["urgency"]["probabilities"]) == ["0", "1", "2"]


def test_noul_is_probability_of_true_regardless_of_dict_order(mixed_body):
    req = parse_request(mixed_body)
    result = list(_MIXED_RESULT)
    result[1] = ("duplicate", {"false": 0.75, "true": 0.25})
    out = encode_response(req, _batch(*result), model_id="m")
    assert out["answers"]["duplicate"]["noul"] == 0.25


def test_choice_tie_selects_first_supplied_option(mixed_body):
    req = parse_request(mixed_body)
    result = list(_MIXED_RESULT)
    result[0] = ("team", {"technical": 0.5, "billing": 0.5})
    out = encode_response(req, _batch(*result), model_id="m")
    assert out["answers"]["team"]["choice"] == "billing"
    assert out["answers"]["team"]["confidence"] == 0.0


def test_precision_is_not_rounded(mixed_body):
    req = parse_request(mixed_body)
    p = 0.123456789012
    result = list(_MIXED_RESULT)
    result[0] = ("team", {"billing": p, "technical": 1 - p})
    out = encode_response(req, _batch(*result), model_id="m")
    assert out["answers"]["team"]["probabilities"]["billing"] == p


def test_roundoff_within_tolerance_is_renormalized(mixed_body):
    req = parse_request(mixed_body)
    drift = NORMALIZATION_TOLERANCE / 2
    result = list(_MIXED_RESULT)
    result[2] = ("urgency", {"0": 0.7 + drift, "1": 0.2, "2": 0.1})
    out = encode_response(req, _batch(*result), model_id="m")
    assert math.fsum(out["answers"]["urgency"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-12)


def test_integer_probabilities_are_accepted(mixed_body):
    req = parse_request(mixed_body)
    result = list(_MIXED_RESULT)
    result[0] = ("team", {"billing": 1, "technical": 0})
    out = encode_response(req, _batch(*result), model_id="m")
    assert out["answers"]["team"]["choice"] == "billing"
    assert out["answers"]["team"]["confidence"] == 1.0


@pytest.mark.parametrize("team_probs", [
    {"billing": 0.9},                                      # missing key
    {"billing": 0.9, "technical": 0.1, "other": 0.0},      # extra key
    {"billing": float("nan"), "technical": 0.1},
    {"billing": float("inf"), "technical": 0.1},
    {"billing": 1.1, "technical": -0.1},                   # negative
    {"billing": 0.9, "technical": 0.1 + 2e-5},             # materially unnormalized
    {"billing": 0.5, "technical": 0.4},
    {"billing": True, "technical": False},                 # bools are not numbers
    {"billing": "0.9", "technical": "0.1"},
    {"billing": 0.0, "technical": 0.0},
])
def test_invalid_distributions_rejected(team_probs, mixed_body):
    req = parse_request(mixed_body)
    result = list(_MIXED_RESULT)
    result[0] = ("team", team_probs)
    with pytest.raises(DecisionBackendError) as exc:
        encode_response(req, _batch(*result), model_id="m")
    assert exc.value.code == "invalid_distribution"


def test_non_dict_distribution_rejected(mixed_body):
    req = parse_request(mixed_body)
    result = list(_MIXED_RESULT)
    result[0] = ("team", [0.9, 0.1])
    with pytest.raises(DecisionBackendError) as exc:
        encode_response(req, _batch(*result), model_id="m")
    assert exc.value.code == "invalid_distribution"


@pytest.mark.parametrize("answers", [
    _MIXED_RESULT[:2],                                       # missing question
    _MIXED_RESULT + (("extra", {"true": 1.0, "false": 0.0}),),
    _MIXED_RESULT + (_MIXED_RESULT[0],),                     # repeated question
])
def test_incomplete_or_repeated_answers_rejected(answers, mixed_body):
    req = parse_request(mixed_body)
    with pytest.raises(DecisionBackendError) as exc:
        encode_response(req, _batch(*answers), model_id="m")
    assert exc.value.code == "incomplete_answers"


@pytest.mark.parametrize("tokens", [-1, True, 1.5, None])
def test_invalid_usage_rejected(tokens, mixed_body):
    req = parse_request(mixed_body)
    with pytest.raises(DecisionBackendError) as exc:
        encode_response(req, _batch(*_MIXED_RESULT, input_tokens=tokens), model_id="m")
    assert exc.value.code == "invalid_usage"


def test_backend_errors_do_not_echo_values_or_ids(encode_body):
    body = encode_body({
        "model": "m", "state": "s",
        "questions": {"SECRETID": {"type": "noul", "instructions": "i"}},
    })
    req = parse_request(body)
    with pytest.raises(DecisionBackendError) as exc:
        encode_response(req, _batch(("SECRETID", {"true": 0.31337, "false": 0.1})), model_id="m")
    assert "SECRETID" not in str(exc.value)
    assert "31337" not in str(exc.value)


def test_non_batch_result_rejected(mixed_body):
    req = parse_request(mixed_body)
    with pytest.raises(DecisionBackendError):
        encode_response(req, {"answers": []}, model_id="m")
