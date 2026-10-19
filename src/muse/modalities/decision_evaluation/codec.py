"""Answer encoding for decision/evaluation.

Backends return one raw probability distribution per question; this
module validates them and builds the wire envelope:

    {"model": id, "answers": {qid: answer, ...},
     "usage": {"input_tokens": n, "output_tokens": 0}}

Choice answers carry the winning original key, every option's
probability and a confidence. Score answers carry the probability-weighted
level index, a legend and the full level distribution keyed by decimal
index strings, plus a confidence. Noul answers carry only `noul`, the
probability of true.

Confidence summarizes how concentrated a distribution is; it is not a
probability of being correct. Exact ties select the first supplied
option or level. Probabilities are not rounded.
"""
from __future__ import annotations

import math

from muse.modalities.decision_evaluation.protocol import (
    DecisionBackendError,
    DecisionBatch,
    DecisionQuestion,
    DecisionRequest,
    JsonValue,
    QuestionDistribution,
)


# A distribution whose sum is within this of 1 is renormalized (float
# roundoff); anything further off is invalid backend output.
NORMALIZATION_TOLERANCE = 1e-5


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def _first_argmax(probabilities: tuple[float, ...]) -> int:
    # max() keeps the first of equal elements.
    return max(range(len(probabilities)), key=probabilities.__getitem__)


def choice_confidence(probabilities: tuple[float, ...]) -> float:
    """(max(p) - 1/K) / (1 - 1/K): 0 for uniform, 1 for one-hot."""
    k = len(probabilities)
    if k < 2:
        raise ValueError("confidence needs at least two outcomes")
    return _clamp_unit((max(probabilities) - 1 / k) / (1 - 1 / k))


def score_confidence(probabilities: tuple[float, ...]) -> float:
    """1 - E|i - m| / U for ordered levels, floored at 0.

    m is the first most-probable level and U = mean |i - (K - 1)/2|, the
    spread of a uniform distribution about the middle level.
    """
    k = len(probabilities)
    if k < 2:
        raise ValueError("confidence needs at least two outcomes")
    m = _first_argmax(probabilities)
    spread = math.fsum(p * abs(i - m) for i, p in enumerate(probabilities))
    uniform = sum(abs(i - (k - 1) / 2) for i in range(k)) / k
    return _clamp_unit(1 - spread / uniform)


def score_expectation(probabilities: tuple[float, ...]) -> float:
    """Probability-weighted level index."""
    return math.fsum(i * p for i, p in enumerate(probabilities))


def _invalid(code: str = "invalid_distribution") -> DecisionBackendError:
    return DecisionBackendError(code)


def _distribution(keys: tuple[str, ...], probabilities: object) -> tuple[float, ...]:
    """Return probabilities in `keys` order, renormalized; raise if invalid."""
    if not isinstance(probabilities, dict) or len(probabilities) != len(keys):
        raise _invalid()
    if set(probabilities) != set(keys):
        raise _invalid()
    values = []
    for key in keys:
        value = probabilities[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _invalid()
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise _invalid()
        values.append(value)
    total = math.fsum(values)
    if abs(total - 1.0) > NORMALIZATION_TOLERANCE:
        raise _invalid()
    if total != 1.0:
        values = [value / total for value in values]
    return tuple(values)


def _encode_answer(question: DecisionQuestion, values: tuple[float, ...]) -> dict[str, JsonValue]:
    keys = question.keys
    if question.kind == "choice":
        return {
            "type": "choice",
            "choice": keys[_first_argmax(values)],
            "probabilities": dict(zip(keys, values)),
            "confidence": choice_confidence(values),
        }
    if question.kind == "score":
        return {
            "type": "score",
            "score": score_expectation(values),
            "legend": dict(question.criteria),
            "probabilities": dict(zip(keys, values)),
            "confidence": score_confidence(values),
        }
    return {"type": "noul", "noul": values[keys.index("true")]}


def encode_response(
    request: DecisionRequest,
    result: DecisionBatch,
    *,
    model_id: str,
) -> dict[str, JsonValue]:
    """Validate a backend result against its request and build the envelope.

    Raises DecisionBackendError (never a partial answer) when any question
    is missing, repeated or extra, any distribution is incomplete,
    non-finite, negative or not normalized, or usage is malformed.
    """
    if not isinstance(result, DecisionBatch):
        raise _invalid("invalid_result")
    tokens = result.input_tokens
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise _invalid("invalid_usage")

    by_id: dict[str, QuestionDistribution] = {}
    for dist in result.answers:
        if not isinstance(dist, QuestionDistribution) or dist.question_id in by_id:
            raise _invalid("incomplete_answers")
        by_id[dist.question_id] = dist
    if set(by_id) != {qid for qid, _ in request.questions}:
        raise _invalid("incomplete_answers")

    answers: dict[str, JsonValue] = {}
    for qid, question in request.questions:
        values = _distribution(question.keys, by_id[qid].probabilities)
        answers[qid] = _encode_answer(question, values)

    return {
        "model": model_id,
        "answers": answers,
        "usage": {"input_tokens": tokens, "output_tokens": 0},
    }
