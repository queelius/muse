"""Records and backend protocol for decision/evaluation.

No FastAPI or ML imports: the strict parser, the codec, the gateway and
the per-model worker all import this module.

A request carries shared `state` plus typed questions (Choice, Score,
Noul). A backend first prepares every question (tokenization, length
checks), then evaluates the prepared batch into one complete probability
distribution per question. The codec, not the backend, turns those
distributions into wire answers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Union, runtime_checkable


JsonValue = Union[
    None, bool, int, float, str, list["JsonValue"], dict[str, "JsonValue"],
]

QuestionKind = Literal["choice", "score", "noul"]
QUESTION_KINDS: tuple[str, ...] = ("choice", "score", "noul")


@dataclass(frozen=True)
class DecisionQuestion:
    """One validated question.

    criteria: ordered (key, description) pairs. Choice keeps the caller's
    option keys and order (description may be None). Score keys are the
    decimal level indices "0".."K-1". Noul keys are exactly ("true",
    "false"), with omitted descriptions filled as "True" / "False".
    """

    kind: QuestionKind
    instructions: str
    criteria: tuple[tuple[str, str | None], ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self.criteria)


@dataclass(frozen=True)
class DecisionRequest:
    """A validated request. `questions` keeps the caller's order and IDs."""

    model: str
    state: str | dict[str, JsonValue] | list[JsonValue]
    questions: tuple[tuple[str, DecisionQuestion], ...]


@dataclass(frozen=True)
class PreparedQuestion:
    """One question after backend preparation.

    payload is adapter-private token/array data, never wire JSON.
    input_tokens counts the logical tokens this question feeds the model.
    """

    question_id: str
    keys: tuple[str, ...]
    payload: object
    input_tokens: int


@dataclass(frozen=True)
class PreparedBatch:
    model_id: str
    questions: tuple[PreparedQuestion, ...]
    input_tokens: int


@dataclass(frozen=True)
class QuestionDistribution:
    """Raw backend output: a probability for every expected key."""

    question_id: str
    probabilities: dict[str, float]


@dataclass(frozen=True)
class DecisionBatch:
    answers: tuple[QuestionDistribution, ...]
    input_tokens: int


class DecisionInputError(Exception):
    """An invalid or unsupported request.

    code and message are safe to return to the client: they never contain
    user-supplied state, IDs, instructions or criteria. status is 400, or
    413 for size limits.
    """

    def __init__(self, code: str, message: str, *, status: int = 400) -> None:
        self.code = code
        self.message = message
        self.status = status
        super().__init__(message)


class DecisionBackendError(Exception):
    """A backend failure or invalid backend output.

    Carries only a safe code; native exception text (which can include
    local paths or user text) is never attached.
    """

    def __init__(
        self,
        code: str,
        message: str = "the decision backend produced an invalid result",
    ) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


@runtime_checkable
class DecisionModel(Protocol):
    """Structural protocol both decision runtimes satisfy."""

    model_id: str

    def prepare(self, request: DecisionRequest) -> PreparedBatch:
        """Tokenize and length-check every question; no inference."""
        ...

    def evaluate(self, batch: PreparedBatch) -> DecisionBatch:
        """Run inference on a batch this instance prepared."""
        ...

    def close(self) -> None:
        ...
