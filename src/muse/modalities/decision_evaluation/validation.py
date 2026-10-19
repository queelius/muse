"""Strict request parsing for POST /v1/systemone.

`parse_request(body)` takes the raw request bytes (the gateway and the
direct worker both call it before any model selection) and returns an
immutable `DecisionRequest`, or raises `DecisionInputError`.

Strictness rules:
  - UTF-8 only; malformed bytes, a BOM and unpaired surrogates are errors.
  - Duplicate object keys are errors at every nesting level.
  - NaN, Infinity and numbers that overflow to infinity are errors.
  - Unknown fields and media inputs are errors, never ignored.
  - IDs and option keys are kept exactly as sent (no Unicode normalization).
  - Byte limits count UTF-8 bytes, not characters.

Error messages never quote user-supplied content (state, IDs, keys,
instructions, criteria, unknown field names); they point at positions.
"""
from __future__ import annotations

import json
import math

from muse.modalities.decision_evaluation.protocol import (
    QUESTION_KINDS,
    DecisionInputError,
    DecisionQuestion,
    DecisionRequest,
    JsonValue,
)


MAX_BODY_BYTES = 128 * 1024
MAX_STATE_BYTES = 64 * 1024
MAX_STATE_DEPTH = 8
MIN_QUESTIONS, MAX_QUESTIONS = 1, 16
MIN_CHOICE_OPTIONS, MAX_CHOICE_OPTIONS = 2, 24
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10
MAX_KEY_BYTES = 128
MAX_INSTRUCTIONS_BYTES = 4 * 1024
MAX_DESCRIPTION_BYTES = 2 * 1024

NOUL_DEFAULTS = (("true", "True"), ("false", "False"))

_REQUEST_FIELDS = frozenset({"model", "state", "questions"})
_QUESTION_FIELDS = frozenset({"type", "instructions", "criteria"})
_MEDIA_FIELDS = frozenset({
    "image", "images", "audio", "video", "media", "file", "files",
    "asset", "assets", "attachments", "image_url",
})


class _Reject(Exception):
    """Raised from json hooks; converted to DecisionInputError."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message


def _reject_duplicates(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    out: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in out:
            raise _Reject("duplicate_key", "request contains a duplicate object key")
        out[key] = value
    return out


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise _Reject("non_finite_number", "request contains a non-finite number")
    return value


def _reject_constant(_text: str):
    raise _Reject("non_finite_number", "request contains a non-finite number")


def _fail(code: str, message: str, *, status: int = 400) -> DecisionInputError:
    return DecisionInputError(code, message, status=status)


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _check_text(value: JsonValue) -> int:
    """Reject unpaired surrogates anywhere; return the deepest container depth.

    Iterative, so a deep (but parser-accepted) document cannot overflow
    the Python stack here.
    """
    deepest = 0
    stack: list[tuple[JsonValue, int]] = [(value, 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, str):
            _encodable(node)
        elif isinstance(node, dict):
            deepest = max(deepest, depth + 1)
            for key, child in node.items():
                _encodable(key)
                stack.append((child, depth + 1))
        elif isinstance(node, list):
            deepest = max(deepest, depth + 1)
            stack.extend((child, depth + 1) for child in node)
    return deepest


def _encodable(text: str) -> None:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise _fail("invalid_unicode", "request contains an unpaired surrogate") from None


def _decode(body: bytes) -> JsonValue:
    if len(body) > MAX_BODY_BYTES:
        raise _fail(
            "body_too_large",
            f"request body exceeds {MAX_BODY_BYTES} bytes",
            status=413,
        )
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        raise _fail("invalid_encoding", "request body is not valid UTF-8") from None
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_float=_finite_float,
            parse_constant=_reject_constant,
        )
    except _Reject as exc:
        raise _fail(exc.code, exc.message) from None
    except RecursionError:
        raise _fail("nesting_too_deep", "request JSON is nested too deeply") from None
    except ValueError:
        # JSONDecodeError, and int() refusing an absurdly long literal.
        raise _fail("invalid_json", "request body is not valid JSON") from None


def _reject_unknown(obj: dict, allowed: frozenset[str], where: str) -> None:
    extra = [key for key in obj if key not in allowed]
    if not extra:
        return
    if any(key in _MEDIA_FIELDS for key in extra):
        raise _fail("unsupported_media", f"{where} contains a media input; only text is supported")
    raise _fail("unsupported_field", f"{where} contains an unsupported field")


def _require(obj: dict, field: str, where: str) -> JsonValue:
    if field not in obj:
        raise _fail("missing_field", f"{where} is missing required field '{field}'")
    return obj[field]


def _bounded_text(value: JsonValue, limit: int, *, allow_empty: bool = False) -> bool:
    return (
        isinstance(value, str)
        and (allow_empty or value != "")
        and _utf8_len(value) <= limit
    )


def _parse_state(value: JsonValue, depth: int) -> str | dict | list:
    if isinstance(value, str):
        if value == "":
            raise _fail("invalid_state", "state must be a nonempty string, object or array")
        size = _utf8_len(value)
    elif isinstance(value, (dict, list)):
        if depth > MAX_STATE_DEPTH:
            raise _fail("state_too_deep", f"state nests deeper than {MAX_STATE_DEPTH} levels")
        size = _utf8_len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    else:
        raise _fail("invalid_state", "state must be a nonempty string, object or array")
    if size > MAX_STATE_BYTES:
        raise _fail("state_too_large", f"state exceeds {MAX_STATE_BYTES} bytes", status=413)
    return value


def _parse_choice(criteria: JsonValue, where: str) -> tuple[tuple[str, str | None], ...]:
    if not isinstance(criteria, dict):
        raise _fail("invalid_criteria", f"{where}.criteria must be an object of options")
    n = len(criteria)
    if not MIN_CHOICE_OPTIONS <= n <= MAX_CHOICE_OPTIONS:
        raise _fail(
            "invalid_criteria",
            f"{where}.criteria has {n} options; expected "
            f"{MIN_CHOICE_OPTIONS}-{MAX_CHOICE_OPTIONS}",
        )
    for i, (key, desc) in enumerate(criteria.items()):
        if not _bounded_text(key, MAX_KEY_BYTES):
            raise _fail(
                "invalid_criteria",
                f"{where}.criteria option {i} key must be 1-{MAX_KEY_BYTES} UTF-8 bytes",
            )
        if desc is not None and not _bounded_text(desc, MAX_DESCRIPTION_BYTES, allow_empty=True):
            raise _fail(
                "invalid_criteria",
                f"{where}.criteria option {i} description must be null or a string "
                f"of at most {MAX_DESCRIPTION_BYTES} UTF-8 bytes",
            )
    return tuple(criteria.items())


def _parse_score(criteria: JsonValue, where: str) -> tuple[tuple[str, str | None], ...]:
    if not isinstance(criteria, list):
        raise _fail("invalid_criteria", f"{where}.criteria must be an array of levels")
    n = len(criteria)
    if not MIN_SCORE_LEVELS <= n <= MAX_SCORE_LEVELS:
        raise _fail(
            "invalid_criteria",
            f"{where}.criteria has {n} levels; expected {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS}",
        )
    for i, desc in enumerate(criteria):
        if not _bounded_text(desc, MAX_DESCRIPTION_BYTES):
            raise _fail(
                "invalid_criteria",
                f"{where}.criteria level {i} must be a nonempty string of at most "
                f"{MAX_DESCRIPTION_BYTES} UTF-8 bytes",
            )
    if len(set(criteria)) != n:
        raise _fail("invalid_criteria", f"{where}.criteria levels must be distinct")
    return tuple((str(i), desc) for i, desc in enumerate(criteria))


def _parse_noul(question: dict, where: str) -> tuple[tuple[str, str | None], ...]:
    if "criteria" not in question:
        return NOUL_DEFAULTS
    criteria = question["criteria"]
    if not isinstance(criteria, dict):
        raise _fail("invalid_criteria", f"{where}.criteria must be an object")
    if any(key not in ("true", "false") for key in criteria):
        raise _fail("invalid_criteria", f"{where}.criteria accepts only 'true' and 'false'")
    for key, desc in criteria.items():
        if not _bounded_text(desc, MAX_DESCRIPTION_BYTES):
            raise _fail(
                "invalid_criteria",
                f"{where}.criteria '{key}' must be a nonempty string of at most "
                f"{MAX_DESCRIPTION_BYTES} UTF-8 bytes",
            )
    return tuple((key, criteria.get(key, default)) for key, default in NOUL_DEFAULTS)


def _parse_question(value: JsonValue, where: str) -> DecisionQuestion:
    if not isinstance(value, dict):
        raise _fail("invalid_question", f"{where} must be an object")
    _reject_unknown(value, _QUESTION_FIELDS, where)
    kind = _require(value, "type", where)
    if kind not in QUESTION_KINDS or not isinstance(kind, str):
        raise _fail(
            "unsupported_question_type",
            f"{where}.type must be one of: {', '.join(QUESTION_KINDS)}",
        )
    instructions = _require(value, "instructions", where)
    if not _bounded_text(instructions, MAX_INSTRUCTIONS_BYTES):
        raise _fail(
            "invalid_instructions",
            f"{where}.instructions must be a nonempty string of at most "
            f"{MAX_INSTRUCTIONS_BYTES} UTF-8 bytes",
        )
    if kind == "noul":
        criteria = _parse_noul(value, where)
    elif kind == "choice":
        criteria = _parse_choice(_require(value, "criteria", where), where)
    else:
        criteria = _parse_score(_require(value, "criteria", where), where)
    return DecisionQuestion(kind=kind, instructions=instructions, criteria=criteria)


def parse_request(body: bytes) -> DecisionRequest:
    """Parse and validate a raw /v1/systemone body. Raises DecisionInputError."""
    doc = _decode(body)
    if not isinstance(doc, dict):
        raise _fail("invalid_request", "request body must be a JSON object")
    _check_text(doc)
    _reject_unknown(doc, _REQUEST_FIELDS, "request")

    model = _require(doc, "model", "request")
    if not _bounded_text(model, MAX_KEY_BYTES):
        raise _fail(
            "invalid_model",
            f"model must be a nonempty catalog id of at most {MAX_KEY_BYTES} UTF-8 bytes",
        )

    raw_state = _require(doc, "state", "request")
    state = _parse_state(raw_state, _check_text(raw_state))

    questions = _require(doc, "questions", "request")
    if not isinstance(questions, dict):
        raise _fail("invalid_questions", "questions must be an object keyed by question id")
    n = len(questions)
    if not MIN_QUESTIONS <= n <= MAX_QUESTIONS:
        raise _fail(
            "invalid_questions",
            f"questions has {n} entries; expected {MIN_QUESTIONS}-{MAX_QUESTIONS}",
        )
    parsed: list[tuple[str, DecisionQuestion]] = []
    for i, (qid, question) in enumerate(questions.items()):
        where = f"questions[{i}]"
        if not _bounded_text(qid, MAX_KEY_BYTES):
            raise _fail(
                "invalid_question_id",
                f"{where} id must be 1-{MAX_KEY_BYTES} UTF-8 bytes",
            )
        parsed.append((qid, _parse_question(question, where)))

    return DecisionRequest(model=model, state=state, questions=tuple(parsed))
