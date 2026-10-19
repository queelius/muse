"""Strict request parsing for POST /v1/systemone."""
from __future__ import annotations

import pytest

from muse.modalities.decision_evaluation.protocol import (
    DecisionInputError,
    DecisionQuestion,
    DecisionRequest,
)
from muse.modalities.decision_evaluation.validation import (
    MAX_BODY_BYTES,
    MAX_DESCRIPTION_BYTES,
    MAX_INSTRUCTIONS_BYTES,
    MAX_KEY_BYTES,
    MAX_STATE_BYTES,
    MAX_STATE_DEPTH,
    parse_request,
)


def _err(body: bytes) -> DecisionInputError:
    with pytest.raises(DecisionInputError) as exc:
        parse_request(body)
    return exc.value


def _choice(n: int) -> dict:
    return {
        "type": "choice",
        "instructions": "Pick one.",
        "criteria": {f"opt{i}": None for i in range(n)},
    }


def _one(question: dict, *, state="s", model="m") -> dict:
    return {"model": model, "state": state, "questions": {"q": question}}


# --- happy path ----------------------------------------------------------


def test_mixed_request_parses_in_order(mixed_body):
    req = parse_request(mixed_body)

    assert isinstance(req, DecisionRequest)
    assert req.model == "decision-0.8b-q8"
    assert req.state == {"message": "I was charged twice and need a refund."}
    assert [qid for qid, _ in req.questions] == ["team", "duplicate", "urgency"]

    team = dict(req.questions)["team"]
    assert team == DecisionQuestion(
        kind="choice",
        instructions="Which team should handle this?",
        criteria=(
            ("billing", "Charges and refunds"),
            ("technical", "Software faults"),
        ),
    )
    duplicate = dict(req.questions)["duplicate"]
    assert duplicate.criteria == (("true", "True"), ("false", "False"))
    urgency = dict(req.questions)["urgency"]
    assert urgency.criteria == (
        ("0", "Routine"),
        ("1", "Needs prompt attention"),
        ("2", "Immediate outage response"),
    )
    assert urgency.keys == ("0", "1", "2")


def test_question_order_is_the_callers_not_sorted(encode_body):
    qs = {qid: _choice(2) for qid in ("z", "a", "m")}
    req = parse_request(encode_body({"model": "m", "state": "s", "questions": qs}))
    assert [qid for qid, _ in req.questions] == ["z", "a", "m"]


def test_choice_option_order_and_null_descriptions_preserved(encode_body):
    q = {
        "type": "choice",
        "instructions": "Pick.",
        "criteria": {"zeta": None, "alpha": "first letter", "mu": None},
    }
    req = parse_request(encode_body(_one(q)))
    assert req.questions[0][1].criteria == (
        ("zeta", None), ("alpha", "first letter"), ("mu", None),
    )


def test_unicode_ids_and_keys_are_preserved_byte_for_byte(encode_body):
    nfc, nfd = "café", "café"
    q = {
        "type": "choice",
        "instructions": "Pick.",
        "criteria": {nfc: None, nfd: None},
    }
    body = encode_body({"model": "m", "state": "s", "questions": {nfd: q, "😀": q}})
    req = parse_request(body)

    assert [qid.encode("utf-8") for qid, _ in req.questions] == [
        nfd.encode("utf-8"), "😀".encode("utf-8"),
    ]
    assert req.questions[0][1].keys == (nfc, nfd)


def test_escaped_surrogate_pair_is_valid_text():
    body = b'{"model":"m","state":"\\ud83d\\ude00","questions":' \
           b'{"q":{"type":"noul","instructions":"i"}}}'
    assert parse_request(body).state == "😀"


@pytest.mark.parametrize("state", [{}, [], {"a": [1, 2.5, None, True]}, [{"x": "y"}]])
def test_container_states_are_valid_including_empty(state, encode_body):
    assert parse_request(encode_body(_one(_choice(2), state=state))).state == state


def test_noul_partial_criteria_fill_the_missing_side(encode_body):
    q = {"type": "noul", "instructions": "i", "criteria": {"true": "It is a refund"}}
    req = parse_request(encode_body(_one(q)))
    assert req.questions[0][1].criteria == (("true", "It is a refund"), ("false", "False"))


def test_noul_criteria_keep_true_false_order_whatever_the_wire_order(encode_body):
    q = {"type": "noul", "instructions": "i", "criteria": {"false": "no", "true": "yes"}}
    req = parse_request(encode_body(_one(q)))
    assert req.questions[0][1].criteria == (("true", "yes"), ("false", "no"))


# --- size limits: boundary passes, boundary + 1 fails ---------------------


def _pad_to(body: bytes, size: int) -> bytes:
    assert body.endswith(b"}")
    return body[:-1] + b" " * (size - len(body)) + b"}"


def test_body_limit(mixed_body):
    assert len(parse_request(_pad_to(mixed_body, MAX_BODY_BYTES)).questions) == 3
    err = _err(_pad_to(mixed_body, MAX_BODY_BYTES + 1))
    assert (err.code, err.status) == ("body_too_large", 413)


def test_string_state_limit_counts_utf8_bytes(encode_body):
    ok = "é" * (MAX_STATE_BYTES // 2)  # 2 bytes each, half as many chars
    assert parse_request(encode_body(_one(_choice(2), state=ok))).state == ok
    err = _err(encode_body(_one(_choice(2), state=ok + "a")))
    assert (err.code, err.status) == ("state_too_large", 413)


def test_container_state_limit_counts_compact_unescaped_serialization(encode_body):
    # Compact '{"k":"..."}' is 8 bytes of framing; non-ASCII counts as
    # its UTF-8 bytes, not as a \\u escape.
    fill = (MAX_STATE_BYTES - 8) // 2
    ok = {"k": "é" * fill}
    assert parse_request(encode_body(_one(_choice(2), state=ok))).state == ok
    err = _err(encode_body(_one(_choice(2), state={"k": "é" * fill + "a"})))
    assert (err.code, err.status) == ("state_too_large", 413)


def _nest(depth: int):
    value = "leaf"
    for i in range(depth):
        value = [value] if i % 2 else {"k": value}
    return value


def test_state_depth_limit(encode_body):
    assert parse_request(encode_body(_one(_choice(2), state=_nest(MAX_STATE_DEPTH))))
    err = _err(encode_body(_one(_choice(2), state=_nest(MAX_STATE_DEPTH + 1))))
    assert (err.code, err.status) == ("state_too_deep", 400)


def test_pathological_nesting_is_a_clean_400():
    body = (
        b'{"model":"m","state":' + b"[" * 50_000 + b"]" * 50_000
        + b',"questions":{"q":{"type":"noul","instructions":"i"}}}'
    )
    assert len(body) <= MAX_BODY_BYTES
    err = _err(body)
    assert err.status == 400
    assert err.code in {"nesting_too_deep", "state_too_deep"}


def test_question_count_limits(encode_body):
    for n in (1, 16):
        qs = {f"q{i}": _choice(2) for i in range(n)}
        assert len(parse_request(encode_body({"model": "m", "state": "s", "questions": qs})).questions) == n
    for n in (0, 17):
        qs = {f"q{i}": _choice(2) for i in range(n)}
        assert _err(encode_body({"model": "m", "state": "s", "questions": qs})).code == "invalid_questions"


def test_choice_option_count_limits(encode_body):
    for n in (2, 24):
        assert len(parse_request(encode_body(_one(_choice(n)))).questions[0][1].keys) == n
    for n in (1, 25):
        assert _err(encode_body(_one(_choice(n)))).code == "invalid_criteria"


def test_score_level_count_limits(encode_body):
    def score(n):
        return {"type": "score", "instructions": "i", "criteria": [f"L{i}" for i in range(n)]}

    for n in (2, 10):
        assert parse_request(encode_body(_one(score(n)))).questions[0][1].keys == tuple(
            str(i) for i in range(n)
        )
    for n in (1, 11):
        assert _err(encode_body(_one(score(n)))).code == "invalid_criteria"


@pytest.mark.parametrize("unit", ["a", "é", "😀"])
def test_question_id_limit_counts_utf8_bytes(unit, encode_body):
    width = len(unit.encode("utf-8"))
    ok = unit * (MAX_KEY_BYTES // width)
    assert parse_request(encode_body({"model": "m", "state": "s", "questions": {ok: _choice(2)}}))
    err = _err(encode_body({"model": "m", "state": "s", "questions": {ok + "a": _choice(2)}}))
    assert err.code == "invalid_question_id"


def test_option_key_limit_counts_utf8_bytes(encode_body):
    ok = "é" * (MAX_KEY_BYTES // 2)
    q = {"type": "choice", "instructions": "i", "criteria": {ok: None, "b": None}}
    assert parse_request(encode_body(_one(q)))
    q["criteria"] = {ok + "a": None, "b": None}
    assert _err(encode_body(_one(q))).code == "invalid_criteria"


def test_model_limit(encode_body):
    assert parse_request(encode_body(_one(_choice(2), model="m" * MAX_KEY_BYTES)))
    assert _err(encode_body(_one(_choice(2), model="m" * (MAX_KEY_BYTES + 1)))).code == "invalid_model"


def test_instructions_limit(encode_body):
    q = _choice(2)
    q["instructions"] = "é" * (MAX_INSTRUCTIONS_BYTES // 2)
    assert parse_request(encode_body(_one(q)))
    q["instructions"] += "a"
    assert _err(encode_body(_one(q))).code == "invalid_instructions"


@pytest.mark.parametrize("kind", ["choice", "score", "noul"])
def test_description_limit(kind, encode_body):
    ok = "é" * (MAX_DESCRIPTION_BYTES // 2)

    def question(desc):
        if kind == "choice":
            return {"type": "choice", "instructions": "i", "criteria": {"a": desc, "b": None}}
        if kind == "score":
            return {"type": "score", "instructions": "i", "criteria": [desc, "other"]}
        return {"type": "noul", "instructions": "i", "criteria": {"true": desc}}

    assert parse_request(encode_body(_one(question(ok))))
    assert _err(encode_body(_one(question(ok + "a")))).code == "invalid_criteria"


# --- duplicate keys at every object level ---------------------------------


_QUESTION = b'{"type":"noul","instructions":"i"}'


@pytest.mark.parametrize("body", [
    b'{"model":"a","model":"b","state":"s","questions":{"q":' + _QUESTION + b"}}",
    b'{"model":"m","state":"s","questions":{"q":' + _QUESTION + b',"q":' + _QUESTION + b"}}",
    b'{"model":"m","state":"s","questions":{"q":'
    b'{"type":"noul","type":"choice","instructions":"i"}}}',
    b'{"model":"m","state":"s","questions":{"q":{"type":"choice","instructions":"i",'
    b'"criteria":{"a":null,"b":null,"a":"again"}}}}',
    b'{"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"i",'
    b'"criteria":{"true":"x","true":"y"}}}}',
    b'{"model":"m","state":{"a":[{"b":{"c":1,"c":2}}]},"questions":{"q":' + _QUESTION + b"}}",
])
def test_duplicate_keys_rejected_at_every_level(body):
    err = _err(body)
    assert (err.code, err.status) == ("duplicate_key", 400)


# --- schema --------------------------------------------------------------


@pytest.mark.parametrize("payload, code", [
    ({"state": "s", "questions": {"q": _choice(2)}}, "missing_field"),
    ({"model": "m", "questions": {"q": _choice(2)}}, "missing_field"),
    ({"model": "m", "state": "s"}, "missing_field"),
    ({"model": "", "state": "s", "questions": {"q": _choice(2)}}, "invalid_model"),
    ({"model": True, "state": "s", "questions": {"q": _choice(2)}}, "invalid_model"),
    ({"model": None, "state": "s", "questions": {"q": _choice(2)}}, "invalid_model"),
    ({"model": "m", "state": "", "questions": {"q": _choice(2)}}, "invalid_state"),
    ({"model": "m", "state": True, "questions": {"q": _choice(2)}}, "invalid_state"),
    ({"model": "m", "state": 5, "questions": {"q": _choice(2)}}, "invalid_state"),
    ({"model": "m", "state": None, "questions": {"q": _choice(2)}}, "invalid_state"),
    ({"model": "m", "state": "s", "questions": [_choice(2)]}, "invalid_questions"),
    ({"model": "m", "state": "s", "questions": {"": _choice(2)}}, "invalid_question_id"),
    ({"model": "m", "state": "s", "questions": {"q": "choice"}}, "invalid_question"),
])
def test_top_level_schema(payload, code, encode_body):
    assert _err(encode_body(payload)).code == code


def test_top_level_must_be_an_object():
    assert _err(b"[]").code == "invalid_request"


@pytest.mark.parametrize("question, code", [
    ({"instructions": "i"}, "missing_field"),
    ({"type": "rank", "instructions": "i"}, "unsupported_question_type"),
    ({"type": "Choice", "instructions": "i", "criteria": {"a": None, "b": None}}, "unsupported_question_type"),
    ({"type": True, "instructions": "i"}, "unsupported_question_type"),
    ({"type": "noul"}, "missing_field"),
    ({"type": "noul", "instructions": ""}, "invalid_instructions"),
    ({"type": "noul", "instructions": 1}, "invalid_instructions"),
    ({"type": "choice", "instructions": "i"}, "missing_field"),
    ({"type": "choice", "instructions": "i", "criteria": ["a", "b"]}, "invalid_criteria"),
    ({"type": "choice", "instructions": "i", "criteria": {"a": True, "b": None}}, "invalid_criteria"),
    ({"type": "choice", "instructions": "i", "criteria": {"a": 1, "b": None}}, "invalid_criteria"),
    ({"type": "choice", "instructions": "i", "criteria": {"": None, "b": None}}, "invalid_criteria"),
    ({"type": "score", "instructions": "i"}, "missing_field"),
    ({"type": "score", "instructions": "i", "criteria": {"0": "a", "1": "b"}}, "invalid_criteria"),
    ({"type": "score", "instructions": "i", "criteria": ["same", "same"]}, "invalid_criteria"),
    ({"type": "score", "instructions": "i", "criteria": ["a", ""]}, "invalid_criteria"),
    ({"type": "score", "instructions": "i", "criteria": ["a", None]}, "invalid_criteria"),
    ({"type": "score", "instructions": "i", "criteria": ["a", 2]}, "invalid_criteria"),
    ({"type": "noul", "instructions": "i", "criteria": None}, "invalid_criteria"),
    ({"type": "noul", "instructions": "i", "criteria": ["yes", "no"]}, "invalid_criteria"),
    ({"type": "noul", "instructions": "i", "criteria": {"maybe": "x"}}, "invalid_criteria"),
    ({"type": "noul", "instructions": "i", "criteria": {"true": None}}, "invalid_criteria"),
    ({"type": "noul", "instructions": "i", "criteria": {"true": True}}, "invalid_criteria"),
    ({"type": "noul", "instructions": "i", "criteria": {"true": ""}}, "invalid_criteria"),
])
def test_question_schema(question, code, encode_body):
    assert _err(encode_body(_one(question))).code == code


@pytest.mark.parametrize("payload, code", [
    ({"model": "m", "state": "s", "questions": {"q": _choice(2)}, "temperature": 0}, "unsupported_field"),
    ({"model": "m", "state": "s", "questions": {"q": _choice(2)}, "stream": True}, "unsupported_field"),
    ({"model": "m", "state": "s", "questions": {"q": _choice(2)}, "images": []}, "unsupported_media"),
    ({"model": "m", "state": "s", "questions": {"q": {**_choice(2), "weight": 2}}}, "unsupported_field"),
    ({"model": "m", "state": "s", "questions": {"q": {**_choice(2), "image": "data:"}}}, "unsupported_media"),
])
def test_unknown_fields_and_media_rejected(payload, code, encode_body):
    assert _err(encode_body(payload)).code == code


# --- numbers, encoding, Unicode ------------------------------------------


@pytest.mark.parametrize("literal", [b"NaN", b"Infinity", b"-Infinity", b"1e400", b"-1e400"])
def test_non_finite_numbers_rejected(literal):
    body = b'{"model":"m","state":{"x":[' + literal + b']},"questions":{"q":' + _QUESTION + b"}}"
    err = _err(body)
    assert (err.code, err.status) == ("non_finite_number", 400)


def test_huge_integer_is_a_clean_400():
    body = b'{"model":"m","state":{"x":' + b"9" * 5000 + b'},"questions":{"q":' + _QUESTION + b"}}"
    assert _err(body).status == 400


@pytest.mark.parametrize("raw", [b"\xff", b"\xc0\xaf", b"\xed\xa0\x80", b"\xe2\x82"])
def test_malformed_utf8_rejected(raw):
    body = b'{"model":"m","state":"' + raw + b'","questions":{"q":' + _QUESTION + b"}}"
    assert _err(body).code == "invalid_encoding"


def test_byte_order_mark_rejected():
    assert _err(b"\xef\xbb\xbf" + b'{"model":"m","state":"s","questions":{"q":' + _QUESTION + b"}}").status == 400


@pytest.mark.parametrize("body", [
    b'{"model":"m","state":"\\ud800","questions":{"q":' + _QUESTION + b"}}",
    b'{"model":"m","state":{"\\udc00":1},"questions":{"q":' + _QUESTION + b"}}",
    b'{"model":"m","state":"s","questions":{"\\ud800":' + _QUESTION + b"}}",
    b'{"model":"m","state":"s","questions":{"q":{"type":"choice","instructions":"i",'
    b'"criteria":{"\\ud83d":null,"b":null}}}}',
    b'{"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":"\\ude00"}}}',
])
def test_unpaired_surrogates_rejected(body):
    assert _err(body).code == "invalid_unicode"


@pytest.mark.parametrize("body", [b"", b"{", b'{"model":"m",}', b"not json"])
def test_invalid_json_rejected(body):
    assert _err(body).code == "invalid_json"


# --- errors never echo user content ---------------------------------------


@pytest.mark.parametrize("payload", [
    {"model": "m", "state": "s", "questions": {"SECRET" * 30: _choice(2)}},
    {"model": "m", "state": "s", "questions": {"q": _choice(2)}, "SECRETFIELD": 1},
    {"model": "m", "state": "s", "questions": {"q": {**_choice(2), "SECRETFIELD": 1}}},
    {"model": "m", "state": "s", "questions": {"q": {"type": "SECRET", "instructions": "i"}}},
    {"model": "m", "state": "s", "questions": {"q": {"type": "noul", "instructions": "SECRET" * 1000}}},
    {"model": "m", "state": "s", "questions": {"q": {"type": "score", "instructions": "i",
                                                     "criteria": ["SECRET", "SECRET"]}}},
    {"model": "SECRET" * 30, "state": "s", "questions": {"q": _choice(2)}},
])
def test_error_messages_do_not_echo_user_content(payload, encode_body):
    err = _err(encode_body(payload))
    assert "SECRET" not in err.message
    assert "SECRET" not in str(err)
