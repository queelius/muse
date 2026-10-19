"""DecisionClient and MuseClient.decide: same body, same envelope."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
import requests

from muse.mcp.client import MuseClient
from muse.modalities.decision_evaluation.client import DecisionClient, encode_request
from muse.modalities.decision_evaluation.protocol import DecisionInputError


ENVELOPE = {
    "model": "decima-base",
    "answers": {"é-id": {"type": "noul", "noul": 0.25}},
    "usage": {"input_tokens": 9, "output_tokens": 0},
}
QUESTIONS = {"é-id": {"type": "noul", "instructions": "Is it urgent?"}}


def _requests_response(status=200, payload=ENVELOPE):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    if status >= 400:
        r.raise_for_status.side_effect = requests.HTTPError(f"{status} error")
    else:
        r.raise_for_status.return_value = None
    return r


def test_decision_client_posts_the_validated_bytes_and_returns_the_envelope():
    with patch("muse.modalities.decision_evaluation.client.requests.post",
               return_value=_requests_response()) as post:
        out = DecisionClient("http://muse/").decide(
            model="decima-base", state={"msg": "café"}, questions=QUESTIONS,
        )
    assert out == ENVELOPE
    (url,), kwargs = post.call_args
    assert url == "http://muse/v1/systemone"
    sent = kwargs["data"]
    assert sent == encode_request(model="decima-base", state={"msg": "café"}, questions=QUESTIONS)
    assert "é-id".encode("utf-8") in sent  # Unicode ids are sent unescaped and unchanged
    assert json.loads(sent)["model"] == "decima-base"


@pytest.mark.parametrize("kwargs, code", [
    ({"model": "", "state": "s", "questions": QUESTIONS}, "invalid_model"),
    ({"model": "m", "state": "", "questions": QUESTIONS}, "invalid_state"),
    ({"model": "m", "state": "s", "questions": {"q": {"type": "rank", "instructions": "i"}}},
     "unsupported_question_type"),
    ({"model": "m", "state": {"x": float("nan")}, "questions": QUESTIONS}, "non_finite_number"),
    ({"model": "m", "state": "s", "questions": {"q": {**QUESTIONS["é-id"], "image": "x"}}},
     "unsupported_media"),
])
def test_invalid_requests_fail_locally_before_http(kwargs, code):
    with patch("muse.modalities.decision_evaluation.client.requests.post") as post:
        with pytest.raises(DecisionInputError) as exc:
            DecisionClient("http://muse").decide(**kwargs)
    assert exc.value.code == code
    post.assert_not_called()


def test_model_is_keyword_only_and_required():
    with pytest.raises(TypeError):
        DecisionClient("http://muse").decide(state="s", questions=QUESTIONS)
    with pytest.raises(TypeError):
        DecisionClient("http://muse").decide("m", "s", QUESTIONS)


@pytest.mark.parametrize("status", [400, 403, 404, 413, 503])
def test_http_errors_raise_rather_than_return_answers(status):
    with patch("muse.modalities.decision_evaluation.client.requests.post",
               return_value=_requests_response(status, {"error": {"code": "x"}})):
        with pytest.raises(requests.HTTPError):
            DecisionClient("http://muse").decide(model="m", state="s", questions=QUESTIONS)


def _mock_httpx(monkeypatch, respond):
    original = httpx.Client
    monkeypatch.setattr(
        "muse.mcp.client.httpx.Client",
        lambda **kw: original(transport=httpx.MockTransport(respond), **kw),
    )


def test_muse_client_sends_the_identical_body(monkeypatch):
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(200, json=ENVELOPE)

    _mock_httpx(monkeypatch, respond)
    out = MuseClient(server_url="http://gateway").decide(
        model="decima-base", state={"msg": "café"}, questions=QUESTIONS,
    )
    assert out == ENVELOPE
    (request,) = seen
    assert request.url.path == "/v1/systemone"
    assert request.content == encode_request(
        model="decima-base", state={"msg": "café"}, questions=QUESTIONS,
    )


def test_muse_client_validates_before_http(monkeypatch):
    seen = []
    _mock_httpx(monkeypatch, lambda r: seen.append(r) or httpx.Response(200, json={}))
    with pytest.raises(DecisionInputError):
        MuseClient(server_url="http://gateway").decide(model="", state="s", questions=QUESTIONS)
    assert seen == []
