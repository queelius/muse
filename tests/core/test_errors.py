"""Tests for OpenAI-style error envelopes.

Regression guard for L10: error_response derives the OpenAI `type` from
the HTTP status (server_error for 5xx) instead of hardcoding
invalid_request_error for every status.
"""
from __future__ import annotations

import json

from muse.core.errors import error_response


def _body(resp) -> dict:
    return json.loads(bytes(resp.body))


class TestErrorResponseType:
    def test_4xx_is_invalid_request_error(self):
        resp = error_response(400, "bad_input", "nope")
        assert resp.status_code == 400
        assert _body(resp)["error"]["type"] == "invalid_request_error"

    def test_404_is_invalid_request_error(self):
        resp = error_response(404, "model_not_found", "ghost")
        assert _body(resp)["error"]["type"] == "invalid_request_error"

    def test_500_is_server_error(self):
        resp = error_response(500, "internal_error", "boom")
        assert resp.status_code == 500
        assert _body(resp)["error"]["type"] == "server_error"

    def test_503_is_server_error(self):
        resp = error_response(503, "model_unservable", "no room")
        assert _body(resp)["error"]["type"] == "server_error"

    def test_code_and_message_preserved(self):
        resp = error_response(500, "internal_error", "the message")
        err = _body(resp)["error"]
        assert err["code"] == "internal_error"
        assert err["message"] == "the message"
