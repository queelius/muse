"""Tests for OpenAI-style error envelopes.

Regression guard for L10: error_response derives the OpenAI `type` from
the HTTP status (server_error for 5xx) instead of hardcoding
invalid_request_error for every status.
"""
from __future__ import annotations

import json

import pytest

from muse.core.errors import error_response, error_type_for_status


def _body(resp) -> dict:
    return json.loads(bytes(resp.body))


class TestErrorTypeForStatus:
    """The shared status->type mapping reused by error_response, the
    gateway's OperationError surface, and the admin auth dependency."""

    @pytest.mark.parametrize("status", [500, 502, 503, 599])
    def test_5xx_is_server_error(self, status):
        assert error_type_for_status(status) == "server_error"

    @pytest.mark.parametrize("status", [200, 400, 401, 403, 404, 429, 499])
    def test_sub_500_is_invalid_request_error(self, status):
        assert error_type_for_status(status) == "invalid_request_error"


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
