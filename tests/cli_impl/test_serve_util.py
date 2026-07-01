"""Tests for muse.cli_impl.serve_util.

The supervisor gateway (port 8000) and each per-model worker both run a
FastAPI app under uvicorn. uvicorn's default `timeout_graceful_shutdown`
is None, meaning the first Ctrl-C waits FOREVER for any in-flight
connection (an SSE chat stream, a long inference, or an idle keep-alive
socket) to drain before the process exits. Until then the process stays
alive holding the listen port, so `muse serve` cannot be restarted
without manually killing the stuck process.

serve_util centralizes uvicorn construction with a BOUNDED graceful
timeout so the first Ctrl-C always exits within a fixed window.
"""
import os
from unittest.mock import patch

import pytest

from muse.cli_impl import serve_util


def test_build_uvicorn_server_sets_bounded_graceful_timeout():
    # The whole point: timeout_graceful_shutdown must be a finite number,
    # never None (the uvicorn default that hangs forever).
    server = serve_util.build_uvicorn_server(
        object(), host="127.0.0.1", port=8000,
    )
    assert server.config.timeout_graceful_shutdown is not None
    assert server.config.timeout_graceful_shutdown == pytest.approx(10.0)
    # host/port must round-trip so the gateway still binds where asked.
    assert server.config.host == "127.0.0.1"
    assert server.config.port == 8000


def test_shutdown_grace_seconds_default():
    with patch.dict(os.environ, {}, clear=True):
        assert serve_util.shutdown_grace_seconds() == pytest.approx(10.0)


def test_shutdown_grace_seconds_env_override():
    with patch.dict(os.environ, {"MUSE_SHUTDOWN_GRACE_SECONDS": "2.5"}):
        assert serve_util.shutdown_grace_seconds() == pytest.approx(2.5)


def test_shutdown_grace_seconds_invalid_falls_back():
    with patch.dict(os.environ, {"MUSE_SHUTDOWN_GRACE_SECONDS": "not-a-number"}):
        assert serve_util.shutdown_grace_seconds() == pytest.approx(10.0)


def test_shutdown_grace_seconds_negative_falls_back():
    # A negative grace is nonsensical; treat it as the default rather than
    # letting it reach uvicorn (which would treat <0 unpredictably).
    with patch.dict(os.environ, {"MUSE_SHUTDOWN_GRACE_SECONDS": "-4"}):
        assert serve_util.shutdown_grace_seconds() == pytest.approx(10.0)
