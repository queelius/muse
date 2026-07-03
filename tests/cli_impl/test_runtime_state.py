"""Tests for the shared public-runtime-state reader.

`muse models list` and `muse models info` both read which models the
running server has loaded from the PUBLIC GET /v1/models endpoint (each
entry carries `loaded: bool` since v0.47.3). This module centralizes that
fetch + parse so both call sites share one implementation.
"""
from __future__ import annotations


class _FakeResp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _fake_httpx_client(*, captured_urls, resp=None, raise_exc=None):
    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            captured_urls.append(url)
            if raise_exc is not None:
                raise raise_exc
            return resp

    return _FakeClient


def test_base_url_from_config(monkeypatch):
    """_base_url() resolves MUSE_SERVER through muse.core.config, rstrip applied."""
    from muse.core import config as cfg

    monkeypatch.setenv("MUSE_SERVER", "http://box:9000/")
    cfg.reset_config()
    try:
        from muse.cli_impl import runtime_state as rs

        assert rs._base_url() == "http://box:9000"
    finally:
        cfg.reset_config()


def test_fetch_public_models_returns_data_list(monkeypatch):
    from muse.cli_impl import runtime_state as rs

    monkeypatch.setenv("MUSE_SERVER", "http://srv:8000")
    urls: list[str] = []
    payload = {"data": [{"id": "a", "loaded": True}, {"id": "b", "loaded": False}]}
    monkeypatch.setattr(
        "httpx.Client",
        _fake_httpx_client(captured_urls=urls, resp=_FakeResp(200, payload)),
    )

    data = rs.fetch_public_models()
    assert data == payload["data"]
    assert urls == ["http://srv:8000/v1/models"]


def test_fetch_public_models_default_base_url(monkeypatch):
    from muse.cli_impl import runtime_state as rs

    monkeypatch.delenv("MUSE_SERVER", raising=False)
    urls: list[str] = []
    monkeypatch.setattr(
        "httpx.Client",
        _fake_httpx_client(captured_urls=urls, resp=_FakeResp(200, {"data": []})),
    )
    assert rs.fetch_public_models() == []
    assert urls == ["http://localhost:8000/v1/models"]


def test_fetch_public_models_unreachable_returns_none(monkeypatch):
    import httpx

    from muse.cli_impl import runtime_state as rs

    monkeypatch.setattr(
        "httpx.Client",
        _fake_httpx_client(
            captured_urls=[], raise_exc=httpx.ConnectError("x", request=None),
        ),
    )
    assert rs.fetch_public_models() is None


def test_fetch_public_models_non_200_returns_none(monkeypatch):
    from muse.cli_impl import runtime_state as rs

    monkeypatch.setattr(
        "httpx.Client",
        _fake_httpx_client(captured_urls=[], resp=_FakeResp(503, {"data": []})),
    )
    assert rs.fetch_public_models() is None


def test_fetch_public_models_malformed_body_returns_none(monkeypatch):
    from muse.cli_impl import runtime_state as rs

    monkeypatch.setattr(
        "httpx.Client",
        _fake_httpx_client(captured_urls=[], resp=_FakeResp(200, ["not", "a", "dict"])),
    )
    assert rs.fetch_public_models() is None


def test_loaded_ids_filters_to_loaded_true():
    from muse.cli_impl import runtime_state as rs

    data = [
        {"id": "x", "loaded": True},
        {"id": "y", "loaded": False},
        {"loaded": True},          # no id
        {"id": "", "loaded": True},  # empty id
        "junk",
        {"id": "z"},               # no loaded key
    ]
    assert rs.loaded_ids(data) == {"x"}


def test_fetch_public_models_quiets_httpx_log(monkeypatch):
    """The per-request httpx INFO line ('HTTP Request: GET ...') must not
    leak into `muse models list` / `info` output: fetch raises the httpx
    logger to >= WARNING (only raising, never lowering)."""
    import logging

    from muse.cli_impl import runtime_state as rs

    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.NOTSET)
    monkeypatch.setattr(
        "httpx.Client",
        _fake_httpx_client(captured_urls=[], resp=_FakeResp(200, {"data": []})),
    )
    rs.fetch_public_models()
    assert httpx_logger.level >= logging.WARNING
