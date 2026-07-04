"""Tests for Task 10: gateway request-telemetry recording + dashboard mount.

Two things land here:

  1. Every forwarded request (director-driven path) records one
     `request` telemetry event via `muse.cli_impl.gateway.record`, timed
     around the forward, with the modality derived structurally from
     the request path (no hardcoded per-route lookup table).
  2. `build_gateway` mounts the dashboard router
     (`GET /dashboard` -> 200) whenever `telemetry.enabled` is true AND
     a SupervisorState is passed; the legacy static-routes mode
     (state=None) has no supervisor to serve telemetry from, so it is
     not mounted there.

Reuses the existing director-path test harness from test_gateway_lazy.py
(`_make_state_with_director`, `_patch_get_manifest`, `_wire_async_client_json`)
rather than inventing a new one.
"""
from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from muse.cli_impl.gateway import build_gateway
from muse.core import config
from tests.cli_impl.test_gateway_lazy import (
    _make_state_with_director,
    _patch_get_manifest,
    _wire_async_client_json,
)


class TestRequestTelemetry:
    def test_forwarded_request_records_one_request_event(self, monkeypatch):
        captured: list[tuple[str, dict]] = []

        def _fake_record(event_type: str, **fields) -> None:
            captured.append((event_type, fields))

        monkeypatch.setattr("muse.cli_impl.gateway.record", _fake_record)

        state = _make_state_with_director(acquire_port=9001)
        app = build_gateway(state=state)
        client = TestClient(app)

        with _patch_get_manifest(), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_async_client_json(mock_cls, response_status=200)
            r = client.post(
                "/v1/chat/completions",
                json={"model": "fake-model", "messages": []},
            )

        assert r.status_code == 200
        request_events = [c for c in captured if c[0] == "request"]
        assert len(request_events) == 1
        _, fields = request_events[0]
        assert fields["model_id"] == "fake-model"
        assert isinstance(fields["latency_ms"], (int, float))
        assert fields["latency_ms"] >= 0
        assert fields["status"] == r.status_code
        assert fields["stream"] is False
        assert fields["modality"] == "chat/completions"

    def test_record_failure_never_breaks_the_forward(self, monkeypatch):
        """record() raising must not propagate: telemetry is fire-and-forget."""
        def _boom(event_type: str, **fields) -> None:
            raise RuntimeError("telemetry backend exploded")

        monkeypatch.setattr("muse.cli_impl.gateway.record", _boom)

        state = _make_state_with_director(acquire_port=9001)
        app = build_gateway(state=state)
        client = TestClient(app)

        with _patch_get_manifest(), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_async_client_json(mock_cls, response_status=200)
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 200


class TestDashboardMount:
    def test_dashboard_mounted_when_telemetry_enabled_and_state_present(self):
        config.reset_config()
        state = _make_state_with_director(acquire_port=9001)
        app = build_gateway(state=state)
        client = TestClient(app)

        r = client.get("/dashboard")
        assert r.status_code == 200

    def test_dashboard_absent_when_telemetry_disabled(self, monkeypatch):
        # With no dashboard router mounted, GET /dashboard falls through to
        # the catch-all proxy, which 400s "model_required" (no `model`
        # field) rather than FastAPI's native 404 -- that 400 IS the
        # "not mounted" signal here.
        monkeypatch.setenv("MUSE_TELEMETRY_ENABLED", "false")
        config.reset_config()
        try:
            state = _make_state_with_director(acquire_port=9001)
            app = build_gateway(state=state)
            client = TestClient(app, raise_server_exceptions=False)

            r = client.get("/dashboard")
            assert r.status_code == 400
            assert r.json()["error"]["code"] == "model_required"
        finally:
            config.reset_config()

    def test_dashboard_absent_in_legacy_static_routes_mode(self):
        """state=None (legacy static-routes mode) has no supervisor to
        serve telemetry from, so the dashboard router is not mounted
        even though telemetry.enabled defaults to true. Falls through to
        the catch-all proxy's 400 model_required, same signal as above."""
        config.reset_config()
        app = build_gateway([])
        client = TestClient(app, raise_server_exceptions=False)

        r = client.get("/dashboard")
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "model_required"
