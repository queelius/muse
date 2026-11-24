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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from muse.cli_impl.gateway import build_gateway
from muse.core import config
from tests.cli_impl.test_gateway_lazy import (
    _make_state_with_director,
    _patch_get_manifest,
    _wire_async_client_json,
)


@pytest.fixture(autouse=True)
def _reset_gateway_config():
    config.reset_config()
    yield
    config.reset_config()


def _capture_records(monkeypatch) -> list[tuple[str, dict]]:
    captured: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "muse.cli_impl.gateway.record",
        lambda event_type, **fields: captured.append((event_type, fields)),
    )
    return captured


class TestRequestTelemetry:
    def test_forwarded_request_records_one_request_event(self, monkeypatch):
        captured = _capture_records(monkeypatch)
        state = _make_state_with_director(acquire_port=9001)
        client = TestClient(build_gateway(state=state))

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
        assert isinstance(fields["queued_ms"], float)
        assert isinstance(fields["request_id"], str) and fields["request_id"]
        assert fields["cold"] is False
        assert fields["latency_ms"] >= fields["forward_ms"]
        assert fields["load_ms"] == 0.0
        assert fields["evicted_models"] == "[]"

    def test_streamed_request_records_stream_true(self, monkeypatch):
        captured: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            "muse.cli_impl.gateway.record",
            lambda event_type, **fields: captured.append((event_type, fields)),
        )
        state = _make_state_with_director(acquire_port=9001)
        client = TestClient(build_gateway(state=state))

        with _patch_get_manifest(), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            mock_client = _wire_async_client_json(
                mock_cls, response_content_type="text/event-stream",
            )
            response = mock_client.stream.return_value.__aenter__.return_value

            async def aiter_raw():
                yield b"data: hi\n\n"

            response.aiter_raw = aiter_raw
            r = client.post(
                "/v1/chat/completions",
                json={"model": "fake-model", "messages": [], "stream": True},
            )

        assert r.status_code == 200
        assert r.content == b"data: hi\n\n"
        [(kind, fields)] = captured
        assert kind == "request"
        assert fields["stream"] is True
        assert fields["status"] == 200
        assert {"model_id", "modality", "latency_ms", "queued_ms"} <= fields.keys()
        state.director.release.assert_called_once_with("fake-model")

    def test_record_failure_never_breaks_the_forward(self, monkeypatch):
        """record() raising must not propagate: telemetry is fire-and-forget."""
        def _boom(event_type: str, **fields) -> None:
            raise RuntimeError("telemetry backend exploded")

        monkeypatch.setattr("muse.cli_impl.gateway.record", _boom)

        state = _make_state_with_director(acquire_port=9001)
        client = TestClient(build_gateway(state=state))

        with _patch_get_manifest(), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_async_client_json(mock_cls, response_status=200)
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert r.status_code == 200

    def test_cold_annotation_crosses_asyncio_to_thread_context(self, monkeypatch):
        from muse.observability.traces import note_model_eviction, note_model_load

        captured = _capture_records(monkeypatch)
        state = _make_state_with_director(acquire_port=9001)

        def acquire(model_id, *, manifest):
            # gateway invokes this through asyncio.to_thread; ContextVar
            # propagation is what links these notes to the outer request.
            note_model_eviction("old-model")
            note_model_load(model_id, 0.5)
            return 9001

        state.director.acquire.side_effect = acquire
        client = TestClient(build_gateway(state=state))
        with _patch_get_manifest(), patch(
            "muse.cli_impl.gateway.httpx.AsyncClient"
        ) as mock_cls:
            _wire_async_client_json(mock_cls, response_status=200)
            response = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "fake-model"},
            )

        assert response.status_code == 200
        fields = [fields for event, fields in captured if event == "request"][0]
        assert fields["cold"] is True
        assert fields["load_ms"] == 500.0
        assert fields["evicted_models"] == '["old-model"]'

    def test_stream_trace_finishes_after_the_response_body(self, monkeypatch):
        captured = _capture_records(monkeypatch)
        chunks_seen = []
        finish_at = []
        state = _make_state_with_director(acquire_port=9001)

        class Tracker:
            def begin(self, request_id):
                self.request_id = request_id

            def finish(self, request_id):
                assert request_id == self.request_id
                finish_at.append(len(chunks_seen))
                return 5.5

        state.telemetry_vram_tracker = Tracker()
        client = TestClient(build_gateway(state=state))
        chunks = [b"data: one\n\n", b"event: done\ndata: \n\n"]
        with _patch_get_manifest(), patch(
            "muse.cli_impl.gateway.httpx.AsyncClient"
        ) as mock_cls:
            mock_client = MagicMock()
            mock_client.aclose = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/event-stream"}

            async def aiter_raw():
                for chunk in chunks:
                    chunks_seen.append(chunk)
                    yield chunk

            mock_response.aiter_raw = aiter_raw
            stream_ctx = MagicMock()
            stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            stream_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream.return_value = stream_ctx
            mock_cls.return_value = mock_client

            response = client.post(
                "/v1/chat/completions",
                json={"model": "fake-model", "messages": [], "stream": True},
            )

        assert response.status_code == 200
        assert finish_at == [len(chunks)]
        fields = [fields for event, fields in captured if event == "request"][0]
        assert fields["stream"] is True
        assert fields["peak_vram_gb"] == 5.5

    def test_disabled_telemetry_skips_trace_creation(self, monkeypatch):
        monkeypatch.setenv("MUSE_TELEMETRY_ENABLED", "false")
        config.reset_config()
        monkeypatch.setattr(
            "muse.cli_impl.gateway.begin_request_trace",
            lambda *_args: (_ for _ in ()).throw(
                AssertionError("trace must not be created when telemetry is disabled")
            ),
        )
        state = _make_state_with_director(acquire_port=9001)
        client = TestClient(build_gateway(state=state))
        with _patch_get_manifest(), patch(
            "muse.cli_impl.gateway.httpx.AsyncClient"
        ) as mock_cls:
            _wire_async_client_json(mock_cls, response_status=200)
            response = client.post(
                "/v1/chat/completions",
                json={"model": "fake-model", "messages": []},
            )
        assert response.status_code == 200

    def test_worker_timeout_records_forward_ms(self, monkeypatch):
        import time
        import httpx

        captured = _capture_records(monkeypatch)
        state = _make_state_with_director(acquire_port=9001)
        client = TestClient(build_gateway(state=state))

        with _patch_get_manifest(), patch(
            "muse.cli_impl.gateway.httpx.AsyncClient"
        ) as mock_cls:
            mock_client = MagicMock()
            mock_client.aclose = AsyncMock()
            stream_ctx = MagicMock()

            async def _slow_timeout():
                time.sleep(0.02)
                raise httpx.ReadTimeout("worker timed out")

            stream_ctx.__aenter__ = AsyncMock(side_effect=_slow_timeout)
            mock_client.stream.return_value = stream_ctx
            mock_cls.return_value = mock_client

            response = client.post(
                "/v1/chat/completions",
                json={"model": "fake-model", "messages": []},
            )

        assert response.status_code == 504
        fields = [fields for event, fields in captured if event == "request"][0]
        assert fields["status"] == 504
        assert fields["forward_ms"] >= 15.0

    def test_queue_timeout_records_queued_ms(self, monkeypatch):
        import asyncio
        from muse.cli_impl.queueing import QueueTimeout

        captured = _capture_records(monkeypatch)
        state = _make_state_with_director(acquire_port=9001)

        async def _slow_queue_timeout(_model_id, _cap, *, deadline, **_kwargs):
            await asyncio.sleep(0.02)
            raise QueueTimeout("fake-model")

        state.concurrency_gate.acquire_slot = _slow_queue_timeout
        client = TestClient(build_gateway(state=state))

        with _patch_get_manifest():
            response = client.post(
                "/v1/chat/completions",
                json={"model": "fake-model", "messages": []},
            )

        assert response.status_code == 503
        fields = [fields for event, fields in captured if event == "request"][0]
        assert fields["status"] == 503
        assert fields["queued_ms"] >= 15.0
        assert fields["forward_ms"] == 0.0

    def test_coalesced_cold_load_waiter_records_queued_ms(self, monkeypatch):
        import asyncio
        from muse.cli_impl.gateway import _acquire_coalesced
        from muse.observability.traces import begin_request_trace, reset_request_trace

        state = _make_state_with_director(acquire_port=9001)

        async def _run():
            loop = asyncio.get_running_loop()
            gate = loop.create_future()
            state.cold_load_gates["fake-model"] = gate

            async def _release_gate():
                await asyncio.sleep(0.02)
                gate.set_result(None)

            trace, token = begin_request_trace("fake-model", "chat/completions")
            try:
                releaser = asyncio.create_task(_release_gate())
                port = await _acquire_coalesced(
                    state, "fake-model", {"model_id": "fake-model"},
                )
                await releaser
                return port, trace.snapshot()
            finally:
                reset_request_trace(token)

        port, snap = asyncio.run(_run())
        assert port == 9001
        assert snap["queued_ms"] >= 15.0


class TestDashboardMount:
    def test_dashboard_mounted_when_telemetry_enabled_and_state_present(self):
        state = _make_state_with_director(acquire_port=9001)
        client = TestClient(build_gateway(state=state))

        r = client.get("/dashboard")
        assert r.status_code == 200

    def test_dashboard_absent_when_telemetry_disabled(self, monkeypatch):
        # With no dashboard router mounted, GET /dashboard falls through to
        # the catch-all proxy, which 400s "model_required" (no `model`
        # field) rather than FastAPI's native 404 -- that 400 IS the
        # "not mounted" signal here.
        monkeypatch.setenv("MUSE_TELEMETRY_ENABLED", "false")
        config.reset_config()
        state = _make_state_with_director(acquire_port=9001)
        client = TestClient(build_gateway(state=state), raise_server_exceptions=False)

        r = client.get("/dashboard")
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "model_required"
