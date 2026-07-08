"""Gateway queueing integration (spec 2026-07-08).

Uses the same fake-director + TestClient style as test_gateway_lazy.py:
a real FastAPI gateway app over a FakeDirector and a stub worker, no
subprocesses. Focus: cap resolution from manifest/config, queue_timeout
and queue_full envelopes, capacity-wait retry on retryable 503s, and
release pairing on success/failure.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

from muse.admin.operations import OperationError
from muse.cli_impl.queueing import ConcurrencyGate, CapacityNotifier


def _effective_cap_for(manifest: dict) -> int | None:
    from muse.cli_impl.gateway import _effective_max_concurrency
    return _effective_max_concurrency(manifest)


class TestEffectiveCap:
    def test_manifest_cap_wins(self, monkeypatch):
        monkeypatch.setenv("MUSE_DEFAULT_MAX_CONCURRENCY", "4")
        assert _effective_cap_for(
            {"capabilities": {"max_concurrency": 1}}) == 1

    def test_config_default_when_undeclared(self, monkeypatch):
        monkeypatch.setenv("MUSE_DEFAULT_MAX_CONCURRENCY", "4")
        assert _effective_cap_for({"capabilities": {}}) == 4

    def test_unlimited_when_neither(self, monkeypatch):
        monkeypatch.delenv("MUSE_DEFAULT_MAX_CONCURRENCY", raising=False)
        assert _effective_cap_for({"capabilities": {}}) is None

    def test_bad_manifest_value_falls_back(self, monkeypatch):
        monkeypatch.delenv("MUSE_DEFAULT_MAX_CONCURRENCY", raising=False)
        assert _effective_cap_for(
            {"capabilities": {"max_concurrency": "not-an-int"}}) is None


class TestCapacityWaitRetry:
    """_acquire_with_capacity_wait: park on retryable 503, retry after
    notify, propagate non-retryable immediately, 503 on deadline.

    Returns `(worker_port, waited_seconds)`: waited_seconds is the total
    time spent parked in the notifier's event across all retries, used by
    the gateway to keep queued_ms precise (excluding the time spent
    inside a successful acquire_once(), e.g. a cold load)."""

    async def test_retryable_then_success_after_notify(self):
        from muse.cli_impl.gateway import _acquire_with_capacity_wait
        notifier = CapacityNotifier()
        calls = []

        async def fake_acquire():
            calls.append(1)
            if len(calls) == 1:
                raise OperationError("model_too_large_for_device",
                                     "full", status=503, retryable=True)
            return 9001

        async def free_capacity_soon():
            await asyncio.sleep(0.05)
            notifier.notify()

        asyncio.create_task(free_capacity_soon())
        port, waited_seconds = await _acquire_with_capacity_wait(
            fake_acquire, notifier, deadline=time.monotonic() + 5,
            model_id="m",
        )
        assert port == 9001 and len(calls) == 2
        assert waited_seconds > 0.0

    async def test_immediate_success_reports_zero_wait(self):
        """No retry needed -> waited_seconds is exactly 0.0 (the gateway
        relies on this so a hot request's queued_ms is 0)."""
        from muse.cli_impl.gateway import _acquire_with_capacity_wait
        notifier = CapacityNotifier()

        async def fake_acquire():
            return 9001

        port, waited_seconds = await _acquire_with_capacity_wait(
            fake_acquire, notifier, deadline=time.monotonic() + 5,
            model_id="m",
        )
        assert port == 9001
        assert waited_seconds == 0.0

    async def test_non_retryable_propagates_immediately(self):
        from muse.cli_impl.gateway import _acquire_with_capacity_wait
        notifier = CapacityNotifier()

        async def fake_acquire():
            raise OperationError("model_too_large_for_device",
                                 "impossible", status=503, retryable=False)

        with pytest.raises(OperationError) as exc:
            await _acquire_with_capacity_wait(
                fake_acquire, notifier, deadline=time.monotonic() + 5,
                model_id="m",
            )
        assert exc.value.retryable is False

    async def test_deadline_exhaustion_raises_queue_timeout(self):
        from muse.cli_impl.gateway import _acquire_with_capacity_wait
        from muse.cli_impl.queueing import QueueTimeout
        notifier = CapacityNotifier()

        async def fake_acquire():
            raise OperationError("model_too_large_for_device",
                                 "full", status=503, retryable=True)

        with pytest.raises(QueueTimeout):
            await _acquire_with_capacity_wait(
                fake_acquire, notifier, deadline=time.monotonic() + 0.1,
                model_id="m",
            )

    async def test_zero_timeout_degrades_to_immediate_503(self):
        """queue_timeout_seconds=0 -> deadline already passed -> the
        retryable 503 surfaces as-is (today's behavior)."""
        from muse.cli_impl.gateway import _acquire_with_capacity_wait
        notifier = CapacityNotifier()

        async def fake_acquire():
            raise OperationError("model_too_large_for_device",
                                 "full", status=503, retryable=True)

        with pytest.raises(OperationError):
            await _acquire_with_capacity_wait(
                fake_acquire, notifier, deadline=time.monotonic(),  # now
                model_id="m",
            )


class TestQueuedMsColumn:
    def test_event_to_row_accepts_queued_ms(self):
        from muse.observability.events import event_to_row
        row = event_to_row("request", 1.0, queued_ms=42.0)
        assert row["queued_ms"] == 42.0

    def test_store_migrates_missing_column(self, tmp_path):
        """A pre-v0.55 telemetry.db (no queued_ms column) must be migrated
        in place by TelemetryStore so insert_many with the new column works."""
        import sqlite3
        from muse.observability.store import TelemetryStore
        from muse.observability.events import event_to_row
        db = tmp_path / "telemetry.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE events (ts REAL NOT NULL, type TEXT NOT NULL, "
            "model_id TEXT)")  # ancient schema: most columns missing
        conn.commit(); conn.close()
        store = TelemetryStore(db)
        store.insert_many([event_to_row("request", 1.0, queued_ms=5.0)])
        # TelemetryStore has no count(); the public row-count accessor is
        # summary_counts()["total"] (adapted from the brief's store.count()).
        assert store.summary_counts()["total"] == 1
        store.close()


# ---------------------------------------------------------------------------
# End-to-end gateway wiring: cap resolution -> gate -> capacity-wait ->
# release pairing, exercised through the real FastAPI app over a fake
# director (mirrors tests/cli_impl/test_gateway_lazy.py).
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

from muse.cli_impl.gateway import build_gateway  # noqa: E402
from muse.cli_impl.supervisor import SupervisorState  # noqa: E402


def _manifest(memory_gb: float = 0.5, device: str = "cpu",
              max_concurrency: int | None = None) -> dict:
    caps: dict = {"memory_gb": memory_gb, "device": device}
    if max_concurrency is not None:
        caps["max_concurrency"] = max_concurrency
    return {"model_id": "fake-model", "modality": "audio/speech",
            "capabilities": caps}


def _state(*, acquire_port: int = 9001) -> SupervisorState:
    state = SupervisorState(workers=[], device="cpu")
    director = MagicMock()
    director.acquire.return_value = acquire_port
    state.director = director
    return state


def _patch_get_manifest(manifest: dict):
    return patch("muse.cli_impl.gateway.get_manifest", return_value=manifest)


def _wire_json_client(mock_cls: MagicMock) -> MagicMock:
    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.aclose = AsyncMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.aread = AsyncMock(return_value=b'{"ok": true}')
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=None)
    mock_client.stream = MagicMock(return_value=ctx)
    mock_cls.return_value = mock_client
    return mock_client


class TestGatewayReleasePairing:
    def test_success_forwards_and_releases_slot(self):
        """Unlimited cap (no gating): request succeeds, director release fires,
        and the gate is left balanced (no stranded slot)."""
        state = _state()
        app = build_gateway(state=state)
        client = TestClient(app)
        with _patch_get_manifest(_manifest()), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_json_client(mock_cls)
            r = client.post("/v1/audio/speech",
                            json={"input": "hi", "model": "fake-model"})
        assert r.status_code == 200
        state.director.release.assert_called_once_with("fake-model")
        assert state.concurrency_gate.depths() == {}

    def test_capped_model_balances_gate_after_request(self):
        """A capped model takes and returns exactly one slot per request."""
        state = _state()
        app = build_gateway(state=state)
        client = TestClient(app)
        with _patch_get_manifest(_manifest(max_concurrency=1)), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_json_client(mock_cls)
            r = client.post("/v1/audio/speech",
                            json={"input": "hi", "model": "fake-model"})
        assert r.status_code == 200
        # slot returned: no residual queue depth and the semaphore is free
        assert state.concurrency_gate.depth("fake-model") == 0
        assert state.concurrency_gate._entered.get("fake-model", 0) == 0

    def test_hot_request_records_zero_queued_ms(self):
        """Uncontended request (free gate slot, no capacity retry): queued_ms
        must be exactly 0.0, not inflated by the (mocked, instant) director
        acquire span -- regression guard for the bug where queued_ms spanned
        the whole gate-to-acquire-return window and would have conflated a
        real cold load's tens-of-seconds with actual queue delay."""
        state = _state()
        app = build_gateway(state=state)
        client = TestClient(app)
        with _patch_get_manifest(_manifest()), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls, \
             patch("muse.cli_impl.gateway.record") as mock_record:
            _wire_json_client(mock_cls)
            r = client.post("/v1/audio/speech",
                            json={"input": "hi", "model": "fake-model"})
        assert r.status_code == 200
        mock_record.assert_called_once()
        _, kwargs = mock_record.call_args
        # Not a strict == 0.0: the no-op gate path still crosses two
        # time.monotonic() calls, so allow for sub-millisecond clock noise
        # while still failing loudly if any real wait leaked in.
        assert 0.0 <= kwargs["queued_ms"] < 1.0


class TestGatewayQueueEnvelopesAsync:
    """Drive _route_via_director on ONE loop so two requests share the gate."""

    async def _route(self, state, model_id="fake-model"):
        from muse.cli_impl.gateway import _route_via_director
        req = MagicMock()
        req.method = "POST"
        req.headers = {"content-type": "application/json"}
        req.body = AsyncMock(
            return_value=b'{"input":"hi","model":"fake-model"}')
        req.query_params = {}
        return await _route_via_director(
            req, "v1/audio/speech", model_id, state, 1.0,
        )

    async def test_queue_timeout_envelope(self, monkeypatch):
        monkeypatch.setenv("MUSE_QUEUE_TIMEOUT_SECONDS", "0.2")
        state = _state()

        release = threading.Event()
        started = threading.Event()

        def blocking_acquire(model_id, *, manifest):
            started.set()
            release.wait(timeout=5)
            return 9001

        state.director.acquire.side_effect = blocking_acquire

        with _patch_get_manifest(_manifest(max_concurrency=1)), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_json_client(mock_cls)
            holder = asyncio.create_task(self._route(state))
            await asyncio.to_thread(started.wait, 5)  # holder owns the slot
            # second request cannot get the slot within 0.2s -> queue_timeout
            resp = await self._route(state)
            assert resp.status_code == 503
            assert resp.body is not None
            import json
            body = json.loads(resp.body)
            assert body["error"]["code"] == "queue_timeout"
            release.set()
            await holder

    async def test_zero_timeout_uncontended_request_does_not_queue_timeout(
        self, monkeypatch,
    ):
        """MUSE_QUEUE_TIMEOUT_SECONDS=0 must not 503 an uncontended capped
        model. A zero wait budget means "do not WAIT", not "do not TRY":
        the concurrency gate has a free slot, so the request should be
        forwarded and succeed rather than fail fast with queue_timeout."""
        monkeypatch.setenv("MUSE_QUEUE_TIMEOUT_SECONDS", "0")
        state = _state()

        with _patch_get_manifest(_manifest(max_concurrency=1)), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_json_client(mock_cls)
            resp = await self._route(state)
        assert resp.status_code == 200
        # slot returned: no residual queue depth on the capped model
        assert state.concurrency_gate.depth("fake-model") == 0

    async def test_queue_full_envelope(self, monkeypatch):
        monkeypatch.setenv("MUSE_QUEUE_TIMEOUT_SECONDS", "5")
        monkeypatch.setenv("MUSE_MAX_QUEUE_DEPTH", "1")
        state = _state()

        release = threading.Event()
        started = threading.Event()

        def blocking_acquire(model_id, *, manifest):
            started.set()
            release.wait(timeout=5)
            return 9001

        state.director.acquire.side_effect = blocking_acquire

        with _patch_get_manifest(_manifest(max_concurrency=1)), \
             patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_json_client(mock_cls)
            holder = asyncio.create_task(self._route(state))
            await asyncio.to_thread(started.wait, 5)  # holder owns the slot
            waiter = asyncio.create_task(self._route(state))  # parks at depth 1
            await asyncio.sleep(0.05)
            # third request: queue is full (depth 1 == max) -> queue_full
            resp = await self._route(state)
            assert resp.status_code == 503
            import json
            body = json.loads(resp.body)
            assert body["error"]["code"] == "queue_full"
            release.set()
            await holder
            await waiter
