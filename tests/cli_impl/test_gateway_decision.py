"""Gateway handling of POST /v1/systemone (decision/evaluation).

Strict parsing happens before model selection, decision models are
clamped to the modality's CONCURRENCY_CEILING, and nothing about the
route needs a hard-coded gateway map.
"""
from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from muse.cli_impl.gateway import (
    _effective_max_concurrency,
    _effective_max_queue_depth,
    _modality_from_path,
    _route_via_director,
    build_gateway,
)
from muse.cli_impl.supervisor import SupervisorState
from muse.core.discovery import (
    concurrency_ceilings,
    model_optional_paths,
    modality_tags,
    strict_request_parsers,
)


DECISION = "decision-0.8b-q8"
BODY = {
    "model": DECISION,
    "state": "I was charged twice.",
    "questions": {"dup": {"type": "noul", "instructions": "A duplicate charge."}},
}


def _manifest(max_concurrency=None, modality="decision/evaluation"):
    caps = {"memory_gb": 0.5, "device": "cpu"}
    if max_concurrency is not None:
        caps["max_concurrency"] = max_concurrency
    return {"model_id": DECISION, "modality": modality, "capabilities": caps}


def _state():
    state = SupervisorState(workers=[], device="cpu")
    state.director = MagicMock()
    state.director.acquire.return_value = 9001
    return state


def _wire_json_client(mock_cls):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.aclose = AsyncMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.aread = AsyncMock(return_value=b'{"ok": true}')
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=None)
    client.stream = MagicMock(return_value=ctx)
    mock_cls.return_value = client
    return client


# --- discovery -------------------------------------------------------------


def test_discovery_reports_the_modality_ceiling_and_strict_parser():
    assert "decision/evaluation" in modality_tags()
    assert concurrency_ceilings()["decision/evaluation"] == {
        "max_concurrency": 1, "max_queue_depth": 32,
    }
    assert strict_request_parsers()["/v1/systemone"].endswith("validation:parse_request")


def test_systemone_is_not_a_default_model_path():
    assert "/v1/systemone" not in model_optional_paths()


def test_telemetry_label_is_path_derived():
    assert _modality_from_path("v1/systemone") == "systemone"


# --- ceilings ---------------------------------------------------------------


@pytest.mark.parametrize("declared, default, expected", [
    (None, None, 1),     # unlimited everywhere -> the ceiling
    (8, None, 1),        # a manifest cannot raise it
    (None, "4", 1),      # neither can the global default
    (1, "4", 1),
])
def test_decision_concurrency_is_clamped(declared, default, expected, monkeypatch):
    if default is None:
        monkeypatch.delenv("MUSE_DEFAULT_MAX_CONCURRENCY", raising=False)
    else:
        monkeypatch.setenv("MUSE_DEFAULT_MAX_CONCURRENCY", default)
    assert _effective_max_concurrency(_manifest(declared)) == expected
    assert _effective_max_queue_depth(_manifest(declared)) == 32


def test_other_modalities_keep_their_caps(monkeypatch):
    monkeypatch.delenv("MUSE_DEFAULT_MAX_CONCURRENCY", raising=False)
    assert _effective_max_concurrency(_manifest(8, modality="chat/completion")) == 8
    assert _effective_max_concurrency(_manifest(None, modality="chat/completion")) is None
    assert _effective_max_queue_depth(_manifest(None, modality="chat/completion")) is None


# --- strict parsing before model selection ----------------------------------


def _post(client, raw, headers=None):
    return client.post("/v1/systemone", content=raw,
                       headers={"content-type": "application/json", **(headers or {})})


@pytest.mark.parametrize("raw, status, code", [
    (b'{"model":"other","model":"decision-0.8b-q8","state":"s",'
     b'"questions":{"q":{"type":"noul","instructions":"i"}}}', 400, "duplicate_key"),
    (b'{"state":"s","questions":{"q":{"type":"noul","instructions":"i"}}}', 400, "missing_field"),
    (b'{"model":"decision-0.8b-q8","state":"s","questions":{}}', 400, "invalid_questions"),
    (b'{"model":"decision-0.8b-q8","state":{"x":NaN},'
     b'"questions":{"q":{"type":"noul","instructions":"i"}}}', 400, "non_finite_number"),
])
def test_invalid_bodies_fail_before_any_catalog_or_director_work(raw, status, code):
    state = _state()
    client = TestClient(build_gateway(state=state))
    with patch("muse.cli_impl.gateway.get_manifest") as get_manifest, \
         patch("muse.cli_impl.gateway._read_catalog") as read_catalog:
        r = _post(client, raw)
    assert r.status_code == status
    assert r.json()["error"]["code"] == code
    get_manifest.assert_not_called()
    read_catalog.assert_not_called()
    state.director.acquire.assert_not_called()


def test_streamed_body_over_128k_is_413_before_model_selection():
    state = _state()
    client = TestClient(build_gateway(state=state))

    def chunks():
        yield b'{"model":"decision-0.8b-q8","state":"'
        for _ in range(40):
            yield b"x" * 4096
        yield b'","questions":{"q":{"type":"noul","instructions":"i"}}}'

    with patch("muse.cli_impl.gateway.get_manifest") as get_manifest:
        r = client.post("/v1/systemone", content=chunks(),
                        headers={"content-type": "application/json"})
    assert r.status_code == 413
    get_manifest.assert_not_called()
    state.director.acquire.assert_not_called()


def test_valid_body_routes_by_the_parsed_model_and_records_no_payload():
    state = _state()
    client = TestClient(build_gateway(state=state))
    with patch("muse.cli_impl.gateway.get_manifest", return_value=_manifest()), \
         patch("muse.cli_impl.gateway._read_catalog", return_value={DECISION: {"enabled": True}}), \
         patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls, \
         patch("muse.cli_impl.gateway.record") as mock_record:
        _wire_json_client(mock_cls)
        r = _post(client, json.dumps(BODY).encode())
    assert r.status_code == 200
    state.director.acquire.assert_called_once()
    assert state.director.acquire.call_args.args[0] == DECISION
    state.director.release.assert_called_once_with(DECISION)
    _, kwargs = mock_record.call_args
    assert kwargs["modality"] == "systemone" and kwargs["model_id"] == DECISION
    recorded = json.dumps(kwargs, default=str)
    assert "charged" not in recorded and "duplicate charge" not in recorded.lower()
    assert state.concurrency_gate._entered.get(DECISION, 0) == 0


def test_disabled_model_keeps_the_existing_contract():
    state = _state()
    client = TestClient(build_gateway(state=state))
    with patch("muse.cli_impl.gateway._read_catalog", return_value={DECISION: {"enabled": False}}):
        r = _post(client, json.dumps(BODY).encode())
    assert r.status_code == 409 and r.json()["error"]["code"] == "model_disabled"
    state.director.acquire.assert_not_called()


# --- queue ceiling through the real routing path ------------------------------


async def _until(predicate, timeout=5.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not predicate():
        assert loop.time() < end, "condition never became true"
        await asyncio.sleep(0.005)


class TestQueueCeiling:
    async def _route(self, state):
        req = MagicMock()
        req.method = "POST"
        req.headers = {"content-type": "application/json"}
        req.body = AsyncMock(return_value=json.dumps(BODY).encode())
        req.query_params = {}
        return await _route_via_director(req, "v1/systemone", DECISION, state, 1.0)

    async def _saturate(self, state, monkeypatch, n_waiters):
        monkeypatch.setenv("MUSE_QUEUE_TIMEOUT_SECONDS", "10")
        release, started = threading.Event(), threading.Event()

        def blocking_acquire(model_id, *, manifest):
            started.set()
            release.wait(timeout=10)
            return 9001

        state.director.acquire.side_effect = blocking_acquire
        first = asyncio.ensure_future(self._route(state))
        await _until(started.is_set)
        waiters = [asyncio.ensure_future(self._route(state)) for _ in range(n_waiters)]
        try:
            await _until(lambda: state.concurrency_gate.depth(DECISION) >= n_waiters)
        except AssertionError:
            await self._finish(release, [first, *waiters])
            raise
        return release, [first, *waiters]

    async def _finish(self, release, tasks):
        for t in tasks:
            t.cancel()
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def test_one_admitted_32_waiting_then_queue_full(self, monkeypatch):
        monkeypatch.setenv("MUSE_MAX_QUEUE_DEPTH", "0")  # global unbounded
        state = _state()
        with patch("muse.cli_impl.gateway.get_manifest", return_value=_manifest(max_concurrency=8)), \
             patch("muse.cli_impl.gateway._read_catalog", return_value={DECISION: {"enabled": True}}):
            release, tasks = await self._saturate(state, monkeypatch, 32)
            r = await self._route(state)
            assert r.status_code == 503
            assert json.loads(r.body)["error"]["code"] == "queue_full"
            await self._finish(release, tasks)

    async def test_positive_global_depth_lowers_the_ceiling(self, monkeypatch):
        monkeypatch.setenv("MUSE_MAX_QUEUE_DEPTH", "3")
        state = _state()
        with patch("muse.cli_impl.gateway.get_manifest", return_value=_manifest()), \
             patch("muse.cli_impl.gateway._read_catalog", return_value={DECISION: {"enabled": True}}):
            release, tasks = await self._saturate(state, monkeypatch, 3)
            r = await self._route(state)
            assert json.loads(r.body)["error"]["code"] == "queue_full"
            await self._finish(release, tasks)


# --- client disconnect -------------------------------------------------------


def test_only_systemone_cancels_its_worker_call_on_disconnect():
    from muse.core.discovery import disconnect_cancellable_paths

    assert disconnect_cancellable_paths() == frozenset({"/v1/systemone"})


def _disconnecting_request(body: bytes, gone: asyncio.Event):
    from starlette.requests import Request

    sent = [{"type": "http.request", "body": body, "more_body": False}]
    received = []

    async def receive():
        received.append(1)
        if sent:
            return sent.pop()
        await gone.wait()
        return {"type": "http.disconnect"}

    scope = {
        "type": "http", "method": "POST", "path": "/v1/systemone", "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }
    return Request(scope, receive), received


def _hanging_client(mock_cls, opened: asyncio.Event, cancelled: asyncio.Event):
    client = MagicMock()
    client.aclose = AsyncMock()

    async def hang():
        opened.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    ctx = MagicMock()
    ctx.__aenter__ = MagicMock(side_effect=lambda: hang())
    ctx.__aexit__ = AsyncMock(return_value=None)
    client.stream = MagicMock(return_value=ctx)
    mock_cls.return_value = client
    return client


async def test_disconnect_drops_the_worker_call_and_releases():
    from muse.cli_impl.gateway import _forward_with_release
    from muse.core.disconnect import ClientDisconnected

    gone, opened, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    request, _ = _disconnecting_request(json.dumps(BODY).encode(), gone)
    director = MagicMock()
    released = []
    with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
        client = _hanging_client(mock_cls, opened, cancelled)
        call = asyncio.ensure_future(_forward_with_release(
            request, "http://127.0.0.1:9001/v1/systemone", 30.0,
            director=director, model_id=DECISION,
            extra_release=lambda: released.append(1), cancel_on_disconnect=True,
        ))
        await asyncio.wait_for(opened.wait(), 5)
        gone.set()
        with pytest.raises(ClientDisconnected):
            await asyncio.wait_for(call, 5)
    assert cancelled.is_set()
    director.release.assert_called_once_with(DECISION)
    assert released == [1]
    client.aclose.assert_awaited_once()


async def test_other_paths_do_not_watch_for_disconnect():
    from muse.cli_impl.gateway import _forward_with_release

    gone, opened, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    request, received = _disconnecting_request(json.dumps(BODY).encode(), gone)
    with patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
        _hanging_client(mock_cls, opened, cancelled)
        call = asyncio.ensure_future(_forward_with_release(
            request, "http://127.0.0.1:9001/v1/x", 30.0,
            director=MagicMock(), model_id=DECISION,
        ))
        await asyncio.wait_for(opened.wait(), 5)
        gone.set()
        await asyncio.sleep(0.05)
        assert not call.done() and len(received) == 1  # only the body was read
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
