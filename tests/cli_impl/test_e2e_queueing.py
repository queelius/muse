"""Slow e2e: a max_concurrency=1 model serializes concurrent requests.

In-process style (mirrors tests/cli_impl/test_gateway_lazy.py and
test_queueing_gateway.py): a real gateway ASGI app (build_gateway) over a
FakeDirector (instant acquire/release -- no real subprocess) plus the REAL
ConcurrencyGate/CapacityNotifier that SupervisorState default-constructs
(Tasks 1-4). The "worker" is stubbed at the httpx.AsyncClient forward
boundary: each forwarded request sleeps ~0.3s before responding (simulating
inference latency) and records its (start, end) service window under a
lock. Four concurrent requests against a max_concurrency=1 model must all
succeed AND be serialized end-to-end by the concurrency gate: non-
overlapping service windows, total wall time ~= 4 * 0.3s (not ~0.3s, which
would mean the gate did nothing).

TestClient MUST be used as a context manager (`with TestClient(app) as
client:`). Outside a `with` block, starlette's TestClient spins up a FRESH
portal (and therefore a fresh asyncio event loop) for every single request
(see `_portal_factory`); the asyncio.Semaphore backing the gate binds to
whichever loop first uses it (Python 3.10+ `_LoopBoundMixin`), so a second
concurrent request landing on a different fresh loop would raise "bound to
a different event loop" instead of actually queueing. The `with` form
reuses ONE persistent portal/loop across every `client.post(...)` call
regardless of which OS thread issues it, so the four threads below
genuinely interleave on that one loop -- the same "waiting ON the event
loop, never in pool threads" invariant the gate itself relies on.
"""
from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from muse.cli_impl.gateway import build_gateway
from muse.cli_impl.supervisor import SupervisorState

pytestmark = pytest.mark.slow

SERVICE_SECONDS = 0.3
N_REQUESTS = 4

_MANIFEST = {
    "model_id": "fake-model",
    "modality": "audio/speech",
    "capabilities": {"max_concurrency": 1, "memory_gb": 0.1},
}


def _make_state() -> SupervisorState:
    """SupervisorState with a FakeDirector (instant acquire/release) and
    the REAL ConcurrencyGate/CapacityNotifier default-constructed by
    SupervisorState. The concurrency cap under test lives entirely in the
    gate (ahead of the director in `_route_via_director`), so a mocked
    director with an instant acquire is sufficient -- no real worker
    process, no real load/eviction machinery needed to exercise it.
    """
    state = SupervisorState(workers=[], device="cpu")
    director = MagicMock()
    director.acquire.return_value = 9001
    state.director = director
    return state


def _slow_worker_stream(windows: list[tuple[float, float]], lock: threading.Lock):
    """side_effect for the patched AsyncClient.stream(...): simulates a
    worker that takes SERVICE_SECONDS to service one request. Returns a
    fresh async-context-manager stub per call (one per forwarded request)
    whose __aenter__ sleeps, records the (start, end) window, then hands
    back a 200 JSON response.
    """
    def _make_ctx(method, url, **kwargs):
        ctx = MagicMock()

        async def _aenter_body():
            start = time.monotonic()
            await asyncio.sleep(SERVICE_SECONDS)
            end = time.monotonic()
            with lock:
                windows.append((start, end))
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {"content-type": "application/json"}
            resp.aread = AsyncMock(return_value=b'{"ok": true}')
            return resp

        # AsyncMock(side_effect=...), not a bare coroutine function: assigning
        # a plain function directly to a MagicMock dunder attribute (e.g.
        # __aenter__) makes unittest.mock treat it as an unbound method and
        # inject `self` as the first positional arg, which blows up a
        # zero-arg body. Wrapping in AsyncMock sidesteps that rewrap (matches
        # the existing __aenter__ = AsyncMock(...) pattern used throughout
        # tests/cli_impl/test_gateway_lazy.py).
        ctx.__aenter__ = AsyncMock(side_effect=_aenter_body)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    return _make_ctx


def _run_four_concurrent(monkeypatch):
    """Fire N_REQUESTS concurrent POSTs at the capped model; return
    (status_codes, windows, elapsed_seconds).
    """
    # Isolate from any queueing env vars a neighboring test left set:
    # unbounded queue depth, the 300s config default timeout (plenty for
    # a ~1.2s serialized burst), and no config-level concurrency default
    # (the manifest's own max_concurrency=1 must be what gates this).
    monkeypatch.delenv("MUSE_MAX_QUEUE_DEPTH", raising=False)
    monkeypatch.delenv("MUSE_QUEUE_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("MUSE_DEFAULT_MAX_CONCURRENCY", raising=False)

    state = _make_state()
    app = build_gateway(state=state)

    windows: list[tuple[float, float]] = []
    windows_lock = threading.Lock()
    results: list[int | None] = [None] * N_REQUESTS

    with patch("muse.cli_impl.gateway.get_manifest", return_value=_MANIFEST), \
         patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.aclose = AsyncMock()
        mock_client.stream = MagicMock(
            side_effect=_slow_worker_stream(windows, windows_lock),
        )
        mock_cls.return_value = mock_client

        with TestClient(app) as client:
            def _fire(i: int) -> None:
                r = client.post(
                    "/v1/audio/speech",
                    json={"input": f"hi-{i}", "model": "fake-model"},
                )
                results[i] = r.status_code

            t0 = time.monotonic()
            threads = [
                threading.Thread(target=_fire, args=(i,))
                for i in range(N_REQUESTS)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)
            elapsed = time.monotonic() - t0

    return results, windows, elapsed


@pytest.mark.timeout(30)
def test_cap_one_serializes_four_concurrent_requests(monkeypatch):
    results, windows, elapsed = _run_four_concurrent(monkeypatch)

    # All four requests completed successfully.
    assert all(status == 200 for status in results), results

    # The stub worker actually serviced all four (no request skipped the
    # forward leg, e.g. by 503ing on queue_full/queue_timeout).
    assert len(windows) == N_REQUESTS

    # Non-overlapping service windows: the cap-1 gate let exactly one
    # request into the "worker" at a time. A small epsilon absorbs
    # scheduling/clock jitter without masking genuine overlap (real
    # overlap would be on the order of SERVICE_SECONDS, not milliseconds).
    epsilon = 0.02
    ordered = sorted(windows, key=lambda w: w[0])
    for (_, end_a), (start_b, _) in zip(ordered, ordered[1:]):
        assert end_a <= start_b + epsilon, (
            f"overlapping service windows (gate did not serialize): {ordered}"
        )

    # Belt: total wall time proves serialization, not just non-overlap
    # bookkeeping. Fully parallel would take ~SERVICE_SECONDS; fully
    # serialized takes ~N_REQUESTS * SERVICE_SECONDS.
    slack = 0.05
    assert elapsed >= N_REQUESTS * SERVICE_SECONDS - slack, (
        f"elapsed {elapsed:.3f}s is too fast for {N_REQUESTS} requests "
        f"serialized at {SERVICE_SECONDS}s each -- looks parallel, not gated"
    )
