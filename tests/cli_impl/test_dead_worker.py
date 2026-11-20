"""A worker the monitor gives up on must stop being "hot" (review I1).

Before the fix the monitor marked the spec dead but the LoadDirector kept
its LoadEntry, so every request was forwarded to the dead port and got a
502 forever (each release() refreshed last_touched_at, so the idle sweeper
never reclaimed it), while /health kept reporting "ok".

These tests drive the real gateway, the real LoadDirector and the real
monitor loop; only the worker process and the HTTP hop are faked.
"""
from __future__ import annotations

import sys
import threading
import time
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.responses import Response

from muse.cli_impl import supervisor as sup
from muse.cli_impl.gateway import build_gateway
from muse.cli_impl.load_director import LoadDirector
from muse.cli_impl.supervisor import SupervisorState, WorkerSpec

MODEL = "dead-model"


@pytest.fixture(autouse=True)
def _isolated_catalog(tmp_path, monkeypatch):
    from muse.core.catalog import (
        _reset_known_models_cache,
        _reset_read_catalog_cache,
        _write_catalog,
    )

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    monkeypatch.setenv("MUSE_TELEMETRY_ENABLED", "false")
    _reset_known_models_cache()
    _reset_read_catalog_cache()
    weights = tmp_path / "weights" / MODEL
    weights.mkdir(parents=True)
    _write_catalog({MODEL: {
        "enabled": True,
        "local_dir": str(weights),
        "python_path": sys.executable,
        "manifest": {
            "model_id": MODEL,
            "modality": "audio/speech",
            "hf_repo": "test/model",
            "backend_path": "muse.fake:Model",
            "capabilities": {"device": "cpu", "memory_gb": 1.0},
        },
    }})
    yield
    _reset_known_models_cache()
    _reset_read_catalog_cache()


class _Probe:
    def gpu_free_gb(self):
        return None

    def cpu_free_gb(self):
        return 64.0

    def cuda_available(self):
        return False


@pytest.fixture(autouse=True)
def _no_host_probe(monkeypatch):
    monkeypatch.setattr(sup, "_MemoryProbeAdapter", _Probe)
    monkeypatch.setattr(
        "muse.cli_impl.supervisor.register_process",
        lambda **kwargs: "rid",
    )
    monkeypatch.setattr(
        "muse.cli_impl.supervisor.unregister_process", lambda rid: True,
    )


def _wire(state: SupervisorState) -> list[int]:
    """A director whose enable_fn behaves like load_model_into_worker."""
    ports: list[int] = []

    def enable_fn(model_id: str) -> int:
        with state.lock:
            # Like load_model_into_worker: a non-running spec is dropped
            # and replaced by a fresh one.
            state.workers[:] = [
                s for s in state.workers
                if not (model_id in s.models and s.status != "running")
            ]
            port = 9100 + len(ports)
            spec = WorkerSpec(models=[model_id], python_path="/p", port=port)
            spec.status = "running"
            spec.process = object()
            state.workers.append(spec)
        ports.append(port)
        return port

    def disable_fn(model_id: str) -> None:
        with state.lock:
            state.workers[:] = [s for s in state.workers if model_id not in s.models]

    state.director = LoadDirector(
        enable_fn=enable_fn, disable_fn=disable_fn, memory_probe=_Probe(),
        cpu_headroom_gb=0.0,
    )
    return ports


def _fake_forward(state: SupervisorState):
    async def forward(request, target_url, timeout, *, release=None, **_kw):
        try:
            port = int(target_url.split(":")[2].split("/")[0])
            with state.lock:
                alive = any(
                    s.port == port and s.status == "running" for s in state.workers
                )
            if not alive:
                raise httpx.ConnectError("connection refused")
            return Response(content=b"{}", media_type="application/json")
        finally:
            if release is not None:
                release()
    return forward


def _kill_and_exhaust(state: SupervisorState) -> WorkerSpec:
    """Run the real monitor until it gives up on the (crashed) worker."""
    spec = next(s for s in state.workers if MODEL in s.models)
    with spec.process_lock:
        spec.process = None  # the worker process is gone
    spec.restart_count = sup._MAX_RESTARTS  # restart budget exhausted

    stop = threading.Event()
    monitor = threading.Thread(
        target=sup._monitor_workers,
        args=(state.workers, stop),
        kwargs={"interval": 0.01, "state": state},
        daemon=True,
    )
    monitor.start()
    deadline = time.monotonic() + 5.0
    while spec.status != "dead" and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)
    stop.set()
    monitor.join(timeout=5.0)
    assert spec.status == "dead"
    return spec


def test_dead_worker_is_cold_loaded_again_instead_of_502_forever():
    state = SupervisorState(workers=[], device="cpu")
    ports = _wire(state)
    app = build_gateway(state=state)
    body = {"model": MODEL, "input": "hi"}
    with patch("muse.cli_impl.gateway._forward", _fake_forward(state)):
        client = TestClient(app)
        assert client.post("/v1/audio/speech", json=body).status_code == 200
        assert MODEL in state.director.loaded

        _kill_and_exhaust(state)

        # The director forgot the dead worker ...
        assert MODEL not in state.director.loaded
        # ... so the next request cold-loads a replacement instead of 502.
        r = client.post("/v1/audio/speech", json=body)
        assert r.status_code == 200, r.text
        assert ports == [9100, 9101]
        assert state.director.loaded[MODEL].worker_port == 9101


def test_health_reports_degraded_while_a_worker_is_dead():
    state = SupervisorState(workers=[], device="cpu")
    _wire(state)
    app = build_gateway(state=state)
    with patch("muse.cli_impl.gateway._forward", _fake_forward(state)):
        client = TestClient(app)
        client.post("/v1/audio/speech", json={"model": MODEL, "input": "hi"})
        with patch("muse.cli_impl.gateway.httpx.AsyncClient.get",
                   side_effect=httpx.ConnectError("no workers in this test")):
            _kill_and_exhaust(state)
            assert client.get("/health").json()["status"] == "degraded"
            models = client.get("/v1/models").json()["data"]
    row = next(m for m in models if m["id"] == MODEL)
    assert row["loaded"] is False


def test_retained_process_keeps_accounting_but_stops_routing():
    """A dead spec whose process could not be released still holds memory:
    keep it accounted but blocked until a cleanup teardown succeeds."""
    state = SupervisorState(workers=[], device="cpu")
    _wire(state)
    state.director.acquire(MODEL, manifest={
        "capabilities": {"device": "cpu", "memory_gb": 1.0},
    })
    state.director.release(MODEL)
    spec = state.workers[0]
    spec.status = "dead"
    sup._reconcile_dead_worker(state, spec)  # process still retained
    assert MODEL in state.director.loaded
    from muse.admin.operations import OperationError
    with pytest.raises(OperationError) as exc:
        state.director.acquire(MODEL, manifest={
            "capabilities": {"device": "cpu", "memory_gb": 1.0},
        })
    assert exc.value.code == "model_cleanup_required"


def test_revived_worker_is_adopted_by_the_director():
    """revive_worker (the admin restart seam) + a successful monitor
    restart leave the model accounted in the director again."""
    state = SupervisorState(workers=[], device="cpu")
    _wire(state)
    state.director.acquire(MODEL, manifest={
        "capabilities": {"device": "cpu", "memory_gb": 1.0},
    })
    state.director.release(MODEL)
    spec = _kill_and_exhaust(state)
    assert MODEL not in state.director.loaded

    assert sup.revive_worker(state, spec) is True
    assert spec.status == "unhealthy" and spec.restart_count == 0

    def respawn(s, **_kw):
        s.process = object()

    stop = threading.Event()
    with patch("muse.cli_impl.supervisor.spawn_worker", side_effect=respawn), \
         patch("muse.cli_impl.supervisor.wait_for_ready"), \
         patch("muse.cli_impl.supervisor._worker_process_state_locked",
               return_value=True), \
         patch("muse.cli_impl.supervisor.check_worker_health", return_value=True):
        monitor = threading.Thread(
            target=sup._monitor_workers, args=(state.workers, stop),
            kwargs={"interval": 0.01, "state": state}, daemon=True,
        )
        monitor.start()
        deadline = time.monotonic() + 5.0
        while spec.status != "running" and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)
        stop.set()
        monitor.join(timeout=5.0)
    assert spec.status == "running"
    entry = state.director.loaded[MODEL]
    assert entry.worker_port == spec.port
    assert entry.refcount == 0 and entry.memory_gb == 1.0


# --- /v1/models and /health catalog enrichment runs off the loop (I4) ------


def _seed_unloaded(n: int) -> None:
    from muse.core.catalog import _read_catalog, _write_catalog

    catalog = dict(_read_catalog())
    template = catalog[MODEL]
    for i in range(n):
        catalog[f"cold-{i}"] = dict(
            template, manifest=dict(template["manifest"], model_id=f"cold-{i}"),
        )
    _write_catalog(catalog)


@pytest.mark.parametrize("path", ["/v1/models", "/health"])
def test_listing_enrichment_runs_off_the_event_loop(path):
    """The per-model servability walk (provenance stats, capacity probes,
    weights walks) must not stall hot inference traffic on the loop."""
    _seed_unloaded(3)
    state = SupervisorState(workers=[], device="cpu")
    _wire(state)
    app = build_gateway(state=state)
    loop_threads: set[int] = set()
    check_threads: list[int] = []
    real = sup.revalidate_servability

    def recording(state_arg, model_id, **kwargs):
        check_threads.append(threading.get_ident())
        return real(state_arg, model_id, **kwargs)

    @app.middleware("http")
    async def note_loop_thread(request, call_next):
        loop_threads.add(threading.get_ident())
        return await call_next(request)

    with patch("muse.cli_impl.gateway.revalidate_servability", recording):
        body = TestClient(app).get(path).json()
    assert len(check_threads) == 4
    assert loop_threads and not loop_threads & set(check_threads)
    if path == "/v1/models":
        assert {m["id"] for m in body["data"]} >= {"cold-0", "cold-1", "cold-2"}


def test_listing_probes_capacity_once_per_request():
    _seed_unloaded(3)
    state = SupervisorState(workers=[], device="cpu")
    _wire(state)
    app = build_gateway(state=state)
    real = sup._capacity_pools
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    with patch("muse.cli_impl.supervisor._capacity_pools", counting):
        TestClient(app).get("/v1/models")
    assert len(calls) == 1
