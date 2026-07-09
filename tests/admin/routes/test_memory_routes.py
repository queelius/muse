"""Tests for GET /v1/admin/memory."""
from __future__ import annotations

import asyncio
import json
import sys

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from muse.admin.auth import ADMIN_TOKEN_ENV, verify_admin_token
from muse.admin.routes.memory import build_memory_router
from muse.cli_impl.supervisor import (
    SupervisorState,
    WorkerSpec,
    clear_supervisor_state,
    set_supervisor_state,
)


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(
        build_memory_router(),
        prefix="/v1/admin",
        dependencies=[Depends(verify_admin_token)],
    )
    return app


@pytest.fixture
def client(app, monkeypatch):
    monkeypatch.setenv(ADMIN_TOKEN_ENV, "test-token")
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def headers():
    return {"Authorization": "Bearer test-token"}


@pytest.fixture(autouse=True)
def _state_reset():
    clear_supervisor_state()
    yield
    clear_supervisor_state()


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    from muse.core.catalog import _reset_known_models_cache
    _reset_known_models_cache()
    yield tmp_path
    _reset_known_models_cache()


def _seed_catalog(data: dict) -> None:
    from muse.core.catalog import _catalog_path, _reset_known_models_cache
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    _reset_known_models_cache()


class TestMemoryRoute:
    def test_no_psutil_no_pynvml_yields_nulls(self, client, headers, monkeypatch):
        # Force both imports to fail so the fallback paths run.
        monkeypatch.setitem(sys.modules, "psutil", None)
        monkeypatch.setitem(sys.modules, "pynvml", None)
        set_supervisor_state(SupervisorState(workers=[], device="cpu"))
        r = client.get("/v1/admin/memory", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body == {"gpu": None, "cpu": None, "recent_decisions": []}

    def test_per_model_breakdown_unit(self, tmp_catalog):
        """Direct call into the helper; bypasses psutil/pynvml entirely."""
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": True,
                "measurements": {
                    "cpu": {"weights_bytes": 1024**3,
                             "peak_bytes": 2 * 1024**3},
                },
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/v/bin/python", port=9001,
        )
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))
        from muse.admin.routes.memory import _per_model_breakdown
        out = _per_model_breakdown("cpu", "cpu")
        assert len(out) == 1
        record = out[0]
        assert record["model_id"] == "kokoro-82m"
        assert record["weights_gb"] == 1.0
        assert record["peak_gb"] == 2.0

    def test_per_model_breakdown_includes_refcount_from_director(self, tmp_catalog):
        """The breakdown surfaces each loaded model's live director refcount,
        so a 503 'no evictable candidates (all loaded models have refcount >
        0)' can be verified against reality: refcount 0 means idle-evictable,
        refcount > 0 means pinned by an in-flight request."""
        import time
        from muse.cli_impl.load_director import LoadDirector, LoadEntry
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": True,
                "measurements": {
                    "cpu": {"weights_bytes": 1024**3, "peak_bytes": 2 * 1024**3},
                },
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/v/bin/python", port=9001,
        )
        director = LoadDirector(
            enable_fn=lambda mid: 9001,
            disable_fn=lambda mid: None,
            memory_probe=type("P", (), {
                "gpu_free_gb": staticmethod(lambda: None),
                "cpu_free_gb": staticmethod(lambda: 64.0),
            })(),
        )
        now = time.monotonic()
        director.loaded["kokoro-82m"] = LoadEntry(
            model_id="kokoro-82m", worker_port=9001, memory_gb=2.0,
            refcount=3, last_touched_at=now, loaded_at=now,
        )
        state = SupervisorState(workers=[spec], device="cpu")
        state.director = director
        set_supervisor_state(state)
        from muse.admin.routes.memory import _per_model_breakdown
        out = _per_model_breakdown("cpu", "cpu")
        assert len(out) == 1
        assert out[0]["refcount"] == 3

    def test_per_model_breakdown_omits_refcount_without_director(self, tmp_catalog):
        """No director bound (gateway-in-isolation / pre-boot): the refcount
        key is omitted rather than falsely reported as 0, which would read as
        'evictable' when the truth is unknown."""
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": True,
                "measurements": {
                    "cpu": {"weights_bytes": 1024**3, "peak_bytes": 2 * 1024**3},
                },
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/v/bin/python", port=9001,
        )
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))
        from muse.admin.routes.memory import _per_model_breakdown
        out = _per_model_breakdown("cpu", "cpu")
        assert len(out) == 1
        assert "refcount" not in out[0]

    def test_per_model_breakdown_includes_queue_depth(self, tmp_catalog):
        """The breakdown surfaces each loaded model's live queue depth from
        the gateway's ConcurrencyGate, so an operator can see parked waiters
        without inferring it from latency alone. depth is excess-over-cap
        (entered - cap, floored at 0): cap=1 with 4 entered means 1 holds
        the slot and 3 are parked waiting."""
        from muse.cli_impl.queueing import ConcurrencyGate
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": True,
                "measurements": {
                    "cpu": {"weights_bytes": 1024**3, "peak_bytes": 2 * 1024**3},
                },
            },
        })
        spec = WorkerSpec(models=["kokoro-82m"],
                          python_path="/v/bin/python", port=9001)
        state = SupervisorState(workers=[spec], device="cpu")
        state.concurrency_gate = ConcurrencyGate()
        # Simulate 3 parked waiters via the gate's real internals: cap=1,
        # 4 entered (1 holding the slot + 3 parked) -> depth = 4 - 1 = 3.
        gate = state.concurrency_gate
        gate._caps["kokoro-82m"] = 1
        gate._sems["kokoro-82m"] = asyncio.Semaphore(1)
        gate._entered["kokoro-82m"] = 4
        assert gate.depth("kokoro-82m") == 3
        set_supervisor_state(state)
        from muse.admin.routes.memory import _per_model_breakdown
        out = _per_model_breakdown("cpu", "cpu")
        assert out[0]["queue_depth"] == 3

    def test_per_model_breakdown_queue_depth_zero_when_gate_bound_but_idle(
        self, tmp_catalog,
    ):
        """A bound gate with no excess waiters reports queue_depth 0 (not
        omitted): 0 is meaningful ("gate exists, nothing queued"), whereas
        omission means "no gate bound, unknown"."""
        from muse.cli_impl.queueing import ConcurrencyGate
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": True,
                "measurements": {
                    "cpu": {"weights_bytes": 1024**3, "peak_bytes": 2 * 1024**3},
                },
            },
        })
        spec = WorkerSpec(models=["kokoro-82m"],
                          python_path="/v/bin/python", port=9001)
        state = SupervisorState(workers=[spec], device="cpu")
        state.concurrency_gate = ConcurrencyGate()
        set_supervisor_state(state)
        from muse.admin.routes.memory import _per_model_breakdown
        out = _per_model_breakdown("cpu", "cpu")
        assert out[0]["queue_depth"] == 0

    def test_per_model_breakdown_omits_queue_depth_without_gate(self, tmp_catalog):
        """No gate bound (bare state with the field cleared, e.g. a Mock in
        another test harness): the queue_depth key is omitted rather than
        falsely reported as 0."""
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": True,
                "measurements": {
                    "cpu": {"weights_bytes": 1024**3, "peak_bytes": 2 * 1024**3},
                },
            },
        })
        spec = WorkerSpec(models=["kokoro-82m"],
                          python_path="/v/bin/python", port=9001)
        state = SupervisorState(workers=[spec], device="cpu")
        state.concurrency_gate = None
        set_supervisor_state(state)
        from muse.admin.routes.memory import _per_model_breakdown
        out = _per_model_breakdown("cpu", "cpu")
        assert "queue_depth" not in out[0]

    def test_per_model_breakdown_shows_cold_start_queuers(self, tmp_catalog):
        """#331: a model with parked waiters but NO live worker yet (its own
        cold start -- exactly when the queue is deepest) must still appear
        in the breakdown, marked loaded=False, so operators can see the
        pressure instead of nothing."""
        import asyncio
        from muse.cli_impl.queueing import ConcurrencyGate
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": True,
                "measurements": {
                    "cpu": {"weights_bytes": 1024**3, "peak_bytes": 2 * 1024**3},
                },
            },
        })
        state = SupervisorState(workers=[], device="cpu")  # nothing loaded
        gate = ConcurrencyGate()
        gate._caps["kokoro-82m"] = 1
        gate._sems["kokoro-82m"] = asyncio.Semaphore(1)
        gate._entered["kokoro-82m"] = 4
        assert gate.depth("kokoro-82m") == 3
        state.concurrency_gate = gate
        set_supervisor_state(state)
        from muse.admin.routes.memory import _per_model_breakdown
        out = _per_model_breakdown("cpu", "cpu")
        rows = [r for r in out if r["model_id"] == "kokoro-82m"]
        assert len(rows) == 1
        assert rows[0]["queue_depth"] == 3
        assert rows[0]["loaded"] is False

    def test_auto_device_model_resolves_to_gpu_side_on_cuda_supervisor(self, tmp_catalog):
        """A device='auto' bundled model (e.g. kokoro-82m) is accounted on
        the GPU side when the supervisor runs --device cuda, and NOT on the
        CPU side. This pins the auto-resolution that the cpu->auto default
        flip depends on (else auto models would always bucket to GPU)."""
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": True,
                "measurements": {
                    "cuda": {"weights_bytes": 1024**3, "peak_bytes": 2 * 1024**3},
                },
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/v/bin/python", port=9001,
        )
        set_supervisor_state(SupervisorState(workers=[spec], device="cuda"))
        from muse.admin.routes.memory import _per_model_breakdown
        gpu_side = _per_model_breakdown("cuda", "gpu")
        cpu_side = _per_model_breakdown("cpu", "cpu")
        assert [r["model_id"] for r in gpu_side] == ["kokoro-82m"]
        assert cpu_side == []

    def test_per_model_breakdown_filters_gpu_when_target_cpu(self, tmp_catalog):
        """A GPU-side measurement should NOT appear in the cpu bucket."""
        # sd-turbo defaults to GPU (capability device != cpu), so a cpu
        # bucket shouldn't include it.
        _seed_catalog({
            "sd-turbo": {
                "pulled_at": "...", "hf_repo": "stabilityai/sd-turbo",
                "local_dir": "/x", "venv_path": "/v",
                "python_path": "/v/bin/python", "enabled": True,
                "measurements": {
                    "cuda": {"weights_bytes": 5 * 1024**3,
                              "peak_bytes": 7 * 1024**3},
                },
            },
        })
        spec = WorkerSpec(
            models=["sd-turbo"], python_path="/v/bin/python", port=9001,
        )
        set_supervisor_state(SupervisorState(workers=[spec], device="cuda"))
        from muse.admin.routes.memory import _per_model_breakdown
        out_cpu = _per_model_breakdown("cpu", "cpu")
        out_gpu = _per_model_breakdown("cuda", "gpu")
        # GPU model not in CPU bucket
        assert all(r["model_id"] != "sd-turbo" for r in out_cpu)
        # GPU model present in GPU bucket
        assert any(r["model_id"] == "sd-turbo" for r in out_gpu)

    def test_auto_device_model_resolves_to_cpu_side_on_cpu_supervisor(self, tmp_catalog):
        """The non-tautological companion to the cuda-supervisor case: on a
        --device cpu supervisor a device='auto' bundled model must bucket to
        the CPU side, NOT the GPU side. Before _per_model_breakdown resolved
        'auto', it fell through to GPU (on_cpu = 'auto' == 'cpu' -> False),
        so this case FAILS against the unfixed code -- which the cuda case
        could not show (on a GPU host 'auto' buckets to GPU either way)."""
        _seed_catalog({
            "kokoro-82m": {
                "pulled_at": "...", "hf_repo": "k", "local_dir": "/k",
                "venv_path": "/v", "python_path": "/v/bin/python",
                "enabled": True,
                "measurements": {
                    "cpu": {"weights_bytes": 1024**3, "peak_bytes": 2 * 1024**3},
                },
            },
        })
        spec = WorkerSpec(
            models=["kokoro-82m"], python_path="/v/bin/python", port=9001,
        )
        set_supervisor_state(SupervisorState(workers=[spec], device="cpu"))
        from muse.admin.routes.memory import _per_model_breakdown
        cpu_side = _per_model_breakdown("cpu", "cpu")
        gpu_side = _per_model_breakdown("cuda", "gpu")
        assert [r["model_id"] for r in cpu_side] == ["kokoro-82m"]
        assert gpu_side == []


class TestResolveAutoSide:
    """Direct unit tests for _resolve_auto_side's full branch table: the
    supervisor --device flag wins first, then live GPU detection decides
    for an 'auto' supervisor, with mps and any probe exception both
    grouping to the CPU side. The breakdown tests above exercise this
    only through the cuda/cpu --device flags; these pin the auto-detection
    and exception branches the flags can't reach."""

    def test_cuda_flag_resolves_gpu(self):
        set_supervisor_state(SupervisorState(workers=[], device="cuda"))
        from muse.admin.routes.memory import _resolve_auto_side
        assert _resolve_auto_side() == "gpu"

    def test_cpu_flag_resolves_cpu(self):
        set_supervisor_state(SupervisorState(workers=[], device="cpu"))
        from muse.admin.routes.memory import _resolve_auto_side
        assert _resolve_auto_side() == "cpu"

    def test_mps_flag_resolves_cpu(self):
        set_supervisor_state(SupervisorState(workers=[], device="mps"))
        from muse.admin.routes.memory import _resolve_auto_side
        assert _resolve_auto_side() == "cpu"

    def test_auto_flag_with_gpu_detected_resolves_gpu(self, monkeypatch):
        set_supervisor_state(SupervisorState(workers=[], device="auto"))
        from muse.core import memory_probe
        monkeypatch.setattr(memory_probe, "gpu_free_gb", lambda: 12.0)
        from muse.admin.routes.memory import _resolve_auto_side
        assert _resolve_auto_side() == "gpu"

    def test_auto_flag_without_gpu_resolves_cpu(self, monkeypatch):
        set_supervisor_state(SupervisorState(workers=[], device="auto"))
        from muse.core import memory_probe
        monkeypatch.setattr(memory_probe, "gpu_free_gb", lambda: None)
        from muse.admin.routes.memory import _resolve_auto_side
        assert _resolve_auto_side() == "cpu"

    def test_auto_flag_probe_exception_resolves_cpu(self, monkeypatch):
        set_supervisor_state(SupervisorState(workers=[], device="auto"))
        from muse.core import memory_probe

        def boom():
            raise RuntimeError("pynvml exploded")

        monkeypatch.setattr(memory_probe, "gpu_free_gb", boom)
        from muse.admin.routes.memory import _resolve_auto_side
        assert _resolve_auto_side() == "cpu"


class TestRecentDecisions:
    """Task G: /v1/admin/memory exposes the director's recent_decisions
    deque (last 20 load/evict events) for operational visibility.

    Each entry is a dict shaped like the serialized DecisionLogEntry
    with an ISO-8601 timestamp string instead of the raw float.
    """

    def test_no_director_returns_empty_recent_decisions(
        self, client, headers, monkeypatch,
    ):
        # Bare SupervisorState (no director) should still produce a
        # well-formed envelope; recent_decisions = [] keeps clients
        # from having to guard against missing fields.
        monkeypatch.setitem(sys.modules, "psutil", None)
        monkeypatch.setitem(sys.modules, "pynvml", None)
        set_supervisor_state(SupervisorState(workers=[], device="cpu"))

        r = client.get("/v1/admin/memory", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert "recent_decisions" in body
        assert body["recent_decisions"] == []

    def test_director_recent_decisions_serialized(
        self, client, headers, monkeypatch,
    ):
        # Populate the director with a couple decisions and verify they
        # land in the response with the right shape.
        from muse.cli_impl.load_director import DecisionLogEntry, LoadDirector

        monkeypatch.setitem(sys.modules, "psutil", None)
        monkeypatch.setitem(sys.modules, "pynvml", None)

        director = LoadDirector(
            enable_fn=lambda mid: 9001,
            disable_fn=lambda mid: None,
            memory_probe=type("P", (), {
                "gpu_free_gb": staticmethod(lambda: 32.0),
                "cpu_free_gb": staticmethod(lambda: 64.0),
            })(),
        )
        director.recent_decisions.append(DecisionLogEntry(
            timestamp=1700000000.0,
            model_id="kokoro-82m",
            action="load",
            memory_gb=0.5,
            free_before_gb=10.0,
            free_after_gb=9.5,
            reason="fit",
            evicted=[],
        ))
        director.recent_decisions.append(DecisionLogEntry(
            timestamp=1700000001.5,
            model_id="sd-turbo",
            action="evict",
            memory_gb=4.0,
            free_before_gb=2.0,
            free_after_gb=6.0,
            reason="evicted_for_newcomer",
            evicted=["sd-turbo"],
        ))

        state = SupervisorState(workers=[], device="cpu")
        state.director = director
        set_supervisor_state(state)

        r = client.get("/v1/admin/memory", headers=headers)
        assert r.status_code == 200
        body = r.json()

        decisions = body["recent_decisions"]
        assert len(decisions) == 2

        d0 = decisions[0]
        assert d0["model_id"] == "kokoro-82m"
        assert d0["action"] == "load"
        assert d0["memory_gb"] == 0.5
        assert d0["free_before_gb"] == 10.0
        assert d0["free_after_gb"] == 9.5
        assert d0["reason"] == "fit"
        assert d0["evicted"] == []
        # ISO-format timestamp string, not a raw float.
        assert isinstance(d0["timestamp"], str)
        assert "T" in d0["timestamp"]  # ISO-8601 with T separator

        d1 = decisions[1]
        assert d1["model_id"] == "sd-turbo"
        assert d1["action"] == "evict"
        assert d1["evicted"] == ["sd-turbo"]

    def test_recent_decisions_limited_to_last_20(
        self, client, headers, monkeypatch,
    ):
        # The deque is maxlen=20 by construction; even if we push 25
        # entries, the response should only contain the most recent 20.
        from muse.cli_impl.load_director import DecisionLogEntry, LoadDirector

        monkeypatch.setitem(sys.modules, "psutil", None)
        monkeypatch.setitem(sys.modules, "pynvml", None)

        director = LoadDirector(
            enable_fn=lambda mid: 9001,
            disable_fn=lambda mid: None,
            memory_probe=type("P", (), {
                "gpu_free_gb": staticmethod(lambda: 32.0),
                "cpu_free_gb": staticmethod(lambda: 64.0),
            })(),
        )
        for i in range(25):
            director.recent_decisions.append(DecisionLogEntry(
                timestamp=1700000000.0 + i,
                model_id=f"model-{i}",
                action="load",
                memory_gb=0.5,
                free_before_gb=10.0,
                free_after_gb=9.5,
                reason="fit",
                evicted=[],
            ))

        state = SupervisorState(workers=[], device="cpu")
        state.director = director
        set_supervisor_state(state)

        r = client.get("/v1/admin/memory", headers=headers)
        body = r.json()

        # Deque maxlen=20 caps the count; we get the most recent 20.
        assert len(body["recent_decisions"]) == 20
        # First in the response is model-5 (the oldest of the surviving 20).
        assert body["recent_decisions"][0]["model_id"] == "model-5"
        # Last is model-24.
        assert body["recent_decisions"][-1]["model_id"] == "model-24"

    def test_handles_none_free_after_gb(self, client, headers, monkeypatch):
        # During an eviction round the partial DecisionLogEntry has
        # free_after_gb=None (set later when the poll completes). The
        # serializer must handle None without crashing.
        from muse.cli_impl.load_director import DecisionLogEntry, LoadDirector

        monkeypatch.setitem(sys.modules, "psutil", None)
        monkeypatch.setitem(sys.modules, "pynvml", None)

        director = LoadDirector(
            enable_fn=lambda mid: 9001,
            disable_fn=lambda mid: None,
            memory_probe=type("P", (), {
                "gpu_free_gb": staticmethod(lambda: 32.0),
                "cpu_free_gb": staticmethod(lambda: 64.0),
            })(),
        )
        director.recent_decisions.append(DecisionLogEntry(
            timestamp=1700000000.0,
            model_id="evicting",
            action="evict",
            memory_gb=4.0,
            free_before_gb=1.0,
            free_after_gb=None,
            reason="evicted_for_newcomer",
            evicted=["evicting"],
        ))

        state = SupervisorState(workers=[], device="cpu")
        state.director = director
        set_supervisor_state(state)

        r = client.get("/v1/admin/memory", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert body["recent_decisions"][0]["free_after_gb"] is None
