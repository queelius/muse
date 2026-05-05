"""End-to-end slow tests for v0.40.0 lazy load + LRU eviction.

Task J of the v0.40.0 plan. Unit tests in `test_load_director.py`,
`test_gateway_lazy.py`, and `test_supervisor_lazy.py` cover each layer
in isolation. These tests verify the cross-cutting integration:

  - LoadDirector + gateway + catalog all wired together.
  - Two-model fit, two-model tight-budget swap, unservable short-circuit,
    and the boot-speed regression watchdog.

Approach: most tests run in-process. We instantiate a real LoadDirector
with mocked `enable_fn` / `disable_fn` (no subprocess spawn) plus a
mocked memory probe whose return values can be reprogrammed mid-test
to simulate cold-load memory consumption + eviction-driven release.
The gateway is built with `build_gateway(state=...)` and driven through
a FastAPI TestClient. `httpx.AsyncClient` inside the gateway is patched
to short-circuit forwarding so no real worker is contacted.

The boot-speed test (J4) is the only subprocess-spawning case: it
launches `muse serve` and asserts the `/health` endpoint becomes
reachable within 1 second of startup. This is the regression watchdog
against accidental re-introduction of eager-boot worker spawning.

Why slow lane:
  - The in-process tests build real LoadDirector instances + run the
    actual three-phase acquire (decide / load / commit) including
    threading.Event coordination. A mock-the-director approach lives in
    `test_gateway_lazy.py`; this file's contract is "the real director
    plus the real gateway plus the real catalog seam reproduce the
    documented end-to-end flow."
  - The subprocess test spawns `muse serve` for ~1-2s.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from muse.cli_impl.gateway import build_gateway
from muse.cli_impl.load_director import LoadDirector
from muse.cli_impl.supervisor import (
    SupervisorState,
    clear_supervisor_state,
)


pytestmark = pytest.mark.slow


# ----------------------------------------------------------------------
# Shared fixtures + helpers
# ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_supervisor_state():
    """Belt-and-suspenders: nothing in this file registers a global
    SupervisorState, but other test files in the slow lane might leave
    one behind. Clear before + after to keep each test hermetic.
    """
    clear_supervisor_state()
    yield
    clear_supervisor_state()


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    """Isolated catalog dir per test; resets module-level cache."""
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    from muse.core.catalog import _reset_known_models_cache
    _reset_known_models_cache()
    yield tmp_path
    _reset_known_models_cache()


def _seed_catalog(data: dict) -> None:
    """Write `data` to the active catalog path and reset the discovery cache."""
    from muse.core.catalog import _catalog_path, _reset_known_models_cache
    p = _catalog_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
    _reset_known_models_cache()


def _manifest(model_id: str, *, memory_gb: float, device: str = "cpu") -> dict:
    """Minimal manifest the LoadDirector needs to route + size a load."""
    return {
        "model_id": model_id,
        "modality": "audio/speech",
        "capabilities": {"memory_gb": memory_gb, "device": device},
    }


def _wire_async_client_json(
    mock_client_cls: MagicMock,
    *,
    response_body: bytes = b'{"ok": true}',
    response_status: int = 200,
    response_content_type: str = "application/json",
) -> dict:
    """Patch httpx.AsyncClient so the gateway's forward sees a 200 JSON.

    Returns a dict carrying the captured `url` of each forwarded request
    so the test can assert which worker port the gateway routed to.
    """
    captured: dict = {"urls": []}

    def _capture_stream(method, url, **kwargs):
        captured["urls"].append(url)
        mock_response = MagicMock()
        mock_response.status_code = response_status
        mock_response.headers = {"content-type": response_content_type}
        mock_response.aread = AsyncMock(return_value=response_body)
        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        stream_ctx.__aexit__ = AsyncMock(return_value=None)
        return stream_ctx

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.aclose = AsyncMock()
    mock_client.stream = MagicMock(side_effect=_capture_stream)
    mock_client_cls.return_value = mock_client

    return captured


# ----------------------------------------------------------------------
# J1: two small models both fit in budget
# ----------------------------------------------------------------------


class TestTwoModelsBothFit:
    """When both catalog rows fit live free memory, lazy-load serves
    requests to either model without eviction. Both end up in
    director.loaded after the second request.
    """

    def test_both_models_loaded_no_eviction(self, tmp_catalog):
        # Memory budget: 16 GB CPU. Models declared at 0.5 GB and 1.0 GB.
        # Headroom 2 GB. Free 16 - 2 = 14 GB available; both fit easily.
        cpu_free_state = {"value": 16.0}
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None  # CPU-only host
        probe.cpu_free_gb.side_effect = lambda: cpu_free_state["value"]

        # enable_fn returns distinct ports per model; loads consume memory.
        port_map = {"model-a": 9001, "model-b": 9002}
        memory_per_load = {"model-a": 0.5, "model-b": 1.0}

        def enable_side_effect(model_id: str) -> int:
            cpu_free_state["value"] -= memory_per_load[model_id]
            return port_map[model_id]

        enable_fn = MagicMock(side_effect=enable_side_effect)
        disable_fn = MagicMock()  # never called in this test

        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            memory_probe=probe,
            cpu_headroom_gb=2.0,
        )

        state = SupervisorState(workers=[], device="cpu")
        state.director = director

        manifest_a = _manifest("model-a", memory_gb=0.5)
        manifest_b = _manifest("model-b", memory_gb=1.0)

        def _patch_get_manifest(model_id: str):
            mapping = {"model-a": manifest_a, "model-b": manifest_b}
            return mapping[model_id]

        app = build_gateway(state=state)
        client = TestClient(app)

        with patch(
            "muse.cli_impl.gateway.get_manifest",
            side_effect=_patch_get_manifest,
        ), patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            captured = _wire_async_client_json(mock_cls)

            r1 = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "model-a"},
            )
            assert r1.status_code == 200
            # First request loaded model-a + forwarded to its port.
            assert captured["urls"][0] == "http://127.0.0.1:9001/v1/audio/speech"

            r2 = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "model-b"},
            )
            assert r2.status_code == 200
            # Second request loaded model-b + forwarded to its port.
            assert captured["urls"][1] == "http://127.0.0.1:9002/v1/audio/speech"

        # Both models in director.loaded; neither was evicted.
        snapshot = director.status()
        assert "model-a" in snapshot
        assert "model-b" in snapshot
        assert disable_fn.call_count == 0
        assert enable_fn.call_count == 2

    def test_second_request_to_already_loaded_model_is_hot(self, tmp_catalog):
        """A second request to the SAME model is hot: enable_fn is NOT
        called again. Refcount returns to 0 between requests (release fires
        in the gateway's finally clause).
        """
        cpu_free_state = {"value": 16.0}
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.side_effect = lambda: cpu_free_state["value"]

        def enable_side_effect(model_id: str) -> int:
            cpu_free_state["value"] -= 0.5
            return 9001

        enable_fn = MagicMock(side_effect=enable_side_effect)
        disable_fn = MagicMock()

        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            memory_probe=probe,
            cpu_headroom_gb=2.0,
        )

        state = SupervisorState(workers=[], device="cpu")
        state.director = director

        manifest_a = _manifest("model-a", memory_gb=0.5)
        app = build_gateway(state=state)
        client = TestClient(app)

        with patch(
            "muse.cli_impl.gateway.get_manifest",
            return_value=manifest_a,
        ), patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_async_client_json(mock_cls)
            # Three requests; only the first triggers enable_fn.
            for _ in range(3):
                r = client.post(
                    "/v1/audio/speech",
                    json={"input": "hi", "model": "model-a"},
                )
                assert r.status_code == 200

        # enable_fn called exactly once; disable_fn never.
        assert enable_fn.call_count == 1
        assert disable_fn.call_count == 0
        # After three release calls, refcount is back to 0 (no leaks).
        snapshot = director.status()
        assert snapshot["model-a"]["refcount"] == 0


# ----------------------------------------------------------------------
# J2: tight budget forces alternating swap
# ----------------------------------------------------------------------


class TestTightBudgetForcesSwap:
    """Two models that don't both fit. Alternating requests A -> B -> A -> B
    each trigger an eviction of the previously loaded model and a fresh
    load of the new one. All four requests must succeed (200), and each
    swap must be visible in director.recent_decisions.
    """

    def test_alternating_requests_swap_one_at_a_time(self, tmp_catalog):
        # Budget: cpu_free = 8 GB, headroom = 2 GB. Available = 6 GB.
        # Each model claims 5 GB. Loading one drops free to 3 GB.
        # Loading the second on top would need (5 - (3 - 2)) = 4 GB
        # shortfall, so the LRU candidate (the first model) gets evicted.
        cpu_free_state = {"value": 8.0}
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.side_effect = lambda: cpu_free_state["value"]

        port_map = {"model-a": 9001, "model-b": 9002}

        def enable_side_effect(model_id: str) -> int:
            cpu_free_state["value"] -= 5.0
            return port_map[model_id]

        enable_fn = MagicMock(side_effect=enable_side_effect)

        evict_calls: list[str] = []

        def disable_side_effect(model_id: str) -> None:
            evict_calls.append(model_id)
            # SIGTERM-driven release simulation: the OS reclaims 5 GB.
            cpu_free_state["value"] += 5.0

        disable_fn = MagicMock(side_effect=disable_side_effect)

        director = LoadDirector(
            enable_fn=enable_fn,
            disable_fn=disable_fn,
            memory_probe=probe,
            cpu_headroom_gb=2.0,
        )

        state = SupervisorState(workers=[], device="cpu")
        state.director = director

        manifest_a = _manifest("model-a", memory_gb=5.0)
        manifest_b = _manifest("model-b", memory_gb=5.0)

        def _patch_get_manifest(model_id: str):
            return {"model-a": manifest_a, "model-b": manifest_b}[model_id]

        app = build_gateway(state=state)
        client = TestClient(app)

        with patch(
            "muse.cli_impl.gateway.get_manifest",
            side_effect=_patch_get_manifest,
        ), patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            captured = _wire_async_client_json(mock_cls)

            # Sequence: A -> B -> A -> B. Each new model evicts the
            # previous. release fires in the gateway's finally clause so
            # the previously loaded model becomes refcount==0 (evictable)
            # by the time the next request arrives.
            sequence = ["model-a", "model-b", "model-a", "model-b"]
            for model_id in sequence:
                r = client.post(
                    "/v1/audio/speech",
                    json={"input": "hi", "model": model_id},
                )
                assert r.status_code == 200, (
                    f"request to {model_id} failed: {r.status_code} {r.text}"
                )

        # All four requests succeeded (already asserted inline).
        # Each request after the first triggered exactly one eviction.
        # First request: cold load A, no eviction.
        # Second: evict A, load B.
        # Third: evict B, load A.
        # Fourth: evict A, load B.
        assert evict_calls == ["model-a", "model-b", "model-a"]

        # enable_fn called once per cold load = 4 times (A, B, A, B).
        # The director re-loads on every alternating request because the
        # previous model was evicted, so each request misses the hot path.
        assert enable_fn.call_count == 4

        # Final state: only model-b is loaded.
        snapshot = director.status()
        assert "model-b" in snapshot
        assert "model-a" not in snapshot

        # The four forwarded URLs alternate between the two ports as the
        # director swaps models on each request.
        assert captured["urls"] == [
            "http://127.0.0.1:9001/v1/audio/speech",
            "http://127.0.0.1:9002/v1/audio/speech",
            "http://127.0.0.1:9001/v1/audio/speech",
            "http://127.0.0.1:9002/v1/audio/speech",
        ]

    def test_swap_writes_decision_log_entries(self, tmp_catalog):
        """Each swap appends an evict + load entry to director.recent_decisions
        so /v1/admin/memory can surface the load behavior to operators.
        """
        cpu_free_state = {"value": 8.0}
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.side_effect = lambda: cpu_free_state["value"]

        def enable_side_effect(model_id: str) -> int:
            cpu_free_state["value"] -= 5.0
            return {"model-a": 9001, "model-b": 9002}[model_id]

        def disable_side_effect(model_id: str) -> None:
            cpu_free_state["value"] += 5.0

        director = LoadDirector(
            enable_fn=MagicMock(side_effect=enable_side_effect),
            disable_fn=MagicMock(side_effect=disable_side_effect),
            memory_probe=probe,
            cpu_headroom_gb=2.0,
        )

        state = SupervisorState(workers=[], device="cpu")
        state.director = director

        manifest_a = _manifest("model-a", memory_gb=5.0)
        manifest_b = _manifest("model-b", memory_gb=5.0)

        def _patch_get_manifest(model_id: str):
            return {"model-a": manifest_a, "model-b": manifest_b}[model_id]

        app = build_gateway(state=state)
        client = TestClient(app)

        with patch(
            "muse.cli_impl.gateway.get_manifest",
            side_effect=_patch_get_manifest,
        ), patch("muse.cli_impl.gateway.httpx.AsyncClient") as mock_cls:
            _wire_async_client_json(mock_cls)
            client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "model-a"},
            )
            client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "model-b"},
            )

        # Decision log: load(model-a), evict(model-a), load(model-b).
        actions = [(d.action, d.model_id) for d in director.recent_decisions]
        assert ("load", "model-a") in actions
        assert ("evict", "model-a") in actions
        assert ("load", "model-b") in actions


# ----------------------------------------------------------------------
# J3: unservable model 503s without calling acquire
# ----------------------------------------------------------------------


class TestUnservable503ImmediatelyAtBoot:
    """A catalog model with no memory data + no probe data is flagged
    unservable by `validate_catalog_at_boot`. The gateway short-circuits
    503 BEFORE calling director.acquire, so there's no load attempt.
    """

    def test_unservable_returns_503_director_acquire_never_called(
        self, tmp_catalog,
    ):
        from muse.cli_impl.supervisor import validate_catalog_at_boot

        # Seed a catalog row with no `capabilities.memory_gb` AND no
        # `measurements`. The boot validation MUST flag it unservable.
        _seed_catalog({
            "no-memory-data": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "venv_path": "/v",
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "no-memory-data",
                    "modality": "audio/speech",
                    "hf_repo": "x/y",
                    "backend_path": "muse.core.runtime:Whatever",
                    "capabilities": {},  # no memory_gb
                },
            },
        })

        # Real director with a mock that records whether acquire was ever
        # called. The gateway's unservable short-circuit must keep it
        # uncalled.
        director_mock = MagicMock()

        state = SupervisorState(workers=[], device="cpu")
        state.director = director_mock

        # Run the actual boot validation seam against the seeded catalog.
        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 16.0
        validate_catalog_at_boot(state, memory_probe=probe)

        # Boot validation flagged the model unservable.
        assert "no-memory-data" in state.unservable_reasons
        assert "no memory estimate" in state.unservable_reasons["no-memory-data"]

        app = build_gateway(state=state)
        client = TestClient(app)

        with patch(
            "muse.cli_impl.gateway.get_manifest",
            return_value=_manifest("no-memory-data", memory_gb=0.5),
        ):
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "no-memory-data"},
            )

        # 503 with the OpenAI envelope shape.
        assert r.status_code == 503
        body = r.json()
        assert "error" in body
        assert "detail" not in body
        assert body["error"]["code"] == "model_unservable"
        # The unservable_reason text is surfaced to the client.
        assert "no memory estimate" in body["error"]["message"]
        # Crucial: the director was NEVER asked to acquire this model.
        director_mock.acquire.assert_not_called()
        director_mock.release.assert_not_called()

    def test_unservable_reason_for_oversized_model(self, tmp_catalog):
        """Second flavor of unservable: declared memory_gb exceeds host
        capacity. Gateway 503s with the device-capacity reason text.
        """
        from muse.cli_impl.supervisor import validate_catalog_at_boot

        _seed_catalog({
            "huge-model": {
                "pulled_at": "...",
                "hf_repo": "x/y",
                "local_dir": "/tmp/x",
                "venv_path": "/v",
                "python_path": "/v/bin/python",
                "enabled": True,
                "manifest": {
                    "model_id": "huge-model",
                    "modality": "audio/speech",
                    "hf_repo": "x/y",
                    "backend_path": "muse.core.runtime:Whatever",
                    "capabilities": {
                        "memory_gb": 200.0,  # way over any sane host
                        "device": "cpu",
                    },
                },
            },
        })

        director_mock = MagicMock()
        state = SupervisorState(workers=[], device="cpu")
        state.director = director_mock

        probe = MagicMock()
        probe.gpu_free_gb.return_value = None
        probe.cpu_free_gb.return_value = 16.0
        validate_catalog_at_boot(state, memory_probe=probe)

        assert "huge-model" in state.unservable_reasons
        assert "exceeds device capacity" in state.unservable_reasons["huge-model"]

        app = build_gateway(state=state)
        client = TestClient(app)

        with patch(
            "muse.cli_impl.gateway.get_manifest",
            return_value=_manifest("huge-model", memory_gb=200.0),
        ):
            r = client.post(
                "/v1/audio/speech",
                json={"input": "hi", "model": "huge-model"},
            )

        assert r.status_code == 503
        body = r.json()
        assert body["error"]["code"] == "model_unservable"
        assert "exceeds device capacity" in body["error"]["message"]
        director_mock.acquire.assert_not_called()


# ----------------------------------------------------------------------
# J4: regression watchdog -- `muse serve` boots quickly with empty catalog
# ----------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_muse_serve_starts_quickly_with_empty_catalog(tmp_path, monkeypatch):
    """Regression watchdog for the v0.40.0 instant-boot guarantee.

    Historical behavior (pre-v0.40.0): supervisor ran plan_workers,
    spawned every enabled model's worker, and waited for at least one
    to pass /health before starting the gateway. With six enabled
    models that took 30-60s of dead-air startup.

    v0.40.0 contract: the gateway is reachable as soon as uvicorn binds
    the port. No worker spawn at boot. The first request to a model
    triggers the cold load.

    This test sets a 5-second budget from subprocess spawn to /health=200.
    The threshold's job is to catch a re-introduction of eager-boot
    worker spawning, which would push the boot well past 30 seconds.
    A pure cold-Python `muse serve` against an empty catalog lands at
    ~1.5-2.5s on a developer laptop (Python interpreter cold + module
    imports + uvicorn binding); 5s is generous enough that CI containers
    won't flake but tight enough that any eager-boot regression trips it
    immediately. Using subprocess (not TestClient) because we need the
    real uvicorn boot path: TestClient doesn't exercise port binding,
    process startup, or lifespan hooks.

    Tolerance: 5.0s. Empirical runs on a developer laptop land at
    ~1-2s; CI containers may be slower but still well under 5s. If
    this test flakes upward, investigate the import-time cost regression
    (the gateway module's heavy-import budget) before relaxing the bound.
    """
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    env = os.environ.copy()
    env["MUSE_CATALOG_DIR"] = str(tmp_path)

    budget_seconds = 5.0
    # Pick an unusual port so no other test in the slow lane collides.
    proc = subprocess.Popen(
        [sys.executable, "-m", "muse.cli", "serve", "--port", "18766"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        spawn_started_at = time.monotonic()
        deadline = spawn_started_at + budget_seconds
        ready_at: float | None = None
        while time.monotonic() < deadline:
            try:
                r = httpx.get(
                    "http://127.0.0.1:18766/health",
                    timeout=0.5,
                )
                if r.status_code == 200:
                    ready_at = time.monotonic()
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.05)

        if ready_at is None:
            stdout, stderr = proc.communicate(timeout=5)
            pytest.fail(
                f"gateway did not become ready within {budget_seconds}s. "
                f"This is the v0.40.0 regression watchdog: a re-introduction "
                f"of eager-boot worker spawning will trip it.\n"
                f"stdout: {stdout.decode()[:1500]}\n"
                f"stderr: {stderr.decode()[:1500]}"
            )

        elapsed = ready_at - spawn_started_at
        assert elapsed < budget_seconds, (
            f"gateway took {elapsed:.3f}s to become ready; budget is "
            f"{budget_seconds}s. Investigate eager-boot regression."
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
