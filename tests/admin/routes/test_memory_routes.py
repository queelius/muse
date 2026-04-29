"""Tests for GET /v1/admin/memory."""
from __future__ import annotations

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
        assert body == {"gpu": None, "cpu": None}

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
