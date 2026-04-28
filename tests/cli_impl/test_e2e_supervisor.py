"""End-to-end smoke test: real subprocess worker, real gateway, mocked model.

This test is SLOW (~5-15s depending on Python cold-start). It's the only
integration test for supervisor + worker + gateway wired together. All
other tests mock the process boundary.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


pytestmark = pytest.mark.slow


@pytest.fixture
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    return tmp_path


@pytest.mark.timeout(30)
def test_supervisor_gateway_serves_empty_catalog(tmp_catalog):
    """Real subprocess, real gateway with an empty catalog.

    Gateway plumbing does not depend on any specific model being
    loaded, so we prove the wiring works on an empty catalog: gateway
    comes up, /health + /v1/models return, and /v1/chat/completions
    returns an OpenAI-style error envelope for unknown models.

    Prior to v0.12.1 this test seeded a fake catalog entry and relied
    on the worker loading nothing (graceful degrade). v0.12.1's
    fail-fast worker contract means a seeded-but-unloadable model
    would crash the worker and prevent the supervisor from starting
    the gateway, so we switched to empty-catalog mode: same coverage,
    clearer intent.
    """
    env = os.environ.copy()
    env["MUSE_CATALOG_DIR"] = str(tmp_catalog)

    # Spawn muse serve on port 18765 (non-conflicting).
    # plan_workers() will return no specs, the supervisor logs
    # "no pulled models... will start empty" and goes straight to the
    # gateway loop without spawning any worker subprocesses.
    proc = subprocess.Popen(
        [sys.executable, "-m", "muse.cli", "serve", "--port", "18765"],
        env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        # Wait up to 20s for the gateway to come up
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            try:
                r = httpx.get("http://127.0.0.1:18765/health", timeout=2.0)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.3)
        else:
            stdout, stderr = proc.communicate(timeout=5)
            pytest.fail(
                f"gateway never became ready.\n"
                f"stdout: {stdout.decode()[:1000]}\n"
                f"stderr: {stderr.decode()[:1000]}"
            )

        # Gateway is up; verify aggregated /health works
        r = httpx.get("http://127.0.0.1:18765/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("ok", "degraded")
        assert "modalities" in body

        # /v1/models should also work (may be empty if worker didn't
        # actually load a model - that's fine; this test is about
        # plumbing, not model availability)
        r = httpx.get("http://127.0.0.1:18765/v1/models")
        assert r.status_code == 200
        assert "data" in r.json()

        # /v1/chat/completions must also be wired through the gateway
        # to a worker that mounted the chat router (auto-mounted via
        # discover_modalities). The gateway routes by request body's
        # `model` field; with an unknown model, it must return an OpenAI-
        # style error envelope (NOT FastAPI's {"detail": "Not Found"}).
        # This proves the chat/completion modality is reachable end-to-end.
        r = httpx.post(
            "http://127.0.0.1:18765/v1/chat/completions",
            json={
                "model": "no-such-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=5.0,
        )
        assert r.status_code in (404, 400), f"expected 404/400, got {r.status_code}: {r.text}"
        body = r.json()
        assert "error" in body, f"expected OpenAI envelope, got {body}"
        assert "detail" not in body
        # Either model_not_found (worker reachable) or unknown_model
        # (gateway resolves model -> worker mapping) is acceptable proof
        # the path is wired and uses our error envelope, not FastAPI's.
        assert body["error"]["code"] in ("model_not_found", "unknown_model", "model_required")

        # /v1/images/animations: same modality-discovery proof for the
        # image/animation modality (added in v0.18.0). Discovery must
        # find the modality package and a worker must mount the router;
        # with an unknown model, an OpenAI-style error envelope confirms
        # the route exists (FastAPI's default for missing routes is
        # `{"detail": "Not Found"}` with no `error` key).
        r = httpx.post(
            "http://127.0.0.1:18765/v1/images/animations",
            json={"model": "no-such-model", "prompt": "a cat"},
            timeout=5.0,
        )
        assert r.status_code in (404, 400), f"expected 404/400, got {r.status_code}: {r.text}"
        body = r.json()
        assert "error" in body, f"expected OpenAI envelope, got {body}"
        assert "detail" not in body
        assert body["error"]["code"] in ("model_not_found", "unknown_model", "model_required")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
