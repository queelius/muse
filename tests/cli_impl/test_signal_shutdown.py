"""Real signals must unwind Muse cleanup, not just stop Uvicorn's socket."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import httpx
import psutil
import pytest

from muse.core.resource_registry import list_resources


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.name != "posix", reason="POSIX subprocess signal semantics"),
]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def _child(tmp_path, command, env):
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            yield proc, log_path
        finally:
            # Only this test's Popen child is signalled, never a host process.
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)


def _wait_ready(proc, url, log_path):
    deadline = time.monotonic() + 20
    with httpx.Client(timeout=0.5, trust_env=False) as client:
        while time.monotonic() < deadline:
            assert proc.poll() is None, log_path.read_text()
            try:
                if client.get(url).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
    pytest.fail(f"test server did not become ready:\n{log_path.read_text()}")


@pytest.mark.parametrize(
    ("shutdown_signal", "custom_handler"),
    [(signal.SIGTERM, False), (signal.SIGTERM, True), (signal.SIGINT, False)],
    ids=["sigterm-default", "sigterm-custom", "sigint"],
)
def test_handled_signal_runs_outer_finally_and_restores_handlers(
    tmp_path, shutdown_signal, custom_handler,
):
    """Replaying SIGTERM kills the caller or spuriously invokes its handler."""
    port = _free_port()
    marker = tmp_path / "cleanup.json"
    script = textwrap.dedent(f"""
        import json
        import signal
        from pathlib import Path
        from fastapi import FastAPI
        from muse.cli_impl.serve_util import run_uvicorn

        calls = []
        if {custom_handler!r}:
            signal.signal(signal.SIGTERM, lambda sig, frame: calls.append(sig))
        previous = {{sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}}
        try:
            run_uvicorn(FastAPI(), host="127.0.0.1", port={port})
        finally:
            Path({str(marker)!r}).write_text(json.dumps({{
                "cleaned": True,
                "handlers_restored": all(signal.getsignal(sig) == old for sig, old in previous.items()),
                "previous_handler_calls": calls,
            }}))
    """)
    with _child(tmp_path, [sys.executable, "-c", script], os.environ.copy()) as (proc, log):
        _wait_ready(proc, f"http://127.0.0.1:{port}/docs", log)
        proc.send_signal(shutdown_signal)
        proc.wait(timeout=10)
        assert marker.exists(), f"outer cleanup was skipped (exit {proc.returncode}):\n{log.read_text()}"
        assert json.loads(marker.read_text()) == {
            "cleaned": True, "handlers_restored": True, "previous_handler_calls": [],
        }
        assert proc.returncode == 0, log.read_text()
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", port)) != 0


@pytest.mark.parametrize("first_signal", [signal.SIGTERM, signal.SIGINT], ids=["sigterm", "sigint"])
def test_second_sigint_still_forces_exit_with_an_active_request(tmp_path, first_signal):
    """Consuming handled signals must not disable the second-Ctrl-C escape."""
    port = _free_port()
    active = tmp_path / "request-active"
    cleaned = tmp_path / "cleaned"
    script = textwrap.dedent(f"""
        import asyncio
        import threading
        from pathlib import Path
        from fastapi import FastAPI
        from muse.cli_impl.serve_util import run_uvicorn

        app = FastAPI()

        @app.get("/slow")
        async def slow():
            Path({str(active)!r}).touch()
            await asyncio.Event().wait()

        try:
            run_uvicorn(app, host="127.0.0.1", port={port}, shutdown_event=threading.Event())
        finally:
            Path({str(cleaned)!r}).touch()
    """)
    env = dict(os.environ, MUSE_SHUTDOWN_GRACE_SECONDS="30")
    with _child(tmp_path, [sys.executable, "-c", script], env) as (proc, log):
        _wait_ready(proc, f"http://127.0.0.1:{port}/docs", log)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(httpx.get, f"http://127.0.0.1:{port}/slow", timeout=10, trust_env=False)
            deadline = time.monotonic() + 5
            while not active.exists() and time.monotonic() < deadline:
                assert proc.poll() is None, log.read_text()
                time.sleep(0.01)
            assert active.exists(), log.read_text()
            proc.send_signal(first_signal)
            # Socket closure proves the first handler has completed and
            # Uvicorn entered draining; an event set before handle_exit's
            # delegation would race the second signal against should_exit.
            deadline = time.monotonic() + 5
            while True:
                assert proc.poll() is None, log.read_text()
                with socket.socket() as sock:
                    sock.settimeout(0.1)
                    if sock.connect_ex(("127.0.0.1", port)) != 0:
                        break
                assert time.monotonic() < deadline, log.read_text()
                time.sleep(0.01)
            # The first signal is waiting on /slow with a 30-second grace.
            # A second Ctrl-C must release it without waiting that long.
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=5)
            future.exception(timeout=5)
        assert cleaned.exists(), log.read_text()
        assert proc.returncode == 0, log.read_text()


@pytest.mark.parametrize("shutdown_signal", [signal.SIGTERM, signal.SIGINT], ids=["sigterm", "sigint"])
@pytest.mark.timeout(60)
def test_supervisor_signal_releases_worker_ports_and_registry(tmp_path, shutdown_signal):
    """The supervisor's real finally must unregister itself and its worker."""
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    weights = tmp_path / "weights"
    weights.mkdir()
    (models_dir / "signal_model.py").write_text(textwrap.dedent("""
        from muse.modalities.embedding_text.protocol import EmbeddingResult

        MANIFEST = {
            "model_id": "signal-test-model", "modality": "embedding/text",
            "hf_repo": "test/signal", "capabilities": {"device": "cpu", "memory_gb": 0.001},
        }

        class Model:
            dimensions = 2

            def __init__(self, model_id, **kwargs):
                self.model_id = model_id

            def embed(self, input, **kwargs):
                return EmbeddingResult(
                    embeddings=[[0.25, 0.75]], dimensions=2,
                    model_id=self.model_id, prompt_tokens=1,
                )
    """))
    (catalog_dir / "catalog.json").write_text(json.dumps({
        "signal-test-model": {
            "enabled": True, "local_dir": str(weights),
            "python_path": sys.executable,
        },
    }))
    env = dict(
        os.environ, MUSE_CATALOG_DIR=str(catalog_dir), MUSE_MODELS_DIR=str(models_dir),
        MUSE_ADMIN_TOKEN="signal-test-token", MUSE_TELEMETRY_ENABLED="0",
        MUSE_SHUTDOWN_GRACE_SECONDS="1", CUDA_VISIBLE_DEVICES="",
    )
    port = _free_port()
    worker = None
    command = [sys.executable, "-m", "muse.cli", "serve", "--host", "127.0.0.1", "--port", str(port), "--device", "cpu"]
    with _child(tmp_path, command, env) as (proc, log):
        try:
            _wait_ready(proc, f"http://127.0.0.1:{port}/health", log)
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=20, trust_env=False) as client:
                response = client.post("/v1/embeddings", json={"model": "signal-test-model", "input": "hello"})
                assert response.status_code == 200, f"{response.text}\n{log.read_text()}"
                assert response.json()["data"][0]["embedding"] == [0.25, 0.75]
                response = client.get("/v1/admin/workers", headers={"Authorization": "Bearer signal-test-token"})
                response.raise_for_status()
                spec, = response.json()["workers"]
            worker = psutil.Process(spec["pid"])
            assert worker.ppid() == proc.pid
            assert {record.pid for record in list_resources(catalog_dir=catalog_dir)} == {proc.pid, worker.pid}

            proc.send_signal(shutdown_signal)
            proc.wait(timeout=15)
            deadline = time.monotonic() + 5
            while worker.is_running() and worker.status() != psutil.STATUS_ZOMBIE and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not worker.is_running() or worker.status() == psutil.STATUS_ZOMBIE
            assert list_resources(catalog_dir=catalog_dir) == [], log.read_text()
            assert proc.returncode == 0, log.read_text()
            for closed_port in (port, spec["port"]):
                with socket.socket() as sock:
                    assert sock.connect_ex(("127.0.0.1", closed_port)) != 0
        finally:
            # The red regression may leave the test worker for its watchdog.
            # psutil guards against PID reuse; no process-table scan is used.
            if worker is not None and worker.is_running():
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except psutil.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
