"""Real worker processes on a POSIX platform without ``os.waitid`` (macOS).

CPython before 3.13 never builds ``os.waitid`` on macOS, so the supervisor
cannot pin an exited worker leader with WNOWAIT there. Deleting
``os.waitid`` runs that exact code path on Linux. Every test also runs with
``waitid`` present to prove the pinned path is unchanged.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

import pytest

from muse.cli_impl import supervisor
from muse.cli_impl.supervisor import WorkerSpec
from muse.core import resource_registry

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")

_FAKE_WORKER = """\
import os, subprocess, sys, time
mode = os.environ["FAKE_WORKER_MODE"]
if mode == "exit":
    sys.exit(3)
if mode == "exit_soon":
    time.sleep(2.0)
    sys.exit(3)
if mode == "grandchild":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(os.environ["FAKE_WORKER_PIDFILE"], "w") as handle:
        handle.write(str(child.pid))
time.sleep(60)
"""


@pytest.fixture(params=["waitid", "no-waitid"])
def platform(request, monkeypatch):
    if request.param == "no-waitid":
        monkeypatch.delattr(os, "waitid", raising=False)
    elif not callable(getattr(os, "waitid", None)):
        pytest.skip("platform has no waitid")
    return request.param


@pytest.fixture
def fake_python(tmp_path):
    """An executable standing in for a venv python; ignores worker argv."""
    script = tmp_path / "fake-python"
    script.write_text(f"#!{sys.executable}\n{_FAKE_WORKER}")
    script.chmod(0o755)
    return str(script)


@pytest.fixture
def spec(fake_python):
    created = WorkerSpec(models=["fake-model"], python_path=fake_python, port=9)
    yield created
    # Never leave a real child behind, whatever the test outcome.
    with created.process_lock:
        proc = created.process
    if proc is not None and proc.returncode is None:
        proc.kill()
        proc.wait(timeout=10.0)


def _process_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        with open(f"/proc/{pid}/stat") as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return False


def _state(spec: WorkerSpec, proc) -> bool | None:
    with spec.process_lock:
        return supervisor._worker_process_state_locked(spec, proc)


def _assert_released(spec: WorkerSpec, proc) -> None:
    assert proc.returncode is not None
    assert spec.process is None
    assert spec.resource_id is None
    assert resource_registry.list_resources() == []


def test_spawned_worker_is_owned_and_torn_down(platform, spec, monkeypatch):
    monkeypatch.setenv("FAKE_WORKER_MODE", "sleep")
    supervisor.spawn_worker(spec, device="cpu")
    proc = spec.process
    assert spec.process_group_id == proc.pid
    assert spec.resource_id is not None

    supervisor._require_worker_generation_alive(spec, proc)
    assert _state(spec, proc) is True

    result = supervisor._shutdown_workers([spec], grace=2.0)
    assert result.complete
    _assert_released(spec, proc)


def test_fast_exiting_worker_reports_its_exit(platform, spec, monkeypatch):
    """The worker exits before its identity is registered."""
    import psutil

    real_register = supervisor.register_process

    def register_after_exit(**kwargs):
        deadline = time.monotonic() + 10.0
        while psutil.Process(kwargs["pid"]).status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline, "worker never exited"
            time.sleep(0.01)
        return real_register(**kwargs)

    monkeypatch.setenv("FAKE_WORKER_MODE", "exit")
    monkeypatch.setattr(supervisor, "register_process", register_after_exit)
    supervisor.spawn_worker(spec, device="cpu")
    proc = spec.process
    assert spec.resource_id is None

    with pytest.raises(RuntimeError, match="exited before readiness"):
        supervisor.wait_for_ready(port=spec.port, timeout=5.0, worker=spec)

    assert supervisor._shutdown_workers([spec], grace=1.0).complete
    _assert_released(spec, proc)
    assert proc.returncode == 3


def test_teardown_kills_the_group_including_a_grandchild(
    platform, spec, tmp_path, monkeypatch,
):
    pid_file = tmp_path / "grandchild.pid"
    monkeypatch.setenv("FAKE_WORKER_MODE", "grandchild")
    monkeypatch.setenv("FAKE_WORKER_PIDFILE", str(pid_file))
    supervisor.spawn_worker(spec, device="cpu")
    proc = spec.process
    deadline = time.monotonic() + 10.0
    while not (pid_file.exists() and pid_file.read_text()):
        assert time.monotonic() < deadline, "worker never started its descendant"
        time.sleep(0.02)
    grandchild = int(pid_file.read_text())

    assert supervisor._shutdown_workers([spec], grace=2.0).complete
    _assert_released(spec, proc)
    deadline = time.monotonic() + 5.0
    while not _process_gone(grandchild):
        assert time.monotonic() < deadline, "descendant survived worker teardown"
        time.sleep(0.02)


def test_admin_restart_signal_reaches_a_live_worker_group(platform, spec, monkeypatch):
    monkeypatch.setenv("FAKE_WORKER_MODE", "sleep")
    supervisor.spawn_worker(spec, device="cpu")
    proc = spec.process
    with spec.process_lock:
        assert supervisor._signal_worker_process(spec, proc, signal.SIGTERM) is True
    # Observe exit through the owner (never a foreign proc.wait(), which
    # would reap the leader outside the identity protocol).
    deadline = time.monotonic() + 10.0
    while _state(spec, proc) is not False:
        assert time.monotonic() < deadline, "worker ignored the signal"
        time.sleep(0.02)
    assert supervisor._shutdown_workers([spec], grace=1.0).complete
    _assert_released(spec, proc)
    assert proc.returncode == -signal.SIGTERM


def test_monitor_detects_worker_death_only_after_exit(platform, spec, monkeypatch):
    monkeypatch.setenv("FAKE_WORKER_MODE", "exit_soon")
    supervisor.spawn_worker(spec, device="cpu")
    proc = spec.process
    spec.status = "running"
    stop = threading.Event()
    restarts: list[bool | None] = []

    def fake_restart(target, **_kwargs):
        # The death verdict must come from the exact process, not ambiguity.
        restarts.append(_state(target, proc))
        stop.set()

    health_checks: list[bool | None] = []

    def healthy_while_alive(**_kwargs):
        # Only reached while the monitor judges the process alive.
        health_checks.append(_state(spec, proc))
        return True

    monkeypatch.setattr(supervisor, "check_worker_health", healthy_while_alive)
    monkeypatch.setattr(supervisor, "_attempt_restart", fake_restart)
    monitor = threading.Thread(
        target=supervisor._monitor_workers,
        args=([spec], stop),
        kwargs={"interval": 0.02},
        daemon=True,
    )
    monitor.start()
    monitor.join(timeout=20.0)
    stop.set()
    assert not monitor.is_alive()
    assert restarts == [False]
    # Alive ticks first (health-checked), then death from the exact process.
    assert health_checks and set(health_checks) == {True}
    assert supervisor._shutdown_workers([spec], grace=1.0).complete
    _assert_released(spec, proc)


@pytest.mark.skipif(
    not callable(getattr(os, "waitid", None)), reason="requires waitid",
)
def test_wnowait_platform_still_fails_closed_for_an_unvalidated_group(
    spec, monkeypatch,
):
    monkeypatch.setenv("FAKE_WORKER_MODE", "sleep")
    monkeypatch.setattr(supervisor, "_validated_worker_process_group", lambda proc: None)
    supervisor.spawn_worker(spec, device="cpu")
    proc = spec.process
    with pytest.raises(supervisor.WorkerIdentityError):
        supervisor._require_worker_generation_alive(spec, proc)
    with spec.process_lock:
        assert supervisor._signal_worker_process(spec, proc, signal.SIGTERM) is False
