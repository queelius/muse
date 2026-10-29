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
from tests import darwin_process_semantics

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")

_FAKE_WORKER = """\
import os, subprocess, sys, time
mode = os.environ["FAKE_WORKER_MODE"]
if mode == "exit":
    sys.exit(3)
if mode == "exit_soon":
    time.sleep(2.0)
    sys.exit(3)
if mode in ("grandchild", "stubborn_grandchild"):
    code = "import time; time.sleep(60)"
    if mode == "stubborn_grandchild":
        code = (
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(60)"
        )
    child = subprocess.Popen([sys.executable, "-c", code])
    with open(os.environ["FAKE_WORKER_PIDFILE"], "w") as handle:
        handle.write(str(child.pid))
time.sleep(60)
"""


@pytest.fixture(params=darwin_process_semantics.PLATFORMS)
def platform(request, monkeypatch):
    """``+darwin`` adds macOS's zombie killpg/getpgid rules (EPERM / ESRCH)."""
    darwin_process_semantics.apply_platform(request.param, monkeypatch)
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
    """Whether a numeric pid has exited (absent, or a zombie awaiting reap)."""
    import psutil

    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:  # includes ZombieProcess
        return True


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


def _wait_for_grandchild(pid_file) -> int:
    deadline = time.monotonic() + 10.0
    while not (pid_file.exists() and pid_file.read_text()):
        assert time.monotonic() < deadline, "worker never started its descendant"
        time.sleep(0.02)
    return int(pid_file.read_text())


def test_term_ignoring_grandchild_is_killed_when_the_leader_exits_on_term(
    platform, spec, tmp_path, monkeypatch,
):
    """The leader honours TERM; the final group KILL must still reach a
    descendant that ignores it, while the exited leader stays pinned."""
    pid_file = tmp_path / "grandchild.pid"
    monkeypatch.setenv("FAKE_WORKER_MODE", "stubborn_grandchild")
    monkeypatch.setenv("FAKE_WORKER_PIDFILE", str(pid_file))
    supervisor.spawn_worker(spec, device="cpu")
    proc = spec.process
    grandchild = _wait_for_grandchild(pid_file)
    time.sleep(0.2)  # let the grandchild install its SIGTERM handler

    result = supervisor._shutdown_workers([spec], grace=2.0)
    assert result.complete
    assert proc.returncode == -signal.SIGTERM
    _assert_released(spec, proc)
    deadline = time.monotonic() + 5.0
    while not _process_gone(grandchild):
        assert time.monotonic() < deadline, "TERM-ignoring descendant survived"
        time.sleep(0.02)


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


@pytest.mark.parametrize("health_delay", [0.0, 0.05], ids=["prompt", "slow-health"])
def test_monitor_detects_worker_death_only_after_exit(
    platform, spec, monkeypatch, health_delay,
):
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

    # The monitor's own liveness verdicts, in order. Re-observing inside the
    # fake health check would race: the monitor releases process_lock
    # before probing health, so the worker can exit in between.
    verdicts: list[bool | None] = []
    real_state_locked = supervisor._worker_process_state_locked

    def recording_state_locked(target, observed, **kwargs):
        state = real_state_locked(target, observed, **kwargs)
        if threading.current_thread() is monitor:
            verdicts.append(state)
        return state

    health_checks: list[bool | None] = []

    def healthy_while_alive(**_kwargs):
        # Only reached while the monitor judges the process alive; record
        # that verdict rather than observing again.
        health_checks.append(verdicts[-1] if verdicts else None)
        # A slow probe widens the window in which the worker exits.
        time.sleep(health_delay)
        return True

    monkeypatch.setattr(
        supervisor, "_worker_process_state_locked", recording_state_locked,
    )
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
    # Alive ticks first (each health-checked), then a death verdict that
    # came from the exited process, never from ambiguity.
    first_death = next(
        index for index, state in enumerate(verdicts) if state is not True
    )
    assert first_death >= 1
    assert verdicts[first_death] is False
    assert health_checks == [True] * first_death
    assert supervisor._shutdown_workers([spec], grace=1.0).complete
    _assert_released(spec, proc)


@pytest.mark.skipif(
    not callable(getattr(os, "waitid", None)), reason="requires waitid",
)
def test_wnowait_platform_still_fails_closed_for_an_unvalidated_group(
    spec, monkeypatch,
):
    monkeypatch.setenv("FAKE_WORKER_MODE", "sleep")
    monkeypatch.setattr(supervisor, "_validated_worker_process_group", lambda proc, **_: None)
    supervisor.spawn_worker(spec, device="cpu")
    proc = spec.process
    with pytest.raises(supervisor.WorkerIdentityError):
        supervisor._require_worker_generation_alive(spec, proc)
    with spec.process_lock:
        assert supervisor._signal_worker_process(spec, proc, signal.SIGTERM) is False
