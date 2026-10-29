"""run_owned_command with real children, with and without ``os.waitid``.

Without waitid (macOS before Python 3.13) the installer runner pins its
leader by psutil zombie status, so a timed-out command's leader that honours
TERM stays unreaped until the final group KILL reaches a descendant that
ignores TERM.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from muse.core import owned_process
from muse.core.owned_command import run_owned_command

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")


@pytest.fixture(params=["waitid", "no-waitid"])
def platform(request, monkeypatch):
    if request.param == "no-waitid":
        monkeypatch.delattr(os, "waitid", raising=False)
        assert owned_process.can_observe_exit_without_reaping()
    elif not callable(getattr(os, "waitid", None)):
        pytest.skip("platform has no waitid")
    return request.param


def _process_gone(pid: int) -> bool:
    import psutil

    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:  # includes ZombieProcess
        return True


def test_timeout_kills_a_term_ignoring_grandchild(platform, tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)'])\n"
        f"with open({str(pid_file)!r}, 'w') as handle:\n"
        "    handle.write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_owned_command([sys.executable, "-c", script], timeout=3.0)
    assert time.monotonic() - started < 20.0
    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 5.0
    while not _process_gone(grandchild):
        assert time.monotonic() < deadline, "TERM-ignoring descendant survived"
        time.sleep(0.02)


def test_fast_nonzero_exit_reports_its_status(platform):
    result = run_owned_command(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
        timeout=10.0,
    )
    assert result.returncode == 3
    assert "boom" in result.stderr
