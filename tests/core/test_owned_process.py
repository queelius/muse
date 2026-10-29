"""Shared owned-process-group primitives (muse.core.owned_process)."""
from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from muse.core import owned_process as op


# WNOWAIT, or psutil zombie status where the interpreter lacks waitid.
_POSIX_WNOWAIT = os.name == "posix" and op.can_observe_exit_without_reaping()


class TestIdentityChecks:
    @pytest.mark.parametrize("value", [None, True, False, 0, 1, -5, 4.2, "42", MagicMock()])
    def test_concrete_safe_id_rejects_unsafe_values(self, value):
        assert op.concrete_safe_id(value) is None

    def test_concrete_safe_id_accepts_concrete_int(self):
        assert op.concrete_safe_id(4242) == 4242

    def test_validated_group_requires_posix_and_pid(self):
        assert op.validated_isolated_group(4242, os_name="nt", label="t") is None
        assert op.validated_isolated_group(None, os_name="posix", label="t") is None

    @pytest.mark.parametrize("pgid,own", [
        (4243, 31337),   # child did not start a new session
        (4242, 4242),    # child shares our group
        (4242, 1),       # own group not concrete
        (MagicMock(), 31337),
    ])
    def test_validated_group_refuses_unproven_identity(self, pgid, own):
        with patch("muse.core.owned_process.os.getpgid", return_value=pgid), \
             patch("muse.core.owned_process.os.getpgrp", return_value=own):
            assert op.validated_isolated_group(4242, os_name="posix", label="t") is None

    def test_validated_group_fails_closed_on_oserror(self):
        with patch(
            "muse.core.owned_process.os.getpgid", side_effect=ProcessLookupError,
        ):
            assert op.validated_isolated_group(4242, os_name="posix", label="t") is None

    def test_validated_group_accepts_isolated_leader(self):
        with patch("muse.core.owned_process.os.getpgid", return_value=4242), \
             patch("muse.core.owned_process.os.getpgrp", return_value=31337):
            assert op.validated_isolated_group(4242, os_name="posix", label="t") == 4242

    def test_mock_process_is_never_pinnable(self):
        assert op.supports_pinned_leader(
            MagicMock(), 4242, 4242, os_name="posix",
        ) is False

    def test_new_process_group_kwargs(self):
        assert op.new_process_group_kwargs("posix") == {"start_new_session": True}
        with patch.object(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, create=True):
            assert op.new_process_group_kwargs("nt", creationflags=0x8) == {
                "creationflags": 0x208,
            }
        assert op.new_process_group_kwargs("java") == {}

    def test_cleanup_deadlines_share_one_budget(self):
        before = time.monotonic()
        term, final = op.cleanup_deadlines(10.0)
        assert before + 6.0 <= term <= final - 3.9
        assert final <= time.monotonic() + 10.0
        term, final = op.cleanup_deadlines(-1.0)
        assert term == final


class TestSignalIsolatedGroup:
    @pytest.mark.parametrize("pid,pgid,own", [
        (4242, None, 31337),
        (4242, 1, 31337),
        (4242, 0, 31337),
        (4242, 4243, 31337),
        (4242, 4242, 4242),
        (4242, 4242, 1),
        (MagicMock(), MagicMock(), 31337),
    ])
    def test_refuses_unsafe_identity(self, pid, pgid, own):
        with patch("muse.core.owned_process.os.getpgrp", return_value=own), \
             patch("muse.core.owned_process.os.killpg") as killpg:
            assert op.signal_isolated_group(
                pid, pgid, signal.SIGTERM, label="t",
            ) is False
        killpg.assert_not_called()

    def test_refuses_when_own_group_unreadable(self):
        with patch("muse.core.owned_process.os.getpgrp", side_effect=OSError), \
             patch("muse.core.owned_process.os.killpg") as killpg:
            assert op.signal_isolated_group(
                4242, 4242, signal.SIGTERM, label="t",
            ) is False
        killpg.assert_not_called()

    def test_last_moment_ownership_check_runs_before_killpg(self):
        with patch("muse.core.owned_process.os.getpgrp", return_value=31337), \
             patch("muse.core.owned_process.os.killpg") as killpg:
            assert op.signal_isolated_group(
                4242, 4242, signal.SIGTERM, label="t", still_owned=lambda: False,
            ) is False
        killpg.assert_not_called()

    @pytest.mark.parametrize("effect,expected", [
        (None, True),
        (ProcessLookupError, True),
        (PermissionError, False),
    ])
    def test_delivery_outcomes(self, effect, expected):
        with patch("muse.core.owned_process.os.getpgrp", return_value=31337), \
             patch("muse.core.owned_process.os.killpg", side_effect=effect) as killpg:
            assert op.signal_isolated_group(
                4242, 4242, signal.SIGTERM, label="t", still_owned=lambda: True,
            ) is expected
        killpg.assert_called_once_with(4242, signal.SIGTERM)


class TestObserveAndReap:
    def test_already_reaped_is_exited(self):
        process = MagicMock(returncode=0)
        assert op.observe_leader(
            process, 4242, pinned=False, os_name="posix", label="t",
        ) is False
        process.poll.assert_not_called()

    def test_real_unpinned_posix_child_fails_closed_without_polling(self, monkeypatch):
        monkeypatch.setattr(op, "platform_supports_pinned_leaders", lambda: True)
        process = MagicMock(spec=subprocess.Popen, returncode=None)
        assert op.observe_leader(
            process, 4242, pinned=False, os_name="posix", label="t",
        ) is None
        process.poll.assert_not_called()

    def test_no_wnowait_platform_polls_the_exact_real_child(self, monkeypatch):
        """macOS before Python 3.13 has no os.waitid: poll the exact handle."""
        monkeypatch.delattr(os, "waitid", raising=False)
        assert op.platform_supports_pinned_leaders() is False
        process = MagicMock(spec=subprocess.Popen, returncode=None)
        # Only an unpinnable child (here: no validated group) may be polled.
        assert op.uses_exact_handle_fallback(process, 4242, None, os_name="posix")
        assert not op.uses_exact_handle_fallback(process, 4242, 4242, os_name="posix")
        assert not op.uses_exact_handle_fallback(process, None, None, os_name="posix")
        assert not op.uses_exact_handle_fallback(MagicMock(), 4242, None, os_name="posix")
        assert not op.uses_exact_handle_fallback(process, 4242, None, os_name="nt")
        process.poll.return_value = None
        assert op.observe_leader(
            process, 4242, pinned=False, os_name="posix", label="t",
            process_group=None,
        ) is True
        process.poll.return_value = 3
        assert op.observe_leader(
            process, 4242, pinned=False, os_name="posix", label="t",
            process_group=None,
        ) is False
        # Without a concrete pid the real child still fails closed.
        assert op.observe_leader(
            MagicMock(spec=subprocess.Popen, returncode=None), None,
            pinned=False, os_name="posix", label="t", process_group=None,
        ) is None

    def test_fallback_never_polls_a_pinnable_or_unknown_group_leader(self, monkeypatch):
        monkeypatch.delattr(os, "waitid", raising=False)
        process = MagicMock(spec=subprocess.Popen, returncode=None)
        # A caller that wrongly passes pinned=False for a pinnable leader, or
        # does not say which group it owns, never gets a reaping poll.
        for kwargs in ({"process_group": 4242}, {}):
            assert op.observe_leader(
                process, 4242, pinned=False, os_name="posix", label="t", **kwargs,
            ) is None
        process.poll.assert_not_called()

    def test_non_reaping_observation_never_polls_a_real_child(self, monkeypatch):
        monkeypatch.delattr(os, "waitid", raising=False)
        monkeypatch.setattr(op, "_psutil_zombie_observer_available", lambda: False)
        process = MagicMock(spec=subprocess.Popen, returncode=None)
        assert op.observe_leader(
            process, 4242, pinned=False, os_name="posix", label="t",
            process_group=None, may_reap=False,
        ) is None
        process.poll.assert_not_called()
        # Test doubles cannot be reaped, so they are still polled.
        double = MagicMock(returncode=None)
        double.poll.return_value = 1
        assert op.observe_leader(
            double, 4242, pinned=False, os_name="posix", label="t", may_reap=False,
        ) is False

    def test_no_wnowait_observes_exit_by_psutil_zombie_status(self, monkeypatch):
        import psutil

        monkeypatch.delattr(os, "waitid", raising=False)
        real = MagicMock(spec=subprocess.Popen, returncode=None)
        assert op.supports_pinned_leader(real, 4242, 4242, os_name="posix")
        assert not op.supports_pinned_leader(real, 4242, None, os_name="posix")

        def proc_with(status=None, error=None, ppid=None):
            handle = MagicMock()
            handle.ppid.return_value = os.getpid() if ppid is None else ppid
            if error is not None:
                handle.status.side_effect = error
            else:
                handle.status.return_value = status
            return handle

        cases = [
            (proc_with(psutil.STATUS_ZOMBIE), True),
            (proc_with(psutil.STATUS_SLEEPING), False),
            (proc_with(error=psutil.ZombieProcess(4242, ppid=os.getpid())), True),
        ]
        for handle, expected in cases:
            with patch("psutil.Process", return_value=handle):
                assert op.leader_exited_nowait(4242) is expected
        # psutil reports on whoever holds the PID now: a process that is not
        # our child is a foreign reap (like waitid's ECHILD), never "ours".
        not_ours = [
            proc_with(psutil.STATUS_ZOMBIE, ppid=1),
            proc_with(psutil.STATUS_SLEEPING, ppid=1),
            proc_with(error=psutil.ZombieProcess(4242, ppid=1)),
            proc_with(error=psutil.ZombieProcess(4242)),
        ]
        for handle in not_ours:
            with patch("psutil.Process", return_value=handle):
                with pytest.raises(ChildProcessError):
                    op.leader_exited_nowait(4242)
                assert op.observe_leader(
                    real, 4242, pinned=True, os_name="posix", label="t",
                ) is None
        with patch("psutil.Process", side_effect=psutil.NoSuchProcess(4242)):
            with pytest.raises(ChildProcessError):
                op.leader_exited_nowait(4242)
            # Reaped elsewhere: never treated as a live or signalable leader.
            assert op.observe_leader(
                real, 4242, pinned=True, os_name="posix", label="t",
            ) is None
        with patch("psutil.Process", return_value=proc_with(error=psutil.AccessDenied(4242))):
            with pytest.raises(OSError):
                op.leader_exited_nowait(4242)
            assert op.observe_leader(
                real, 4242, pinned=True, os_name="posix", label="t",
            ) is None
        real.poll.assert_not_called()

    def test_no_wnowait_and_no_psutil_is_not_pinnable(self, monkeypatch):
        monkeypatch.delattr(os, "waitid", raising=False)
        monkeypatch.setattr(op, "_psutil_zombie_observer_available", lambda: False)
        real = MagicMock(spec=subprocess.Popen, returncode=None)
        assert not op.supports_pinned_leader(real, 4242, 4242, os_name="posix")
        assert op.uses_exact_handle_fallback(real, 4242, 4242, os_name="posix")

    @pytest.mark.skipif(os.name != "posix", reason="POSIX zombie")
    def test_real_zombie_stays_pinned_without_waitid(self, monkeypatch):
        monkeypatch.delattr(os, "waitid", raising=False)
        process = subprocess.Popen(
            [sys.executable, "-c", "pass"], **op.new_process_group_kwargs("posix"),
        )
        try:
            deadline = time.monotonic() + 10.0
            while not op.leader_exited_nowait(process.pid):
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert process.returncode is None  # observed, not reaped
            assert op.leader_exited_nowait(process.pid) is True
        finally:
            process.wait(timeout=10.0)

    def test_wnowait_platform_never_uses_the_fallback(self, monkeypatch):
        monkeypatch.setattr(op, "platform_supports_pinned_leaders", lambda: True)
        process = MagicMock(spec=subprocess.Popen, returncode=None)
        for group in (None, 4242):
            assert not op.uses_exact_handle_fallback(process, 4242, group, os_name="posix")

    @pytest.mark.skipif(os.name != "posix", reason="POSIX process tree")
    def test_psutil_observer_rejects_a_pid_that_is_not_our_child(self, monkeypatch):
        """A real non-child (our child's child) is never observed as ours."""
        monkeypatch.delattr(os, "waitid", raising=False)
        process = subprocess.Popen(
            [
                sys.executable, "-c",
                "import subprocess, sys, time\n"
                "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                "print(g.pid, flush=True)\n"
                "time.sleep(30)\n",
            ],
            stdout=subprocess.PIPE, text=True,
            **op.new_process_group_kwargs("posix"),
        )
        try:
            grandchild = int(process.stdout.readline())
            with pytest.raises(ChildProcessError):
                op.leader_exited_nowait(grandchild)
            assert op.leader_exited_nowait(process.pid) is False
        finally:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10.0)
            process.stdout.close()

    def test_live_fallback_signal_targets_group_else_exact_child(self):
        process = MagicMock()
        with patch.object(op, "signal_isolated_group", return_value=True) as group:
            assert op.signal_live_fallback_leader(
                process, 4242, 4242, signal.SIGTERM, label="t",
            ) is True
        group.assert_called_once_with(4242, 4242, signal.SIGTERM, label="t")
        process.send_signal.assert_not_called()

        assert op.signal_live_fallback_leader(
            process, 4242, None, signal.SIGTERM, label="t",
        ) is True
        process.send_signal.assert_called_once_with(signal.SIGTERM)
        process.send_signal.side_effect = ProcessLookupError
        assert op.signal_live_fallback_leader(
            process, 4242, None, signal.SIGKILL, label="t",
        ) is True
        process.send_signal.side_effect = OSError("denied")
        assert op.signal_live_fallback_leader(
            process, 4242, None, signal.SIGKILL, label="t",
        ) is False

    def test_double_poll_mirrors_returncode(self):
        process = MagicMock(returncode=None)
        process.poll.return_value = 3
        assert op.observe_leader(
            process, 4242, pinned=False, os_name="posix", label="t",
        ) is False
        assert process.returncode == 3

    def test_poll_failure_is_ambiguous(self):
        process = MagicMock(returncode=None)
        process.poll.side_effect = RuntimeError("boom")
        assert op.observe_leader(
            process, 4242, pinned=False, os_name="posix", label="t",
        ) is None

    @pytest.mark.parametrize("effect,expected", [
        (InterruptedError, True),
        (ChildProcessError, None),
        (OSError, None),
    ])
    def test_pinned_waitid_errors(self, effect, expected):
        with patch(
            "muse.core.owned_process.leader_exited_nowait", side_effect=effect,
        ):
            assert op.observe_leader(
                MagicMock(returncode=None), 4242,
                pinned=True, os_name="posix", label="t",
            ) is expected

    def test_reap_mirrors_status_and_fails_closed(self):
        process = MagicMock(returncode=None)
        process.wait.return_value = 7
        assert op.reap_exited_leader(process, 4242, label="t") == 7
        assert process.returncode == 7
        process.wait.assert_called_once_with(timeout=0.0)

        stuck = MagicMock(returncode=None)
        stuck.wait.side_effect = subprocess.TimeoutExpired("x", 0.0)
        assert op.reap_exited_leader(stuck, 4242, label="t") is None

    @pytest.mark.skipif(not _POSIX_WNOWAIT, reason="requires POSIX WNOWAIT")
    def test_real_leader_stays_pinned_until_reaped(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            **op.new_process_group_kwargs("posix"),
        )
        try:
            pid = op.concrete_safe_id(process.pid)
            pgid = op.validated_isolated_group(pid, os_name="posix", label="t")
            assert pgid == pid
            assert op.supports_pinned_leader(process, pid, pgid, os_name="posix")
            assert op.observe_leader(
                process, pid, pinned=True, os_name="posix", label="t",
            ) is True
            assert op.signal_isolated_group(pid, pgid, signal.SIGTERM, label="t")
            deadline = time.monotonic() + 10.0
            while not op.leader_exited_nowait(pid):
                assert time.monotonic() < deadline
                time.sleep(0.01)
            # Observed, not reaped: the zombie still pins PID == PGID.
            assert process.returncode is None
            assert os.getpgid(pid) == pgid
            assert op.signal_isolated_group(
                pid, pgid, op.kill_signal(), label="t",
            ) is True
            assert op.reap_exited_leader(process, pid, label="t") == -signal.SIGTERM
        finally:
            if process.returncode is None:
                process.kill()
                process.wait(timeout=5)


class TestBoundedTail:
    def test_keeps_tail_and_counts_discarded(self):
        tail = op.BoundedTail(4)
        tail.append(b"ab")
        tail.append(b"cdef")
        tail.append("g", encoding="utf-8")
        tail.append(b"")
        assert tail.snapshot() == (b"defg", 3, None)

    def test_zero_limit_retains_nothing(self):
        tail = op.BoundedTail(0)
        tail.append(b"abc")
        assert tail.snapshot() == (b"", 3, None)

    def test_records_read_error_name(self):
        tail = op.BoundedTail(8)
        tail.record_read_error(ValueError("closed"))
        assert tail.snapshot()[2] == "ValueError"

    def test_drain_reads_to_eof_and_propagates_errors(self):
        tail = op.BoundedTail(5)
        op.drain_stream_into(io.BytesIO(b"0123456789"), tail, read_bytes=3)
        assert tail.snapshot() == (b"56789", 5, None)

        broken = MagicMock()
        broken.read.side_effect = OSError("gone")
        with pytest.raises(OSError):
            op.drain_stream_into(broken, op.BoundedTail(5), read_bytes=3)
