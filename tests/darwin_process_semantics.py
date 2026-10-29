"""Emulate macOS (XNU) zombie process-group semantics on Linux.

Two calls behave differently for a zombie on macOS than on Linux, and muse's
owned-process code meets both whenever an exited leader is pinned unreaped:

* ``killpg(pgid, sig)`` (libc's UNIX03 ``killpg`` is ``__kill(-pgid, sig,
  posix=1)``, Libc ``compat-43/FreeBSD/killpg.c``) reaches XNU's ``killpg1``
  (``bsd/kern/kern_sig.c``). Its ``pgrp_iterate`` filter skips members with
  ``p_stat == SZOMB``, so a group whose members are all zombies has
  ``nfound == 0`` and returns ``posix ? EPERM : ESRCH``, i.e. EPERM. The
  group itself still exists: a zombie leaves its process group only when it
  is reaped (``pgrp_leave_locked`` in ``reap_child_locked``,
  ``bsd/kern/kern_exit.c``), so ``pgrp_find`` succeeds and ESRCH is returned
  only once no member, zombie or not, is left. Linux instead signals the
  zombie (``__kill_pgrp_info`` in ``kernel/signal.c`` succeeds if any member
  accepted the signal) and returns 0.
* ``getpgid(pid)`` (``bsd/kern/kern_prot.c``) looks the pid up with
  ``proc_find``, which takes no reference on an exited process, so a zombie
  is ESRCH. Linux's ``find_task_by_vpid`` still finds a zombie.

``install`` patches ``os.killpg``, ``os.getpgid`` and negative-pid
``os.kill`` with those rules, reading member states from ``/proc`` (Linux
only). Every other case delegates to the real call, so a group with a live
member is really signalled. Setting ``MUSE_TEST_DARWIN_PROCESS_SEMANTICS=1``
applies it to the whole suite (``tests/conftest.py``).
"""
from __future__ import annotations

import errno
import os
import sys

ENV_VAR = "MUSE_TEST_DARWIN_PROCESS_SEMANTICS"


def supported() -> bool:
    """The emulation reads /proc, so it runs only on Linux."""
    return sys.platform.startswith("linux") and os.path.isdir("/proc/self")


def globally_enabled() -> bool:
    return os.environ.get(ENV_VAR) == "1"


def _stat(pid: int) -> tuple[str, int] | None:
    """``(state, pgrp)`` from /proc/<pid>/stat, or None when absent."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii", errors="replace") as handle:
            data = handle.read()
        fields = data[data.rindex(")") + 2:].split()
        return fields[0], int(fields[2])
    except (OSError, ValueError, IndexError):
        # Vanished (or was reaped) mid-read: no longer a member of anything.
        return None


def _member_states(pgid: int) -> list[str]:
    states = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        stat = _stat(int(entry))
        if stat is not None and stat[1] == pgid:
            states.append(stat[0])
    return states


def _zombie_only(pgid: int) -> bool:
    states = _member_states(pgid)
    return bool(states) and all(state in ("Z", "X") for state in states)


def _eperm() -> PermissionError:
    return PermissionError(errno.EPERM, os.strerror(errno.EPERM))


def install(monkeypatch) -> None:
    """Patch ``os`` with XNU's zombie rules for the current test."""
    if not supported():
        raise RuntimeError("darwin process-semantics emulation needs Linux /proc")
    real_killpg = os.killpg
    real_getpgid = os.getpgid
    real_kill = os.kill

    def killpg(pgid, sig):
        if type(pgid) is int and pgid > 1 and _zombie_only(pgid):
            raise _eperm()
        return real_killpg(pgid, sig)

    def kill(pid, sig):
        if type(pid) is int and pid < -1 and _zombie_only(-pid):
            raise _eperm()
        return real_kill(pid, sig)

    def getpgid(pid):
        if type(pid) is int and pid > 0:
            stat = _stat(pid)
            if stat is not None and stat[0] in ("Z", "X"):
                raise ProcessLookupError(errno.ESRCH, os.strerror(errno.ESRCH))
        return real_getpgid(pid)

    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(os, "kill", kill)
    monkeypatch.setattr(os, "getpgid", getpgid)


def apply_platform(param: str, monkeypatch) -> None:
    """Apply a ``waitid`` / ``no-waitid`` [``+darwin``] platform fixture param."""
    import pytest

    base, _, variant = param.partition("+")
    if variant == "darwin":
        if not supported():
            pytest.skip("darwin process-semantics emulation needs Linux /proc")
        install(monkeypatch)
    if base == "no-waitid":
        monkeypatch.delattr(os, "waitid", raising=False)
    elif not callable(getattr(os, "waitid", None)):
        pytest.skip("platform has no waitid")


PLATFORMS = ["waitid", "no-waitid", "waitid+darwin", "no-waitid+darwin"]
