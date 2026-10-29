"""Shared primitives for subprocesses that own an isolated process group.

Three call sites own child process groups: the bounded installer runner
(``muse.core.owned_command.run_owned_command``), the admin job store
(``muse.admin.jobs.JobStore``), and the supervisor's worker teardown
(``muse.cli_impl.supervisor``). Each keeps its own lifecycle policy (lock
discipline, timing, fallback rules); this module holds only the identity
and signalling primitives they must agree on:

* a spawned child is placed in a new session (POSIX) or process group
  (Windows) via :func:`new_process_group_kwargs`;
* its group is recorded only when :func:`validated_isolated_group` proves
  ``PGID == PID`` and that the group is not this process's own (for a
  leader that is already a pinned zombie, from its new-session spawn: macOS
  ``getpgid`` cannot read a zombie);
* a real POSIX leader is observed WITHOUT reaping it
  (:func:`leader_exited_nowait`, :func:`observe_leader`): with
  ``waitid(WNOWAIT)`` where the interpreter has it, else (macOS before
  Python 3.13) by psutil's zombie status. Either way the unreaped leader
  keeps its PID==PGID pinned against reuse while the group is signalled
  (:func:`signal_isolated_group`);
* ESRCH or EPERM from that signal counts as delivered only while the
  leader is still pinned and observed exited, re-checked right after the
  signal (:func:`signal_isolated_group` with ``pinned=True``): macOS
  returns EPERM for a group whose members are all zombies, ours included;
* the exact leader is reaped (:func:`reap_exited_leader`) only after the
  final group signal;
* only where neither observer works (no WNOWAIT and no psutil, or no
  validated group) is the exact Popen handle polled under the owner's lock
  (:func:`uses_exact_handle_fallback`, :func:`signal_live_fallback_leader`),
  and a ``poll()``-reaped leader's number is never signalled again;
* :class:`BoundedTail` and :func:`drain_stream_into` retain a bounded tail
  of a child pipe that is drained continuously.

A numeric PGID is only ever passed to ``os.killpg`` after it has been
re-checked as a concrete integer > 1 equal to the leader PID and distinct
from the caller's own group: ``killpg(0)`` signals our own group and
``killpg(1)``/``kill(-1)`` broadcast. Stdlib-only at import (psutil is
imported lazily, only where WNOWAIT is missing); safe to import from
``muse --help`` paths.
"""
from __future__ import annotations

import locale
import logging
import os
import signal
import subprocess
import threading
import time
from typing import Any, BinaryIO, Callable

logger = logging.getLogger(__name__)

# Captured at import so tests that patch ``subprocess.Popen`` with a mock
# factory cannot make a mock look like a real, WNOWAIT-pinnable child.
REAL_POPEN_TYPE = subprocess.Popen

# Set in an admin-job child's environment. Muse helpers running inside that
# job stay in the job's authoritative outer group instead of starting a
# nested session the job store could not drain.
MANAGED_JOB_GROUP_ENV = "MUSE_MANAGED_JOB_PROCESS_GROUP"


def concrete_safe_id(value: Any) -> int | None:
    """Return a concrete signal-safe PID/PGID, rejecting bools, mocks and <= 1."""
    if type(value) is not int or value <= 1:
        return None
    return value


def kill_signal() -> signal.Signals:
    """The terminal signal: SIGKILL where it exists, else SIGTERM."""
    return getattr(signal, "SIGKILL", signal.SIGTERM)


def cleanup_deadlines(
    timeout: float, *, term_fraction: float = 0.6,
) -> tuple[float, float]:
    """Split one cleanup budget into ``(term_deadline, deadline)``.

    TERM and the final KILL share one budget instead of each getting a full
    timeout, so cleanup wall time stays bounded by ``timeout``.
    """
    budget = max(0.0, timeout)
    started = time.monotonic()
    return started + budget * term_fraction, started + budget


def new_process_group_kwargs(
    os_name: str, *, creationflags: int = 0,
) -> dict[str, Any]:
    """Popen kwargs that start a child in its own session / process group."""
    if os_name == "posix":
        return {"start_new_session": True}
    if os_name == "nt":
        flags = int(creationflags) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0,
        )
        if flags:
            return {"creationflags": flags}
    return {}


def _pinned_exited_leader(pid: int) -> bool:
    """Whether ``pid`` is our unreaped child, observed exited without reaping.

    A True answer is the pinned-zombie state: :func:`leader_exited_nowait`
    proves the PID is still our child (waitid ``P_PID``, or psutil
    ``ppid() == os.getpid()``) and that it has exited, and only its owner
    reaps it, so the PID (and an equal PGID) cannot be reused. Any failure to
    prove that is False.
    """
    try:
        return leader_exited_nowait(pid)
    except (InterruptedError, ChildProcessError, OSError) as exc:
        logger.warning("could not prove leader %r is a pinned zombie: %s", pid, exc)
        return False


def _leader_group(pid: int, *, new_session: bool) -> int | None:
    """``getpgid(pid)``, or ``pid`` for a pinned zombie started in a new session.

    macOS cannot read a zombie's group: XNU's ``getpgid`` looks the pid up
    with ``proc_find``, which refuses an exited process, so it is ESRCH
    (``bsd/kern/kern_prot.c``); Linux still finds a zombie. A child spawned
    with ``start_new_session=True`` ran ``setsid()`` before exec (CPython's
    Popen raises in the parent when that fails), so it became a session
    leader with PGID == PID, and a session leader can never change its group
    (``setpgid`` is EPERM). While that child is our pinned zombie its PID
    cannot be reused, so PGID == PID still holds. ``new_session`` is the
    caller's statement that it passed ``start_new_session=True``.
    """
    try:
        return concrete_safe_id(os.getpgid(pid))
    except ProcessLookupError:
        if new_session and _pinned_exited_leader(pid):
            return pid
        raise


def validated_isolated_group(
    pid: int | None, *, os_name: str, label: str, new_session: bool = False,
) -> int | None:
    """Return the child's isolated PGID only when positively proven.

    Proven means POSIX, ``getpgid(pid) == pid`` (the child leads a new
    session), and that group differs from this process's own group. Where
    ``getpgid`` cannot read an exited leader (macOS), a child the caller
    started with ``start_new_session=True`` (``new_session``) that is still
    our pinned zombie is proven by its spawn instead (:func:`_leader_group`).
    """
    if os_name != "posix" or pid is None:
        return None
    try:
        process_group = _leader_group(pid, new_session=new_session)
        own_group = concrete_safe_id(os.getpgrp())
    except OSError as exc:
        logger.warning(
            "could not validate %s process group for pid %s: %s", label, pid, exc,
        )
        return None
    if process_group != pid or own_group is None or process_group == own_group:
        logger.error(
            "refusing %s process-group ownership pid=%r pgid=%r own_pgid=%r",
            label, pid, process_group, own_group,
        )
        return None
    return process_group


def platform_supports_pinned_leaders() -> bool:
    """Whether this interpreter exposes ``waitid(WNOWAIT)`` at all.

    CPython before 3.13 never builds ``os.waitid`` on macOS (posixmodule.c
    guarded it with ``defined(HAVE_WAITID) && !defined(__APPLE__)`` until
    gh-113536) even though the ``P_PID`` / ``WEXITED`` / ``WNOWAIT``
    constants are still exported, so an exited leader cannot be observed
    there without reaping it.
    """
    return bool(
        callable(getattr(os, "waitid", None))
        and getattr(os, "P_PID", None) is not None
        and getattr(os, "WEXITED", None) is not None
        and getattr(os, "WNOHANG", None) is not None
        and getattr(os, "WNOWAIT", None) is not None
    )


def _psutil_zombie_observer_available() -> bool:
    """Whether psutil can report an unreaped child's zombie status."""
    try:
        import psutil  # noqa: F401 - lazy: a muse[server] dependency
    except ImportError:
        return False
    return True


def can_observe_exit_without_reaping() -> bool:
    """WNOWAIT, or (where the interpreter lacks it) psutil zombie status."""
    return platform_supports_pinned_leaders() or _psutil_zombie_observer_available()


def supports_pinned_leader(
    process: Any, pid: int | None, process_group: int | None, *, os_name: str,
) -> bool:
    """Whether this exact real POSIX PID==PGID leader can be pinned.

    Pinned means its exit can be observed without reaping it
    (:func:`leader_exited_nowait`), so the zombie keeps PID==PGID reserved
    through the final group signal.
    """
    return bool(
        os_name == "posix"
        and pid is not None
        and process_group == pid
        and isinstance(process, REAL_POPEN_TYPE)
        and can_observe_exit_without_reaping()
    )


def uses_exact_handle_fallback(
    process: Any, pid: int | None, process_group: int | None, *, os_name: str,
) -> bool:
    """Whether a real POSIX child must be owned through its Popen handle alone.

    True only on a platform with no ``waitid(WNOWAIT)`` at all (see
    :func:`platform_supports_pinned_leaders`), for a child that is not
    pinned by :func:`supports_pinned_leader` (no validated group, or no
    psutil). There the exact Popen handle is the ownership token:
    :func:`observe_leader` uses ``poll()``, which may reap. Because only the
    owner reaps, under its owning lock, a ``None`` poll proves the leader,
    and so its PID and any PID==PGID group, are still unreaped and cannot
    have been reused; a leader that exits right after that poll stays an
    unreaped zombie until the owner reaps it. Once ``poll()`` has reaped
    the leader its numeric identity is never signalled again, so
    descendants that outlive it are not drained (best effort, like the
    installer runner's fallback). Where WNOWAIT exists an unpinned real
    child keeps failing closed.
    """
    return bool(
        os_name == "posix"
        and pid is not None
        and isinstance(process, REAL_POPEN_TYPE)
        and not platform_supports_pinned_leaders()
        and not supports_pinned_leader(
            process, pid, process_group, os_name=os_name,
        )
    )


def signal_live_fallback_leader(
    process: Any,
    pid: int | None,
    process_group: int | None,
    sig: signal.Signals,
    *,
    label: str,
) -> bool:
    """Signal an exact-handle-fallback child just observed alive by ``poll()``.

    The caller holds the owning lock and has just seen
    :func:`observe_leader` return True (an unreaped leader), so a validated
    PID==PGID group is still pinned and is signalled whole, descendants
    included. Without a validated group only the exact child is signalled,
    through ``Popen.send_signal``, which itself skips a reaped child.
    """
    if process_group is not None:
        return signal_isolated_group(pid, process_group, sig, label=label)
    try:
        process.send_signal(sig)
    except ProcessLookupError:
        return True
    except Exception as exc:  # noqa: BLE001 - platform Popen handles vary
        logger.warning("could not signal %s child %r: %s", label, pid, exc)
        return False
    return True


def leader_exited_nowait(pid: int) -> bool:
    """Whether ``pid`` has exited, observed WITHOUT reaping it.

    The exited leader stays a zombie, so its PID (and the equal PGID) cannot
    be reused until the owner reaps it. ``InterruptedError``,
    ``ChildProcessError`` (another reaper took it) and other ``OSError`` are
    left to the caller, whose fail-closed policy differs per site.

    Uses ``waitid(WNOWAIT)`` where the interpreter has it, else psutil's
    process status: an unreaped child of ours is either running or a
    zombie, and its PID cannot be reused, so ``STATUS_ZOMBIE`` is an exact,
    non-reaping exit observation. Unlike ``waitid(P_PID)``, psutil reports
    on whatever process holds the PID now, so the same ``oneshot()`` also
    requires ``ppid() == os.getpid()``: a PID that is not our child (it was
    reaped elsewhere and possibly reused) or that vanished is a
    ``ChildProcessError``, like waitid's ECHILD; AccessDenied or any other
    psutil failure is an ``OSError``.
    """
    if platform_supports_pinned_leaders():
        flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
        status = os.waitid(os.P_PID, pid, flags)
        return status is not None and getattr(status, "si_pid", pid) == pid
    try:
        import psutil
    except ImportError as exc:
        raise OSError("no non-reaping exit observer (no WNOWAIT, no psutil)") from exc
    own_pid = os.getpid()
    try:
        observed = psutil.Process(pid)
        with observed.oneshot():
            parent = observed.ppid()
            status = observed.status()
    except psutil.ZombieProcess as exc:
        if getattr(exc, "ppid", None) == own_pid:
            return True
        raise ChildProcessError(
            f"zombie pid {pid} cannot be proven to be our child"
        ) from exc
    except psutil.NoSuchProcess as exc:
        raise ChildProcessError(f"pid {pid} was reaped outside its owner") from exc
    except psutil.Error as exc:
        raise OSError(f"cannot observe pid {pid}: {exc}") from exc
    if parent != own_pid:
        raise ChildProcessError(
            f"pid {pid} is no longer our child (parent {parent}); "
            "it was reaped outside its owner"
        )
    return status == psutil.STATUS_ZOMBIE


_UNKNOWN_GROUP: Any = object()


def observe_leader(
    process: Any,
    pid: int | None,
    *,
    pinned: bool,
    os_name: str,
    label: str,
    process_group: Any = _UNKNOWN_GROUP,
    may_reap: bool = True,
) -> bool | None:
    """Return True alive, False exited, or None when identity is ambiguous.

    ``pinned`` is the caller's :func:`supports_pinned_leader` verdict. A
    pinned leader is observed without reaping (WNOWAIT, or psutil zombie
    status where WNOWAIT is missing). A real POSIX child that is NOT
    pinned fails closed (None): polling would reap it and release an
    unknown numeric group before its descendants are drained. The exception
    is an unpinnable child on a platform with no WNOWAIT at all
    (:func:`uses_exact_handle_fallback`), where ``poll()`` under the owning
    lock is the only exact observation; the caller must then pass the
    ``process_group`` it owns (omitted means unknown, which never polls a
    real child, so a pinnable leader can never be reaped by mistake).
    ``may_reap=False`` is for observers that own nothing (e.g. a status
    route): a real child is then never polled, and an unpinnable one is
    reported ambiguous (None) instead.
    Anything else (test doubles, non-POSIX) falls back to ``poll()``,
    mirroring Popen's side effect of storing the polled status on
    ``returncode``. Callers decide whether an unsafe (None) PID is
    ambiguous before calling: the exact Popen handle is still a valid
    ownership token for ``poll()``.
    """
    if type(getattr(process, "returncode", None)) is int:
        return False
    if pinned:
        try:
            return not leader_exited_nowait(pid)
        except InterruptedError:
            return True
        except (ChildProcessError, OSError) as exc:
            logger.warning(
                "could not observe %s leader %r without reaping: %s",
                label, pid, exc,
            )
            return None
    if isinstance(process, REAL_POPEN_TYPE):
        if not may_reap:
            return None
        if os_name == "posix" and (
            process_group is _UNKNOWN_GROUP
            or not uses_exact_handle_fallback(
                process, pid, process_group, os_name=os_name,
            )
        ):
            return None
    try:
        observed = process.poll()
        if type(observed) is int and type(
            getattr(process, "returncode", None)
        ) is not int:
            process.returncode = observed
        return observed is None
    except Exception as exc:  # noqa: BLE001 - platform/test handles vary
        logger.warning("could not inspect %s leader %r: %s", label, pid, exc)
        return None


def signal_isolated_group(
    pid: int | None,
    process_group: int | None,
    sig: signal.Signals,
    *,
    label: str,
    still_owned: Callable[[], bool] | None = None,
    pinned: bool = False,
) -> bool:
    """Signal a validated, still-owned isolated group exactly once.

    The caller must guarantee the group identity is pinned (an unreaped
    PID==PGID leader under the caller's lock) or freshly revalidated. This
    function re-checks the numeric identity anyway: the PGID must be a
    concrete int > 1, equal to the leader PID, and not our own group.
    ``still_owned`` is an optional last-moment ownership check run after
    validation and immediately before ``killpg``.

    Returns True when the signal was delivered; False when refused or
    delivery failed. Never falls through to a second numeric identity.

    ``pinned=True`` is the caller's statement that the leader is pinned
    (:func:`supports_pinned_leader`, observed under its owning lock). Then,
    and only then, ESRCH or EPERM also counts as delivered, provided a
    non-reaping observation made right after the failed ``killpg`` (still
    under the caller's lock) shows the leader is our unreaped, exited child
    (:func:`_pinned_exited_leader`). That zombie keeps the PID, so no other
    group can carry this id, and the failure proves the group has no live
    member this process may signal:

    * macOS returns EPERM here. XNU's ``killpg1`` (``bsd/kern/kern_sig.c``)
      skips members with ``p_stat == SZOMB`` and, finding none, returns
      ``posix ? EPERM : ESRCH``; libc's ``killpg`` passes ``posix=1``. The
      group cannot be missing (ESRCH) while its zombie leader is unreaped,
      because a process leaves its group only when reaped
      (``reap_child_locked``, ``bsd/kern/kern_exit.c``).
    * Linux never fails here: ``__kill_pgrp_info`` (``kernel/signal.c``)
      succeeds once any member, the zombie leader included, accepts the
      signal. So a live member this process may not signal (a descendant
      that became another user) is not detected on Linux either; treating
      EPERM as delivered gives macOS exactly Linux's guarantee.

    A live leader, a leader that cannot be observed, or one reaped
    elsewhere keeps the failure fail-closed, as does ESRCH/EPERM whenever
    ``pinned`` is False.
    """
    try:
        own_group = concrete_safe_id(os.getpgrp())
    except OSError:
        own_group = None
    if (
        concrete_safe_id(process_group) is None
        or process_group != pid
        or own_group is None
        or process_group == own_group
    ):
        logger.error(
            "refusing unsafe %s group signal pid=%r pgid=%r own_pgid=%r",
            label, pid, process_group, own_group,
        )
        return False
    if still_owned is not None and not still_owned():
        logger.warning(
            "refusing %s group signal after leader exit during group "
            "revalidation (pid=%r pgid=%r)",
            label, pid, process_group,
        )
        return False
    try:
        os.killpg(process_group, sig)
    except (ProcessLookupError, PermissionError) as exc:
        if pinned and _pinned_exited_leader(pid):
            logger.debug(
                "%s group %s has no live signalable member (%s) while its "
                "exited leader is pinned; counting the signal as delivered",
                label, process_group, exc,
            )
            return True
        logger.warning(
            "could not signal %s process group %s: %s", label, process_group, exc,
        )
        return False
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(
            "could not signal %s process group %s: %s", label, process_group, exc,
        )
        return False
    return True


def reap_exited_leader(process: Any, pid: int | None, *, label: str) -> int | None:
    """Non-blockingly reap an exact leader already observed as exited.

    Returns the exit status, or None when the reap could not be proven.
    Mirrors the status onto ``returncode`` for lightweight test doubles.
    """
    try:
        returncode = process.wait(timeout=0.0)
    except Exception as exc:  # noqa: BLE001 - TimeoutExpired, ChildProcessError, OSError, ...
        logger.warning("could not reap %s leader %r: %s", label, pid, exc)
        return None
    if type(returncode) is not int:
        return None
    if type(getattr(process, "returncode", None)) is not int:
        process.returncode = returncode
    return returncode


class BoundedTail:
    """Thread-safe tail of one byte stream, accounting for what it dropped."""

    def __init__(self, limit: int) -> None:
        self._limit = max(0, int(limit))
        self._buffer = bytearray()
        self._discarded = 0
        self._read_error: str | None = None
        self._lock = threading.Lock()

    def append(self, chunk: bytes | str, *, encoding: str | None = None) -> None:
        """Append a chunk; ``str`` is encoded with ``encoding`` (default locale)."""
        if isinstance(chunk, str):
            data = chunk.encode(
                encoding or locale.getpreferredencoding(False), errors="replace",
            )
        else:
            data = bytes(chunk)
        if not data:
            return
        with self._lock:
            self._buffer.extend(data)
            excess = len(self._buffer) - self._limit
            if excess > 0:
                del self._buffer[:excess]
                self._discarded += excess

    def record_read_error(self, error: BaseException) -> None:
        with self._lock:
            self._read_error = type(error).__name__

    def snapshot(self) -> tuple[bytes, int, str | None]:
        """Return ``(retained bytes, bytes discarded, read error name)``."""
        with self._lock:
            return bytes(self._buffer), self._discarded, self._read_error


def drain_stream_into(
    stream: BinaryIO,
    tail: BoundedTail,
    *,
    read_bytes: int,
    encoding: str | None = None,
) -> None:
    """Read ``stream`` to EOF into ``tail``. Read errors propagate."""
    while True:
        chunk = stream.read(read_bytes)
        if not chunk:
            return
        tail.append(chunk, encoding=encoding)
