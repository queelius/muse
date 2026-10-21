"""Shared primitives for subprocesses that own an isolated process group.

Three call sites own child process groups: the bounded installer runner
(``muse.core.venv.run_owned_command``), the admin job store
(``muse.admin.jobs.JobStore``), and the supervisor's worker teardown
(``muse.cli_impl.supervisor``). Each keeps its own lifecycle policy (lock
discipline, timing, fallback rules); this module holds only the identity
and signalling primitives they must agree on:

* a spawned child is placed in a new session (POSIX) or process group
  (Windows) via :func:`new_process_group_kwargs`;
* its group is recorded only when :func:`validated_isolated_group` proves
  ``PGID == PID`` and that the group is not this process's own;
* a real POSIX leader is observed with ``waitid(WNOWAIT)``
  (:func:`leader_exited_nowait`, :func:`observe_leader`), never ``poll()``,
  so the unreaped leader keeps its PID==PGID pinned against reuse while
  the group is signalled (:func:`signal_isolated_group`);
* the exact leader is reaped (:func:`reap_exited_leader`) only after the
  final group signal;
* :class:`BoundedTail` and :func:`drain_stream_into` retain a bounded tail
  of a child pipe that is drained continuously.

A numeric PGID is only ever passed to ``os.killpg`` after it has been
re-checked as a concrete integer > 1 equal to the leader PID and distinct
from the caller's own group: ``killpg(0)`` signals our own group and
``killpg(1)``/``kill(-1)`` broadcast. Stdlib-only; safe to import from
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


def validated_isolated_group(
    pid: int | None, *, os_name: str, label: str,
) -> int | None:
    """Return the child's isolated PGID only when positively proven.

    Proven means POSIX, ``getpgid(pid) == pid`` (the child leads a new
    session), and that group differs from this process's own group.
    """
    if os_name != "posix" or pid is None:
        return None
    try:
        process_group = concrete_safe_id(os.getpgid(pid))
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


def supports_pinned_leader(
    process: Any, pid: int | None, process_group: int | None, *, os_name: str,
) -> bool:
    """Whether WNOWAIT can pin this exact real POSIX PID==PGID leader."""
    return bool(
        os_name == "posix"
        and pid is not None
        and process_group == pid
        and isinstance(process, REAL_POPEN_TYPE)
        and callable(getattr(os, "waitid", None))
        and getattr(os, "P_PID", None) is not None
        and getattr(os, "WEXITED", None) is not None
        and getattr(os, "WNOHANG", None) is not None
        and getattr(os, "WNOWAIT", None) is not None
    )


def leader_exited_nowait(pid: int) -> bool:
    """Whether ``pid`` has exited, observed WITHOUT reaping it.

    The exited leader stays a zombie, so its PID (and the equal PGID) cannot
    be reused until the owner reaps it. ``InterruptedError``,
    ``ChildProcessError`` (another reaper took it) and other ``OSError`` are
    left to the caller, whose fail-closed policy differs per site.
    """
    flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
    status = os.waitid(os.P_PID, pid, flags)
    return status is not None and getattr(status, "si_pid", pid) == pid


def observe_leader(
    process: Any, pid: int | None, *, pinned: bool, os_name: str, label: str,
) -> bool | None:
    """Return True alive, False exited, or None when identity is ambiguous.

    ``pinned`` is the caller's :func:`supports_pinned_leader` verdict. A
    pinned leader is observed with WNOWAIT. A real POSIX child that is NOT
    pinned fails closed (None): polling would reap it and release an
    unknown numeric group before its descendants are drained. Anything else
    (test doubles, non-POSIX) falls back to ``poll()``, mirroring Popen's
    side effect of storing the polled status on ``returncode``. Callers
    decide whether an unsafe (None) PID is ambiguous before calling: the
    exact Popen handle is still a valid ownership token for ``poll()``.
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
    if os_name == "posix" and isinstance(process, REAL_POPEN_TYPE):
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
) -> bool:
    """Signal a validated, still-owned isolated group exactly once.

    The caller must guarantee the group identity is pinned (an unreaped
    PID==PGID leader under the caller's lock) or freshly revalidated. This
    function re-checks the numeric identity anyway: the PGID must be a
    concrete int > 1, equal to the leader PID, and not our own group.
    ``still_owned`` is an optional last-moment ownership check run after
    validation and immediately before ``killpg``.

    Returns True when the signal was delivered or ESRCH proved the group has
    no members left; False when refused or delivery failed. Never falls
    through to a second numeric identity.
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
    except ProcessLookupError:
        return True
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
