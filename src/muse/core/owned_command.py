"""Bounded, ownership-safe runner for installer subprocesses.

`run_owned_command` starts one command in an isolated process group, keeps
only a bounded tail of its output, and on timeout, error or cancellation
terminates the whole group within a shared cleanup budget. On POSIX
platforms that can observe exit without reaping (`waitid(WNOWAIT)`, or
psutil zombie status where waitid is missing, e.g. macOS before Python 3.13)
the exited leader stays unreaped until the group is cleaned, so its PID/PGID
cannot be reused underneath the cleanup.
Venv creation, pip installs and reviewed-source git checkouts all run
through it. The identity and signalling primitives it builds on live in
`muse.core.owned_process`.
"""
from __future__ import annotations

import locale
import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, BinaryIO, Sequence

from muse.core.owned_process import (
    MANAGED_JOB_GROUP_ENV,
    REAL_POPEN_TYPE,
    BoundedTail,
    cleanup_deadlines,
    concrete_safe_id,
    drain_stream_into,
    kill_signal,
    leader_exited_nowait,
    new_process_group_kwargs,
    signal_isolated_group,
    supports_pinned_leader,
    validated_isolated_group,
)

logger = logging.getLogger(__name__)


# Quiet installs drain both pipes continuously so a chatty installer cannot
# deadlock, but retain only a bounded tail for diagnostics.  The limit is per
# stream, so one child retains at most 1 MiB plus small bookkeeping overhead.
_CAPTURE_LIMIT_BYTES = 512 * 1024
_CAPTURE_READ_BYTES = 64 * 1024
_CAPTURE_JOIN_TIMEOUT = 1.0


# Cleanup adds at most this much wall time after the command's own deadline.
# TERM and KILL share the budget instead of each receiving a full timeout.
_CLEANUP_TIMEOUT = 5.0
_OS_NAME = os.name
_REAL_POPEN_TYPE = REAL_POPEN_TYPE
_MANAGED_JOB_GROUP_ENV = MANAGED_JOB_GROUP_ENV


@dataclass
class _OwnedProcess:
    """One exact child plus its positively-validated isolated group.

    On supported POSIX platforms the leader is observed without reaping
    (waitid(WNOWAIT), or psutil zombie status where waitid is missing),
    which deliberately keeps it unreaped while the owned group is cleaned.
    That live/zombie leader pins both its PID and equal PGID against reuse.
    """

    process: subprocess.Popen
    pid: int | None
    process_group: int | None
    leader_exit_observed: bool = False
    leader_reaped: bool = False


class _OwnedProcessCleanupError(RuntimeError):
    """The runner could not safely complete its required cleanup sequence."""


class _BoundedCapture(BoundedTail):
    """Keep the tail of one binary pipe while accounting for truncation.

    Installer output is decoded with the locale encoding. Read errors are
    not rendered: installer readers swallow them when cleanup closes a pipe.
    """

    def __init__(self, limit: int = _CAPTURE_LIMIT_BYTES) -> None:
        super().__init__(limit)

    def text(self) -> str:
        data, discarded, _read_error = self.snapshot()
        body = data.decode(locale.getpreferredencoding(False), errors="replace")
        if not discarded:
            return body
        return f"[... {discarded} output bytes truncated ...]\n{body}"


@dataclass
class _CaptureReader:
    stream: BinaryIO
    capture: _BoundedCapture
    thread: threading.Thread


_concrete_safe_id = concrete_safe_id


def _validated_isolated_group(
    pid: int | None, *, new_session: bool = False,
) -> int | None:
    """Return a child-owned POSIX process group only when fully proven."""
    return validated_isolated_group(
        pid, os_name=_OS_NAME, label="installer", new_session=new_session,
    )


def _spawn_owned(
    cmd: Sequence[str],
    *,
    capture_output: bool,
    env: Mapping[str, str] | None = None,
) -> _OwnedProcess:
    """Start one installer in an isolated group and retain its exact handle."""
    popen_kwargs: dict[str, Any] = {}
    # Inside an admin job the job's outer group is authoritative: a nested
    # session would escape the group JobStore drains, so the installer stays
    # in it and owns no group of its own (exact-handle fallback cleanup).
    if not (
        _OS_NAME == "posix"
        and os.environ.get(_MANAGED_JOB_GROUP_ENV) == "1"
    ):
        popen_kwargs.update(new_process_group_kwargs(_OS_NAME))
    if capture_output:
        # Unbuffered FileIO lets cleanup close the stream object itself. A raw
        # os.close(fd) would leave a Python wrapper holding the old number and
        # could later close an unrelated descriptor after numeric reuse.
        popen_kwargs.update(
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    if env is not None:
        popen_kwargs["env"] = dict(env)

    process = subprocess.Popen(list(cmd), **popen_kwargs)
    pid = _concrete_safe_id(getattr(process, "pid", None))
    return _OwnedProcess(
        process=process,
        pid=pid,
        process_group=_validated_isolated_group(
            pid, new_session=popen_kwargs.get("start_new_session") is True,
        ),
    )


def _drain_pipe(stream: BinaryIO, capture: _BoundedCapture) -> None:
    """Continuously drain one child pipe, retaining only its bounded tail."""
    try:
        drain_stream_into(stream, capture, read_bytes=_CAPTURE_READ_BYTES)
    except (OSError, ValueError):
        # Cleanup may close a pipe to release a reader whose writer did not
        # exit cleanly. The captured prefix/tail remains useful diagnostics.
        pass
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def _start_capture(target: _OwnedProcess) -> list[_CaptureReader]:
    readers: list[_CaptureReader] = []
    try:
        for name in ("stdout", "stderr"):
            stream = getattr(target.process, name, None)
            if stream is None:
                continue
            capture = _BoundedCapture()
            thread = threading.Thread(
                target=_drain_pipe,
                args=(stream, capture),
                name=f"muse-installer-{name}",
                daemon=True,
            )
            try:
                thread.start()
            except BaseException:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
                raise
            readers.append(_CaptureReader(stream=stream, capture=capture, thread=thread))
    except BaseException:
        _finish_capture(readers)
        raise
    return readers


def _finish_capture(readers: list[_CaptureReader]) -> tuple[str, str]:
    """Bound pipe-reader shutdown so inherited descriptors cannot hang Muse."""
    stuck: list[_CaptureReader] = []
    for reader in readers:
        reader.thread.join(timeout=_CAPTURE_JOIN_TIMEOUT)
        if reader.thread.is_alive():
            stuck.append(reader)

    if stuck:
        logger.warning(
            "installer cleanup left %d capture reader(s) still open; "
            "releasing Muse-owned streams",
            len(stuck),
        )

    # Supported POSIX cleanup has already signaled the group while its leader
    # pinned the identity. Unsupported-platform fallback may leave a surviving
    # descendant with a writer. At this point no numeric identity is safe to
    # revisit, so close only Muse's exact read descriptors and give the daemon
    # drain threads one final bounded opportunity to unwind.
    for reader in stuck:
        try:
            reader.stream.close()
        except (OSError, ValueError):
            pass

    for reader in stuck:
        reader.thread.join(timeout=_CAPTURE_JOIN_TIMEOUT)
        if reader.thread.is_alive():
            logger.warning(
                "installer %s reader did not close after its read fd was released",
                reader.thread.name,
            )

    values = [reader.capture.text() for reader in readers]
    while len(values) < 2:
        values.append("")
    return values[0], values[1]


def _target_alive(target: _OwnedProcess) -> bool:
    """Best-effort exact-handle liveness for unsupported platforms/tests."""
    if target.leader_reaped:
        return False
    try:
        returncode = target.process.poll()
    except Exception as exc:  # noqa: BLE001 - platform Popen handles vary
        logger.warning("could not prove installer leader is alive: %s", exc)
        return False
    if returncode is not None:
        # Popen.poll() reaps a POSIX child. Record that identity is no longer
        # safe before any caller could consider a numeric group signal.
        target.leader_reaped = True
        return False
    return True


def _supports_pinned_leader(target: _OwnedProcess) -> bool:
    """Whether this exact real POSIX child identity can be pinned.

    Pinned means its exit is observable without reaping:
    ``waitid(WNOWAIT)``, or psutil zombie status where waitid is missing.
    """
    return supports_pinned_leader(
        target.process, target.pid, target.process_group, os_name=_OS_NAME,
    )


def _observe_owned_leader(
    target: _OwnedProcess,
    *,
    timeout: float,
    command: Sequence[str],
) -> None:
    """Observe child exit without reaping, preserving PID/PGID ownership."""
    if not _supports_pinned_leader(target):
        raise _OwnedProcessCleanupError(
            "non-reaping observation requested for an unpinnable installer child"
        )
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            exited = leader_exited_nowait(target.pid)
        except InterruptedError:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(list(command), timeout)
            continue
        except ChildProcessError as exc:
            # Another reaper has released the identity; fail closed and never
            # use the stored numeric PID/PGID again.
            target.leader_reaped = True
            raise _OwnedProcessCleanupError(
                f"installer leader {target.pid} was reaped outside its owner"
            ) from exc
        except OSError as exc:
            raise _OwnedProcessCleanupError(
                f"could not observe installer leader {target.pid} safely"
            ) from exc
        if exited:
            target.leader_exit_observed = True
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(list(command), timeout)
        time.sleep(min(0.01, remaining))


def _wait_owned_leader(
    target: _OwnedProcess,
    *,
    timeout: float,
    command: Sequence[str],
) -> int | None:
    """Wait for a leader, retaining a supported POSIX child as a zombie."""
    if _supports_pinned_leader(target):
        _observe_owned_leader(target, timeout=timeout, command=command)
        return None
    returncode = target.process.wait(timeout=timeout)
    target.leader_reaped = True
    return returncode


def _reap_owned_leader(target: _OwnedProcess, *, deadline: float) -> int:
    """Reap only the exact child after the final safe group signal."""
    if target.leader_reaped:
        returncode = getattr(target.process, "returncode", None)
        if type(returncode) is int:
            return returncode
        raise _OwnedProcessCleanupError(
            f"installer leader {target.pid} was already reaped without status"
        )
    try:
        returncode = target.process.wait(
            timeout=max(0.0, deadline - time.monotonic())
        )
    except subprocess.TimeoutExpired as exc:
        raise _OwnedProcessCleanupError(
            f"installer leader {target.pid} could not be reaped before "
            "the cleanup deadline"
        ) from exc
    except (OSError, ValueError, ChildProcessError) as exc:
        raise _OwnedProcessCleanupError(
            f"could not reap installer leader {target.pid}"
        ) from exc
    target.leader_reaped = True
    return returncode


def _signal_exact_process(target: _OwnedProcess, *, force: bool) -> bool:
    if target.pid is None or target.leader_reaped:
        logger.error("refusing installer child signal with unsafe pid=%r", target.pid)
        return False
    try:
        if force:
            target.process.kill()
        else:
            target.process.terminate()
    except ProcessLookupError:
        return True
    except Exception as exc:  # noqa: BLE001 - platform Popen implementations vary
        logger.warning("could not signal installer child %s: %s", target.pid, exc)
        return False
    return True


def _signal_owned(target: _OwnedProcess, sig: signal.Signals, *, force: bool) -> bool:
    """Signal a pinned group, or use a documented best-effort fallback."""
    process_group = target.process_group
    pinned = _supports_pinned_leader(target) and not target.leader_reaped
    # Mock children retain the legacy double-poll path so safety behavior is
    # deterministic in unit tests. A real child that cannot be pinned (no
    # WNOWAIT and no psutil, or no validated group) deliberately avoids
    # numeric group signaling and falls back to the exact Popen handle.
    mocked_group = (
        process_group is not None
        and _OS_NAME == "posix"
        and not isinstance(target.process, _REAL_POPEN_TYPE)
    )
    if pinned or mocked_group:
        if mocked_group and not _target_alive(target):
            logger.warning(
                "refusing installer signal after exact leader exit "
                "(pid=%r pgid=%r)",
                target.pid, process_group,
            )
            return False
        # A verified group that failed signaling must not fall through to a
        # second numeric identity; the caller reports cleanup failure. Mock
        # children re-poll between validation and killpg (double poll).
        # ESRCH/EPERM counts as delivered only for a pinned leader that is
        # still our unreaped, exited child right after the signal.
        return signal_isolated_group(
            target.pid,
            process_group,
            sig,
            label="installer",
            still_owned=(
                (lambda: _target_alive(target)) if mocked_group else None
            ),
            pinned=pinned,
        )

    if not _target_alive(target):
        logger.warning(
            "refusing installer signal after exact leader exit (pid=%r pgid=%r)",
            target.pid, target.process_group,
        )
        return False
    return _signal_exact_process(target, force=force)


def _wait_until_gone(target: _OwnedProcess, deadline: float) -> bool:
    """Best-effort exact-child wait for unsupported platforms/tests."""
    while _target_alive(target):
        delay = min(0.05, deadline - time.monotonic())
        if delay <= 0:
            return False
        time.sleep(delay)
    return True


def _terminate_owned_fallback(
    target: _OwnedProcess, *, deadline: float, term_deadline: float,
) -> None:
    """Bounded exact-handle cleanup for a child that cannot be pinned."""
    if _target_alive(target):
        _signal_owned(target, signal.SIGTERM, force=False)
        if not _wait_until_gone(target, term_deadline):
            _signal_owned(target, kill_signal(), force=True)
            if not _wait_until_gone(target, deadline):
                raise _OwnedProcessCleanupError(
                    f"installer child {target.pid} did not exit before "
                    "the cleanup deadline"
                )
    try:
        target.process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired as exc:
        raise _OwnedProcessCleanupError(
            f"installer child {target.pid} could not be reaped before "
            "the cleanup deadline"
        ) from exc
    except (OSError, ValueError) as exc:
        raise _OwnedProcessCleanupError(
            f"could not reap installer child {target.pid}"
        ) from exc
    target.leader_reaped = True


def _terminate_owned(target: _OwnedProcess, *, timeout: float | None = None) -> None:
    """TERM then KILL an owned install tree within one shared time budget.

    Where the leader can be pinned (``waitid(WNOWAIT)``, or psutil zombie
    status where waitid is missing), it stays unreaped until both signals
    have been sent to its validated group,
    eliminating PID/PGID reuse. A child that cannot be pinned (no WNOWAIT
    and no psutil, or no validated group) gets bounded best-effort cleanup
    through the exact Popen handle; descendant cleanup cannot be guaranteed
    there.
    """
    if timeout is None:
        timeout = _CLEANUP_TIMEOUT
    term_deadline, deadline = cleanup_deadlines(timeout)
    if not _supports_pinned_leader(target):
        _terminate_owned_fallback(
            target, deadline=deadline, term_deadline=term_deadline,
        )
        return

    term_ok = _signal_owned(target, signal.SIGTERM, force=False)
    observed = target.leader_exit_observed
    if not observed:
        try:
            _observe_owned_leader(
                target,
                timeout=max(0.0, term_deadline - time.monotonic()),
                command=getattr(target.process, "args", ()),
            )
            observed = True
        except subprocess.TimeoutExpired:
            pass

    # Always send the terminal signal while the leader identity is pinned:
    # the leader may have honored TERM while a descendant ignored it.
    kill_ok = _signal_owned(target, kill_signal(), force=True)
    if not term_ok:
        logger.warning(
            "could not deliver TERM to installer group pid=%r pgid=%r",
            target.pid, target.process_group,
        )
    if not kill_ok:
        raise _OwnedProcessCleanupError(
            f"could not deliver final cleanup signal to installer group "
            f"{target.process_group}"
        )
    if not observed:
        try:
            _observe_owned_leader(
                target,
                timeout=max(0.0, deadline - time.monotonic()),
                command=getattr(target.process, "args", ()),
            )
        except subprocess.TimeoutExpired as exc:
            raise _OwnedProcessCleanupError(
                f"installer group {target.process_group} did not terminate "
                "before the cleanup deadline"
            ) from exc
    _reap_owned_leader(target, deadline=deadline)


def _run_owned(
    cmd: Sequence[str], *, timeout: float, capture_output: bool,
    check: bool = True, env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one bounded installer command and clean its tree on every failure."""
    target = _spawn_owned(cmd, capture_output=capture_output, env=env)
    readers: list[_CaptureReader] = []
    try:
        if capture_output:
            readers = _start_capture(target)
        returncode = _wait_owned_leader(
            target, timeout=timeout, command=cmd,
        )
        if returncode is None:
            # The exited leader remains an unreaped zombie, pinning the
            # validated equal PID/PGID. Kill any descendants that survived
            # their leader before releasing that identity.
            if not _signal_owned(target, kill_signal(), force=True):
                raise _OwnedProcessCleanupError(
                    f"could not clean completed installer group "
                    f"{target.process_group}"
                )
            returncode = _reap_owned_leader(
                target, deadline=time.monotonic() + _CLEANUP_TIMEOUT,
            )
    except subprocess.TimeoutExpired as exc:
        try:
            _terminate_owned(target)
        except BaseException as cleanup_exc:
            stdout, stderr = _finish_capture(readers)
            exc.stdout = stdout
            exc.stderr = stderr
            raise cleanup_exc from exc
        stdout, stderr = _finish_capture(readers)
        exc.stdout = stdout
        exc.stderr = stderr
        raise
    except BaseException as exc:
        try:
            _terminate_owned(target)
        except BaseException as cleanup_exc:
            _finish_capture(readers)
            raise cleanup_exc from exc
        _finish_capture(readers)
        raise

    # The exact leader is reaped only after the final group signal. No numeric
    # PID/PGID is inspected or signaled beyond this point.
    stdout, stderr = _finish_capture(readers)
    result = subprocess.CompletedProcess(list(cmd), returncode, stdout, stderr)
    if returncode != 0 and check:
        raise subprocess.CalledProcessError(
            returncode, list(cmd), output=stdout, stderr=stderr,
        )
    return result


def run_owned_command(
    cmd: Sequence[str],
    *,
    timeout: float,
    capture_output: bool = True,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command in an isolated group with bounded ownership-safe cleanup.

    This is the ownership-safe analogue of the small subprocess.run subset
    used by Muse. Where the leader can be pinned (``waitid(WNOWAIT)``, or
    psutil zombie status where waitid is missing), the validated group is
    cleaned while its leader pins the PID/PGID;
    otherwise a bounded exact-child best-effort fallback remains. Captured
    output keeps a bounded tail; check controls whether a non-zero exit is
    returned or raised.
    """
    return _run_owned(
        cmd,
        timeout=timeout,
        capture_output=capture_output,
        check=check,
    )
