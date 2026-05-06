"""Venv management helpers.

Each pulled model gets its own venv under ~/.muse/venvs/<model-id>/.
This module handles creation and pip-install-into-venv; the catalog
records the resulting Python interpreter path per model.

Output discipline (v0.40.3+): `install_into_venv` defaults to quiet
mode (`pip install -q` + captured stdout/stderr, only emitted on
non-zero exit). The CLI's `muse pull -v` / `--verbose` flips into
pass-through mode via the `install_output_mode` context manager.
The verbose mode preserves the v0.40.2 firehose for dep-resolution
debugging.
"""
from __future__ import annotations

import contextlib
import logging
import socket
import subprocess
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


# 30 minutes: long enough for a slow PyPI mirror to finish a torch
# install on a fresh venv; short enough to detect a hung mirror before
# the user gives up. TimeoutExpired propagates to the catalog.pull
# caller, which surfaces it as a pull failure.
_PIP_TIMEOUT = 1800
_VENV_CREATE_TIMEOUT = 120


# Thread-local verbose flag. The CLI flips this via
# `install_output_mode(verbose=True)` when `muse pull -v` is set;
# default behavior (no flag) is quiet. Thread-local so concurrent
# pulls in different threads don't stomp each other (not a current
# muse use-case but cheap correctness insurance).
_local = threading.local()


def _is_verbose() -> bool:
    return bool(getattr(_local, "verbose", False))


@contextlib.contextmanager
def install_output_mode(verbose: bool):
    """Toggle pip + HF download output for everything called inside.

    Used by the CLI's `pull` command:

        with install_output_mode(verbose=args.verbose):
            pull(identifier)

    Inside the block: `install_into_venv` runs pip without `-q` and
    streams pip's stdout/stderr to the user; outside (or with
    verbose=False), pip is run with `-q` and stdout/stderr is
    captured, only emitted on non-zero exit.

    Catalog.py wraps snapshot_download similarly using the
    `HF_HUB_DISABLE_PROGRESS_BARS` env var whose value is read from
    `_is_verbose()`.
    """
    prev = getattr(_local, "verbose", False)
    _local.verbose = bool(verbose)
    try:
        yield
    finally:
        _local.verbose = prev


def venv_python(venv_path: Path) -> Path:
    """Return the Python interpreter path inside a venv.

    POSIX layout only (bin/python). The Windows layout (Scripts/python.exe)
    is not supported because muse is Linux/macOS-focused.
    """
    return venv_path / "bin" / "python"


def create_venv(target: Path) -> None:
    """Create a fresh venv at `target`, using the same Python that muse runs on.

    Using `sys.executable` guarantees ABI compatibility: the venv's Python
    is the same version as the muse-supervisor's Python, so torch/CUDA
    wheels built for one will load in the other.

    Honors `install_output_mode(verbose=...)`: in quiet mode captures
    stdout/stderr and only emits on error.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("creating venv at %s", target)
    cmd = [sys.executable, "-m", "venv", str(target)]
    if _is_verbose():
        subprocess.run(cmd, check=True, timeout=_VENV_CREATE_TIMEOUT)
        return
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=_VENV_CREATE_TIMEOUT,
    )
    if result.returncode != 0:
        if result.stdout:
            sys.stderr.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        result.check_returncode()


def install_into_venv(venv_path: Path, packages: list[str]) -> None:
    """pip-install `packages` using the venv's own pip.

    Shells out to `<venv>/bin/python -m pip install ...` so installs
    land in the target venv, not the supervisor's env.

    Output mode (v0.40.3+):
      - Quiet (default): adds `-q` to pip and captures stdout/stderr.
        On non-zero exit, both are emitted to stderr so the user can
        diagnose. On success, the user sees only the stage marker
        from the caller's `logger.info`.
      - Verbose (`muse pull -v`): no `-q`, no capture; pip's full
        output streams to the user's terminal as before.

    The mode flips via the `install_output_mode(verbose=...)` context
    manager (set by the CLI on each pull invocation).
    """
    if not packages:
        return
    py = venv_python(venv_path)
    verbose = _is_verbose()
    logger.info("installing %s into %s", packages, venv_path)
    cmd = [str(py), "-m", "pip", "install"]
    if not verbose:
        cmd.append("-q")
    cmd.extend(packages)

    if verbose:
        # Pass through: pip writes directly to stdout/stderr.
        subprocess.run(cmd, check=True, timeout=_PIP_TIMEOUT)
        return

    # Quiet: capture + emit-on-error.
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=_PIP_TIMEOUT,
    )
    if result.returncode != 0:
        if result.stdout:
            sys.stderr.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        result.check_returncode()


def find_free_port(start: int = 9001, end: int = 9999) -> int:
    """Find an unbound local port in [start, end]. Raises RuntimeError if exhausted.

    The port is probed by briefly binding then releasing it; the returned
    number is a *hint*, not a reservation. A TOCTOU window exists between
    this function returning and the caller binding the port, so the caller
    MUST verify the worker actually bound it (e.g., via /health check)
    and retry with a different port on startup failure.
    """
    for port in range(start, end + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"no free port in range [{start}, {end}]")
