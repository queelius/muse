"""Venv management helpers.

Each pulled model gets its own venv under ~/.muse/venvs/<model-id>/.
This module handles creation and pip-install-into-venv; the catalog
records the resulting Python interpreter path per model.
"""
from __future__ import annotations

import logging
import socket
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("creating venv at %s", target)
    subprocess.run(
        [sys.executable, "-m", "venv", str(target)],
        check=True,
    )


def install_into_venv(venv_path: Path, packages: list[str]) -> None:
    """pip-install `packages` using the venv's own pip.

    Shells out to `<venv>/bin/python -m pip install ...` so installs
    land in the target venv, not the supervisor's env.
    """
    if not packages:
        return
    py = venv_python(venv_path)
    logger.info("installing %s into %s", packages, venv_path)
    subprocess.run(
        [str(py), "-m", "pip", "install", *packages],
        check=True,
    )


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
