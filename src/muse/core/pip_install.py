"""pip installs into a model's venv.

`install_into_venv` runs the venv's own pip through the ownership-safe
installer runner, quiet by default (see `muse.core.venv.install_output_mode`).
`pip_install_commands` yields the pip command sequence, including the
prebuilt CUDA llama-cpp-python wheel steps when `install.llama_cpp_wheel`
is set.
"""
from __future__ import annotations

import collections.abc
import logging
import subprocess
from pathlib import Path

from muse.core.owned_command import run_owned_command
from muse.core.venv import _emit_failure_output, _is_verbose, venv_python

logger = logging.getLogger(__name__)


# 30 minutes: long enough for a slow PyPI mirror to finish a torch
# install on a fresh venv; short enough to detect a hung mirror before
# the user gives up. TimeoutExpired propagates to the catalog.pull
# caller, which surfaces it as a pull failure.
_PIP_TIMEOUT = 1800


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
    def run(cmd: list[str]) -> None:
        try:
            run_owned_command(
                cmd, timeout=_PIP_TIMEOUT,
                capture_output=not verbose, check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if not verbose:
                _emit_failure_output(exc)
            raise

    for cmd in pip_install_commands(str(py), packages, quiet=not verbose):
        run(cmd)


def pip_install_commands(
    python: str, packages: list[str], *, upgrade: bool = False, quiet: bool = False,
) -> collections.abc.Iterator[list[str]]:
    """Yield the pip commands that install `packages`; run each before
    requesting the next.

    Usually one command. With `install.llama_cpp_wheel` set and a
    llama-cpp-python requirement present, llama-cpp-python installs as a
    second command from the prebuilt CUDA wheel index. pip keeps an existing
    CPU build of a satisfying version, so after those run, the build now
    installed is inspected and a CPU one is force-reinstalled (without deps).
    That check is why the commands are yielded lazily.
    """
    from muse.core import config, llama_cuda

    plan = llama_cuda.install_plan(list(packages), config.get("install.llama_cpp_wheel"))
    base = [python, "-m", "pip", "install"]
    if quiet:
        base.append("-q")
    if upgrade:
        base.append("--upgrade")
    for args in plan:
        yield [*base, *args]
    if len(plan) > 1 and not _is_cuda_llama_build(python):
        logger.info("replacing the CPU llama-cpp-python build in %s", python)
        yield [*base, *llama_cuda.reinstall_args(plan[1])]


def _is_cuda_llama_build(python: str) -> bool:
    from muse.core import llama_cuda

    try:
        result = run_owned_command(
            [python, "-c", llama_cuda.IS_CUDA_BUILD_SCRIPT],
            timeout=60, capture_output=True, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False  # unknown: reinstalling the wheel is the safe side
    return result.returncode == 0
