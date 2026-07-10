"""Runtime package installation helpers.

Keeps the CLI dependency graph slim: pull-a-model may install pip extras
on demand rather than forcing users to install everything upfront.

Note: this module used to also export `install_pip_extras`, which
installed missing packages into `sys.executable`'s own environment (the
supervisor process's env). It was dead code -- nothing in muse called
it -- and if it HAD been called, it would have defeated per-model venv
isolation (every model pull installs into its own venv via
`muse.core.venv.install_into_venv`, never into the process running the
CLI/supervisor). Removed rather than fixed in place, since the correct
call site already exists elsewhere.
"""
from __future__ import annotations

import shutil


def check_system_packages(packages: list[str]) -> list[str]:
    """Return the subset of system packages not found on PATH."""
    return [p for p in packages if shutil.which(p) is None]
