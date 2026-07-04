"""Public API re-export contract for muse.observability.

The package must re-export the full public surface (store, recorder,
logs, dashboard) directly from `muse.observability`, so callers do not
need to know which submodule a name lives in. It must ALSO stay
import-light: importing a stdlib-only submodule (e.g. `recorder`, used
from the hot request path) must not drag fastapi in as a side effect of
running `__init__.py`.
"""
from __future__ import annotations

import subprocess
import sys


def test_public_api_reexports():
    from muse.observability import (
        record,
        get_recorder,
        reset_recorder,
        init_recorder,
        LogHub,
        build_dashboard_router,
        TelemetryStore,
    )

    assert callable(record)
    assert callable(get_recorder)
    assert callable(reset_recorder)
    assert callable(init_recorder)
    assert LogHub is not None
    assert callable(build_dashboard_router)
    assert TelemetryStore is not None


def test_recorder_import_is_fastapi_free_in_a_clean_interpreter():
    """Guard the import-light contract in a fresh interpreter.

    An in-process check can't prove fastapi wasn't already imported by
    an earlier test in the same session, so this shells out to a clean
    `python -c` subprocess.
    """
    code = (
        "import sys; "
        "import muse.observability.recorder; "
        "assert 'fastapi' not in sys.modules, "
        "'fastapi leaked into import-light path'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
