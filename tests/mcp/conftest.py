"""Test fixtures + pytest hooks for the muse mcp test suite.

Skips the entire mcp test suite when the ``mcp`` Python package is not
installed. The rest of muse's tests run regardless.
"""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Mark every mcp test as skipped if the SDK isn't importable."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(
            reason="mcp Python package not installed; "
                   "install via `pip install 'muse[server]'`",
        )
        for item in items:
            if "/tests/mcp/" in str(item.fspath) or "\\tests\\mcp\\" in str(item.fspath):
                item.add_marker(skip)
