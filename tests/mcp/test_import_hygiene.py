"""`muse mcp` (stdio) must import nothing from muse[server]."""
from __future__ import annotations

import subprocess
import sys


def test_stdio_import_is_light():
    out = subprocess.run([sys.executable, "-c",
        "import sys, muse.cli_impl.mcp_server, muse.mcp.server, muse.mcp.generated;"
        "print('fastapi' in sys.modules, 'muse.cli_impl.gateway' in sys.modules)"],
        capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]
