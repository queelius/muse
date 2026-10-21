"""Local TCP port selection for per-model workers."""
from __future__ import annotations

import socket


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
