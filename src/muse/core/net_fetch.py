"""Hardened URL fetch primitive shared by image routes and the MCP binary-IO resolver.

Public API:
    validate_public_host(url)     -- raises ValueError for private/loopback/etc. IPs.
    fetch_url_bytes(url, ...)     -- sync, host-validated, size-capped, manual-redirect.
    afetch_url_bytes(url, ...)    -- async variant (delegates to thread to stay DRY).

Design decisions:
  - follow_redirects=False + manual per-hop re-validation closes the open-redirect
    / DNS-rebind gap (H7) where the initial host check passes but the redirect
    target is a private IP.
  - Body is streamed chunk-by-chunk so the size cap fires before the full body
    is buffered into RAM (C3).
  - MUSE_ALLOW_PRIVATE_FETCH=1 escape hatch is preserved for operators on
    trusted internal networks (same as the original image_input convention).

IPv6 caveat (inherited from image_input._validate_public_host):
  socket.gethostbyname is IPv4-only. A hostname that resolves ONLY to an AAAA
  record will fail gethostbyname with OSError, causing a ValueError ("DNS
  resolution failed") before any fetch is attempted -- conservative, not
  permissive. A future hardening could use socket.getaddrinfo and reject if
  any address is non-public.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import socket
import urllib.parse

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSRF host validation
# ---------------------------------------------------------------------------


def validate_public_host(url: str) -> None:
    """Reject URLs whose host resolves to a non-public IP.

    Without this check, an untrusted ``url`` field can reach:
      - link-local:  169.254.169.254 (cloud instance metadata / IMDS)
      - loopback:    127.0.0.1, ::1  (worker's own admin/health port)
      - private LAN: 10.x, 172.16-31.x, 192.168.x (lateral scan)
      - multicast / reserved (uncommon but untrusted)

    Operators on a trusted network who want to fetch from internal services
    can opt out by setting MUSE_ALLOW_PRIVATE_FETCH=1.

    Raises:
        ValueError: with an actionable message that names MUSE_ALLOW_PRIVATE_FETCH.
    """
    if os.environ.get("MUSE_ALLOW_PRIVATE_FETCH") == "1":
        return
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    if not host:
        raise ValueError(f"URL has no hostname: {url!r}")
    try:
        ip_str = socket.gethostbyname(host)
    except OSError as e:
        raise ValueError(f"DNS resolution of {host!r} failed: {e}") from e
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError as e:
        raise ValueError(
            f"host {host!r} resolved to non-IP {ip_str!r}: {e}"
        ) from e
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        raise ValueError(
            f"refusing to fetch URL whose host {host!r} resolves to "
            f"non-public IP {ip!s}; set MUSE_ALLOW_PRIVATE_FETCH=1 to override"
        )


# ---------------------------------------------------------------------------
# Synchronous fetch
# ---------------------------------------------------------------------------


def fetch_url_bytes(
    url: str,
    *,
    max_bytes: int,
    timeout: float = 30.0,
    max_redirects: int = 3,
) -> bytes:
    """Fetch ``url`` synchronously, with SSRF guard, size cap, and safe redirect handling.

    - Validates the host of the initial URL with ``validate_public_host``.
    - Uses ``follow_redirects=False``; manually follows up to ``max_redirects``
      hops, re-validating the redirect target's host on every hop.
    - Streams the body and raises ``ValueError`` as soon as accumulated bytes
      exceed ``max_bytes`` (does NOT buffer the full body before checking).

    Raises:
        ValueError: SSRF block, size exceeded, too many redirects, HTTP error.
        httpx.HTTPStatusError: non-2xx final response (callers may re-raise as ValueError).
    """
    validate_public_host(url)

    current_url = url
    hops = 0

    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        while True:
            request = client.build_request("GET", current_url)
            response = client.send(request, stream=True)

            if response.is_redirect:
                hops += 1
                if hops > max_redirects:
                    raise ValueError(
                        f"too many redirects fetching {url!r} "
                        f"(limit {max_redirects})"
                    )
                location = response.headers.get("location", "")
                if not location:
                    raise ValueError(
                        f"redirect response from {current_url!r} has no Location header"
                    )
                # Resolve relative redirects against the current URL.
                next_url = urllib.parse.urljoin(current_url, location)
                # Re-validate: the redirect target might be a private IP
                # even if the original host was public (open-redirect / DNS-rebind).
                validate_public_host(next_url)
                current_url = next_url
                continue  # follow the hop

            # Non-redirect response: stream the body with a size cap.
            response.raise_for_status()
            buf = bytearray()
            for chunk in response.iter_bytes():
                buf += chunk
                if len(buf) > max_bytes:
                    raise ValueError(
                        f"response body exceeds max ({len(buf)} > {max_bytes} bytes) "
                        f"while fetching {url!r}"
                    )
            return bytes(buf)


# ---------------------------------------------------------------------------
# Async fetch (thin wrapper -- shares all logic with the sync variant)
# ---------------------------------------------------------------------------


async def afetch_url_bytes(
    url: str,
    *,
    max_bytes: int,
    timeout: float = 30.0,
    max_redirects: int = 3,
) -> bytes:
    """Async variant of ``fetch_url_bytes`` for use in async route handlers.

    Delegates to ``fetch_url_bytes`` via ``asyncio.to_thread`` so the event
    loop is not blocked during the synchronous httpx I/O. All SSRF, size-cap,
    and redirect semantics are identical to the sync version.
    """
    return await asyncio.to_thread(
        fetch_url_bytes,
        url,
        max_bytes=max_bytes,
        timeout=timeout,
        max_redirects=max_redirects,
    )
