"""Hardened URL fetch primitive shared by image routes and the MCP binary-IO resolver.

Public API:
    validate_public_host(url)     -- raises ValueError for private/loopback/etc. IPs.
    fetch_url_bytes(url, ...)     -- sync, host-validated, size-capped, manual-redirect.
    afetch_url_bytes(url, ...)    -- async variant (delegates to thread to stay DRY).

Design decisions:
  - follow_redirects=False + manual per-hop re-validation closes the open-redirect
    / DNS-rebind gap (H7) where the initial host check passes but the redirect
    target is a private IP.
  - IP pinning closes the DNS-rebinding TOCTOU (M4): each hop resolves the host
    to an IP, verifies it is public, then dials THAT EXACT IP (never re-resolving
    the hostname), so a low-TTL DNS flip or a multi-A-record host cannot make the
    connect land on 127.0.0.1 / 169.254.169.254 after the check passed. The
    original hostname rides in the Host header (for virtual hosting) and in the
    TLS SNI extension (for certificate verification).
  - Body is streamed chunk-by-chunk so the size cap fires before the full body
    is buffered into RAM (C3).
  - MUSE_ALLOW_PRIVATE_FETCH=1 escape hatch is preserved for operators on
    trusted internal networks (same as the original image_input convention);
    when set, resolution + pinning are skipped and the hostname is dialed as-is.

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


def resolve_public_ip(url: str) -> str | None:
    """Resolve the host of ``url``, verify the IP is public, and RETURN it.

    Returning (not discarding) the resolved IP is what lets the fetch PIN the
    connection to it: the IP that was checked is the exact IP that gets
    dialed, so a DNS-rebind flip (or a multi-A-record host) between check and
    connect cannot land on a private address (M4).

    Without this check, an untrusted ``url`` field can reach:
      - link-local:  169.254.169.254 (cloud instance metadata / IMDS)
      - loopback:    127.0.0.1, ::1  (worker's own admin/health port)
      - private LAN: 10.x, 172.16-31.x, 192.168.x (lateral scan)
      - multicast / reserved (uncommon but untrusted)

    Returns:
        The validated public IP string, or ``None`` when the SSRF guard is
        disabled (MUSE_ALLOW_PRIVATE_FETCH=1), signalling "dial the hostname
        as-is, no pinning."

    Raises:
        ValueError: missing hostname, DNS failure, or a non-public IP; the
        message names MUSE_ALLOW_PRIVATE_FETCH so operators can opt out.
    """
    if os.environ.get("MUSE_ALLOW_PRIVATE_FETCH") == "1":
        return None
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
    return ip_str


def validate_public_host(url: str) -> None:
    """Reject URLs whose host resolves to a non-public IP.

    Thin wrapper over :func:`resolve_public_ip` that discards the resolved
    IP; retained for callers that only need the raise-on-private check and
    do their own connection.

    Operators on a trusted network who want to fetch from internal services
    can opt out by setting MUSE_ALLOW_PRIVATE_FETCH=1.

    Raises:
        ValueError: with an actionable message that names MUSE_ALLOW_PRIVATE_FETCH.
    """
    resolve_public_ip(url)


def _pin_url_to_ip(url: str, ip: str) -> tuple[str, str]:
    """Rewrite ``url`` so its host is the pinned ``ip``.

    Returns ``(pinned_url, host_header)``: the request is sent to the IP, but
    the Host header carries the original hostname (plus explicit port) so
    virtual-hosting servers still route correctly. TLS SNI is set separately
    by the caller from the original hostname so certificate verification
    still checks the name, not the IP.
    """
    parsed = urllib.parse.urlsplit(url)
    ip_host = f"[{ip}]" if ":" in ip else ip  # bracket IPv6 literals
    netloc = ip_host if parsed.port is None else f"{ip_host}:{parsed.port}"
    pinned = urllib.parse.urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )
    host_header = parsed.hostname or ""
    if parsed.port is not None:
        host_header = f"{host_header}:{parsed.port}"
    return pinned, host_header


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

    - Resolves + validates the current host on every hop and PINS the
      connection to the validated IP (Host header + TLS SNI carry the real
      hostname), so no re-resolution can flip to a private IP (M4).
    - Uses ``follow_redirects=False``; manually follows up to ``max_redirects``
      hops, re-resolving + re-validating the redirect target's host each hop.
    - Streams the body and raises ``ValueError`` as soon as accumulated bytes
      exceed ``max_bytes`` (does NOT buffer the full body before checking).

    Raises:
        ValueError: SSRF block, size exceeded, too many redirects, HTTP error.
        httpx.HTTPStatusError: non-2xx final response (callers may re-raise as ValueError).
    """
    current_url = url
    hops = 0

    with httpx.Client(timeout=timeout, follow_redirects=False) as client:
        while True:
            # Resolve + validate the host, then dial THAT exact IP. Skipped
            # (pinned_ip is None) only when MUSE_ALLOW_PRIVATE_FETCH=1.
            pinned_ip = resolve_public_ip(current_url)
            if pinned_ip is not None:
                request_url, host_header = _pin_url_to_ip(current_url, pinned_ip)
                request = client.build_request(
                    "GET", request_url, headers={"Host": host_header},
                )
                # TLS SNI + cert verification use the real hostname, not the
                # pinned IP; httpcore reads this extension for server_hostname.
                request.extensions["sni_hostname"] = (
                    urllib.parse.urlparse(current_url).hostname or ""
                )
            else:
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
                # Resolve relative redirects against the current URL. The next
                # loop iteration resolves + validates + pins the target, so an
                # open-redirect / DNS-rebind to a private IP is caught there.
                next_url = urllib.parse.urljoin(current_url, location)
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
