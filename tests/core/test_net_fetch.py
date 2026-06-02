"""Tests for muse.core.net_fetch -- the shared hardened fetch primitive.

Coverage:
  - validate_public_host: rejects private/loopback/link-local IPs
  - validate_public_host: allows public IPs
  - validate_public_host: honors MUSE_ALLOW_PRIVATE_FETCH=1
  - fetch_url_bytes: raises when body exceeds max_bytes (streaming check)
  - fetch_url_bytes: rejects a redirect whose Location resolves to a private IP (H7 regression)
  - fetch_url_bytes: follows a redirect to a public host successfully
  - fetch_url_bytes: raises after max_redirects hops
"""
from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

from muse.core.net_fetch import validate_public_host, fetch_url_bytes


# ---------------------------------------------------------------------------
# validate_public_host
# ---------------------------------------------------------------------------


def _patch_dns(ip: str):
    return patch(
        "muse.core.net_fetch.socket.gethostbyname",
        return_value=ip,
    )


class TestValidatePublicHost:
    def test_rejects_loopback_127(self, monkeypatch):
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with _patch_dns("127.0.0.1"):
            with pytest.raises(ValueError, match="non-public"):
                validate_public_host("http://localhost/path")

    def test_rejects_link_local_aws_imds(self, monkeypatch):
        """169.254.169.254 is the classic cloud IMDS SSRF target."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with _patch_dns("169.254.169.254"):
            with pytest.raises(ValueError, match="non-public"):
                validate_public_host("http://169.254.169.254/latest/meta-data/")

    def test_rejects_private_192_168(self, monkeypatch):
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with _patch_dns("192.168.1.1"):
            with pytest.raises(ValueError, match="non-public"):
                validate_public_host("http://router.local/")

    def test_rejects_private_10_x(self, monkeypatch):
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with _patch_dns("10.0.0.50"):
            with pytest.raises(ValueError, match="non-public"):
                validate_public_host("https://internal.corp/data")

    def test_rejects_loopback_ipv6(self, monkeypatch):
        """::1 (IPv6 loopback) should be rejected.
        Note: we only test this when the OS resolves it via gethostbyname.
        The IPv6 caveat is documented in net_fetch -- gethostbyname is IPv4-only;
        this test covers the case where the OS returns an IPv4-mapped form."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        # Simulate resolving ::1 as the IPv4-mapped "0.0.0.1" -- but
        # gethostbyname for "::1" may return "::1" itself as a string on some
        # systems. We patch it to return "127.0.0.1" (same loopback intent).
        with _patch_dns("127.0.0.1"):
            with pytest.raises(ValueError, match="non-public"):
                validate_public_host("http://[::1]/path")

    def test_allows_public_ip(self, monkeypatch):
        """A public IP (example.com -> 93.184.216.34) passes through."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with _patch_dns("93.184.216.34"):
            validate_public_host("https://example.com/image.png")  # must not raise

    def test_honors_allow_private_fetch_env(self, monkeypatch):
        """MUSE_ALLOW_PRIVATE_FETCH=1 bypasses all host checks."""
        monkeypatch.setenv("MUSE_ALLOW_PRIVATE_FETCH", "1")
        # No DNS patch needed -- the env bypass returns before resolution.
        validate_public_host("http://169.254.169.254/iam/")  # must not raise

    def test_rejects_url_with_no_hostname(self, monkeypatch):
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with pytest.raises(ValueError, match="hostname"):
            validate_public_host("http:///no-host")

    def test_error_message_names_escape_hatch(self, monkeypatch):
        """Error should mention MUSE_ALLOW_PRIVATE_FETCH so operators know
        how to opt out on trusted networks."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with _patch_dns("10.1.2.3"):
            with pytest.raises(ValueError, match="MUSE_ALLOW_PRIVATE_FETCH"):
                validate_public_host("https://internal.corp/x")


# ---------------------------------------------------------------------------
# fetch_url_bytes  (sync variant)
# ---------------------------------------------------------------------------


def _make_sync_response(
    *,
    status_code: int = 200,
    content: bytes = b"hello",
    headers: dict[str, str] | None = None,
    iter_chunks: list[bytes] | None = None,
):
    """Build a minimal httpx.Response-shaped double."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {"content-type": "application/octet-stream"}
    resp.is_redirect = status_code in (301, 302, 303, 307, 308)
    resp.next_request = None  # not used by our impl
    resp.raise_for_status = MagicMock()
    # iter_bytes yields chunks; fall back to full content
    if iter_chunks is not None:
        resp.iter_bytes = MagicMock(return_value=iter(iter_chunks))
    else:
        resp.iter_bytes = MagicMock(return_value=iter([content]))
    return resp


def _make_redirect_response(location: str, status_code: int = 302):
    resp = _make_sync_response(status_code=status_code, content=b"")
    resp.headers = {"location": location}
    resp.is_redirect = True
    return resp


class TestFetchUrlBytes:
    def test_successful_fetch(self, monkeypatch):
        """Happy path: public host, response under cap."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        monkeypatch.setattr(
            "muse.core.net_fetch.socket.gethostbyname",
            lambda host: "93.184.216.34",
        )
        resp = _make_sync_response(content=b"binary-data")
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=MagicMock(
            send=MagicMock(return_value=resp),
            build_request=MagicMock(return_value=MagicMock()),
        ))
        mock_client_ctx.__exit__ = MagicMock(return_value=False)
        with patch("muse.core.net_fetch.httpx.Client", return_value=mock_client_ctx):
            result = fetch_url_bytes(
                "https://example.com/data.bin", max_bytes=1024,
            )
        assert result == b"binary-data"

    def test_raises_when_body_exceeds_max_bytes(self, monkeypatch):
        """Streaming: raises ValueError as soon as bytes exceed cap,
        WITHOUT buffering the full body first."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        monkeypatch.setattr(
            "muse.core.net_fetch.socket.gethostbyname",
            lambda host: "93.184.216.34",
        )
        # Simulate a response streamed in two chunks: 500 + 700 = 1200 bytes.
        big_chunk_1 = b"A" * 500
        big_chunk_2 = b"B" * 700
        resp = _make_sync_response(
            iter_chunks=[big_chunk_1, big_chunk_2],
        )

        inner_client = MagicMock()
        inner_client.send = MagicMock(return_value=resp)
        inner_client.build_request = MagicMock(return_value=MagicMock())
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=inner_client)
        mock_client_ctx.__exit__ = MagicMock(return_value=False)

        with patch("muse.core.net_fetch.httpx.Client", return_value=mock_client_ctx):
            with pytest.raises(ValueError, match="exceeds"):
                fetch_url_bytes(
                    "https://example.com/huge.bin", max_bytes=999,
                )

    def test_redirect_to_private_ip_rejected(self, monkeypatch):
        """H7 regression: a 302 redirect whose Location resolves to a
        private IP must be rejected by re-validating the redirect host."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)

        call_count = {"n": 0}

        def dns_side_effect(host):
            call_count["n"] += 1
            if "example.com" in host:
                return "93.184.216.34"  # public
            if "internal" in host:
                return "10.0.0.1"  # private -- the redirect target
            return "93.184.216.34"

        monkeypatch.setattr(
            "muse.core.net_fetch.socket.gethostbyname",
            dns_side_effect,
        )

        redirect_resp = _make_redirect_response(
            "http://internal.corp/secret", status_code=302,
        )

        inner_client = MagicMock()
        inner_client.send = MagicMock(return_value=redirect_resp)
        inner_client.build_request = MagicMock(return_value=MagicMock())
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=inner_client)
        mock_client_ctx.__exit__ = MagicMock(return_value=False)

        with patch("muse.core.net_fetch.httpx.Client", return_value=mock_client_ctx):
            with pytest.raises(ValueError, match="non-public"):
                fetch_url_bytes(
                    "https://example.com/redirect-me", max_bytes=1024,
                )

    def test_redirect_to_public_host_followed(self, monkeypatch):
        """A 302 redirect to another public host is followed transparently."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        monkeypatch.setattr(
            "muse.core.net_fetch.socket.gethostbyname",
            lambda host: "93.184.216.34",
        )

        redirect_resp = _make_redirect_response(
            "https://cdn.example.com/data.bin", status_code=302,
        )
        final_resp = _make_sync_response(content=b"payload")

        send_calls = []

        def fake_send(request, **kwargs):
            send_calls.append(request)
            if len(send_calls) == 1:
                return redirect_resp
            return final_resp

        inner_client = MagicMock()
        inner_client.send = fake_send
        inner_client.build_request = MagicMock(return_value=MagicMock())
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=inner_client)
        mock_client_ctx.__exit__ = MagicMock(return_value=False)

        with patch("muse.core.net_fetch.httpx.Client", return_value=mock_client_ctx):
            result = fetch_url_bytes(
                "https://example.com/data", max_bytes=1024,
            )
        assert result == b"payload"
        assert len(send_calls) == 2  # initial + redirect

    def test_raises_after_max_redirects(self, monkeypatch):
        """Redirect loop or excessive hops must raise ValueError."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        monkeypatch.setattr(
            "muse.core.net_fetch.socket.gethostbyname",
            lambda host: "93.184.216.34",
        )

        redirect_resp = _make_redirect_response(
            "https://example.com/loop", status_code=302,
        )

        inner_client = MagicMock()
        # Always return a redirect -- every send is a loop.
        inner_client.send = MagicMock(return_value=redirect_resp)
        inner_client.build_request = MagicMock(return_value=MagicMock())
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=inner_client)
        mock_client_ctx.__exit__ = MagicMock(return_value=False)

        with patch("muse.core.net_fetch.httpx.Client", return_value=mock_client_ctx):
            with pytest.raises(ValueError, match="redirect"):
                fetch_url_bytes(
                    "https://example.com/loop", max_bytes=1024, max_redirects=3,
                )
