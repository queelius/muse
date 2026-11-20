"""Tests for muse.core.net_fetch -- the shared hardened fetch primitive.

Coverage:
  - resolve_public_ip: rejects private/loopback/link-local IPs
  - resolve_public_ip: allows public IPs
  - resolve_public_ip: honors MUSE_ALLOW_PRIVATE_FETCH=1
  - fetch_url_bytes: raises when body exceeds max_bytes (streaming check)
  - fetch_url_bytes: rejects a redirect whose Location resolves to a private IP (H7 regression)
  - fetch_url_bytes: follows a redirect to a public host successfully
  - fetch_url_bytes: raises after max_redirects hops
"""
from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

from muse.core.net_fetch import ResponseTooLargeError, fetch_url_bytes, resolve_public_ip


# ---------------------------------------------------------------------------
# resolve_public_ip
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
                resolve_public_ip("http://localhost/path")

    def test_rejects_link_local_aws_imds(self, monkeypatch):
        """169.254.169.254 is the classic cloud IMDS SSRF target."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with _patch_dns("169.254.169.254"):
            with pytest.raises(ValueError, match="non-public"):
                resolve_public_ip("http://169.254.169.254/latest/meta-data/")

    def test_rejects_private_192_168(self, monkeypatch):
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with _patch_dns("192.168.1.1"):
            with pytest.raises(ValueError, match="non-public"):
                resolve_public_ip("http://router.local/")

    def test_rejects_private_10_x(self, monkeypatch):
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with _patch_dns("10.0.0.50"):
            with pytest.raises(ValueError, match="non-public"):
                resolve_public_ip("https://internal.corp/data")

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
                resolve_public_ip("http://[::1]/path")

    def test_allows_public_ip(self, monkeypatch):
        """A public IP (example.com -> 93.184.216.34) passes through."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with _patch_dns("93.184.216.34"):
            resolve_public_ip("https://example.com/image.png")  # must not raise

    def test_honors_allow_private_fetch_env(self, monkeypatch):
        """MUSE_ALLOW_PRIVATE_FETCH=1 bypasses all host checks."""
        monkeypatch.setenv("MUSE_ALLOW_PRIVATE_FETCH", "1")
        # No DNS patch needed -- the env bypass returns before resolution.
        resolve_public_ip("http://169.254.169.254/iam/")  # must not raise

    def test_rejects_url_with_no_hostname(self, monkeypatch):
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with pytest.raises(ValueError, match="hostname"):
            resolve_public_ip("http:///no-host")

    def test_error_message_names_escape_hatch(self, monkeypatch):
        """Error should mention MUSE_ALLOW_PRIVATE_FETCH so operators know
        how to opt out on trusted networks."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with _patch_dns("10.1.2.3"):
            with pytest.raises(ValueError, match="MUSE_ALLOW_PRIVATE_FETCH"):
                resolve_public_ip("https://internal.corp/x")

    def test_rejects_cgnat_100_64(self, monkeypatch):
        """100.64.0.0/10 (Carrier-Grade NAT / shared address space, RFC
        6598) is not `is_private`/`is_loopback`/etc but IS non-global.
        The old explicit disjunction let it through; the guard must
        reject it like any other non-public range."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with _patch_dns("100.64.0.1"):
            with pytest.raises(ValueError, match="non-public"):
                resolve_public_ip("http://cgnat.example/path")

    def test_allows_cgnat_when_allow_private_fetch_set(self, monkeypatch):
        """The MUSE_ALLOW_PRIVATE_FETCH opt-out still applies to CGNAT."""
        monkeypatch.setenv("MUSE_ALLOW_PRIVATE_FETCH", "1")
        resolve_public_ip("http://100.64.0.1/path")  # must not raise


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
            with pytest.raises(ResponseTooLargeError, match="exceeds"):
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
            "https://internal.corp/secret", status_code=302,
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

    def test_connects_to_pinned_ip_not_rehostname(self, monkeypatch):
        """M4: after validating the host resolves to a public IP, fetch must
        dial THAT pinned IP rather than re-resolving the hostname, so a
        DNS-rebind flip (or a multi-A-record host) between the check and the
        connect cannot reach 127.0.0.1 / 169.254.169.254. The original
        hostname is preserved in the Host header for virtual hosting."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        monkeypatch.setattr(
            "muse.core.net_fetch.socket.gethostbyname",
            lambda host: "93.184.216.34",
        )
        resp = _make_sync_response(content=b"ok")
        inner_client = MagicMock()
        inner_client.build_request = MagicMock(return_value=MagicMock())
        inner_client.send = MagicMock(return_value=resp)
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=inner_client)
        mock_client_ctx.__exit__ = MagicMock(return_value=False)

        with patch("muse.core.net_fetch.httpx.Client", return_value=mock_client_ctx):
            fetch_url_bytes("https://example.com/data.bin", max_bytes=1024)

        call = inner_client.build_request.call_args
        # ("GET", <url>): the request URL must target the pinned IP.
        url_arg = call.args[1]
        assert "93.184.216.34" in url_arg
        assert "example.com" not in url_arg
        # Host header carries the original hostname so vhosts still route.
        headers = call.kwargs.get("headers") or {}
        assert headers.get("Host") == "example.com"

    def test_pinning_disabled_when_private_fetch_allowed(self, monkeypatch):
        """MUSE_ALLOW_PRIVATE_FETCH=1 skips SSRF resolution and connects to
        the hostname as-is (no pinning), preserving the internal-network
        escape hatch."""
        monkeypatch.setenv("MUSE_ALLOW_PRIVATE_FETCH", "1")
        # gethostbyname must NOT be consulted when the guard is disabled.
        monkeypatch.setattr(
            "muse.core.net_fetch.socket.gethostbyname",
            MagicMock(side_effect=AssertionError("should not resolve")),
        )
        resp = _make_sync_response(content=b"ok")
        inner_client = MagicMock()
        inner_client.build_request = MagicMock(return_value=MagicMock())
        inner_client.send = MagicMock(return_value=resp)
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=inner_client)
        mock_client_ctx.__exit__ = MagicMock(return_value=False)

        with patch("muse.core.net_fetch.httpx.Client", return_value=mock_client_ctx):
            fetch_url_bytes("http://internal.corp/x", max_bytes=1024)

        url_arg = inner_client.build_request.call_args.args[1]
        assert "internal.corp" in url_arg

    def test_redirect_response_is_closed_before_following(self, monkeypatch):
        """A streamed redirect response must be closed before the next
        hop is dialed, else the underlying connection leaks on every
        redirect (only the final, fully-consumed response was closed
        before this fix)."""
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
            fetch_url_bytes("https://example.com/data", max_bytes=1024)

        redirect_resp.close.assert_called_once()

    def test_response_closed_when_too_many_redirects(self, monkeypatch):
        """The error path (redirect loop) must close the response before
        raising, not just leave the streamed connection dangling."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        monkeypatch.setattr(
            "muse.core.net_fetch.socket.gethostbyname",
            lambda host: "93.184.216.34",
        )

        redirect_resp = _make_redirect_response(
            "https://example.com/loop", status_code=302,
        )

        inner_client = MagicMock()
        inner_client.send = MagicMock(return_value=redirect_resp)
        inner_client.build_request = MagicMock(return_value=MagicMock())
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=inner_client)
        mock_client_ctx.__exit__ = MagicMock(return_value=False)

        with patch("muse.core.net_fetch.httpx.Client", return_value=mock_client_ctx):
            with pytest.raises(ValueError, match="redirect"):
                fetch_url_bytes(
                    "https://example.com/loop", max_bytes=1024, max_redirects=1,
                )

        assert redirect_resp.close.called

    def test_response_closed_when_size_exceeded(self, monkeypatch):
        """The size-cap error path must close the streamed response
        before raising, since iter_bytes was aborted mid-stream (not
        exhausted, so httpx's own auto-close on exhaustion never fires)."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        monkeypatch.setattr(
            "muse.core.net_fetch.socket.gethostbyname",
            lambda host: "93.184.216.34",
        )
        big_chunk_1 = b"A" * 500
        big_chunk_2 = b"B" * 700
        resp = _make_sync_response(iter_chunks=[big_chunk_1, big_chunk_2])

        inner_client = MagicMock()
        inner_client.send = MagicMock(return_value=resp)
        inner_client.build_request = MagicMock(return_value=MagicMock())
        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=inner_client)
        mock_client_ctx.__exit__ = MagicMock(return_value=False)

        with patch("muse.core.net_fetch.httpx.Client", return_value=mock_client_ctx):
            with pytest.raises(ValueError, match="exceeds"):
                fetch_url_bytes("https://example.com/huge.bin", max_bytes=999)

        resp.close.assert_called_once()

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


def _client_ctx(*responses):
    inner = MagicMock()
    inner.send = MagicMock(side_effect=list(responses))
    inner.build_request = MagicMock(return_value=MagicMock(extensions={}))
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=inner)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx, inner


class TestFetchSchemeAndDeadline:
    @pytest.fixture(autouse=True)
    def _public_dns(self, monkeypatch):
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        monkeypatch.setattr(
            "muse.core.net_fetch.socket.gethostbyname", lambda host: "93.184.216.34",
        )

    def test_non_http_initial_scheme_rejected(self):
        ctx, inner = _client_ctx()
        with patch("muse.core.net_fetch.httpx.Client", return_value=ctx):
            with pytest.raises(ValueError, match="scheme"):
                fetch_url_bytes("ftp://example.com/x", max_bytes=10)
        inner.send.assert_not_called()

    def test_redirect_to_non_http_scheme_rejected(self):
        ctx, inner = _client_ctx(_make_redirect_response("ftp://example.com/x"))
        with patch("muse.core.net_fetch.httpx.Client", return_value=ctx):
            with pytest.raises(ValueError, match="scheme"):
                fetch_url_bytes("https://example.com/a", max_bytes=10)
        assert inner.send.call_count == 1

    def test_https_to_http_downgrade_rejected(self):
        ctx, inner = _client_ctx(_make_redirect_response("http://example.com/x"))
        with patch("muse.core.net_fetch.httpx.Client", return_value=ctx):
            with pytest.raises(ValueError, match="downgrade"):
                fetch_url_bytes("https://example.com/a", max_bytes=10)
        assert inner.send.call_count == 1

    def test_http_to_https_redirect_followed(self):
        ctx, inner = _client_ctx(
            _make_redirect_response("https://example.com/x"),
            _make_sync_response(content=b"ok"),
        )
        with patch("muse.core.net_fetch.httpx.Client", return_value=ctx):
            assert fetch_url_bytes("http://example.com/a", max_bytes=10) == b"ok"

    def test_total_deadline_stops_a_slow_drip(self, monkeypatch):
        clock = [1000.0]
        monkeypatch.setattr("muse.core.net_fetch.time.monotonic", lambda: clock[0])

        def drip():
            for _ in range(100):
                clock[0] += 29.0  # each chunk arrives just under a read timeout
                yield b"x"

        resp = _make_sync_response()
        resp.iter_bytes = MagicMock(return_value=drip())
        ctx, _ = _client_ctx(resp)
        with patch("muse.core.net_fetch.httpx.Client", return_value=ctx):
            with pytest.raises(ValueError, match="deadline"):
                fetch_url_bytes("https://example.com/a", max_bytes=1000,
                                total_timeout=60.0)
        resp.close.assert_called()

    def test_per_hop_timeout_capped_by_remaining_budget(self, monkeypatch):
        clock = [1000.0]
        monkeypatch.setattr("muse.core.net_fetch.time.monotonic", lambda: clock[0])
        redirect = _make_redirect_response("https://example.com/b")

        def send(request, stream):
            if inner.send.call_count == 1:
                clock[0] += 50.0
                return redirect
            return _make_sync_response(content=b"ok")

        ctx, inner = _client_ctx()
        inner.send = MagicMock(side_effect=send)
        with patch("muse.core.net_fetch.httpx.Client", return_value=ctx):
            assert fetch_url_bytes("https://example.com/a", max_bytes=10,
                                   timeout=30.0, total_timeout=60.0) == b"ok"
        timeouts = [c.kwargs["timeout"] for c in inner.build_request.call_args_list]
        assert timeouts[0] == 30.0
        assert timeouts[1] == pytest.approx(10.0)

    def test_deadline_exhausted_before_next_hop(self, monkeypatch):
        clock = [1000.0]
        monkeypatch.setattr("muse.core.net_fetch.time.monotonic", lambda: clock[0])

        def send(request, stream):
            clock[0] += 61.0
            return _make_redirect_response("https://example.com/b")

        ctx, inner = _client_ctx()
        inner.send = MagicMock(side_effect=send)
        with patch("muse.core.net_fetch.httpx.Client", return_value=ctx):
            with pytest.raises(ValueError, match="deadline"):
                fetch_url_bytes("https://example.com/a", max_bytes=10,
                                total_timeout=60.0)
        assert inner.send.call_count == 1


def test_response_too_large_is_a_typed_value_error():
    from muse.core.net_fetch import ResponseTooLargeError

    assert issubclass(ResponseTooLargeError, ValueError)
