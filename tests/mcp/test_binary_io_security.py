"""Security tests for muse.mcp.binary_io -- C1, C2, C3 fixes.

These complement the existing test_binary_io.py (which covers the basic
b64 / data-url / path / output-packing contract). This file focuses on:

  C1: URL fetch now has SSRF guard (no raw httpx.get)
  C3: URL fetch now has size cap (streaming, not r.content)
  C2: path input is now gated by MUSE_MCP_ALLOWED_PATH_PREFIXES

Run alongside the existing test_binary_io.py tests.
"""
from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from muse.mcp.binary_io import resolve_binary_input


SAMPLE_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20


# ---------------------------------------------------------------------------
# C1 / C3: URL path -- SSRF guard + size cap
# ---------------------------------------------------------------------------


def _patch_dns_net_fetch(ip: str):
    return patch(
        "muse.core.net_fetch.socket.gethostbyname",
        return_value=ip,
    )


class TestUrlSecurity:
    def test_ssrf_private_ip_rejected(self, monkeypatch):
        """C1: URL input must not be able to reach private IPs (SSRF)."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with _patch_dns_net_fetch("10.0.0.1"):
            with pytest.raises(ValueError, match="non-public"):
                resolve_binary_input(
                    url="http://internal.corp/secret.bin",
                    field_name="audio",
                )

    def test_ssrf_loopback_rejected(self, monkeypatch):
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with _patch_dns_net_fetch("127.0.0.1"):
            with pytest.raises(ValueError, match="non-public"):
                resolve_binary_input(
                    url="http://localhost/token",
                    field_name="image",
                )

    def test_ssrf_imds_rejected(self, monkeypatch):
        """The classic cloud IMDS SSRF target must be rejected."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        with _patch_dns_net_fetch("169.254.169.254"):
            with pytest.raises(ValueError, match="non-public"):
                resolve_binary_input(
                    url="http://169.254.169.254/latest/meta-data/",
                    field_name="audio",
                )

    def test_size_cap_enforced(self, monkeypatch):
        """C3: URL fetch must raise when body exceeds the size cap."""
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        # Set a tiny cap via the env var that binary_io reads.
        monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "100")

        # Simulate a public host so DNS passes.
        with _patch_dns_net_fetch("93.184.216.34"):
            # The inner fetch_url_bytes will raise ValueError("exceeds") once
            # accumulated bytes exceed the cap. We patch httpx.Client to
            # stream a body that is too large.
            big_chunk = b"A" * 200

            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {"content-type": "application/octet-stream"}
            resp.is_redirect = False
            resp.raise_for_status = MagicMock()
            resp.iter_bytes = MagicMock(return_value=iter([big_chunk]))

            inner = MagicMock()
            inner.send = MagicMock(return_value=resp)
            inner.build_request = MagicMock(return_value=MagicMock())
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=inner)
            ctx.__exit__ = MagicMock(return_value=False)

            with patch("muse.core.net_fetch.httpx.Client", return_value=ctx):
                with pytest.raises(ValueError, match="exceeds"):
                    resolve_binary_input(
                        url="http://big.example.com/huge.bin",
                        field_name="audio",
                    )

    def test_http_url_no_raw_httpx_get(self, monkeypatch):
        """Confirm binary_io no longer imports or uses httpx.get directly.

        After the refactor, the URL path should route through net_fetch
        (fetch_url_bytes), not via a bare httpx.get call. Patching
        httpx.get to raise should NOT interfere with a valid public-URL
        fetch (because httpx.get is never called).
        """
        monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
        # Public IP resolves.
        with _patch_dns_net_fetch("93.184.216.34"):
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {"content-type": "application/octet-stream"}
            resp.is_redirect = False
            resp.raise_for_status = MagicMock()
            resp.iter_bytes = MagicMock(return_value=iter([SAMPLE_BYTES]))

            inner = MagicMock()
            inner.send = MagicMock(return_value=resp)
            inner.build_request = MagicMock(return_value=MagicMock())
            ctx = MagicMock()
            ctx.__enter__ = MagicMock(return_value=inner)
            ctx.__exit__ = MagicMock(return_value=False)

            sentinel = {}

            def explode(*a, **kw):  # noqa: ANN002
                sentinel["called"] = True
                raise AssertionError("httpx.get must not be called directly")

            with (
                patch("muse.core.net_fetch.httpx.Client", return_value=ctx),
                patch("httpx.get", side_effect=explode),
            ):
                result = resolve_binary_input(
                    url="http://public.example.com/audio.bin",
                    field_name="audio",
                )
            assert result == SAMPLE_BYTES
            assert "called" not in sentinel, "httpx.get was called -- refactor incomplete"


# ---------------------------------------------------------------------------
# Size cap on the b64 / data: URL branches (the URL-fetch cap already
# applied via net_fetch; this closes the gap for the two branches that
# base64.b64decode with no ceiling).
# ---------------------------------------------------------------------------


class TestB64AndDataUrlSizeCap:
    def test_b64_input_over_cap_rejected(self, monkeypatch):
        monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "100")
        big = base64.b64encode(b"A" * 200).decode("ascii")
        with pytest.raises(ValueError, match="exceeds"):
            resolve_binary_input(b64=big, field_name="audio")

    def test_b64_input_under_cap_allowed(self, monkeypatch):
        monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "1000")
        small = base64.b64encode(b"A" * 50).decode("ascii")
        result = resolve_binary_input(b64=small, field_name="audio")
        assert result == b"A" * 50

    def test_b64_slot_with_data_prefix_over_cap_rejected(self, monkeypatch):
        # Same b64 slot, but the LLM included a leading data: prefix.
        monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "100")
        big = "data:audio/wav;base64," + base64.b64encode(b"A" * 200).decode("ascii")
        with pytest.raises(ValueError, match="exceeds"):
            resolve_binary_input(b64=big, field_name="audio")

    def test_data_url_over_cap_rejected(self, monkeypatch):
        monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "100")
        big = "data:audio/wav;base64," + base64.b64encode(b"A" * 200).decode("ascii")
        with pytest.raises(ValueError, match="exceeds"):
            resolve_binary_input(url=big, field_name="audio")

    def test_data_url_under_cap_allowed(self, monkeypatch):
        monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "1000")
        small = "data:audio/wav;base64," + base64.b64encode(b"A" * 50).decode("ascii")
        result = resolve_binary_input(url=small, field_name="audio")
        assert result == b"A" * 50


# ---------------------------------------------------------------------------
# C2: path allowlist
# ---------------------------------------------------------------------------


class TestPathAllowlist:
    def test_path_denied_by_default(self, monkeypatch, tmp_path):
        """When MUSE_MCP_ALLOWED_PATH_PREFIXES is unset, all paths are denied."""
        monkeypatch.delenv("MUSE_MCP_ALLOWED_PATH_PREFIXES", raising=False)
        f = tmp_path / "test.bin"
        f.write_bytes(SAMPLE_BYTES)
        with pytest.raises(ValueError, match="MUSE_MCP_ALLOWED_PATH_PREFIXES"):
            resolve_binary_input(path=str(f), field_name="audio")

    def test_path_allowed_when_prefix_set(self, monkeypatch, tmp_path):
        """Path under an allowed prefix is read successfully."""
        monkeypatch.setenv("MUSE_MCP_ALLOWED_PATH_PREFIXES", str(tmp_path))
        f = tmp_path / "audio.wav"
        f.write_bytes(SAMPLE_BYTES)
        result = resolve_binary_input(path=str(f), field_name="audio")
        assert result == SAMPLE_BYTES

    def test_path_outside_prefix_denied(self, monkeypatch, tmp_path):
        """A path that exists but is outside the allowed prefix is rejected."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        f = outside / "evil.bin"
        f.write_bytes(SAMPLE_BYTES)
        monkeypatch.setenv("MUSE_MCP_ALLOWED_PATH_PREFIXES", str(allowed))
        with pytest.raises(ValueError, match="not within an allowed prefix"):
            resolve_binary_input(path=str(f), field_name="audio")

    def test_traversal_escape_rejected(self, monkeypatch, tmp_path):
        """../  traversal that escapes the allowed prefix must be rejected."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "secret.bin"
        outside.write_bytes(b"top-secret")
        monkeypatch.setenv("MUSE_MCP_ALLOWED_PATH_PREFIXES", str(allowed))
        # Construct a traversal path: <allowed>/../secret.bin
        traversal = str(allowed / ".." / "secret.bin")
        with pytest.raises(ValueError, match="not within an allowed prefix"):
            resolve_binary_input(path=traversal, field_name="audio")

    def test_symlink_escape_rejected(self, monkeypatch, tmp_path):
        """A symlink inside the allowed prefix that points outside is rejected."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        secret = tmp_path / "secret.bin"
        secret.write_bytes(b"top-secret")
        link = allowed / "link.bin"
        link.symlink_to(secret)
        monkeypatch.setenv("MUSE_MCP_ALLOWED_PATH_PREFIXES", str(allowed))
        with pytest.raises(ValueError, match="not within an allowed prefix"):
            resolve_binary_input(path=str(link), field_name="audio")

    def test_multiple_prefixes_colon_separated(self, monkeypatch, tmp_path):
        """Multiple allowed prefixes are colon-separated (OS pathsep)."""
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        dir_b = tmp_path / "b"
        dir_b.mkdir()
        f = dir_b / "audio.wav"
        f.write_bytes(SAMPLE_BYTES)
        monkeypatch.setenv(
            "MUSE_MCP_ALLOWED_PATH_PREFIXES",
            os.pathsep.join([str(dir_a), str(dir_b)]),
        )
        result = resolve_binary_input(path=str(f), field_name="audio")
        assert result == SAMPLE_BYTES

    def test_path_not_found_after_allowlist_passes(self, monkeypatch, tmp_path):
        """A path that clears the allowlist but doesn't exist on disk
        still raises ValueError with 'not found'."""
        monkeypatch.setenv("MUSE_MCP_ALLOWED_PATH_PREFIXES", str(tmp_path))
        nonexistent = str(tmp_path / "ghost.bin")
        with pytest.raises(ValueError, match="not found"):
            resolve_binary_input(path=nonexistent, field_name="audio")
