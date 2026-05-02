"""Tests for image_input: parsing user-supplied images for img2img.

The helper accepts either:
  - a data URL: data:image/{png,jpeg,webp};base64,...
  - an http(s):// URL fetched via httpx (size-capped, content-type-checked,
    SSRF-protected: hostname must resolve to a public IP unless
    MUSE_ALLOW_PRIVATE_FETCH=1 is set).

Returns a PIL.Image. Decode failures raise ValueError so the route layer
can surface them as 400s.

`decode_image_input` is async (the http path uses httpx.AsyncClient to
avoid blocking the event loop on the calling worker), so all tests
that touch it are marked asyncio.
"""
import base64
import io

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from muse.modalities.image_generation.image_input import decode_image_input


def _png_bytes(width=64, height=64, color=(0, 128, 255)):
    """Build minimal PNG bytes via PIL."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_decode_data_url_png():
    raw = _png_bytes()
    data_url = f"data:image/png;base64,{base64.b64encode(raw).decode()}"
    img = await decode_image_input(data_url)
    assert img.size == (64, 64)
    assert img.mode in ("RGB", "RGBA")


@pytest.mark.asyncio
async def test_decode_data_url_jpeg():
    from PIL import Image
    rgb = Image.new("RGB", (32, 32), (255, 0, 0))
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG")
    data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    img = await decode_image_input(data_url)
    assert img.size == (32, 32)


def _async_response(content: bytes, headers: dict):
    """Build an awaitable AsyncClient.get response double."""
    fake_response = MagicMock()
    fake_response.content = content
    fake_response.headers = headers
    fake_response.raise_for_status = MagicMock()
    return fake_response


def _patch_async_client(fake_response):
    """Patch httpx.AsyncClient so its .get returns the fake response."""
    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)
    fake_client.get = AsyncMock(return_value=fake_response)
    return patch(
        "muse.modalities.image_generation.image_input.httpx.AsyncClient",
        return_value=fake_client,
    )


def _patch_dns(public_ip: str = "8.8.8.8"):
    """Patch socket.gethostbyname to return a public IP for any host."""
    return patch(
        "muse.modalities.image_generation.image_input.socket.gethostbyname",
        return_value=public_ip,
    )


@pytest.mark.asyncio
async def test_decode_http_url_fetches_via_async_httpx():
    raw = _png_bytes()
    fake_response = _async_response(raw, {"content-type": "image/png"})
    with _patch_dns(), _patch_async_client(fake_response):
        img = await decode_image_input("https://example.com/cat.png")
    assert img.size == (64, 64)


@pytest.mark.asyncio
async def test_decode_rejects_oversize_data_url():
    huge = b"\x00" * (11 * 1024 * 1024)  # 11MB
    data_url = f"data:image/png;base64,{base64.b64encode(huge).decode()}"
    with pytest.raises(ValueError, match="exceeds"):
        await decode_image_input(data_url, max_bytes=10 * 1024 * 1024)


@pytest.mark.asyncio
async def test_decode_data_url_rejects_oversize_before_b64_decode(monkeypatch):
    """Pre-b64-decode size guard: a 1GB b64 string must reject without
    inflating into RAM."""
    huge_b64 = "A" * (15 * 1024 * 1024)  # 15MB of b64 chars
    data_url = f"data:image/png;base64,{huge_b64}"
    # If b64decode were called we'd allocate ~11MB, but the pre-check
    # should reject first. Spy on b64decode to confirm.
    from muse.modalities.image_generation import image_input
    spy = MagicMock(side_effect=AssertionError("b64decode reached unexpectedly"))
    monkeypatch.setattr(image_input.base64, "b64decode", spy)
    with pytest.raises(ValueError, match="exceeds"):
        await decode_image_input(data_url, max_bytes=10 * 1024 * 1024)
    spy.assert_not_called()


@pytest.mark.asyncio
async def test_decode_rejects_non_image_http_content_type():
    fake_response = _async_response(
        b"<html>nope</html>", {"content-type": "text/html"},
    )
    with _patch_dns(), _patch_async_client(fake_response):
        with pytest.raises(ValueError, match="content-type"):
            await decode_image_input("https://example.com/page.html")


@pytest.mark.asyncio
async def test_decode_rejects_unknown_data_url_mime():
    raw = b"some text"
    data_url = f"data:text/plain;base64,{base64.b64encode(raw).decode()}"
    with pytest.raises(ValueError, match="MIME"):
        await decode_image_input(data_url)


@pytest.mark.asyncio
async def test_decode_rejects_invalid_url_shape():
    with pytest.raises(ValueError, match="must be"):
        await decode_image_input("ftp://example.com/img.png")


@pytest.mark.asyncio
async def test_decode_rejects_corrupt_image_bytes():
    raw = b"not really a png"
    data_url = f"data:image/png;base64,{base64.b64encode(raw).decode()}"
    with pytest.raises(ValueError, match="decode"):
        await decode_image_input(data_url)


# ---------------- SSRF: blocked on private/loopback/link-local ----------------


@pytest.mark.asyncio
async def test_fetch_blocks_loopback(monkeypatch):
    """A URL whose host resolves to 127.0.0.1 must reject without fetching."""
    monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
    fake_response = _async_response(b"", {"content-type": "image/png"})
    with _patch_dns("127.0.0.1"), _patch_async_client(fake_response) as p:
        with pytest.raises(ValueError, match="non-public"):
            await decode_image_input("https://attacker.example/payload.png")
    # AsyncClient should not have been instantiated; SSRF check fires first.
    p.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_blocks_link_local_aws_metadata(monkeypatch):
    """The classic SSRF target: 169.254.169.254 (AWS instance metadata)."""
    monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
    with _patch_dns("169.254.169.254"):
        with pytest.raises(ValueError, match="non-public"):
            await decode_image_input("http://metadata.example/iam/")


@pytest.mark.asyncio
async def test_fetch_blocks_private_lan(monkeypatch):
    """Private RFC1918 ranges (10.x, 172.16-31.x, 192.168.x) reject."""
    monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
    with _patch_dns("192.168.0.5"):
        with pytest.raises(ValueError, match="non-public"):
            await decode_image_input("https://router.lan/admin.png")


@pytest.mark.asyncio
async def test_fetch_blocks_internal_10_range(monkeypatch):
    monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
    with _patch_dns("10.0.0.1"):
        with pytest.raises(ValueError, match="non-public"):
            await decode_image_input("https://internal.corp/x.png")


@pytest.mark.asyncio
async def test_fetch_allows_override_via_env(monkeypatch):
    """Operators on a trusted network can opt out via MUSE_ALLOW_PRIVATE_FETCH=1."""
    monkeypatch.setenv("MUSE_ALLOW_PRIVATE_FETCH", "1")
    raw = _png_bytes()
    fake_response = _async_response(raw, {"content-type": "image/png"})
    # DNS doesn't even need to be patched: the env-var bypass returns
    # before resolution. But patch anyway to confirm the request goes
    # through despite the loopback IP.
    with _patch_dns("127.0.0.1"), _patch_async_client(fake_response):
        img = await decode_image_input("http://localhost:9001/internal.png")
    assert img.size == (64, 64)


@pytest.mark.asyncio
async def test_fetch_rejects_url_without_hostname(monkeypatch):
    """A schemed-but-empty-host URL has no IP to resolve."""
    monkeypatch.delenv("MUSE_ALLOW_PRIVATE_FETCH", raising=False)
    with pytest.raises(ValueError, match="hostname"):
        await decode_image_input("http:///no-host.png")


# ---------------- decode_image_file (multipart UploadFile) ----------------


class _FakeUploadFile:
    """Minimal async-file shape compatible with decode_image_file.

    Honors the size argument to .read() so the bounded-read fix is
    actually exercised. Records the requested size for assertions.
    """

    def __init__(self, data: bytes):
        self._data = data
        self.read_size: int | None = None

    async def read(self, size: int | None = None) -> bytes:
        self.read_size = size
        if size is None:
            return self._data
        return self._data[:size]


@pytest.mark.asyncio
async def test_decode_image_file_reads_png_from_upload():
    from muse.modalities.image_generation.image_input import decode_image_file
    raw = _png_bytes(width=48, height=24)
    upload = _FakeUploadFile(raw)
    img = await decode_image_file(upload)
    assert img.size == (48, 24)


@pytest.mark.asyncio
async def test_decode_image_file_rejects_empty():
    from muse.modalities.image_generation.image_input import decode_image_file
    upload = _FakeUploadFile(b"")
    with pytest.raises(ValueError, match="empty"):
        await decode_image_file(upload)


@pytest.mark.asyncio
async def test_decode_image_file_rejects_oversized():
    from muse.modalities.image_generation.image_input import decode_image_file
    huge = b"\x00" * (11 * 1024 * 1024)
    upload = _FakeUploadFile(huge)
    with pytest.raises(ValueError, match="exceeds"):
        await decode_image_file(upload, max_bytes=10 * 1024 * 1024)


@pytest.mark.asyncio
async def test_decode_image_file_uses_bounded_read():
    """The fix: read() is called with `max_bytes + 1`, not unbounded.

    A malicious 1GB upload must not buffer fully into worker RAM
    before the size check fires.
    """
    from muse.modalities.image_generation.image_input import decode_image_file
    # 11MB of zeroes; cap at 10MB. Read should request only the first
    # 10MB+1 bytes; oversize check fires after.
    huge = b"\x00" * (11 * 1024 * 1024)
    upload = _FakeUploadFile(huge)
    cap = 10 * 1024 * 1024
    with pytest.raises(ValueError, match="exceeds"):
        await decode_image_file(upload, max_bytes=cap)
    assert upload.read_size == cap + 1, (
        f"expected bounded read of {cap + 1}, got {upload.read_size}"
    )


@pytest.mark.asyncio
async def test_decode_image_file_rejects_undecodable():
    from muse.modalities.image_generation.image_input import decode_image_file
    upload = _FakeUploadFile(b"not really an image at all")
    with pytest.raises(ValueError, match="decode"):
        await decode_image_file(upload)
