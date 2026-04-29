"""Tests for muse.mcp.binary_io.

Cover the three input forms (b64 / url / path) plus error cases. Output
packing is tested for image and audio variants.
"""
from __future__ import annotations

import base64
import os
import tempfile

import pytest

from muse.mcp.binary_io import (
    binary_input_schema,
    pack_audio_output,
    pack_image_output,
    pack_text_output,
    resolve_binary_input,
)


SAMPLE_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10


class TestResolveBinaryInput:
    def test_b64(self):
        b64 = base64.b64encode(SAMPLE_BYTES).decode("ascii")
        out = resolve_binary_input(b64=b64, field_name="image")
        assert out == SAMPLE_BYTES

    def test_b64_with_data_prefix(self):
        b64 = "data:image/png;base64," + base64.b64encode(SAMPLE_BYTES).decode("ascii")
        out = resolve_binary_input(b64=b64, field_name="image")
        assert out == SAMPLE_BYTES

    def test_data_url(self):
        url = "data:image/png;base64," + base64.b64encode(SAMPLE_BYTES).decode("ascii")
        out = resolve_binary_input(url=url, field_name="image")
        assert out == SAMPLE_BYTES

    def test_path_roundtrip(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            f.write(SAMPLE_BYTES)
            path = f.name
        try:
            out = resolve_binary_input(path=path, field_name="image")
            assert out == SAMPLE_BYTES
        finally:
            os.unlink(path)

    def test_missing_input_raises(self):
        with pytest.raises(ValueError, match="missing audio input"):
            resolve_binary_input(field_name="audio")

    def test_too_many_inputs_raises(self):
        b64 = base64.b64encode(b"x").decode("ascii")
        with pytest.raises(ValueError, match="too many image inputs"):
            resolve_binary_input(b64=b64, url="data:," + b64, field_name="image")

    def test_unsupported_url_scheme(self):
        with pytest.raises(ValueError, match="unsupported"):
            resolve_binary_input(url="ftp://nope/path", field_name="image")

    def test_path_not_found(self):
        with pytest.raises(ValueError, match="not found"):
            resolve_binary_input(
                path="/tmp/__definitely_not_a_real_path__.bin",
                field_name="image",
            )

    def test_malformed_b64_raises(self):
        with pytest.raises(ValueError, match="malformed base64"):
            resolve_binary_input(b64="not!!!base64!!!", field_name="image")

    def test_malformed_data_b64_no_comma(self):
        with pytest.raises(ValueError, match="malformed data URL"):
            resolve_binary_input(b64="data:image/png;base64", field_name="image")

    def test_malformed_data_url_no_comma(self):
        with pytest.raises(ValueError, match="malformed data URL"):
            resolve_binary_input(url="data:image/png;base64", field_name="image")

    def test_http_url_uses_httpx(self, monkeypatch):
        # Patch httpx.get to return SAMPLE_BYTES directly (no network).
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.content = SAMPLE_BYTES
        mock_resp.raise_for_status = MagicMock()
        called = {}

        def fake_get(url, **kwargs):
            called["url"] = url
            return mock_resp

        import muse.mcp.binary_io as binary_io_mod
        monkeypatch.setattr(binary_io_mod.httpx, "get", fake_get)
        out = resolve_binary_input(
            url="http://example.com/img.png", field_name="image",
        )
        assert out == SAMPLE_BYTES
        assert called["url"] == "http://example.com/img.png"


class TestPackOutputs:
    def test_pack_image(self):
        block = pack_image_output(SAMPLE_BYTES)
        assert block["type"] == "image"
        assert block["mimeType"] == "image/png"
        assert base64.b64decode(block["data"]) == SAMPLE_BYTES

    def test_pack_image_custom_mime(self):
        block = pack_image_output(SAMPLE_BYTES, mime="image/webp")
        assert block["mimeType"] == "image/webp"

    def test_pack_audio(self):
        block = pack_audio_output(b"RIFFxxxxxx")
        assert block["type"] == "audio"
        assert block["mimeType"] == "audio/wav"
        assert base64.b64decode(block["data"]) == b"RIFFxxxxxx"

    def test_pack_audio_custom_mime(self):
        block = pack_audio_output(b"\x00", mime="audio/opus")
        assert block["mimeType"] == "audio/opus"

    def test_pack_text(self):
        block = pack_text_output("hello")
        assert block == {"type": "text", "text": "hello"}


class TestSchema:
    def test_schema_has_three_fields(self):
        s = binary_input_schema("image")
        assert set(s.keys()) == {"image_b64", "image_url", "image_path"}
        for v in s.values():
            assert v["type"] == "string"
            assert "description" in v

    def test_schema_field_name_substituted(self):
        s = binary_input_schema("audio")
        assert "audio_b64" in s
        assert "audio_url" in s
        assert "audio_path" in s
