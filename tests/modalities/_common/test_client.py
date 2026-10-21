"""Tests for the shared client construction helpers."""
from __future__ import annotations

import io

import pytest

from muse.modalities._common.client import resolve_server_url, upload_bytes


def test_resolve_server_url_prefers_arg_and_strips_slash(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://env:9/")
    assert resolve_server_url("http://explicit:1/") == "http://explicit:1"
    assert resolve_server_url(None) == "http://env:9"


def test_upload_bytes_coerces_bytes_path_and_filelike(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"RIFF")
    assert upload_bytes(b"x", kind="audio") == b"x"
    assert upload_bytes(p, kind="audio") == b"RIFF"
    assert upload_bytes(str(p), kind="audio") == b"RIFF"
    assert upload_bytes(io.BytesIO(b"y"), kind="audio") == b"y"


class _FakeImage:
    def save(self, buf, format):
        assert format == "PNG"
        buf.write(b"PNGDATA")


def test_upload_bytes_pil_only_when_allowed():
    assert upload_bytes(_FakeImage(), kind="image", allow_pil=True) == b"PNGDATA"
    with pytest.raises(TypeError, match="unsupported image type: _FakeImage"):
        upload_bytes(_FakeImage(), kind="image")
    with pytest.raises(TypeError, match="unsupported audio type: int"):
        upload_bytes(3, kind="audio")
