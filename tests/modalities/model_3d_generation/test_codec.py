"""Codec tests for 3d/generation."""
import base64
import os

import pytest

from muse.modalities.model_3d_generation.codec import encode_3d_response
from muse.modalities.model_3d_generation.protocol import Generation3DResult


def test_b64_json_envelope_shape():
    results = [Generation3DResult(glb_bytes=b"GLB", model_id="triposr")]
    body = encode_3d_response(
        results, model_id="triposr", response_format="b64_json",
    )
    assert body["model"] == "triposr"
    assert body["id"].startswith("3d-")
    assert isinstance(body["created"], int)
    assert len(body["data"]) == 1
    entry = body["data"][0]
    assert entry["format"] == "glb"
    assert "b64_json" in entry
    assert "url" not in entry


def test_b64_json_round_trip():
    payload = b"\x67\x6c\x54\x46" + b"\x00" * 10  # mock GLB header + zeros
    results = [Generation3DResult(glb_bytes=payload, model_id="m")]
    body = encode_3d_response(results, model_id="m")
    decoded = base64.b64decode(body["data"][0]["b64_json"])
    assert decoded == payload


def test_url_mode_writes_tempfile():
    payload = b"GLB-bytes-for-url-mode"
    results = [Generation3DResult(glb_bytes=payload, model_id="m")]
    body = encode_3d_response(
        results, model_id="m", response_format="url",
    )
    entry = body["data"][0]
    assert entry["format"] == "glb"
    assert "url" in entry
    assert "b64_json" not in entry
    assert entry["url"].startswith("file://")
    path = entry["url"][len("file://"):]
    try:
        assert os.path.exists(path), "url-mode tempfile must exist"
        with open(path, "rb") as f:
            assert f.read() == payload
        assert path.endswith(".glb")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_default_response_format_is_b64_json():
    results = [Generation3DResult(glb_bytes=b"x", model_id="m")]
    body = encode_3d_response(results, model_id="m")  # no response_format
    assert "b64_json" in body["data"][0]


def test_invalid_response_format_raises():
    results = [Generation3DResult(glb_bytes=b"x", model_id="m")]
    with pytest.raises(ValueError, match="response_format"):
        encode_3d_response(results, model_id="m", response_format="json")


def test_unique_id_per_call():
    results = [Generation3DResult(glb_bytes=b"x", model_id="m")]
    a = encode_3d_response(results, model_id="m")
    b = encode_3d_response(results, model_id="m")
    assert a["id"] != b["id"]


def test_multiple_results_each_get_their_own_entry():
    results = [
        Generation3DResult(glb_bytes=b"first", model_id="m"),
        Generation3DResult(glb_bytes=b"second", model_id="m"),
    ]
    body = encode_3d_response(results, model_id="m")
    assert len(body["data"]) == 2
    assert base64.b64decode(body["data"][0]["b64_json"]) == b"first"
    assert base64.b64decode(body["data"][1]["b64_json"]) == b"second"


def test_format_field_passes_through():
    """Future formats (USDZ, etc.) round-trip through the codec."""
    results = [Generation3DResult(
        glb_bytes=b"x", model_id="m", format="usdz",
    )]
    body = encode_3d_response(results, model_id="m")
    assert body["data"][0]["format"] == "usdz"
