"""Codec tests for 3d/generation."""
import base64
from unittest.mock import MagicMock

import pytest

from muse.modalities.model_3d_generation.codec import encode_3d_response, mesh_to_glb_result
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


def test_url_mode_returns_data_url():
    """url mode emits a data URL with model/gltf-binary MIME (per glTF spec).

    Avoids unreachable file:// URLs from a remote muse server and an
    unbounded server-side tempfile leak.
    """
    payload = b"GLB-bytes-for-url-mode"
    results = [Generation3DResult(glb_bytes=payload, model_id="m")]
    body = encode_3d_response(
        results, model_id="m", response_format="url",
    )
    entry = body["data"][0]
    assert entry["format"] == "glb"
    assert "url" in entry
    assert "b64_json" not in entry
    prefix = "data:model/gltf-binary;base64,"
    assert entry["url"].startswith(prefix)
    encoded = entry["url"][len(prefix):]
    assert base64.b64decode(encoded) == payload


def test_url_mode_does_not_write_tempfile(tmp_path, monkeypatch):
    """Regression: url mode must not leak server-side files."""
    import tempfile as _tempfile

    calls = []
    real_named = _tempfile.NamedTemporaryFile

    def _track(*args, **kwargs):
        calls.append((args, kwargs))
        return real_named(*args, **kwargs)

    monkeypatch.setattr(_tempfile, "NamedTemporaryFile", _track)
    results = [Generation3DResult(glb_bytes=b"GLB", model_id="m")]
    encode_3d_response(results, model_id="m", response_format="url")
    assert calls == [], (
        "url mode must not allocate a server-side tempfile; "
        f"saw {len(calls)} NamedTemporaryFile call(s)"
    )


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


# ---------------- mesh_to_glb_result helper ----------------


def test_mesh_to_glb_result_returns_correct_result():
    """mesh_to_glb_result wraps a trimesh-like mesh into a Generation3DResult."""
    fake_mesh = MagicMock()
    sentinel_bytes = b"fake-glb-sentinel"
    fake_mesh.export = MagicMock(return_value=sentinel_bytes)

    result = mesh_to_glb_result(fake_mesh, "some-model")

    fake_mesh.export.assert_called_once_with(file_type="glb")
    assert isinstance(result, Generation3DResult)
    assert result.glb_bytes == sentinel_bytes
    assert result.model_id == "some-model"
    assert result.format == "glb"


def test_mesh_to_glb_result_coerces_bytearray_to_bytes():
    """bytes() coercion preserves safety when trimesh returns bytearray/memoryview."""
    fake_mesh = MagicMock()
    fake_mesh.export = MagicMock(return_value=bytearray(b"ba-payload"))

    result = mesh_to_glb_result(fake_mesh, "m")

    assert isinstance(result.glb_bytes, bytes)
    assert result.glb_bytes == b"ba-payload"


def test_mesh_to_glb_result_raises_on_empty_export():
    """An empty GLB export (degenerate / no-geometry mesh) must fail loud,
    not hand the client a zero-byte 'success' it cannot open."""
    fake_mesh = MagicMock()
    fake_mesh.export = MagicMock(return_value=b"")

    with pytest.raises(ValueError, match="zero bytes"):
        mesh_to_glb_result(fake_mesh, "m")
