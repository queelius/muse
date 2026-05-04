"""Tests for OcrClient HTTP client."""
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_response(body: dict, status: int = 200):
    mock = MagicMock(
        status_code=status,
        headers={"content-type": "application/json"},
    )
    mock.json = MagicMock(return_value=body)
    mock.text = json.dumps(body)
    mock.raise_for_status = MagicMock()
    return mock


def test_default_server_url():
    from muse.modalities.image_ocr import OcrClient
    c = OcrClient()
    assert c.server_url == "http://localhost:8000"


def test_trailing_slash_stripped():
    from muse.modalities.image_ocr import OcrClient
    c = OcrClient(server_url="http://lan:8000/")
    assert c.server_url == "http://lan:8000"


def test_muse_server_env_fallback(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://custom:9999")
    from muse.modalities.image_ocr import OcrClient
    c = OcrClient()
    assert c.server_url == "http://custom:9999"


def test_ocr_call_with_bytes_input():
    body = {"id": "ocr-1", "model": "trocr",
            "text": "hello", "usage": {"completion_tokens": 3}}
    with patch("muse.modalities.image_ocr.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_ocr import OcrClient
        out = OcrClient().ocr(b"\x89PNG-fake-bytes", model="trocr")
    assert out["text"] == "hello"
    assert mock_post.call_args.kwargs["files"]["image"][1] == b"\x89PNG-fake-bytes"


def test_ocr_call_with_path_input(tmp_path):
    body = {"id": "ocr-1", "model": "m", "text": "ok", "usage": {"completion_tokens": 1}}
    img_path = tmp_path / "img.png"
    img_path.write_bytes(b"\x89PNG-payload")
    with patch("muse.modalities.image_ocr.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_ocr import OcrClient
        OcrClient().ocr(str(img_path))
    assert mock_post.call_args.kwargs["files"]["image"][1] == b"\x89PNG-payload"


def test_ocr_call_with_pathlib_input(tmp_path):
    body = {"id": "ocr-1", "model": "m", "text": "ok", "usage": {"completion_tokens": 1}}
    img_path = tmp_path / "img.png"
    img_path.write_bytes(b"data")
    with patch("muse.modalities.image_ocr.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_ocr import OcrClient
        OcrClient().ocr(img_path)
    assert mock_post.call_args.kwargs["files"]["image"][1] == b"data"


def test_ocr_call_with_file_like_input():
    body = {"id": "ocr-1", "model": "m", "text": "ok", "usage": {"completion_tokens": 1}}
    buf = io.BytesIO(b"file-like-bytes")
    with patch("muse.modalities.image_ocr.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_ocr import OcrClient
        OcrClient().ocr(buf)
    assert mock_post.call_args.kwargs["files"]["image"][1] == b"file-like-bytes"


def test_ocr_call_with_pil_image():
    """PIL.Image.save serializes to PNG bytes via the .save fallback.

    Use a plain class (not MagicMock) so the client's `hasattr(read)`
    check doesn't false-positive on MagicMock's auto-attributes."""
    class _FakePIL:
        def save(self, buf, format):
            buf.write(b"png-from-pil")
    body = {"id": "ocr-1", "model": "m", "text": "ok", "usage": {"completion_tokens": 1}}
    with patch("muse.modalities.image_ocr.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_ocr import OcrClient
        OcrClient().ocr(_FakePIL())
    assert mock_post.call_args.kwargs["files"]["image"][1] == b"png-from-pil"


def test_ocr_call_passes_model_and_prompt():
    body = {"id": "ocr-1", "model": "nougat-base",
            "text": "x", "usage": {"completion_tokens": 1}}
    with patch("muse.modalities.image_ocr.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_ocr import OcrClient
        OcrClient().ocr(
            b"img", model="nougat-base",
            prompt="<s_doc>", max_new_tokens=1024,
        )
    sent_data = mock_post.call_args.kwargs["data"]
    assert sent_data["model"] == "nougat-base"
    assert sent_data["prompt"] == "<s_doc>"
    assert sent_data["max_new_tokens"] == "1024"


def test_ocr_call_omits_optional_fields_when_default():
    body = {"id": "ocr-1", "model": "m", "text": "x", "usage": {"completion_tokens": 1}}
    with patch("muse.modalities.image_ocr.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_ocr import OcrClient
        OcrClient().ocr(b"img")
    sent_data = mock_post.call_args.kwargs["data"]
    assert "model" not in sent_data
    assert "prompt" not in sent_data
    assert "max_new_tokens" not in sent_data


def test_unsupported_input_type_raises():
    from muse.modalities.image_ocr import OcrClient
    import pytest
    with pytest.raises(TypeError, match="unsupported"):
        OcrClient().ocr(12345)


def test_raise_for_status_invoked():
    import requests
    with patch("muse.modalities.image_ocr.client.requests.post") as mock_post:
        resp = _make_response({"error": {"code": "model_not_found"}}, status=404)
        resp.raise_for_status = MagicMock(
            side_effect=requests.HTTPError("404 model_not_found"),
        )
        mock_post.return_value = resp
        from muse.modalities.image_ocr import OcrClient
        import pytest
        with pytest.raises(requests.HTTPError):
            OcrClient().ocr(b"img", model="ghost")
