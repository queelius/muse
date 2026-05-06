"""Tests for Generation3DClient."""
import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_response(body: dict, status: int = 200):
    mock = MagicMock(
        status_code=status,
        headers={"content-type": "application/json"},
    )
    mock.json = MagicMock(return_value=body)
    mock.text = json.dumps(body)
    mock.raise_for_status = MagicMock()
    return mock


# ---------------- server URL resolution ----------------


def test_default_server_url():
    from muse.modalities.model_3d_generation import Generation3DClient
    c = Generation3DClient()
    assert c.server_url == "http://localhost:8000"


def test_muse_server_env_fallback(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://lan:7000")
    from muse.modalities.model_3d_generation import Generation3DClient
    assert Generation3DClient().server_url == "http://lan:7000"


def test_explicit_server_url_strips_trailing_slash():
    from muse.modalities.model_3d_generation import Generation3DClient
    c = Generation3DClient(server_url="http://lan:9000/")
    assert c.server_url == "http://lan:9000"


def test_explicit_server_url_overrides_env(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://lan:7000")
    from muse.modalities.model_3d_generation import Generation3DClient
    c = Generation3DClient(server_url="http://other:8000")
    assert c.server_url == "http://other:8000"


# ---------------- from_text ----------------


def test_from_text_builds_json_body_and_posts():
    body = {"id": "3d-x", "created": 1, "model": "trellis", "data": []}
    with patch(
        "muse.modalities.model_3d_generation.client.requests.post",
    ) as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.model_3d_generation import Generation3DClient
        out = Generation3DClient().from_text(
            "a wooden chair", model="trellis", n=2, seed=42,
        )
    assert out == body
    sent = mock_post.call_args
    assert sent.args[0].endswith("/v1/3d/generations")
    sent_json = sent.kwargs["json"]
    assert sent_json["prompt"] == "a wooden chair"
    assert sent_json["model"] == "trellis"
    assert sent_json["n"] == 2
    assert sent_json["seed"] == 42
    assert sent_json["response_format"] == "b64_json"


def test_from_text_omits_optional_when_none():
    body = {"id": "3d-x", "created": 1, "model": "trellis", "data": []}
    with patch(
        "muse.modalities.model_3d_generation.client.requests.post",
    ) as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.model_3d_generation import Generation3DClient
        Generation3DClient().from_text("hi")
    sent_json = mock_post.call_args.kwargs["json"]
    assert "model" not in sent_json
    assert "seed" not in sent_json
    assert sent_json["prompt"] == "hi"
    assert sent_json["n"] == 1


def test_from_text_response_format_url_passes_through():
    body = {"id": "3d-x", "created": 1, "model": "t", "data": []}
    with patch(
        "muse.modalities.model_3d_generation.client.requests.post",
    ) as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.model_3d_generation import Generation3DClient
        Generation3DClient().from_text("x", response_format="url")
    sent_json = mock_post.call_args.kwargs["json"]
    assert sent_json["response_format"] == "url"


def test_from_text_raises_on_http_error():
    with patch(
        "muse.modalities.model_3d_generation.client.requests.post",
    ) as mock_post:
        mock_post.return_value = MagicMock(
            status_code=400,
            text="bad request",
        )
        from muse.modalities.model_3d_generation import Generation3DClient
        with pytest.raises(RuntimeError, match="400"):
            Generation3DClient().from_text("x")


# ---------------- from_image ----------------


def test_from_image_with_bytes():
    body = {"id": "3d-x", "created": 1, "model": "triposr", "data": []}
    with patch(
        "muse.modalities.model_3d_generation.client.requests.post",
    ) as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.model_3d_generation import Generation3DClient
        out = Generation3DClient().from_image(
            b"PNG-bytes", model="triposr", n=1, seed=7,
        )
    assert out == body
    sent = mock_post.call_args
    assert sent.args[0].endswith("/v1/3d/from-image")
    files = sent.kwargs["files"]
    assert files["image"][1] == b"PNG-bytes"
    data = dict(sent.kwargs["data"])
    assert data["model"] == "triposr"
    assert data["n"] == "1"
    assert data["seed"] == "7"
    assert data["response_format"] == "b64_json"


def test_from_image_with_path(tmp_path):
    p = tmp_path / "scene.png"
    p.write_bytes(b"FROM-PATH")
    body = {"id": "3d-x", "created": 1, "model": "m", "data": []}
    with patch(
        "muse.modalities.model_3d_generation.client.requests.post",
    ) as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.model_3d_generation import Generation3DClient
        Generation3DClient().from_image(p)
    files = mock_post.call_args.kwargs["files"]
    assert files["image"][1] == b"FROM-PATH"


def test_from_image_with_str_path(tmp_path):
    p = tmp_path / "scene.png"
    p.write_bytes(b"STR-PATH")
    body = {"id": "3d-x", "created": 1, "model": "m", "data": []}
    with patch(
        "muse.modalities.model_3d_generation.client.requests.post",
    ) as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.model_3d_generation import Generation3DClient
        Generation3DClient().from_image(str(p))
    files = mock_post.call_args.kwargs["files"]
    assert files["image"][1] == b"STR-PATH"


def test_from_image_with_file_like():
    body = {"id": "3d-x", "created": 1, "model": "m", "data": []}
    buf = io.BytesIO(b"FILE-LIKE")
    with patch(
        "muse.modalities.model_3d_generation.client.requests.post",
    ) as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.model_3d_generation import Generation3DClient
        Generation3DClient().from_image(buf)
    files = mock_post.call_args.kwargs["files"]
    assert files["image"][1] == b"FILE-LIKE"


def test_from_image_omits_optional_when_none():
    body = {"id": "3d-x", "created": 1, "model": "m", "data": []}
    with patch(
        "muse.modalities.model_3d_generation.client.requests.post",
    ) as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.model_3d_generation import Generation3DClient
        Generation3DClient().from_image(b"PNG")
    data = dict(mock_post.call_args.kwargs["data"])
    assert "model" not in data
    assert "seed" not in data
    assert data["n"] == "1"
    assert data["response_format"] == "b64_json"


def test_from_image_rejects_unsupported_type():
    from muse.modalities.model_3d_generation import Generation3DClient
    with pytest.raises(TypeError, match="unsupported"):
        Generation3DClient().from_image(12345)


def test_from_image_rejects_unsupported_dict():
    from muse.modalities.model_3d_generation import Generation3DClient
    with pytest.raises(TypeError, match="unsupported"):
        Generation3DClient().from_image({"foo": "bar"})


def test_from_image_raises_on_http_error():
    with patch(
        "muse.modalities.model_3d_generation.client.requests.post",
    ) as mock_post:
        mock_post.return_value = MagicMock(
            status_code=500, text="boom",
        )
        from muse.modalities.model_3d_generation import Generation3DClient
        with pytest.raises(RuntimeError, match="500"):
            Generation3DClient().from_image(b"png")


def test_from_image_response_format_url_passes_through():
    body = {"id": "3d-x", "created": 1, "model": "m", "data": []}
    with patch(
        "muse.modalities.model_3d_generation.client.requests.post",
    ) as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.model_3d_generation import Generation3DClient
        Generation3DClient().from_image(b"png", response_format="url")
    data = dict(mock_post.call_args.kwargs["data"])
    assert data["response_format"] == "url"
