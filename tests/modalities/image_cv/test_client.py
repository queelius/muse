"""Tests for DepthClient, KeypointClient, ObjectDetectionClient."""
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


# ---------- DepthClient ----------


def test_depth_client_default_server_url():
    from muse.modalities.image_cv import DepthClient
    c = DepthClient()
    assert c.server_url == "http://localhost:8000"


def test_depth_client_muse_server_env_fallback(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://lan:7777")
    from muse.modalities.image_cv import DepthClient
    assert DepthClient().server_url == "http://lan:7777"


def test_depth_client_estimate_depth_request_body():
    body = {
        "id": "depth-1", "model": "m", "depth_map": "AAAA",
        "format": "png16", "width": 64, "height": 32,
        "min_depth": 0.0, "max_depth": 1.0, "metric_depth": False,
    }
    with patch("muse.modalities.image_cv.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_cv import DepthClient
        out = DepthClient().estimate_depth(b"PNG-bytes", model="m")
    assert out["format"] == "png16"
    sent_data = mock_post.call_args.kwargs["data"]
    assert sent_data["model"] == "m"
    assert sent_data["response_format"] == "png16"


def test_depth_client_response_format_float32():
    body = {"id": "d", "model": "m", "depth_map": "x", "format": "float32"}
    with patch("muse.modalities.image_cv.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_cv import DepthClient
        DepthClient().estimate_depth(
            b"png", model="zoedepth", response_format="float32",
        )
    sent_data = mock_post.call_args.kwargs["data"]
    assert sent_data["response_format"] == "float32"


def test_depth_client_url_target():
    body = {"id": "d", "model": "m", "depth_map": ""}
    with patch("muse.modalities.image_cv.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_cv import DepthClient
        DepthClient(server_url="http://x:9000").estimate_depth(b"png")
    args, _ = mock_post.call_args
    assert args[0] == "http://x:9000/v1/images/depth"


# ---------- KeypointClient ----------


def test_keypoint_client_request_body():
    body = {"id": "kp-1", "model": "vp", "image_size": [64, 64], "detections": []}
    with patch("muse.modalities.image_cv.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_cv import KeypointClient
        KeypointClient().detect_keypoints(b"png", model="vp", threshold=0.5)
    sent_data = mock_post.call_args.kwargs["data"]
    assert sent_data["model"] == "vp"
    assert sent_data["threshold"] == "0.5"


def test_keypoint_client_omits_threshold_when_none():
    body = {"id": "kp-1", "model": "vp", "image_size": [1, 1], "detections": []}
    with patch("muse.modalities.image_cv.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_cv import KeypointClient
        KeypointClient().detect_keypoints(b"png", model="vp")
    sent_data = mock_post.call_args.kwargs["data"]
    assert "threshold" not in sent_data


def test_keypoint_client_url_target():
    body = {"id": "kp-1", "model": "m", "image_size": [1, 1], "detections": []}
    with patch("muse.modalities.image_cv.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_cv import KeypointClient
        KeypointClient(server_url="http://x:9000/").detect_keypoints(b"png")
    args, _ = mock_post.call_args
    assert args[0] == "http://x:9000/v1/images/keypoints"


# ---------- ObjectDetectionClient ----------


def test_detection_client_request_body():
    body = {"id": "det-1", "model": "m", "image_size": [1, 1], "detections": []}
    with patch("muse.modalities.image_cv.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_cv import ObjectDetectionClient
        ObjectDetectionClient().detect_objects(
            b"png", model="detr", threshold=0.6, max_detections=20,
        )
    sent_data = mock_post.call_args.kwargs["data"]
    assert sent_data["model"] == "detr"
    assert sent_data["threshold"] == "0.6"
    assert sent_data["max_detections"] == "20"


def test_detection_client_omits_optional_when_none():
    body = {"id": "d", "model": "m", "image_size": [1, 1], "detections": []}
    with patch("muse.modalities.image_cv.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_cv import ObjectDetectionClient
        ObjectDetectionClient().detect_objects(b"png")
    sent_data = mock_post.call_args.kwargs["data"]
    assert "model" not in sent_data
    assert "threshold" not in sent_data
    assert "max_detections" not in sent_data


def test_detection_client_url_target():
    body = {"id": "d", "model": "m", "image_size": [1, 1], "detections": []}
    with patch("muse.modalities.image_cv.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_cv import ObjectDetectionClient
        ObjectDetectionClient(server_url="http://x:9000").detect_objects(b"png")
    args, _ = mock_post.call_args
    assert args[0] == "http://x:9000/v1/images/detect"


# ---------- shared input handling ----------


def test_clients_accept_path_input(tmp_path):
    img_path = tmp_path / "img.png"
    img_path.write_bytes(b"png-bytes")
    body = {"id": "d", "model": "m", "depth_map": "", "format": "png16"}
    with patch("muse.modalities.image_cv.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_cv import DepthClient
        DepthClient().estimate_depth(str(img_path))
    assert mock_post.call_args.kwargs["files"]["image"][1] == b"png-bytes"


def test_clients_accept_pathlib_input(tmp_path):
    img_path = tmp_path / "img.png"
    img_path.write_bytes(b"data")
    body = {"id": "d", "model": "m", "depth_map": "", "format": "png16"}
    with patch("muse.modalities.image_cv.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_cv import DepthClient
        DepthClient().estimate_depth(img_path)
    assert mock_post.call_args.kwargs["files"]["image"][1] == b"data"


def test_clients_accept_file_like():
    buf = io.BytesIO(b"file-like-bytes")
    body = {"id": "d", "model": "m", "depth_map": "", "format": "png16"}
    with patch("muse.modalities.image_cv.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.image_cv import DepthClient
        DepthClient().estimate_depth(buf)
    assert mock_post.call_args.kwargs["files"]["image"][1] == b"file-like-bytes"


def test_clients_reject_unsupported_input_type():
    from muse.modalities.image_cv import DepthClient
    with pytest.raises(TypeError, match="unsupported"):
        DepthClient().estimate_depth(12345)


def test_clients_propagate_http_error():
    import requests
    with patch("muse.modalities.image_cv.client.requests.post") as mock_post:
        resp = _make_response({"error": {"code": "model_not_found"}}, status=404)
        resp.raise_for_status = MagicMock(
            side_effect=requests.HTTPError("404"),
        )
        mock_post.return_value = resp
        from muse.modalities.image_cv import DepthClient
        with pytest.raises(requests.HTTPError):
            DepthClient().estimate_depth(b"png", model="ghost")
