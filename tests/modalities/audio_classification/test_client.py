"""Tests for AudioClassificationsClient."""
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


def test_default_server_url():
    from muse.modalities.audio_classification import AudioClassificationsClient
    c = AudioClassificationsClient()
    assert c.server_url == "http://localhost:8000"


def test_muse_server_env_fallback(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://lan:7000")
    from muse.modalities.audio_classification import AudioClassificationsClient
    assert AudioClassificationsClient().server_url == "http://lan:7000"


def test_classify_request_body():
    body = {"id": "a", "model": "ast", "results": [[]]}
    with patch("muse.modalities.audio_classification.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.audio_classification import AudioClassificationsClient
        AudioClassificationsClient().classify(
            b"WAV-bytes", model="ast", top_k=5,
        )
    sent = mock_post.call_args.kwargs
    assert sent["data"]["model"] == "ast"
    assert sent["data"]["top_k"] == "5"
    assert sent["files"]["file"][1] == b"WAV-bytes"


def test_classify_omits_optional():
    body = {"id": "a", "model": "ast", "results": [[]]}
    with patch("muse.modalities.audio_classification.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.audio_classification import AudioClassificationsClient
        AudioClassificationsClient().classify(b"bytes")
    sent_data = mock_post.call_args.kwargs["data"]
    assert "model" not in sent_data
    assert "top_k" not in sent_data


def test_classify_accepts_path(tmp_path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"PAYLOAD")
    body = {"id": "a", "model": "ast", "results": [[]]}
    with patch("muse.modalities.audio_classification.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.audio_classification import AudioClassificationsClient
        AudioClassificationsClient().classify(p)
    assert mock_post.call_args.kwargs["files"]["file"][1] == b"PAYLOAD"


def test_classify_accepts_file_like():
    body = {"id": "a", "model": "ast", "results": [[]]}
    buf = io.BytesIO(b"FROM-FILE-LIKE")
    with patch("muse.modalities.audio_classification.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.audio_classification import AudioClassificationsClient
        AudioClassificationsClient().classify(buf)
    assert mock_post.call_args.kwargs["files"]["file"][1] == b"FROM-FILE-LIKE"


def test_classify_rejects_unsupported_input():
    from muse.modalities.audio_classification import AudioClassificationsClient
    with pytest.raises(TypeError, match="unsupported"):
        AudioClassificationsClient().classify(12345)


def test_classify_propagates_http_error():
    import requests
    with patch("muse.modalities.audio_classification.client.requests.post") as mock_post:
        resp = _make_response({"error": {"code": "model_not_found"}}, status=404)
        resp.raise_for_status = MagicMock(
            side_effect=requests.HTTPError("404"),
        )
        mock_post.return_value = resp
        from muse.modalities.audio_classification import AudioClassificationsClient
        with pytest.raises(requests.HTTPError):
            AudioClassificationsClient().classify(b"bytes", model="ghost")
