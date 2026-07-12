import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.audio_quality import AudioQualityClient


def _response(body):
    response = MagicMock()
    response.json.return_value = body
    return response


def test_default_server_url():
    assert AudioQualityClient().server_url == "http://localhost:8000"


def test_assess_sends_multipart_request():
    body = {"model": "utmos", "scores": {}}
    with patch(
        "muse.modalities.audio_quality.client.requests.post",
        return_value=_response(body),
    ) as post:
        result = AudioQualityClient().assess(b"WAV", model="utmos")
    assert result == body
    sent = post.call_args.kwargs
    assert sent["files"]["file"][1] == b"WAV"
    assert sent["data"] == {"model": "utmos"}
    post.return_value.raise_for_status.assert_called_once()


def test_assess_accepts_path_and_file_like(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"PATH")
    with patch(
        "muse.modalities.audio_quality.client.requests.post",
        return_value=_response({"scores": {}}),
    ) as post:
        AudioQualityClient().assess(path)
        assert post.call_args.kwargs["files"]["file"][1] == b"PATH"
        AudioQualityClient().assess(io.BytesIO(b"FILE"))
        assert post.call_args.kwargs["files"]["file"][1] == b"FILE"


def test_assess_rejects_unsupported_input():
    with pytest.raises(TypeError, match="unsupported audio type"):
        AudioQualityClient().assess(42)
