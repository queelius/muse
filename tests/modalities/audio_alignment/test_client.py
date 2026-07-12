import io
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.audio_alignment import AudioAlignmentClient


def _response(body):
    response = MagicMock()
    response.json.return_value = body
    return response


def test_default_server_url():
    assert AudioAlignmentClient().server_url == "http://localhost:8000"


def test_align_sends_multipart_request():
    body = {"model": "qwen-aligner", "words": []}
    with patch(
        "muse.modalities.audio_alignment.client.requests.post",
        return_value=_response(body),
    ) as post:
        result = AudioAlignmentClient().align(
            b"WAV", "Hello", model="qwen-aligner", language="en",
        )
    assert result == body
    sent = post.call_args.kwargs
    assert sent["files"]["file"][1] == b"WAV"
    assert sent["data"] == {
        "text": "Hello", "model": "qwen-aligner", "language": "en",
    }
    post.return_value.raise_for_status.assert_called_once()


def test_align_accepts_file_like():
    with patch(
        "muse.modalities.audio_alignment.client.requests.post",
        return_value=_response({"words": []}),
    ) as post:
        AudioAlignmentClient().align(io.BytesIO(b"FILE"), "test")
    assert post.call_args.kwargs["files"]["file"][1] == b"FILE"


def test_align_rejects_unsupported_input():
    with pytest.raises(TypeError, match="unsupported audio type"):
        AudioAlignmentClient().align(42, "test")
