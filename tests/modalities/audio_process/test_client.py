"""Unit tests for ``AudioProcessClient``."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.audio_process import AudioProcessClient


def _ok_json(payload=None):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = payload or {
        "object": "audio.process",
        "model": "audio-tools",
        "format": "wav",
        "b64_audio": "UklGRg==",
        "metrics": {"duration_s": 0.5},
    }
    r.content = b"RIFFFAKE"
    r.raise_for_status.return_value = None
    return r


def test_client_default_and_env_server_url(monkeypatch):
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    assert AudioProcessClient().server_url == "http://localhost:8000"

    monkeypatch.setenv("MUSE_SERVER", "http://env:9000/")
    assert AudioProcessClient().server_url == "http://env:9000"

    assert AudioProcessClient("http://explicit:7777/").server_url == "http://explicit:7777"


def test_client_process_json_posts_multipart():
    with patch(
        "muse.modalities.audio_process.client.requests.post",
        return_value=_ok_json(),
    ) as fake_post:
        c = AudioProcessClient("http://x:8000")
        out = c.process(
            b"RIFF1234",
            model="audio-tools",
            trim_silence=True,
            loop_fade_s=0.5,
            target_peak_dbfs=-3.0,
            mono=True,
            sample_rate=22050,
            output_format="flac",
        )
    assert out["object"] == "audio.process"
    url = fake_post.call_args.args[0]
    kwargs = fake_post.call_args.kwargs
    assert url == "http://x:8000/v1/audio/process"
    assert kwargs["files"]["file"] == ("audio.wav", b"RIFF1234", "audio/wav")
    assert kwargs["data"] == {
        "model": "audio-tools",
        "trim_silence": "true",
        "loop_fade_s": "0.5",
        "target_peak_dbfs": "-3.0",
        "mono": "true",
        "sample_rate": "22050",
        "output_format": "flac",
    }


def test_client_process_raw_audio_response_and_path_input(tmp_path):
    wav_file = tmp_path / "sample.wav"
    wav_file.write_bytes(b"RIFFRAW")

    with patch(
        "muse.modalities.audio_process.client.requests.post",
        return_value=_ok_json(),
    ) as fake_post:
        c = AudioProcessClient("http://x:8000", timeout=45.0)
        out = c.process(wav_file, response_format="audio")

    assert out == b"RIFFFAKE"
    kwargs = fake_post.call_args.kwargs
    assert kwargs["timeout"] == 45.0
    assert kwargs["files"]["file"][1] == b"RIFFRAW"
    assert kwargs["data"] == {"response_format": "audio"}


def test_client_propagates_http_errors():
    resp = MagicMock()
    resp.raise_for_status.side_effect = RuntimeError("HTTP 400")
    with patch("muse.modalities.audio_process.client.requests.post", return_value=resp):
        c = AudioProcessClient("http://x:8000")
        with pytest.raises(RuntimeError, match="HTTP 400"):
            c.process(b"RIFF")
