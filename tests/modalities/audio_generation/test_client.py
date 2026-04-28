"""Tests for MusicClient + SFXClient.

Both clients post to their respective routes (/v1/audio/music,
/v1/audio/sfx) and return raw audio bytes. Tests use a requests.post
mock and verify URL, body, kwargs plumbing.
"""
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.audio_generation import MusicClient, SFXClient


def _ok_response(content=b"FAKEWAVBYTES"):
    r = MagicMock()
    r.status_code = 200
    r.content = content
    r.raise_for_status.return_value = None
    return r


def test_music_client_default_server_url():
    c = MusicClient()
    assert c.server_url == "http://localhost:8000"


def test_sfx_client_default_server_url():
    c = SFXClient()
    assert c.server_url == "http://localhost:8000"


def test_client_explicit_server_url_overrides_default():
    c = MusicClient(server_url="http://example.com:7777")
    assert c.server_url == "http://example.com:7777"


def test_client_strips_trailing_slash():
    c = MusicClient(server_url="http://example.com/")
    assert c.server_url == "http://example.com"


def test_client_honors_muse_server_env(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://env-host:9999")
    c = MusicClient()
    assert c.server_url == "http://env-host:9999"


def test_explicit_url_beats_env(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://env-host:9999")
    c = MusicClient(server_url="http://explicit:1111")
    assert c.server_url == "http://explicit:1111"


def test_music_client_posts_to_v1_audio_music():
    with patch("muse.modalities.audio_generation.client.requests.post",
               return_value=_ok_response()) as fake_post:
        c = MusicClient(server_url="http://x:8000")
        out = c.generate("ambient piano")
    args, kwargs = fake_post.call_args
    assert args[0] == "http://x:8000/v1/audio/music"
    assert kwargs["json"] == {"prompt": "ambient piano"}
    assert out == b"FAKEWAVBYTES"


def test_sfx_client_posts_to_v1_audio_sfx():
    with patch("muse.modalities.audio_generation.client.requests.post",
               return_value=_ok_response()) as fake_post:
        c = SFXClient(server_url="http://x:8000")
        out = c.generate("footsteps on gravel")
    args, kwargs = fake_post.call_args
    assert args[0] == "http://x:8000/v1/audio/sfx"
    assert kwargs["json"] == {"prompt": "footsteps on gravel"}
    assert out == b"FAKEWAVBYTES"


def test_music_client_kwargs_passed_through():
    with patch("muse.modalities.audio_generation.client.requests.post",
               return_value=_ok_response()) as fake_post:
        c = MusicClient(server_url="http://x:8000")
        c.generate(
            "p",
            model="stable-audio-open-1.0",
            duration=10.0,
            seed=42,
            response_format="flac",
            steps=30,
            guidance=5.0,
            negative_prompt="noise",
        )
    body = fake_post.call_args.kwargs["json"]
    assert body["prompt"] == "p"
    assert body["model"] == "stable-audio-open-1.0"
    assert body["duration"] == 10.0
    assert body["seed"] == 42
    assert body["response_format"] == "flac"
    assert body["steps"] == 30
    assert body["guidance"] == 5.0
    assert body["negative_prompt"] == "noise"


def test_sfx_client_kwargs_passed_through():
    with patch("muse.modalities.audio_generation.client.requests.post",
               return_value=_ok_response()) as fake_post:
        c = SFXClient(server_url="http://x:8000")
        c.generate(
            "p",
            model="stable-audio-open-1.0",
            duration=3.0,
            seed=7,
            response_format="mp3",
        )
    body = fake_post.call_args.kwargs["json"]
    assert body["model"] == "stable-audio-open-1.0"
    assert body["duration"] == 3.0
    assert body["seed"] == 7
    assert body["response_format"] == "mp3"


def test_default_response_format_omits_field_for_wav_compactness():
    """Default wav: leave field off the wire (server defaults to wav)."""
    with patch("muse.modalities.audio_generation.client.requests.post",
               return_value=_ok_response()) as fake_post:
        c = MusicClient(server_url="http://x:8000")
        c.generate("p")
    body = fake_post.call_args.kwargs["json"]
    assert "response_format" not in body


def test_client_propagates_http_errors():
    fake_response = MagicMock()
    err = Exception("HTTP 400")
    fake_response.raise_for_status.side_effect = err
    with patch("muse.modalities.audio_generation.client.requests.post",
               return_value=fake_response):
        c = MusicClient()
        with pytest.raises(Exception, match="HTTP 400"):
            c.generate("p")


def test_client_timeout_default_300():
    with patch("muse.modalities.audio_generation.client.requests.post",
               return_value=_ok_response()) as fake_post:
        c = MusicClient(server_url="http://x:8000")
        c.generate("p")
    assert fake_post.call_args.kwargs["timeout"] == 300.0


def test_client_timeout_override():
    with patch("muse.modalities.audio_generation.client.requests.post",
               return_value=_ok_response()) as fake_post:
        c = SFXClient(server_url="http://x:8000", timeout=60.0)
        c.generate("p")
    assert fake_post.call_args.kwargs["timeout"] == 60.0


def test_client_returns_response_content_unchanged():
    with patch("muse.modalities.audio_generation.client.requests.post",
               return_value=_ok_response(b"\x00\x01\x02")):
        c = MusicClient()
        out = c.generate("p")
    assert out == b"\x00\x01\x02"


def test_omits_optional_fields_when_none():
    """Optional kwargs left at their defaults must not appear in the body."""
    with patch("muse.modalities.audio_generation.client.requests.post",
               return_value=_ok_response()) as fake_post:
        c = MusicClient()
        c.generate("p")
    body = fake_post.call_args.kwargs["json"]
    # Only `prompt` should be present; everything else should be omitted.
    assert set(body.keys()) == {"prompt"}
