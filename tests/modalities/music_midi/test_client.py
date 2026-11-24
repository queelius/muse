"""Tests for ``MidiClient`` HTTP client (``client.py``)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from muse.modalities.music_midi.client import MidiClient


def test_midi_client_json_request_and_response():
    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "id": "midi-123",
        "object": "music.midi",
        "model": "midi-composer",
        "midi_b64": "TVRoZA==",
        "score": {"bpm": 110, "tracks": []},
        "metadata": {},
    }
    with patch("muse.modalities.music_midi.client.requests.post", return_value=fake_resp) as post:
        client = MidiClient("http://127.0.0.1:8000")
        out = client.generate(
            "village theme",
            model="midi-composer",
            key="D",
            mode="dorian",
            bpm=110,
            time_signature="3/4",
            bars=12,
            instruments=["lute", "flute"],
            layers=["base", "night"],
            seed=42,
            temperature=0.8,
        )

    assert out["object"] == "music.midi"
    url = post.call_args.args[0]
    body = post.call_args.kwargs["json"]
    assert url == "http://127.0.0.1:8000/v1/music/midi"
    assert body["prompt"] == "village theme"
    assert body["model"] == "midi-composer"
    assert body["key"] == "D"
    assert body["mode"] == "dorian"
    assert body["bpm"] == 110
    assert body["time_signature"] == "3/4"
    assert body["bars"] == 12
    assert body["instruments"] == ["lute", "flute"]
    assert body["layers"] == ["base", "night"]
    assert body["seed"] == 42
    assert body["temperature"] == 0.8
    assert body["response_format"] == "json"


def test_midi_client_generate_bytes_returns_raw_midi():
    fake_resp = MagicMock()
    fake_resp.content = b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0"
    with patch("muse.modalities.music_midi.client.requests.post", return_value=fake_resp) as post:
        client = MidiClient("http://127.0.0.1:8000")
        raw = client.generate_bytes("harp lullaby", bars=4, seed=0)

    assert raw[:4] == b"MThd"
    assert post.call_args.kwargs["json"]["response_format"] == "midi"
