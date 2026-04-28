"""Integration tests for /v1/audio/music + /v1/audio/sfx against a live muse server.

Opt-in via MUSE_REMOTE_SERVER. Skipped when the server is unreachable
or no audio/generation model is loaded.

Targets the configurable audio-gen model id (default
stable-audio-open-1.0, override via MUSE_AUDIO_GENERATION_MODEL_ID).

Note: Stable Audio Open is GPU-bound (10s+ wall on a 12GB GPU; way
slower on CPU). These tests use short durations (1-3s) to keep wall
time under ~60s per test.
"""
from __future__ import annotations

import io
import wave

from muse.modalities.audio_generation import MusicClient, SFXClient


def test_protocol_audio_generation_music_returns_audio_bytes(
    remote_url, audio_generation_model,
):
    """Hard claim: posting to /v1/audio/music returns audio bytes."""
    client = MusicClient(remote_url, timeout=600.0)
    out = client.generate(
        "ambient piano",
        model=audio_generation_model,
        duration=2.0,
        steps=10,
    )
    assert isinstance(out, bytes)
    # Default response_format is wav; expect a RIFF header.
    assert out[:4] == b"RIFF"
    assert out[8:12] == b"WAVE"


def test_protocol_audio_generation_sfx_returns_audio_bytes(
    remote_url, audio_generation_model,
):
    """Hard claim: posting to /v1/audio/sfx returns audio bytes."""
    client = SFXClient(remote_url, timeout=600.0)
    out = client.generate(
        "footsteps on gravel",
        model=audio_generation_model,
        duration=2.0,
        steps=10,
    )
    assert isinstance(out, bytes)
    assert out[:4] == b"RIFF"


def test_protocol_audio_generation_music_wav_is_parseable(
    remote_url, audio_generation_model,
):
    """The returned WAV must be openable via wave.open."""
    client = MusicClient(remote_url, timeout=600.0)
    out = client.generate(
        "ambient", model=audio_generation_model,
        duration=1.0, steps=10,
    )
    with wave.open(io.BytesIO(out), "rb") as w:
        assert w.getframerate() in (44100, 48000, 32000, 22050, 16000)
        assert w.getnchannels() in (1, 2)
        # At least 0.5 seconds of audio.
        assert w.getnframes() / w.getframerate() >= 0.5


def test_observe_audio_generation_music_seed_is_reproducible(
    remote_url, audio_generation_model,
):
    """Soft observation: same prompt + seed should give same bytes.

    Models with non-deterministic ops may not honor seed perfectly;
    this is a watchdog rather than a hard claim.
    """
    client = MusicClient(remote_url, timeout=600.0)
    a = client.generate(
        "calm ambient", model=audio_generation_model,
        duration=1.0, steps=10, seed=42,
    )
    b = client.generate(
        "calm ambient", model=audio_generation_model,
        duration=1.0, steps=10, seed=42,
    )
    # Soft assert: don't fail the suite if not reproducible.
    if a == b:
        return  # ok
    # Different bytes; record but don't fail.
    print("note: seeded music generation not bit-reproducible")


def test_protocol_audio_generation_404_on_unknown_model(remote_url):
    """Hard claim: an unknown model id returns 404."""
    import requests
    r = requests.post(
        f"{remote_url}/v1/audio/music",
        json={"prompt": "hi", "model": "definitely-not-a-real-model"},
        timeout=10.0,
    )
    assert r.status_code == 404
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == "model_not_found"
