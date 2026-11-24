"""Route tests for /v1/audio/music and /v1/audio/sfx.

Both routes share the handler. Tests verify:
- 200 happy path on each route returns audio bytes + content-type.
- 400 capability gate on each route when manifest forbids.
- 400 unsupported response_format triggers UnsupportedFormatError.
- 404 model_not_found envelope on unknown model.
- Pydantic field validation: empty prompt, range checks, format pattern.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.audio_generation import (
    MODALITY,
    AudioGenerationResult,
    build_router,
)
from muse.modalities.audio_generation.codec import UnsupportedFormatError


def _fake_backend(*, model_id="stable-audio-open-1.0", samples=4410, sr=44100, channels=1):
    """Return a MagicMock satisfying AudioGenerationModel.

    Default: 0.1s of silence, mono, 44.1kHz.
    """
    backend = MagicMock()
    backend.model_id = model_id

    if channels == 1:
        audio = np.zeros(samples, dtype=np.float32)
    else:
        audio = np.zeros((samples, channels), dtype=np.float32)

    result = AudioGenerationResult(
        audio=audio, sample_rate=sr, channels=channels,
        duration_seconds=samples / sr,
    )
    backend.generate.return_value = result
    return backend


def _make_client(backend, capabilities=None):
    reg = ModalityRegistry()
    manifest = {
        "model_id": backend.model_id,
        "modality": MODALITY,
        "capabilities": capabilities or {},
    }
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


def test_music_returns_audio_bytes_with_wav_content_type():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={"prompt": "ambient"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
    assert r.content[8:12] == b"WAVE"


def test_sfx_returns_audio_bytes_with_wav_content_type():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/sfx", json={"prompt": "footsteps"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"


def test_music_400_when_supports_music_false():
    backend = _fake_backend()
    client = _make_client(backend, capabilities={"supports_music": False})
    r = client.post("/v1/audio/music", json={"prompt": "x"})
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_parameter"
    assert "music" in body["error"]["message"]


def test_sfx_400_when_supports_sfx_false():
    backend = _fake_backend()
    client = _make_client(backend, capabilities={"supports_sfx": False})
    r = client.post("/v1/audio/sfx", json={"prompt": "x"})
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_parameter"
    assert "sfx" in body["error"]["message"]


def test_music_succeeds_when_supports_music_missing_default_true():
    """When the capability flag is not set, default is True."""
    backend = _fake_backend()
    client = _make_client(backend, capabilities={})
    r = client.post("/v1/audio/music", json={"prompt": "x"})
    assert r.status_code == 200


def test_sfx_succeeds_when_supports_sfx_missing_default_true():
    backend = _fake_backend()
    client = _make_client(backend, capabilities={})
    r = client.post("/v1/audio/sfx", json={"prompt": "x"})
    assert r.status_code == 200


def test_music_404_on_unknown_model_envelope():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={"prompt": "x", "model": "nope"})
    assert r.status_code == 404
    body = r.json()
    # OpenAI-style envelope, not FastAPI's {"detail": ...}
    assert "error" in body
    assert body["error"]["code"] == "model_not_found"


def test_sfx_404_on_unknown_model_envelope():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/sfx", json={"prompt": "x", "model": "nope"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_400_on_empty_prompt():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={"prompt": ""})
    assert r.status_code == 422  # Pydantic min_length


def test_400_on_invalid_response_format():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "x", "response_format": "aac",
    })
    assert r.status_code == 422  # Pydantic pattern


def test_400_on_duration_out_of_range_low():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "x", "duration": 0.1,
    })
    assert r.status_code == 422


def test_400_on_duration_out_of_range_high():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "x", "duration": 999.0,
    })
    assert r.status_code == 422


def test_400_on_steps_out_of_range():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "x", "steps": 1000,
    })
    assert r.status_code == 422


def test_400_on_guidance_out_of_range():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "x", "guidance": 99.0,
    })
    assert r.status_code == 422


def test_negative_seed_rejected():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "x", "seed": -1,
    })
    assert r.status_code == 422


def test_request_kwargs_forwarded_to_backend():
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "ambient piano",
        "duration": 8.0, "seed": 42, "steps": 30, "guidance": 5.0,
        "negative_prompt": "noise",
    })
    assert r.status_code == 200
    args, kwargs = backend.generate.call_args
    assert args[0] == "ambient piano"
    assert kwargs["duration"] == 8.0
    assert kwargs["seed"] == 42
    assert kwargs["steps"] == 30
    assert kwargs["guidance"] == 5.0
    assert kwargs["negative_prompt"] == "noise"


def test_lyrics_field_forwarded_to_backend():
    """ACE-Step's lyrics field must reach model.generate (added v0.49.0).

    The route forwards a fixed kwargs set; lyrics has to be in it or it is
    silently dropped before the backend.
    """
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "pop, female vocal, upbeat",
        "lyrics": "[verse]\nhello world\n[chorus]\nla la la",
    })
    assert r.status_code == 200
    kwargs = backend.generate.call_args.kwargs
    assert kwargs["lyrics"] == "[verse]\nhello world\n[chorus]\nla la la"


def test_lyrics_defaults_to_none_when_omitted():
    """Omitting lyrics forwards None (the runtime treats None as instrumental)."""
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={"prompt": "ambient piano"})
    assert r.status_code == 200
    assert backend.generate.call_args.kwargs["lyrics"] is None


def test_long_duration_accepted_up_to_240s():
    """ACE-Step needs long clips; the route ceiling is 240s (was 120s)."""
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={"prompt": "x", "duration": 200.0})
    assert r.status_code == 200
    assert backend.generate.call_args.kwargs["duration"] == 200.0


def test_duration_above_240s_still_rejected():
    """The raised ceiling is 240s, not unbounded."""
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={"prompt": "x", "duration": 300.0})
    assert r.status_code == 422


def test_music_and_sfx_call_backend_with_same_kwargs():
    """Routes share the handler; only the capability key differs."""
    backend = _fake_backend()
    client = _make_client(backend)
    client.post("/v1/audio/music", json={"prompt": "p"})
    music_kwargs = backend.generate.call_args.kwargs
    backend.reset_mock()
    backend.generate.return_value = backend.generate.return_value
    backend.generate.return_value = AudioGenerationResult(
        audio=np.zeros(4410, dtype=np.float32),
        sample_rate=44100, channels=1, duration_seconds=0.1,
    )
    client.post("/v1/audio/sfx", json={"prompt": "p"})
    sfx_kwargs = backend.generate.call_args.kwargs
    assert music_kwargs == sfx_kwargs


def test_flac_response_when_soundfile_available():
    pytest.importorskip("soundfile")
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "x", "response_format": "flac",
    })
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/flac"
    assert r.content[:4] == b"fLaC"


def test_flac_400_when_soundfile_missing():
    backend = _fake_backend()
    client = _make_client(backend)
    with patch(
        "muse.modalities.audio_generation.codec._try_import_soundfile",
        return_value=None,
    ):
        r = client.post("/v1/audio/music", json={
            "prompt": "x", "response_format": "flac",
        })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"
    assert "soundfile" in r.json()["error"]["message"]


def test_mp3_400_when_pydub_missing():
    backend = _fake_backend()
    client = _make_client(backend)
    with patch(
        "muse.modalities.audio_generation.codec._try_import_pydub",
        return_value=None,
    ):
        r = client.post("/v1/audio/music", json={
            "prompt": "x", "response_format": "mp3",
        })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"
    assert "pydub" in r.json()["error"]["message"]


def test_internal_error_returns_500_envelope():
    """Backend exception translates to a 500 with OpenAI envelope."""
    backend = _fake_backend()
    backend.generate.side_effect = RuntimeError("CUDA OOM")
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={"prompt": "x"})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"


def test_response_is_bytes_not_json():
    """No streaming: single Response, not SSE."""
    backend = _fake_backend()
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={"prompt": "x"})
    assert r.headers["content-type"].startswith("audio/")
    assert "content-length" in {k.lower() for k in r.headers}


# H2 regression guard: backend.generate must be called under _inference_lock.

def test_music_generate_called_under_inference_lock():
    """backend.generate must execute while _inference_lock is held (H2 fix).

    A fake backend asserts the lock is locked when generate() is called.
    """
    import threading

    class _LockAssertingBackend:
        model_id = "lock-test"

        def __init__(self):
            self._inference_lock = threading.Lock()
            self.lock_was_held = False
            audio = np.zeros(4410, dtype=np.float32)
            self._result = AudioGenerationResult(
                audio=audio, sample_rate=44100, channels=1, duration_seconds=0.1,
            )

        def generate(self, prompt, **kwargs):
            self.lock_was_held = self._inference_lock.locked()
            return self._result

    backend = _LockAssertingBackend()
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={
        "model_id": backend.model_id, "modality": MODALITY, "capabilities": {},
    })
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)
    r = client.post("/v1/audio/music", json={"prompt": "test"})
    assert r.status_code == 200, r.text
    assert backend.lock_was_held, (
        "backend.generate was called without holding _inference_lock"
    )


def test_encoding_runs_off_the_event_loop(monkeypatch):
    """I6: pydub/ffmpeg/soundfile encoding must not block the worker loop."""
    import asyncio
    from muse.modalities.audio_generation import routes as routes_mod

    seen = []

    def _encode(fmt, result):
        try:
            asyncio.get_running_loop()
            seen.append(True)
        except RuntimeError:
            seen.append(False)
        return b"RIFF"

    monkeypatch.setattr(routes_mod, "_encode", _encode)
    client = _make_client(_fake_backend())
    r = client.post("/v1/audio/music", json={"prompt": "x"})
    assert r.status_code == 200
    assert seen == [False]


# Optional mastering flags (delegating to muse.modalities.audio_process.dsp).

def _tone_backend(*, sr=16000, duration_s=0.6, lead_s=0.2, tail_s=0.2, channels=2, amp=0.2):
    """Return a fake backend emitting a padded stereo/mono sine clip."""
    backend = MagicMock()
    backend.model_id = "stable-audio-open-1.0"
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr
    tone = (amp * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    mono = np.concatenate([
        np.zeros(int(round(lead_s * sr)), dtype=np.float32),
        tone,
        np.zeros(int(round(tail_s * sr)), dtype=np.float32),
    ])
    audio = np.stack([mono, mono * 0.5], axis=1) if channels == 2 else mono
    backend.generate.return_value = AudioGenerationResult(
        audio=audio, sample_rate=sr, channels=channels,
        duration_seconds=len(mono) / sr,
    )
    return backend


def test_sfx_mastering_default_off_preserves_raw_output():
    from muse.modalities.audio_process.codec import decode_audio_bytes

    backend = _tone_backend(sr=16000, duration_s=0.4, lead_s=0.2, tail_s=0.2, channels=2, amp=0.2)
    client = _make_client(backend)
    r = client.post("/v1/audio/sfx", json={"prompt": "click"})
    assert r.status_code == 200
    decoded, sr = decode_audio_bytes(r.content)
    assert sr == 16000
    assert decoded.ndim == 2 and decoded.shape[1] == 2
    assert len(decoded) == int(round(0.8 * 16000))
    assert float(np.max(np.abs(decoded))) == pytest.approx(0.2, abs=1e-2)


def test_sfx_mastering_trim_normalize_mono_resample():
    from muse.modalities.audio_process.codec import decode_audio_bytes

    backend = _tone_backend(sr=16000, duration_s=0.4, lead_s=0.25, tail_s=0.25, channels=2, amp=0.2)
    client = _make_client(backend)
    r = client.post("/v1/audio/sfx", json={
        "prompt": "footstep",
        "trim_silence": True,
        "target_peak_dbfs": -3.0,
        "mono": True,
        "sample_rate": 22050,
    })
    assert r.status_code == 200
    decoded, sr = decode_audio_bytes(r.content)
    assert sr == 22050
    assert decoded.ndim == 1
    # Trimmed duration (~0.4s + 85ms pad) is strictly shorter than raw 0.9s.
    assert len(decoded) / sr < 0.7
    expected_peak = 10.0 ** (-3.0 / 20.0)
    assert float(np.max(np.abs(decoded))) == pytest.approx(expected_peak, abs=0.02)


def test_music_mastering_loop_fade():
    from muse.modalities.audio_process.codec import decode_audio_bytes

    backend = _tone_backend(sr=16000, duration_s=1.0, lead_s=0.0, tail_s=0.0, channels=1, amp=0.3)
    client = _make_client(backend)
    r = client.post("/v1/audio/music", json={
        "prompt": "ambient bed",
        "loop_fade_s": 0.25,
        "target_peak_dbfs": -6.0,
    })
    assert r.status_code == 200
    decoded, sr = decode_audio_bytes(r.content)
    assert sr == 16000
    assert len(decoded) == pytest.approx(int(0.75 * 16000), abs=4)
    expected_peak = 10.0 ** (-6.0 / 20.0)
    assert float(np.max(np.abs(decoded))) == pytest.approx(expected_peak, abs=0.02)


@pytest.mark.parametrize(
    "bad_fields",
    [
        {"loop_fade_s": -1.0},
        {"loop_fade_s": 31.0},
        {"target_peak_dbfs": 1.0},
        {"target_peak_dbfs": -60.0},
        {"sample_rate": 4000},
        {"sample_rate": 192000},
    ],
)
def test_audio_generation_mastering_validation_bounds(bad_fields):
    client = _make_client(_fake_backend())
    r = client.post("/v1/audio/sfx", json={"prompt": "x", **bad_fields})
    assert r.status_code == 422
