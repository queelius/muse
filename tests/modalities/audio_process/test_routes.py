"""Route and codec tests for ``POST /v1/audio/process``."""
from __future__ import annotations

import base64
import io
import threading
import wave
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities._common.audio_codec import pcm16_wav_bytes
from muse.modalities.audio_process import (
    MODALITY,
    AudioProcessResult,
    build_router,
)
from muse.modalities.audio_process.codec import decode_audio_bytes
from muse.modalities.audio_process.runtime import AudioTools


def _make_wav_bytes(
    *,
    sr: int = 16000,
    duration_s: float = 0.5,
    channels: int = 1,
    lead_silence_s: float = 0.0,
    tail_silence_s: float = 0.0,
    amp: float = 0.25,
) -> bytes:
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr
    tone = (amp * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    parts = []
    if lead_silence_s > 0:
        parts.append(np.zeros(int(round(lead_silence_s * sr)), dtype=np.float32))
    parts.append(tone)
    if tail_silence_s > 0:
        parts.append(np.zeros(int(round(tail_silence_s * sr)), dtype=np.float32))
    mono = np.concatenate(parts)
    if channels == 2:
        audio = np.stack([mono, mono * 0.5], axis=1)
    else:
        audio = mono
    return pcm16_wav_bytes(audio, sr, channels=channels)


def _make_client(backend=None):
    if backend is None:
        backend = AudioTools(model_id="audio-tools")
    reg = ModalityRegistry()
    reg.register(
        MODALITY,
        backend,
        manifest={"model_id": backend.model_id, "modality": MODALITY, "capabilities": {}},
    )
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


def test_process_json_happy_path_default_model():
    client = _make_client()
    wav = _make_wav_bytes(lead_silence_s=0.25, tail_silence_s=0.25, duration_s=0.3, amp=0.2)
    r = client.post(
        "/v1/audio/process",
        files={"file": ("clip.wav", wav, "audio/wav")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "audio.process"
    assert body["model"] == "audio-tools"
    assert body["format"] == "wav"
    out_bytes = base64.b64decode(body["b64_audio"])
    assert out_bytes[:4] == b"RIFF"
    assert out_bytes[8:12] == b"WAVE"
    metrics = body["metrics"]
    assert metrics["sample_rate_hz"] == 16000
    assert metrics["channels"] == 1
    assert metrics["trimmed_leading_ms"] > 150.0
    assert metrics["trimmed_trailing_ms"] > 100.0
    assert metrics["peak_dbfs"] == pytest.approx(-3.0, abs=0.1)
    assert metrics["clipped_samples"] == 0
    assert metrics["looped"] is False


def test_process_raw_audio_response_format():
    client = _make_client()
    wav = _make_wav_bytes(duration_s=0.3, channels=2)
    r = client.post(
        "/v1/audio/process",
        files={"file": ("stereo.wav", wav, "audio/wav")},
        data={
            "mono": "true",
            "sample_rate": "22050",
            "loop_fade_s": "0.1",
            "response_format": "audio",
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF"
    decoded, sr = decode_audio_bytes(r.content)
    assert sr == 22050
    assert decoded.ndim == 1


def test_process_loop_mode_metrics():
    client = _make_client()
    wav = _make_wav_bytes(duration_s=1.0, sr=16000)
    r = client.post(
        "/v1/audio/process",
        files={"file": ("bed.wav", wav, "audio/wav")},
        data={"loop_fade_s": "0.25", "target_peak_dbfs": "-6.0"},
    )
    assert r.status_code == 200, r.text
    metrics = r.json()["metrics"]
    assert metrics["looped"] is True
    assert metrics["duration_s"] == pytest.approx(0.75, abs=1e-2)
    assert metrics["peak_dbfs"] == pytest.approx(-6.0, abs=0.1)


def test_process_404_unknown_model_envelope():
    client = _make_client()
    wav = _make_wav_bytes(duration_s=0.1)
    r = client.post(
        "/v1/audio/process",
        files={"file": ("x.wav", wav, "audio/wav")},
        data={"model": "missing-model"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


@pytest.mark.parametrize(
    "data",
    [
        {"output_format": "mp3"},
        {"response_format": "xml"},
        {"target_peak_dbfs": "1.5"},
        {"target_peak_dbfs": "-60.0"},
        {"sample_rate": "4000"},
        {"sample_rate": "192000"},
        {"loop_fade_s": "-0.1"},
        {"loop_fade_s": "45.0"},
        {"pad_lead_ms": "-5.0"},
        {"fade_in_ms": "-1.0"},
    ],
)
def test_process_400_on_invalid_parameters(data):
    client = _make_client()
    wav = _make_wav_bytes(duration_s=0.1)
    r = client.post(
        "/v1/audio/process",
        files={"file": ("x.wav", wav, "audio/wav")},
        data=data,
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_process_400_on_empty_upload():
    client = _make_client()
    r = client.post(
        "/v1/audio/process",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_process_415_on_undecodable_audio():
    client = _make_client()
    r = client.post(
        "/v1/audio/process",
        files={"file": ("garbage.bin", b"not-a-valid-audio-file", "application/octet-stream")},
    )
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_media_type"


def test_process_flac_when_soundfile_available():
    pytest.importorskip("soundfile")
    client = _make_client()
    wav = _make_wav_bytes(duration_s=0.2)
    r = client.post(
        "/v1/audio/process",
        files={"file": ("clip.wav", wav, "audio/wav")},
        data={"output_format": "flac", "response_format": "audio"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/flac"
    assert r.content[:4] == b"fLaC"


def test_process_flac_400_when_soundfile_missing():
    client = _make_client()
    wav = _make_wav_bytes(duration_s=0.2)
    with patch(
        "muse.modalities.audio_process.codec._try_import_soundfile",
        return_value=None,
    ):
        r = client.post(
            "/v1/audio/process",
            files={"file": ("clip.wav", wav, "audio/wav")},
            data={"output_format": "flac"},
        )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"
    assert "soundfile" in r.json()["error"]["message"]


def test_process_opus_400_when_ffmpeg_missing():
    client = _make_client()
    wav = _make_wav_bytes(duration_s=0.2)
    with patch("muse.modalities._common.audio_codec.shutil.which", return_value=None):
        r = client.post(
            "/v1/audio/process",
            files={"file": ("clip.wav", wav, "audio/wav")},
            data={"output_format": "opus"},
        )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"
    assert "ffmpeg" in r.json()["error"]["message"]


def test_process_runs_under_inference_lock():
    class _LockAssertingBackend:
        model_id = "lock-audio-tools"

        def __init__(self):
            self._inference_lock = threading.Lock()
            self.lock_was_held = False
            self._inner = AudioTools(model_id=self.model_id)

        def process(self, audio, sr, options=None) -> AudioProcessResult:
            self.lock_was_held = self._inference_lock.locked()
            return self._inner.process(audio, sr, options)

    backend = _LockAssertingBackend()
    client = _make_client(backend)
    wav = _make_wav_bytes(duration_s=0.1)
    r = client.post("/v1/audio/process", files={"file": ("x.wav", wav, "audio/wav")})
    assert r.status_code == 200
    assert backend.lock_was_held


def test_codec_decodes_8bit_and_24bit_wav():
    # 8-bit unsigned PCM WAV
    buf8 = io.BytesIO()
    with wave.open(buf8, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(8000)
        w.writeframes(bytes([128, 255, 0, 128]))
    arr8, sr8 = decode_audio_bytes(buf8.getvalue())
    assert sr8 == 8000
    assert arr8.shape == (4,)
    assert arr8[0] == pytest.approx(0.0, abs=1e-3)
    assert arr8[1] > 0.9
    assert arr8[2] == pytest.approx(-1.0, abs=1e-3)

    # 24-bit signed PCM WAV (1 frame: 0x400000 = +0.5, 0xC00000 = -0.5)
    buf24 = io.BytesIO()
    with wave.open(buf24, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00\x40\x00\x00\xc0")
    arr24, sr24 = decode_audio_bytes(buf24.getvalue())
    assert sr24 == 16000
    assert arr24[0] == pytest.approx(0.5, abs=1e-4)
    assert arr24[1] == pytest.approx(-0.5, abs=1e-4)
