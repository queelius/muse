"""End-to-end: multipart upload flows through FastAPI + codec correctly.

Uses a fake TranscriptionModel backend; no real weights. This is the
only test that exercises the full UploadFile -> tempfile -> backend ->
codec chain together. Fast-lane unit tests cover each step in
isolation; this test catches integration bugs.
"""
import io
import wave

import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.audio_transcription import (
    MODALITY,
    Segment,
    TranscriptionResult,
    build_router,
)


pytestmark = pytest.mark.slow


def _make_sine_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    """Synthesize a silent WAV in memory (backend is mocked; content doesn't matter)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


class _FakeWhisper:
    def __init__(self):
        self.called_with = None
        self.model_id = "whisper-tiny"

    def transcribe(self, audio_path, **kwargs):
        self.called_with = (audio_path, kwargs)
        return TranscriptionResult(
            text="mocked transcript",
            language="en",
            duration=1.0,
            task=kwargs.get("task", "transcribe"),
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=1.0,
                    text="mocked transcript",
                    words=None,
                )
            ],
        )


@pytest.mark.timeout(10)
def test_multipart_flow_srt_end_to_end():
    fake = _FakeWhisper()
    reg = ModalityRegistry()
    reg.register(MODALITY, fake, manifest={"model_id": "whisper-tiny"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})

    wav = _make_sine_wav(1.0)
    client = TestClient(app)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", wav, "audio/wav")},
        data={"model": "whisper-tiny", "response_format": "srt"},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/x-subrip")
    assert "mocked transcript" in r.text
    # Backend saw a real file path
    audio_path, kwargs = fake.called_with
    assert audio_path.endswith(".wav")
    assert kwargs["task"] == "transcribe"
