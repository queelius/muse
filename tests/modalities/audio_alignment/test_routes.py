import os
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from muse.core import config
from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.audio_alignment import (
    AlignmentWord,
    AudioAlignmentDecodeError,
    AudioAlignmentDurationExceededError,
    AudioAlignmentResult,
    MODALITY,
    UnalignableTextError,
    UnsupportedAlignmentLanguageError,
    build_router,
)


class _FakeAlignmentModel:
    def __init__(self, model_id="align-fake"):
        self.model_id = model_id
        self.last_path = None
        self.transcript = None
        self.language = None
        self.max_duration_seconds = None

    def align(
        self, audio_path, transcript, *, language=None,
        max_duration_seconds=None,
    ):
        self.last_path = audio_path
        self.transcript = transcript
        self.language = language
        self.max_duration_seconds = max_duration_seconds
        return AudioAlignmentResult(
            text=transcript,
            language="English",
            duration_seconds=1.0,
            words=[AlignmentWord("Hello", 0.1, 0.5, 0.95)],
            metadata={"family": "fake"},
        )


def _client(backend):
    registry = ModalityRegistry()
    registry.register(MODALITY, backend)
    app = create_app(
        registry=registry,
        routers={MODALITY: build_router(registry)},
    )
    return TestClient(app)


def _post(client, *, audio=b"WAV", text="Hello", **data):
    return client.post(
        "/v1/audio/alignments",
        files={"file": ("clip.wav", audio, "audio/wav")},
        data={"text": text, **data},
    )


def test_returns_word_alignment_and_cleans_temp_file():
    backend = _FakeAlignmentModel()
    response = _post(
        _client(backend), model=backend.model_id, language="en",
        text="  Hello  ",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "audio.alignment"
    assert body["words"][0] == {
        "word": "Hello", "start": 0.1, "end": 0.5,
        "confidence": 0.95,
    }
    assert backend.transcript == "Hello"
    assert backend.language == "en"
    assert backend.max_duration_seconds == 300.0
    assert backend.last_path is not None
    assert not os.path.exists(backend.last_path)


def test_default_model_is_optional():
    assert _post(_client(_FakeAlignmentModel())).status_code == 200


def test_unknown_model_returns_404():
    response = _post(_client(_FakeAlignmentModel("real")), model="ghost")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_empty_inputs_return_400():
    client = _client(_FakeAlignmentModel())
    assert _post(client, audio=b"").status_code == 400
    assert _post(client, text="   ").status_code == 400


def test_size_and_text_caps(monkeypatch):
    monkeypatch.setenv("MUSE_AUDIO_ALIGNMENT_MAX_BYTES", "3")
    monkeypatch.setenv("MUSE_AUDIO_ALIGNMENT_MAX_TEXT_CHARS", "4")
    config.reset_config()
    try:
        client = _client(_FakeAlignmentModel())
        audio_response = _post(client, audio=b"1234", text="Hey")
        text_response = _post(client, text="12345")
    finally:
        config.reset_config()
    assert audio_response.status_code == 413
    assert text_response.status_code == 400


def test_live_duration_cap_is_forwarded(monkeypatch):
    monkeypatch.setenv("MUSE_AUDIO_ALIGNMENT_MAX_DURATION_SECONDS", "42.5")
    config.reset_config()
    backend = _FakeAlignmentModel()
    try:
        response = _post(_client(backend))
    finally:
        config.reset_config()
    assert response.status_code == 200
    assert backend.max_duration_seconds == 42.5


def test_input_errors_return_400():
    for exc in (
        UnsupportedAlignmentLanguageError("xx", supported=("English",)),
        UnalignableTextError("no alignable words"),
    ):
        backend = MagicMock()
        backend.model_id = "bad-input"
        backend.align.side_effect = exc
        response = _post(_client(backend))
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_parameter"


def test_decoded_duration_limit_returns_413():
    backend = MagicMock()
    backend.model_id = "too-long"
    backend.align.side_effect = AudioAlignmentDurationExceededError(
        maximum_seconds=300, actual_seconds=500,
    )
    response = _post(_client(backend))
    assert response.status_code == 413
    assert "500.000s" in response.json()["error"]["message"]


def test_decode_and_backend_errors_are_generic():
    decoder = MagicMock()
    decoder.model_id = "decoder"
    decoder.align.side_effect = AudioAlignmentDecodeError(
        "decoder leaked /secret/path"
    )
    decode_response = _post(_client(decoder))
    assert decode_response.status_code == 415
    assert "/secret/path" not in decode_response.text

    backend = MagicMock()
    backend.model_id = "broken"
    backend.align.side_effect = RuntimeError(
        "decode_forced_alignment returned private backend detail"
    )
    backend_response = _post(_client(backend))
    assert backend_response.status_code == 500
    assert "private backend detail" not in backend_response.text
