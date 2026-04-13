"""Tests for narro.models.xtts: XTTS v2 adapter."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from narro.protocol import AudioChunk, AudioResult, TTSModel


@pytest.fixture(autouse=True)
def _isolate_voices(tmp_path, monkeypatch):
    monkeypatch.setenv("NARRO_HOME", str(tmp_path / "narro"))


class TestXttsModel:
    def _make_adapter(self):
        from narro.models.xtts import XttsModel

        mock_tts = MagicMock()
        mock_tts.tts.return_value = [0.1] * 24000

        adapter = object.__new__(XttsModel)
        adapter._tts = mock_tts
        adapter._device = "cpu"
        return adapter

    def test_protocol_conformance(self):
        assert isinstance(self._make_adapter(), TTSModel)

    def test_model_id(self):
        assert self._make_adapter().model_id == "xtts-v2"

    def test_sample_rate(self):
        assert self._make_adapter().sample_rate == 24000

    def test_synthesize_returns_audio_result(self):
        result = self._make_adapter().synthesize("Hello")
        assert isinstance(result, AudioResult)
        assert result.sample_rate == 24000
        assert len(result.audio) == 24000

    def test_synthesize_with_language(self):
        adapter = self._make_adapter()
        adapter.synthesize("Hola mundo", language="es")
        call_kwargs = adapter._tts.tts.call_args[1]
        assert call_kwargs["language"] == "es"

    def test_stream_yields_one_chunk(self):
        chunks = list(self._make_adapter().synthesize_stream("Hello"))
        assert len(chunks) == 1
        assert isinstance(chunks[0], AudioChunk)

    def test_unknown_voice_raises(self):
        with pytest.raises(ValueError, match="Unknown voice"):
            self._make_adapter().synthesize("Hello", voice="nonexistent")


class TestXttsVoiceCloning:
    def _make_adapter(self):
        from narro.models.xtts import XttsModel

        mock_tts = MagicMock()
        mock_tts.tts.return_value = [0.1] * 24000

        adapter = object.__new__(XttsModel)
        adapter._tts = mock_tts
        adapter._device = "cpu"
        return adapter

    def test_create_voice(self, tmp_path):
        adapter = self._make_adapter()
        wav_path = tmp_path / "ref.wav"
        wav_path.write_bytes(b"RIFF" + b"\x00" * 100)

        adapter.create_voice("alex", str(wav_path))
        assert "alex" in adapter.list_voices()

    def test_create_voice_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            self._make_adapter().create_voice("ghost", "/nonexistent.wav")

    def test_custom_voice_used_in_synthesis(self, tmp_path):
        adapter = self._make_adapter()
        wav_path = tmp_path / "ref.wav"
        wav_path.write_bytes(b"RIFF" + b"\x00" * 100)
        adapter.create_voice("alex", str(wav_path))

        adapter.synthesize("Hello", voice="alex")
        call_kwargs = adapter._tts.tts.call_args[1]
        assert "alex.wav" in call_kwargs["speaker_wav"]
