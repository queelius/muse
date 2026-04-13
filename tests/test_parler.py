"""Tests for narro.models.parler: Parler-TTS adapter."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from narro.protocol import AudioChunk, AudioResult, TTSModel


class TestParlerModel:
    def _make_adapter(self):
        from narro.models.parler import ParlerModel

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.randn(1, 44100)
        mock_model.config.sampling_rate = 44100

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(input_ids=torch.zeros(1, 10, dtype=torch.long))

        adapter = object.__new__(ParlerModel)
        adapter._model = mock_model
        adapter._tokenizer = mock_tokenizer
        adapter._device = "cpu"
        adapter._sample_rate = 44100
        return adapter

    def test_protocol_conformance(self):
        assert isinstance(self._make_adapter(), TTSModel)

    def test_model_id(self):
        assert self._make_adapter().model_id == "parler-tts"

    def test_sample_rate(self):
        assert self._make_adapter().sample_rate == 44100

    def test_synthesize_returns_audio_result(self):
        result = self._make_adapter().synthesize("Hello")
        assert isinstance(result, AudioResult)
        assert result.sample_rate == 44100

    def test_voice_as_description(self):
        adapter = self._make_adapter()
        adapter.synthesize("Hello", voice="A deep male voice with slow pace")
        call_args = adapter._tokenizer.call_args_list[0]
        assert "deep male voice" in call_args[0][0]

    def test_voice_as_speaker_name(self):
        adapter = self._make_adapter()
        adapter.synthesize("Hello", voice="Laura")
        call_args = adapter._tokenizer.call_args_list[0]
        assert "Laura" in call_args[0][0]

    def test_default_voice_description(self):
        adapter = self._make_adapter()
        adapter.synthesize("Hello")
        call_args = adapter._tokenizer.call_args_list[0]
        assert "neutral" in call_args[0][0].lower() or "clear" in call_args[0][0].lower()

    def test_voices_list(self):
        from narro.models.parler import PARLER_VOICES
        assert "Laura" in PARLER_VOICES
        assert "Jon" in PARLER_VOICES
