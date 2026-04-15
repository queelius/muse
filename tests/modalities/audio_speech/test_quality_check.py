"""Tests for quality_check — multi-signal failure detection."""
import torch
import inspect
import pytest

from muse.modalities.audio_speech.tts import Narro


@pytest.fixture
def tts():
    n = Narro.__new__(Narro)
    n.device = 'cpu'
    return n


class TestQualityCheck:
    def test_passes_good_output(self, tts):
        response = {
            'hidden_state': torch.randn(50, 512),
            'token_entropy': torch.full((50,), 2.0),
            'finish_reason': 'stop',
        }
        assert tts.quality_check(response, "A normal sentence.") is None

    def test_detects_repetition(self, tts):
        response = {
            'hidden_state': torch.zeros(50, 512),
            'token_entropy': torch.full((50,), 2.0),
            'finish_reason': 'stop',
        }
        assert tts.quality_check(response, "A sentence.") == 'repetition'

    def test_detects_high_entropy(self, tts):
        response = {
            'hidden_state': torch.randn(50, 512),
            'token_entropy': torch.full((50,), 20.0),
            'finish_reason': 'stop',
        }
        assert tts.quality_check(response, "A sentence.") == 'garbled'

    def test_detects_truncation(self, tts):
        response = {
            'hidden_state': torch.randn(50, 512),
            'token_entropy': torch.full((50,), 2.0),
            'finish_reason': 'length',
        }
        assert tts.quality_check(response, "A sentence.") == 'truncated'

    def test_detects_too_many_tokens(self, tts):
        response = {
            'hidden_state': torch.randn(200, 512),
            'token_entropy': torch.full((200,), 2.0),
            'finish_reason': 'stop',
        }
        assert tts.quality_check(response, "Short.") == 'length_anomaly'

    def test_detects_too_few_tokens(self, tts):
        response = {
            'hidden_state': torch.randn(1, 512),
            'token_entropy': torch.full((1,), 2.0),
            'finish_reason': 'stop',
        }
        assert tts.quality_check(response, "A" * 100) == 'length_anomaly'


class TestRetriesDefault:
    def test_encode_batch_defaults_to_retries_1(self):
        """encode_batch should default to retries=1 (not 0)."""
        sig = inspect.signature(Narro.encode_batch)
        assert sig.parameters['retries'].default == 1
