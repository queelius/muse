"""Tests for narro/encoded.py — IR dataclasses."""

import numpy as np

from muse.modalities.audio_speech.encoded import (
    SentenceEncoding,
    EncodedSpeech,
    FORMAT_VERSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sentence(seq_len=20, text_index=0, sentence_index=0, text="hello world",
                   finish_reason='stop', attention=None, hidden_dim=512):
    """Create a SentenceEncoding with random data."""
    return SentenceEncoding(
        hidden_states=np.random.randn(seq_len, hidden_dim).astype(np.float32),
        token_ids=np.random.randint(0, 8192, size=seq_len, dtype=np.int32),
        token_entropy=np.random.rand(seq_len).astype(np.float32),
        finish_reason=finish_reason,
        text=text,
        text_index=text_index,
        sentence_index=sentence_index,
        attention_weights=attention,
    )


def _make_encoded(sentences=None, model_id='test/model'):
    """Create an EncodedSpeech with default params."""
    if sentences is None:
        sentences = [_make_sentence()]
    return EncodedSpeech(
        sentences=sentences,
        model_id=model_id,
    )


# ---------------------------------------------------------------------------
# SentenceEncoding tests
# ---------------------------------------------------------------------------

class TestSentenceEncoding:

    def test_construction(self):
        s = _make_sentence(seq_len=10, text="test sentence")
        assert s.hidden_states.shape == (10, 512)
        assert s.token_ids.shape == (10,)
        assert s.token_entropy.shape == (10,)
        assert s.finish_reason == 'stop'
        assert s.text == "test sentence"
        assert s.text_index == 0
        assert s.sentence_index == 0

    def test_optional_fields_default_none(self):
        s = _make_sentence()
        assert s.attention_weights is None
        assert s.input_token_offsets is None

    def test_with_attention(self):
        attn = np.random.randn(20, 15).astype(np.float32)
        s = _make_sentence(seq_len=20, attention=attn)
        assert s.attention_weights is not None
        assert s.attention_weights.shape == (20, 15)

    def test_finish_reason_length(self):
        s = _make_sentence(finish_reason='length')
        assert s.finish_reason == 'length'


# ---------------------------------------------------------------------------
# EncodedSpeech tests
# ---------------------------------------------------------------------------

class TestEncodedSpeech:

    def test_total_tokens(self):
        s1 = _make_sentence(seq_len=10)
        s2 = _make_sentence(seq_len=20)
        enc = _make_encoded(sentences=[s1, s2])
        assert enc.total_tokens == 30

    def test_mean_entropy(self):
        # Use known entropy values
        s = _make_sentence(seq_len=5)
        s.token_entropy = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        enc = _make_encoded(sentences=[s])
        assert abs(enc.mean_entropy - 3.0) < 1e-6

    def test_mean_entropy_empty(self):
        s = _make_sentence(seq_len=0)
        s.hidden_states = np.zeros((0, 512), dtype=np.float32)
        s.token_ids = np.zeros(0, dtype=np.int32)
        s.token_entropy = np.zeros(0, dtype=np.float32)
        enc = _make_encoded(sentences=[s])
        assert enc.mean_entropy == 0.0

    def test_estimated_duration(self):
        # 100 tokens * 2048 samples/token / 32000 Hz = 6.4 seconds
        s = _make_sentence(seq_len=100)
        enc = _make_encoded(sentences=[s])
        assert abs(enc.estimated_duration - 6.4) < 0.01

    def test_num_texts_single(self):
        s = _make_sentence(text_index=0)
        enc = _make_encoded(sentences=[s])
        assert enc.num_texts == 1

    def test_num_texts_multiple(self):
        s1 = _make_sentence(text_index=0)
        s2 = _make_sentence(text_index=1)
        s3 = _make_sentence(text_index=2)
        enc = _make_encoded(sentences=[s1, s2, s3])
        assert enc.num_texts == 3

    def test_num_texts_empty(self):
        enc = _make_encoded(sentences=[])
        assert enc.num_texts == 0

    def test_defaults(self):
        enc = _make_encoded()
        assert enc.format_version == FORMAT_VERSION
        assert enc.sample_rate == 32000
        assert enc.token_audio_samples == 2048
        assert enc.hidden_dim == 512

    def test_generation_params(self):
        enc = EncodedSpeech(
            sentences=[],
            model_id='test',
            top_p=0.8,
            temperature=0.5,
            repetition_penalty=1.5,
        )
        assert enc.top_p == 0.8
        assert enc.temperature == 0.5
        assert enc.repetition_penalty == 1.5

