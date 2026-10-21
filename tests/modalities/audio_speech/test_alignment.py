"""Tests for narro/alignment.py — paragraph-level alignment from hidden-state counts."""

import numpy as np

from muse.modalities.audio_speech.alignment import extract_paragraph_alignment
from muse.modalities.audio_speech.encoded import SentenceEncoding, EncodedSpeech


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sentence(text, seq_len, text_index=0, sentence_index=0):
    """Create a SentenceEncoding with random hidden states (no attention)."""
    return SentenceEncoding(
        hidden_states=np.random.randn(seq_len, 512).astype(np.float32),
        token_ids=np.random.randint(0, 8192, size=seq_len, dtype=np.int32),
        token_entropy=np.random.rand(seq_len).astype(np.float32),
        finish_reason='stop',
        text=text,
        text_index=text_index,
        sentence_index=sentence_index,
        attention_weights=None,
    )


TOKEN_DURATION = 2048 / 32000  # 0.064 seconds


# ---------------------------------------------------------------------------
# extract_paragraph_alignment tests
# ---------------------------------------------------------------------------

class TestExtractParagraphAlignment:

    def test_single_paragraph_single_sentence(self):
        """One paragraph with one sentence produces one entry."""
        sent = _make_sentence("Hello world.", 10, text_index=0, sentence_index=0)
        encoded = EncodedSpeech(sentences=[sent], model_id='test/model')

        result = extract_paragraph_alignment(encoded)

        assert len(result) == 1
        assert result[0]['paragraph'] == 0
        assert result[0]['start'] == 0.0
        assert result[0]['end'] == round(10 * TOKEN_DURATION, 3)

    def test_single_paragraph_multiple_sentences(self):
        """Multiple sentences in one paragraph produce one entry spanning all."""
        T1, T2 = 10, 15
        sent1 = _make_sentence("Hello.", T1, text_index=0, sentence_index=0)
        sent2 = _make_sentence("World.", T2, text_index=0, sentence_index=1)
        encoded = EncodedSpeech(sentences=[sent1, sent2], model_id='test/model')

        result = extract_paragraph_alignment(encoded)

        assert len(result) == 1
        assert result[0]['paragraph'] == 0
        assert result[0]['start'] == 0.0
        assert result[0]['end'] == round((T1 + T2) * TOKEN_DURATION, 3)

    def test_multiple_paragraphs(self):
        """Each text_index becomes one paragraph entry with correct timing."""
        T1, T2, T3 = 10, 15, 20
        sent1 = _make_sentence("Para one.", T1, text_index=0, sentence_index=0)
        sent2 = _make_sentence("Para two A.", T2, text_index=1, sentence_index=0)
        sent3 = _make_sentence("Para two B.", T3, text_index=1, sentence_index=1)
        encoded = EncodedSpeech(sentences=[sent1, sent2, sent3], model_id='test/model')

        result = extract_paragraph_alignment(encoded)

        assert len(result) == 2
        assert result[0]['paragraph'] == 0
        assert result[1]['paragraph'] == 1

        # Paragraph 0: just sent1
        assert result[0]['start'] == 0.0
        p0_end = round(T1 * TOKEN_DURATION, 3)
        assert result[0]['end'] == p0_end

        # Paragraph 1: sent2 + sent3
        assert result[1]['start'] == p0_end
        assert result[1]['end'] == round(p0_end + (T2 + T3) * TOKEN_DURATION, 3)

    def test_empty_sentences_list(self):
        """No sentences produces empty alignment."""
        encoded = EncodedSpeech(sentences=[], model_id='test/model')
        result = extract_paragraph_alignment(encoded)
        assert result == []

    def test_empty_text_advances_offset(self):
        """Sentences with empty text still advance the cumulative offset."""
        T1, T2, T3 = 10, 5, 10
        sent1 = _make_sentence("Para one.", T1, text_index=0, sentence_index=0)
        sent_empty = _make_sentence("", T2, text_index=0, sentence_index=1)
        sent3 = _make_sentence("Para two.", T3, text_index=1, sentence_index=0)
        encoded = EncodedSpeech(
            sentences=[sent1, sent_empty, sent3], model_id='test/model',
        )

        result = extract_paragraph_alignment(encoded)

        # Paragraph 0 includes the empty sentence's duration
        assert result[0]['end'] == round((T1 + T2) * TOKEN_DURATION, 3)
        # Paragraph 1 starts after both
        assert result[1]['start'] == round((T1 + T2) * TOKEN_DURATION, 3)

    def test_timestamps_are_rounded(self):
        """All timestamps should be rounded to 3 decimal places."""
        T = 7
        sent = _make_sentence("Test.", T, text_index=0)
        encoded = EncodedSpeech(sentences=[sent], model_id='test/model')

        result = extract_paragraph_alignment(encoded)

        for entry in result:
            assert entry['start'] == round(entry['start'], 3)
            assert entry['end'] == round(entry['end'], 3)

