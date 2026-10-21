"""Intermediate representation for Muse Soprano TTS encoder/decoder pipeline.

The IR decouples the LLM encoder from the Vocos decoder, enabling
quality estimation via per-token entropy and sentence/paragraph
alignment from per-sentence token counts.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

FORMAT_VERSION = 1


@dataclass
class SentenceEncoding:
    """One sentence's encoder output.

    Attributes:
        hidden_states: Decoder input, shape (T, 512) float32.
        token_ids: Generated token IDs, shape (T,) int32.
        token_entropy: Per-token entropy in nats, shape (T,) float32.
        finish_reason: 'stop' (EOS reached) or 'length' (max tokens).
        text: Original sentence text before wrapping.
        text_index: Index within the input texts batch.
        sentence_index: Index within sentences of that text.
        attention_weights: Last-layer attention averaged over heads, shape (T, input_len). Opt-in.
        input_token_offsets: Char offsets for input tokens, for word-level highlighting.
    """
    hidden_states: np.ndarray
    token_ids: np.ndarray
    token_entropy: np.ndarray
    finish_reason: str
    text: str
    text_index: int
    sentence_index: int
    attention_weights: Optional[np.ndarray] = None
    input_token_offsets: Optional[list] = None


@dataclass
class EncodedSpeech:
    """Full encoding result from the Narro encoder.

    Attributes:
        sentences: All sentence encodings, ordered by (text_index, sentence_index).
        model_id: HuggingFace model identifier.
        format_version: Wire format version for forward compatibility.
        sample_rate: Output audio sample rate (Hz).
        token_audio_samples: Audio samples per decoder token.
        hidden_dim: LLM hidden state dimension.
        top_p: Nucleus sampling threshold used during generation.
        temperature: Sampling temperature used during generation.
        repetition_penalty: Repetition penalty used during generation.
    """
    sentences: list
    model_id: str
    format_version: int = FORMAT_VERSION
    sample_rate: int = 32000
    token_audio_samples: int = 2048
    hidden_dim: int = 512
    top_p: float = 0.95
    temperature: float = 0.0
    repetition_penalty: float = 1.2

    @property
    def total_tokens(self) -> int:
        return sum(len(s.hidden_states) for s in self.sentences)

    @property
    def mean_entropy(self) -> float:
        all_entropy = np.concatenate([s.token_entropy for s in self.sentences])
        if len(all_entropy) == 0:
            return 0.0
        return float(np.mean(all_entropy))

    @property
    def estimated_duration(self) -> float:
        """Estimated audio duration in seconds."""
        return self.total_tokens * self.token_audio_samples / self.sample_rate

    @property
    def num_texts(self) -> int:
        if not self.sentences:
            return 0
        return max(s.text_index for s in self.sentences) + 1


