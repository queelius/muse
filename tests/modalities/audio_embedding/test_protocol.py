"""Tests for the audio/embedding Protocol + AudioEmbeddingResult dataclass."""
from muse.modalities.audio_embedding.protocol import (
    AudioEmbeddingModel,
    AudioEmbeddingResult,
)


def test_result_dataclass_required_fields():
    r = AudioEmbeddingResult(
        embeddings=[[0.1, 0.2, 0.3]],
        dimensions=3,
        model_id="mert-v1-95m",
        n_audio_clips=1,
    )
    assert r.embeddings == [[0.1, 0.2, 0.3]]
    assert r.dimensions == 3
    assert r.model_id == "mert-v1-95m"
    assert r.n_audio_clips == 1
    # metadata defaults to {}
    assert r.metadata == {}


def test_result_metadata_defaults_to_empty_dict():
    r = AudioEmbeddingResult(
        embeddings=[],
        dimensions=768,
        model_id="x",
        n_audio_clips=0,
    )
    assert r.metadata == {}


def test_result_metadata_accepts_arbitrary_dict():
    r = AudioEmbeddingResult(
        embeddings=[[1.0]],
        dimensions=1,
        model_id="x",
        n_audio_clips=1,
        metadata={"source": "transformers", "sample_rate_used": 24000},
    )
    assert r.metadata["source"] == "transformers"
    assert r.metadata["sample_rate_used"] == 24000


def test_protocol_runtime_checkable_via_duck_typing():
    """A class with the right shape passes isinstance(...) without inheritance."""

    class Fake:
        @property
        def model_id(self):
            return "fake"

        @property
        def dimensions(self):
            return 768

        def embed(self, audio_bytes_list):
            return AudioEmbeddingResult(
                embeddings=[[0.0] * 768] * len(audio_bytes_list),
                dimensions=768,
                model_id="fake",
                n_audio_clips=len(audio_bytes_list),
            )

    assert isinstance(Fake(), AudioEmbeddingModel)


def test_protocol_rejects_wrong_shape():
    """A class missing `embed` is not a valid AudioEmbeddingModel."""

    class NotEmbedder:
        model_id = "x"
        dimensions = 0

    assert not isinstance(NotEmbedder(), AudioEmbeddingModel)


def test_result_supports_batched_embeddings():
    """N clips -> N embeddings; dimensions identical across rows."""
    r = AudioEmbeddingResult(
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        dimensions=3,
        model_id="x",
        n_audio_clips=2,
    )
    assert len(r.embeddings) == 2
    assert all(len(row) == r.dimensions for row in r.embeddings)


def test_result_dataclass_field_independence():
    """Mutating one AudioEmbeddingResult's metadata doesn't bleed to others."""
    a = AudioEmbeddingResult(
        embeddings=[], dimensions=0, model_id="x", n_audio_clips=0,
    )
    b = AudioEmbeddingResult(
        embeddings=[], dimensions=0, model_id="x", n_audio_clips=0,
    )
    a.metadata["foo"] = "bar"
    assert "foo" not in b.metadata
