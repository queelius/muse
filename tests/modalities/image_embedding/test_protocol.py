"""Tests for the image/embedding Protocol + ImageEmbeddingResult dataclass."""
from muse.modalities.image_embedding.protocol import (
    ImageEmbeddingModel,
    ImageEmbeddingResult,
)


def test_result_dataclass_required_fields():
    r = ImageEmbeddingResult(
        embeddings=[[0.1, 0.2, 0.3]],
        dimensions=3,
        model_id="dinov2-small",
        n_images=1,
    )
    assert r.embeddings == [[0.1, 0.2, 0.3]]
    assert r.dimensions == 3
    assert r.model_id == "dinov2-small"
    assert r.n_images == 1
    # metadata defaults to {}
    assert r.metadata == {}


def test_result_metadata_defaults_to_empty_dict():
    r = ImageEmbeddingResult(
        embeddings=[],
        dimensions=384,
        model_id="x",
        n_images=0,
    )
    assert r.metadata == {}


def test_result_metadata_accepts_arbitrary_dict():
    r = ImageEmbeddingResult(
        embeddings=[[1.0]],
        dimensions=1,
        model_id="x",
        n_images=1,
        metadata={"source": "transformers", "truncation_warning": False},
    )
    assert r.metadata["source"] == "transformers"
    assert r.metadata["truncation_warning"] is False


def test_protocol_runtime_checkable_via_duck_typing():
    """A class with the right shape passes isinstance(...) without inheritance."""

    class Fake:
        @property
        def model_id(self):
            return "fake"

        @property
        def dimensions(self):
            return 384

        def embed(self, images, *, dimensions=None):
            return ImageEmbeddingResult(
                embeddings=[[0.0] * (dimensions or 384)] * len(images),
                dimensions=dimensions or 384,
                model_id="fake",
                n_images=len(images),
            )

    assert isinstance(Fake(), ImageEmbeddingModel)


def test_protocol_rejects_wrong_shape():
    """A class missing `embed` is not a valid ImageEmbeddingModel."""

    class NotEmbedder:
        model_id = "x"
        dimensions = 0

    assert not isinstance(NotEmbedder(), ImageEmbeddingModel)


def test_result_supports_batched_embeddings():
    """N images -> N embeddings; dimensions identical across rows."""
    r = ImageEmbeddingResult(
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        dimensions=3,
        model_id="x",
        n_images=2,
    )
    assert len(r.embeddings) == 2
    assert all(len(row) == r.dimensions for row in r.embeddings)


def test_result_dataclass_field_independence():
    """Mutating one ImageEmbeddingResult's metadata doesn't bleed to others."""
    a = ImageEmbeddingResult(
        embeddings=[], dimensions=0, model_id="x", n_images=0,
    )
    b = ImageEmbeddingResult(
        embeddings=[], dimensions=0, model_id="x", n_images=0,
    )
    a.metadata["foo"] = "bar"
    assert "foo" not in b.metadata
