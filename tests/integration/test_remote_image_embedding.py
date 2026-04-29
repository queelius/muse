"""Integration tests for /v1/images/embeddings against a live muse server.

Opt-in via MUSE_REMOTE_SERVER. Skipped when the server is unreachable
or no image embedder is loaded.

Targets the configurable image-embedding model id (default
dinov2-small, override via MUSE_IMAGE_EMBEDDING_MODEL_ID).
"""
from __future__ import annotations

import base64
import io

import pytest

from muse.modalities.image_embedding import ImageEmbeddingsClient


pytest.importorskip("PIL.Image")


def _png_data_url(width=32, height=32, color=(0, 128, 255)):
    from PIL import Image
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def test_protocol_basic_image_embedding(remote_url, image_embedding_model):
    """Hard claim: an embed call returns the OpenAI envelope shape."""
    c = ImageEmbeddingsClient(remote_url)
    out = c.embed_envelope(_png_data_url(), model=image_embedding_model)
    assert out["object"] == "list"
    assert "data" in out
    assert len(out["data"]) == 1
    assert "model" in out
    assert "usage" in out


def test_protocol_data_entries_have_object_marker(remote_url, image_embedding_model):
    """Every entry in `data` carries object='embedding'."""
    c = ImageEmbeddingsClient(remote_url)
    out = c.embed_envelope(_png_data_url(), model=image_embedding_model)
    for entry in out["data"]:
        assert entry["object"] == "embedding"


def test_protocol_indices_zero_based_and_ordered(remote_url, image_embedding_model):
    c = ImageEmbeddingsClient(remote_url)
    out = c.embed_envelope(
        [_png_data_url(), _png_data_url(), _png_data_url()],
        model=image_embedding_model,
    )
    indices = [d["index"] for d in out["data"]]
    assert indices == [0, 1, 2]


def test_protocol_embed_returns_list_of_floats(remote_url, image_embedding_model):
    c = ImageEmbeddingsClient(remote_url)
    out = c.embed(_png_data_url(), model=image_embedding_model)
    assert isinstance(out, list)
    assert len(out) == 1
    assert isinstance(out[0], list)
    assert all(isinstance(v, (int, float)) for v in out[0])


def test_protocol_base64_roundtrip(remote_url, image_embedding_model):
    """encoding_format='base64' decodes to the same dim as 'float'."""
    c = ImageEmbeddingsClient(remote_url)
    via_float = c.embed(
        _png_data_url(), model=image_embedding_model, encoding_format="float",
    )
    via_b64 = c.embed(
        _png_data_url(), model=image_embedding_model, encoding_format="base64",
    )
    assert len(via_float) == len(via_b64) == 1
    assert len(via_float[0]) == len(via_b64[0])


def test_protocol_response_model_is_catalog_id(remote_url, image_embedding_model):
    """The response 'model' field should be the catalog id, not the HF
    repo path."""
    c = ImageEmbeddingsClient(remote_url)
    out = c.embed_envelope(_png_data_url(), model=image_embedding_model)
    assert out["model"] == image_embedding_model


def test_protocol_usage_is_zero_for_image_inputs(remote_url, image_embedding_model):
    """Image embedding has no text tokenization; usage stays 0."""
    c = ImageEmbeddingsClient(remote_url)
    out = c.embed_envelope(_png_data_url(), model=image_embedding_model)
    assert out["usage"]["prompt_tokens"] == 0
    assert out["usage"]["total_tokens"] == 0


def test_protocol_dimensions_truncation_honored(remote_url, image_embedding_model):
    """Requesting smaller dimensions truncates the output vector."""
    c = ImageEmbeddingsClient(remote_url)
    full = c.embed(_png_data_url(), model=image_embedding_model)
    full_dim = len(full[0])
    if full_dim <= 1:
        pytest.skip(f"native dim too small to truncate ({full_dim})")
    truncated = c.embed(
        _png_data_url(), model=image_embedding_model,
        dimensions=max(1, full_dim // 2),
    )
    assert len(truncated[0]) == max(1, full_dim // 2)
    assert len(truncated[0]) < full_dim


def test_observe_same_image_yields_similar_embeddings(remote_url, image_embedding_model):
    """Sanity: identical input twice produces identical (or near-identical)
    embeddings. Records what the model actually did rather than asserting
    exact equality, since models with stochastic eval modes may drift."""
    c = ImageEmbeddingsClient(remote_url)
    img = _png_data_url()
    a = c.embed(img, model=image_embedding_model)[0]
    b = c.embed(img, model=image_embedding_model)[0]
    assert len(a) == len(b)
    # Compute cosine similarity; identical inputs should be very close to 1.
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        pytest.skip("embedding has zero norm; cannot compute cosine similarity")
    cos = dot / (norm_a * norm_b)
    assert cos > 0.99, f"identical inputs gave cos={cos:.4f}"


def test_observe_batch_size_one_matches_single_input(remote_url, image_embedding_model):
    """Sanity: batch [a] should produce one row equal to single input a."""
    c = ImageEmbeddingsClient(remote_url)
    img = _png_data_url(color=(123, 45, 67))
    single = c.embed(img, model=image_embedding_model)
    batch = c.embed([img], model=image_embedding_model)
    assert len(single) == 1
    assert len(batch) == 1
    # Allow tiny numerical drift; cosine similarity should still be ~1.
    dot = sum(x * y for x, y in zip(single[0], batch[0]))
    norm_a = sum(x * x for x in single[0]) ** 0.5
    norm_b = sum(y * y for y in batch[0]) ** 0.5
    if norm_a == 0 or norm_b == 0:
        pytest.skip("embedding has zero norm; cannot compute cosine similarity")
    cos = dot / (norm_a * norm_b)
    assert cos > 0.99


def test_protocol_default_model_resolves(remote_url, image_embedding_model):
    """Omitting `model` should resolve to the default registered model."""
    c = ImageEmbeddingsClient(remote_url)
    # We don't pass model= here; if the server has any image_embedding
    # model loaded, this should succeed regardless of which one is the
    # registered default.
    out = c.embed_envelope(_png_data_url())
    assert "model" in out
    assert out["object"] == "list"
