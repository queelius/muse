"""Tests for ImageEmbeddingsClient (HTTP wrapper)."""
import base64
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.image_embedding.client import (
    ImageEmbeddingsClient,
    _bytes_to_data_url,
)
from muse.modalities.image_embedding.codec import embedding_to_base64


def _envelope(vectors):
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": v, "index": i}
            for i, v in enumerate(vectors)
        ],
        "model": "dinov2-small",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def test_init_uses_default_server_when_no_url():
    c = ImageEmbeddingsClient()
    # Default depends on env; we just want a non-empty url.
    assert c.server_url


def test_init_strips_trailing_slash():
    c = ImageEmbeddingsClient("http://example.com:8000/")
    assert c.server_url == "http://example.com:8000"


def test_init_uses_explicit_url_over_env(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://env-url:9999")
    c = ImageEmbeddingsClient("http://explicit:7000")
    assert c.server_url == "http://explicit:7000"


def test_init_uses_env_when_no_explicit_url(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://from-env:8888")
    c = ImageEmbeddingsClient()
    assert c.server_url == "http://from-env:8888"


def test_init_sets_timeout():
    c = ImageEmbeddingsClient(timeout=42.0)
    assert c.timeout == 42.0


def test_bytes_to_data_url_default_png():
    raw = b"some-bytes"
    url = _bytes_to_data_url(raw)
    assert url.startswith("data:image/png;base64,")
    payload = url.split(",", 1)[1]
    assert base64.b64decode(payload) == raw


def test_bytes_to_data_url_jpeg_mime():
    url = _bytes_to_data_url(b"x", mime="image/jpeg")
    assert url.startswith("data:image/jpeg;base64,")


def test_normalize_input_string_passthrough():
    out = ImageEmbeddingsClient._normalize_input("data:image/png;base64,xyz", mime="image/png")
    assert out == "data:image/png;base64,xyz"


def test_normalize_input_bytes_encoded_as_data_url():
    out = ImageEmbeddingsClient._normalize_input(b"raw-bytes", mime="image/png")
    assert out.startswith("data:image/png;base64,")


def test_normalize_input_list_preserves_order():
    out = ImageEmbeddingsClient._normalize_input(["a", "b", "c"], mime="image/png")
    assert out == ["a", "b", "c"]


def test_normalize_input_mixed_list_encodes_bytes_only():
    out = ImageEmbeddingsClient._normalize_input(
        ["url-a", b"raw", "url-c"], mime="image/png",
    )
    assert out[0] == "url-a"
    assert out[1].startswith("data:image/png;base64,")
    assert out[2] == "url-c"


def test_normalize_input_rejects_unsupported_type():
    with pytest.raises(TypeError):
        ImageEmbeddingsClient._normalize_input(123, mime="image/png")


def test_embed_returns_list_of_vectors_for_float_format():
    c = ImageEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = _envelope([[0.1, 0.2], [0.3, 0.4]])

    with patch("muse.modalities.image_embedding.client.requests.post",
               return_value=fake_response) as post:
        out = c.embed(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]
    # POSTed to /v1/images/embeddings
    args, kwargs = post.call_args
    assert args[0] == "http://srv/v1/images/embeddings"
    assert kwargs["json"]["input"] == ["a", "b"]
    assert kwargs["json"]["encoding_format"] == "float"


def test_embed_decodes_base64_responses():
    c = ImageEmbeddingsClient("http://srv")
    encoded = embedding_to_base64([0.5, 0.6, 0.7])
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": encoded, "index": 0}],
        "model": "dinov2-small",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
    with patch("muse.modalities.image_embedding.client.requests.post",
               return_value=fake_response):
        out = c.embed("data:image/png;base64,xx", encoding_format="base64")
    assert len(out) == 1
    for got, want in zip(out[0], [0.5, 0.6, 0.7]):
        assert got == pytest.approx(want, rel=1e-6)


def test_embed_passes_model_and_dimensions_when_set():
    c = ImageEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = _envelope([[0.0]])
    with patch("muse.modalities.image_embedding.client.requests.post",
               return_value=fake_response) as post:
        c.embed("x", model="custom", dimensions=128)
    body = post.call_args.kwargs["json"]
    assert body["model"] == "custom"
    assert body["dimensions"] == 128


def test_embed_omits_model_and_dimensions_when_none():
    c = ImageEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = _envelope([[0.0]])
    with patch("muse.modalities.image_embedding.client.requests.post",
               return_value=fake_response) as post:
        c.embed("x")
    body = post.call_args.kwargs["json"]
    assert "model" not in body
    assert "dimensions" not in body


def test_embed_raises_on_non_200():
    c = ImageEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 500
    fake_response.text = "boom"
    with patch("muse.modalities.image_embedding.client.requests.post",
               return_value=fake_response):
        with pytest.raises(RuntimeError, match="500"):
            c.embed("x")


def test_embed_invalid_encoding_format_raises():
    c = ImageEmbeddingsClient("http://srv")
    with pytest.raises(ValueError):
        c.embed("x", encoding_format="WRONG")


def test_embed_envelope_returns_full_openai_shape():
    c = ImageEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = _envelope([[0.1]])
    with patch("muse.modalities.image_embedding.client.requests.post",
               return_value=fake_response):
        out = c.embed_envelope("x")
    assert out["object"] == "list"
    assert "data" in out
    assert "model" in out
    assert "usage" in out


def test_embed_envelope_passes_model_and_dimensions():
    c = ImageEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = _envelope([[0.1]])
    with patch("muse.modalities.image_embedding.client.requests.post",
               return_value=fake_response) as post:
        c.embed_envelope("x", model="my-model", dimensions=64)
    body = post.call_args.kwargs["json"]
    assert body["model"] == "my-model"
    assert body["dimensions"] == 64


def test_embed_envelope_invalid_encoding_format_raises():
    c = ImageEmbeddingsClient("http://srv")
    with pytest.raises(ValueError):
        c.embed_envelope("x", encoding_format="WRONG")


def test_embed_envelope_raises_on_non_200():
    c = ImageEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 503
    fake_response.text = "service unavailable"
    with patch("muse.modalities.image_embedding.client.requests.post",
               return_value=fake_response):
        with pytest.raises(RuntimeError, match="503"):
            c.embed_envelope("x")


def test_embed_bytes_input_auto_encoded_as_data_url():
    """Calling embed(b"...") should JSON-encode as a data URL."""
    c = ImageEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = _envelope([[0.0]])
    with patch("muse.modalities.image_embedding.client.requests.post",
               return_value=fake_response) as post:
        c.embed(b"raw-png-bytes")
    body = post.call_args.kwargs["json"]
    assert body["input"].startswith("data:image/png;base64,")


def test_embed_jpeg_mime_override():
    c = ImageEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = _envelope([[0.0]])
    with patch("muse.modalities.image_embedding.client.requests.post",
               return_value=fake_response) as post:
        c.embed(b"raw-jpeg-bytes", mime="image/jpeg")
    body = post.call_args.kwargs["json"]
    assert body["input"].startswith("data:image/jpeg;base64,")


def test_embed_uses_request_timeout():
    c = ImageEmbeddingsClient("http://srv", timeout=11.5)
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = _envelope([[0.0]])
    with patch("muse.modalities.image_embedding.client.requests.post",
               return_value=fake_response) as post:
        c.embed("x")
    assert post.call_args.kwargs["timeout"] == 11.5


def test_client_exposed_on_modality_package_init():
    """The modality __init__ should re-export ImageEmbeddingsClient."""
    from muse.modalities import image_embedding
    assert hasattr(image_embedding, "ImageEmbeddingsClient")
    assert image_embedding.ImageEmbeddingsClient is ImageEmbeddingsClient
