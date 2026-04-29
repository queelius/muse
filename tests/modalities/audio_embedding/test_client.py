"""Tests for AudioEmbeddingsClient (HTTP wrapper, multipart upload)."""
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.audio_embedding.client import AudioEmbeddingsClient
from muse.modalities.audio_embedding.codec import embedding_to_base64


def _envelope(vectors):
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": v, "index": i}
            for i, v in enumerate(vectors)
        ],
        "model": "mert-v1-95m",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def test_init_uses_default_server_when_no_url():
    c = AudioEmbeddingsClient()
    assert c.server_url


def test_init_strips_trailing_slash():
    c = AudioEmbeddingsClient("http://example.com:8000/")
    assert c.server_url == "http://example.com:8000"


def test_init_uses_explicit_url_over_env(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://env-url:9999")
    c = AudioEmbeddingsClient("http://explicit:7000")
    assert c.server_url == "http://explicit:7000"


def test_init_uses_env_when_no_explicit_url(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://from-env:8888")
    c = AudioEmbeddingsClient()
    assert c.server_url == "http://from-env:8888"


def test_init_sets_timeout():
    c = AudioEmbeddingsClient(timeout=42.0)
    assert c.timeout == 42.0


def test_build_files_single_bytes_returns_one_part():
    files = AudioEmbeddingsClient._build_files(
        b"raw-audio", filename="x.wav", content_type="audio/wav",
    )
    assert len(files) == 1
    assert files[0][0] == "file"
    assert files[0][1] == ("x.wav", b"raw-audio", "audio/wav")


def test_build_files_list_of_bytes_returns_one_part_per_clip():
    files = AudioEmbeddingsClient._build_files(
        [b"a", b"b", b"c"], filename="x.wav", content_type="audio/wav",
    )
    assert len(files) == 3
    for fname, payload in files:
        assert fname == "file"
    assert files[0][1][1] == b"a"
    assert files[1][1][1] == b"b"
    assert files[2][1][1] == b"c"


def test_build_files_rejects_non_bytes_in_list():
    with pytest.raises(TypeError):
        AudioEmbeddingsClient._build_files(
            [b"a", "not-bytes"], filename="x", content_type="y",
        )


def test_build_files_rejects_unsupported_type():
    with pytest.raises(TypeError):
        AudioEmbeddingsClient._build_files(
            123, filename="x", content_type="y",
        )


def test_embed_returns_list_of_vectors_for_float_format():
    c = AudioEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = _envelope([[0.1, 0.2], [0.3, 0.4]])

    with patch("muse.modalities.audio_embedding.client.requests.post",
               return_value=fake_response) as post:
        out = c.embed([b"audio-a", b"audio-b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]
    args, kwargs = post.call_args
    assert args[0] == "http://srv/v1/audio/embeddings"
    # encoding_format default is "float"
    data = kwargs["data"]
    assert ("encoding_format", "float") in data
    # Two file parts
    files = kwargs["files"]
    assert len(files) == 2
    assert all(name == "file" for name, _ in files)


def test_embed_decodes_base64_responses():
    c = AudioEmbeddingsClient("http://srv")
    encoded = embedding_to_base64([0.5, 0.6, 0.7])
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": encoded, "index": 0}],
        "model": "mert-v1-95m",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
    with patch("muse.modalities.audio_embedding.client.requests.post",
               return_value=fake_response):
        out = c.embed(b"audio-bytes", encoding_format="base64")
    assert len(out) == 1
    for got, want in zip(out[0], [0.5, 0.6, 0.7]):
        assert got == pytest.approx(want, rel=1e-6)


def test_embed_passes_model_when_set():
    c = AudioEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = _envelope([[0.0]])
    with patch("muse.modalities.audio_embedding.client.requests.post",
               return_value=fake_response) as post:
        c.embed(b"audio", model="custom")
    data = post.call_args.kwargs["data"]
    assert ("model", "custom") in data


def test_embed_omits_model_when_none():
    c = AudioEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = _envelope([[0.0]])
    with patch("muse.modalities.audio_embedding.client.requests.post",
               return_value=fake_response) as post:
        c.embed(b"audio")
    data = post.call_args.kwargs["data"]
    keys = {k for k, _ in data}
    assert "model" not in keys


def test_embed_raises_on_non_200():
    c = AudioEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 500
    fake_response.text = "boom"
    with patch("muse.modalities.audio_embedding.client.requests.post",
               return_value=fake_response):
        with pytest.raises(RuntimeError, match="500"):
            c.embed(b"audio")


def test_embed_invalid_encoding_format_raises():
    c = AudioEmbeddingsClient("http://srv")
    with pytest.raises(ValueError):
        c.embed(b"audio", encoding_format="WRONG")


def test_embed_envelope_returns_full_openai_shape():
    c = AudioEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = _envelope([[0.1]])
    with patch("muse.modalities.audio_embedding.client.requests.post",
               return_value=fake_response):
        out = c.embed_envelope(b"audio")
    assert out["object"] == "list"
    assert "data" in out
    assert "model" in out
    assert "usage" in out


def test_embed_envelope_passes_model():
    c = AudioEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = _envelope([[0.1]])
    with patch("muse.modalities.audio_embedding.client.requests.post",
               return_value=fake_response) as post:
        c.embed_envelope(b"audio", model="my-model")
    data = post.call_args.kwargs["data"]
    assert ("model", "my-model") in data


def test_embed_envelope_invalid_encoding_format_raises():
    c = AudioEmbeddingsClient("http://srv")
    with pytest.raises(ValueError):
        c.embed_envelope(b"audio", encoding_format="WRONG")


def test_embed_envelope_raises_on_non_200():
    c = AudioEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 503
    fake_response.text = "service unavailable"
    with patch("muse.modalities.audio_embedding.client.requests.post",
               return_value=fake_response):
        with pytest.raises(RuntimeError, match="503"):
            c.embed_envelope(b"audio")


def test_embed_uses_request_timeout():
    c = AudioEmbeddingsClient("http://srv", timeout=11.5)
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = _envelope([[0.0]])
    with patch("muse.modalities.audio_embedding.client.requests.post",
               return_value=fake_response) as post:
        c.embed(b"audio")
    assert post.call_args.kwargs["timeout"] == 11.5


def test_embed_filename_and_content_type_overridable():
    c = AudioEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = _envelope([[0.0]])
    with patch("muse.modalities.audio_embedding.client.requests.post",
               return_value=fake_response) as post:
        c.embed(b"audio", filename="song.mp3", content_type="audio/mpeg")
    files = post.call_args.kwargs["files"]
    assert len(files) == 1
    name_tuple = files[0][1]
    assert name_tuple[0] == "song.mp3"
    assert name_tuple[2] == "audio/mpeg"


def test_client_exposed_on_modality_package_init():
    """The modality __init__ should re-export AudioEmbeddingsClient."""
    from muse.modalities import audio_embedding
    assert hasattr(audio_embedding, "AudioEmbeddingsClient")
    assert audio_embedding.AudioEmbeddingsClient is AudioEmbeddingsClient


def test_embed_batch_passes_all_files():
    c = AudioEmbeddingsClient("http://srv")
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = _envelope([[0.0], [0.1], [0.2]])
    with patch("muse.modalities.audio_embedding.client.requests.post",
               return_value=fake_response) as post:
        out = c.embed([b"a", b"b", b"c"])
    files = post.call_args.kwargs["files"]
    assert len(files) == 3
    assert len(out) == 3
