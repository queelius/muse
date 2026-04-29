"""Integration tests for /v1/audio/embeddings against a live muse server.

Opt-in via MUSE_REMOTE_SERVER. Skipped when the server is unreachable
or no audio embedder is loaded.

Targets the configurable audio-embedding model id (default
mert-v1-95m, override via MUSE_AUDIO_EMBEDDING_MODEL_ID).
"""
from __future__ import annotations

import io
import wave

import pytest

from muse.modalities.audio_embedding import AudioEmbeddingsClient


def _sine_wave_wav(duration: float = 1.0, sr: int = 24000, freq: float = 440.0) -> bytes:
    """Generate a small mono sine wave WAV for live tests."""
    import numpy as np

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


def test_protocol_basic_audio_embedding(remote_url, audio_embedding_model):
    """Hard claim: an embed call returns the OpenAI envelope shape."""
    c = AudioEmbeddingsClient(remote_url)
    out = c.embed_envelope(_sine_wave_wav(), model=audio_embedding_model)
    assert out["object"] == "list"
    assert "data" in out
    assert len(out["data"]) == 1
    assert "model" in out
    assert "usage" in out


def test_protocol_data_entries_have_object_marker(remote_url, audio_embedding_model):
    """Every entry in `data` carries object='embedding'."""
    c = AudioEmbeddingsClient(remote_url)
    out = c.embed_envelope(_sine_wave_wav(), model=audio_embedding_model)
    for entry in out["data"]:
        assert entry["object"] == "embedding"


def test_protocol_indices_zero_based_and_ordered(remote_url, audio_embedding_model):
    c = AudioEmbeddingsClient(remote_url)
    out = c.embed_envelope(
        [_sine_wave_wav(), _sine_wave_wav(), _sine_wave_wav()],
        model=audio_embedding_model,
    )
    indices = [d["index"] for d in out["data"]]
    assert indices == [0, 1, 2]


def test_protocol_embed_returns_list_of_floats(remote_url, audio_embedding_model):
    c = AudioEmbeddingsClient(remote_url)
    out = c.embed(_sine_wave_wav(), model=audio_embedding_model)
    assert isinstance(out, list)
    assert len(out) == 1
    assert isinstance(out[0], list)
    assert all(isinstance(v, (int, float)) for v in out[0])


def test_protocol_base64_roundtrip(remote_url, audio_embedding_model):
    """encoding_format='base64' decodes to the same dim as 'float'."""
    c = AudioEmbeddingsClient(remote_url)
    via_float = c.embed(
        _sine_wave_wav(), model=audio_embedding_model, encoding_format="float",
    )
    via_b64 = c.embed(
        _sine_wave_wav(), model=audio_embedding_model, encoding_format="base64",
    )
    assert len(via_float) == len(via_b64) == 1
    assert len(via_float[0]) == len(via_b64[0])


def test_protocol_response_model_is_catalog_id(remote_url, audio_embedding_model):
    """The response 'model' field should be the catalog id, not the HF
    repo path."""
    c = AudioEmbeddingsClient(remote_url)
    out = c.embed_envelope(_sine_wave_wav(), model=audio_embedding_model)
    assert out["model"] == audio_embedding_model


def test_protocol_usage_is_zero_for_audio_inputs(remote_url, audio_embedding_model):
    """Audio embedding has no text tokenization; usage stays 0."""
    c = AudioEmbeddingsClient(remote_url)
    out = c.embed_envelope(_sine_wave_wav(), model=audio_embedding_model)
    assert out["usage"]["prompt_tokens"] == 0
    assert out["usage"]["total_tokens"] == 0


def test_observe_same_audio_yields_similar_embeddings(remote_url, audio_embedding_model):
    """Sanity: identical input twice produces identical (or near-identical)
    embeddings. Records what the model actually did rather than asserting
    exact equality, since some models have non-deterministic eval modes."""
    c = AudioEmbeddingsClient(remote_url)
    audio = _sine_wave_wav()
    a = c.embed(audio, model=audio_embedding_model)[0]
    b = c.embed(audio, model=audio_embedding_model)[0]
    assert len(a) == len(b)
    # Cosine similarity; identical inputs should be very close to 1.
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        pytest.skip("embedding has zero norm; cannot compute cosine similarity")
    cos = dot / (norm_a * norm_b)
    assert cos > 0.99, f"identical inputs gave cos={cos:.4f}"


def test_observe_batch_size_one_matches_single_input(remote_url, audio_embedding_model):
    """Sanity: batch [a] should produce one row equal to single input a."""
    c = AudioEmbeddingsClient(remote_url)
    audio = _sine_wave_wav(freq=523.25)  # C5
    single = c.embed(audio, model=audio_embedding_model)
    batch = c.embed([audio], model=audio_embedding_model)
    assert len(single) == 1
    assert len(batch) == 1
    dot = sum(x * y for x, y in zip(single[0], batch[0]))
    norm_a = sum(x * x for x in single[0]) ** 0.5
    norm_b = sum(y * y for y in batch[0]) ** 0.5
    if norm_a == 0 or norm_b == 0:
        pytest.skip("embedding has zero norm; cannot compute cosine similarity")
    cos = dot / (norm_a * norm_b)
    assert cos > 0.99


def test_protocol_default_model_resolves(remote_url, audio_embedding_model):
    """Omitting `model` should resolve to the default registered model."""
    c = AudioEmbeddingsClient(remote_url)
    out = c.embed_envelope(_sine_wave_wav())
    assert "model" in out
    assert out["object"] == "list"
