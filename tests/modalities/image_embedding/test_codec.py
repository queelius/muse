"""Tests for the image/embedding codec.

The codec is a thin re-export of embedding_text's base64 helpers; we
test that the float<->base64 roundtrip works end-to-end and that the
byte layout is bit-identical (little-endian float32) so OpenAI SDK
clients consuming muse's responses see what they'd see from OpenAI's
own /v1/embeddings.
"""
import base64
import struct

import pytest

from muse.modalities.image_embedding.codec import (
    base64_to_embedding,
    embedding_to_base64,
)


def test_empty_embedding_encodes_to_empty_string():
    assert embedding_to_base64([]) == ""


def test_empty_string_decodes_to_empty_list():
    assert base64_to_embedding("") == []


def test_float_to_base64_roundtrip():
    original = [0.1, 0.2, 0.3, -1.5, 42.0]
    encoded = embedding_to_base64(original)
    decoded = base64_to_embedding(encoded)
    for orig, dec in zip(original, decoded):
        assert orig == pytest.approx(dec, rel=1e-6)


def test_byte_layout_is_little_endian_float32():
    """Bytes must match struct's '<f' (little-endian float32) packing."""
    raw = embedding_to_base64([1.0, 2.0])
    decoded_bytes = base64.b64decode(raw)
    expected = struct.pack("<ff", 1.0, 2.0)
    assert decoded_bytes == expected


def test_decoded_byte_length_must_be_multiple_of_4():
    """Corrupt input (non-4-aligned) raises ValueError."""
    # 5 bytes is not a multiple of 4.
    bogus = base64.b64encode(b"hello").decode("ascii")
    with pytest.raises(ValueError, match="multiple of 4"):
        base64_to_embedding(bogus)


def test_long_vector_roundtrip_preserves_dimension():
    """A 768-dim vector (CLIP base) survives encode/decode unchanged."""
    original = [float(i) * 0.001 for i in range(768)]
    encoded = embedding_to_base64(original)
    decoded = base64_to_embedding(encoded)
    assert len(decoded) == 768
    for orig, dec in zip(original, decoded):
        assert orig == pytest.approx(dec, rel=1e-6)


def test_base64_string_does_not_contain_newlines():
    """Some base64 implementations chunk; ours must not (single string)."""
    encoded = embedding_to_base64([float(i) for i in range(100)])
    assert "\n" not in encoded


def test_codec_module_reexports_helpers():
    """Verify the module surface matches the documented contract."""
    from muse.modalities.image_embedding import codec
    assert hasattr(codec, "embedding_to_base64")
    assert hasattr(codec, "base64_to_embedding")
