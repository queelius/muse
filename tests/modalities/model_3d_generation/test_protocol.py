"""Protocol + dataclass tests for 3d/generation."""
from muse.modalities.model_3d_generation.protocol import (
    Generation3DBackend,
    Generation3DResult,
)


def test_result_construction_default_format():
    r = Generation3DResult(glb_bytes=b"GLB-payload", model_id="triposr")
    assert r.glb_bytes == b"GLB-payload"
    assert r.model_id == "triposr"
    assert r.format == "glb"


def test_result_construction_explicit_format():
    r = Generation3DResult(
        glb_bytes=b"x", model_id="m", format="usdz",
    )
    assert r.format == "usdz"


def test_protocol_accepts_full_duck_type():
    """A class with both image_to_3d and text_to_3d satisfies the protocol."""
    class _Both:
        def image_to_3d(self, image_path, **kwargs):
            return [Generation3DResult(glb_bytes=b"x", model_id="m")]

        def text_to_3d(self, prompt, **kwargs):
            return [Generation3DResult(glb_bytes=b"y", model_id="m")]

    assert isinstance(_Both(), Generation3DBackend)


def test_protocol_rejects_missing_image_method():
    """A class with only text_to_3d does not satisfy the structural check."""
    class _TextOnly:
        def text_to_3d(self, prompt, **kwargs):
            return []

    assert not isinstance(_TextOnly(), Generation3DBackend)


def test_protocol_rejects_missing_text_method():
    """A class with only image_to_3d does not satisfy the structural check."""
    class _ImageOnly:
        def image_to_3d(self, image_path, **kwargs):
            return []

    assert not isinstance(_ImageOnly(), Generation3DBackend)


def test_protocol_rejects_unrelated_class():
    class _NoMethods:
        pass

    assert not isinstance(_NoMethods(), Generation3DBackend)
