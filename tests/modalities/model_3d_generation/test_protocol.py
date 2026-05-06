"""Protocol + dataclass tests for 3d/generation.

The protocol is split into two single-direction protocols
(`ImageTo3DBackend` and `TextTo3DBackend`) so backends like TripoSR
that implement only one direction pass their respective isinstance
check. The capability flag on the manifest gates which direction the
route layer accepts; the protocol mirrors that asymmetry instead of
forcing every backend to declare both methods.
"""
from muse.modalities.model_3d_generation.protocol import (
    Generation3DBackend,
    Generation3DResult,
    ImageTo3DBackend,
    TextTo3DBackend,
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


# ---------------- ImageTo3DBackend ----------------


def test_image_only_satisfies_image_protocol():
    """A class with only image_to_3d (TripoSR-shaped) passes its check."""
    class _ImageOnly:
        def image_to_3d(self, image_path, **kwargs):
            return [Generation3DResult(glb_bytes=b"x", model_id="m")]

    assert isinstance(_ImageOnly(), ImageTo3DBackend)


def test_image_only_fails_text_protocol():
    """The same image-only class is not a TextTo3DBackend."""
    class _ImageOnly:
        def image_to_3d(self, image_path, **kwargs):
            return []

    assert not isinstance(_ImageOnly(), TextTo3DBackend)


def test_image_protocol_rejects_unrelated_class():
    class _NoMethods:
        pass

    assert not isinstance(_NoMethods(), ImageTo3DBackend)


# ---------------- TextTo3DBackend ----------------


def test_text_only_satisfies_text_protocol():
    """A class with only text_to_3d (Shap-E-shaped) passes its check."""
    class _TextOnly:
        def text_to_3d(self, prompt, **kwargs):
            return [Generation3DResult(glb_bytes=b"y", model_id="m")]

    assert isinstance(_TextOnly(), TextTo3DBackend)


def test_text_only_fails_image_protocol():
    """The same text-only class is not an ImageTo3DBackend."""
    class _TextOnly:
        def text_to_3d(self, prompt, **kwargs):
            return []

    assert not isinstance(_TextOnly(), ImageTo3DBackend)


def test_text_protocol_rejects_unrelated_class():
    class _NoMethods:
        pass

    assert not isinstance(_NoMethods(), TextTo3DBackend)


# ---------------- both directions ----------------


def test_both_satisfies_both_protocols():
    """A class with both methods (TRELLIS-shaped) passes BOTH checks."""
    class _Both:
        def image_to_3d(self, image_path, **kwargs):
            return [Generation3DResult(glb_bytes=b"x", model_id="m")]

        def text_to_3d(self, prompt, **kwargs):
            return [Generation3DResult(glb_bytes=b"y", model_id="m")]

    obj = _Both()
    assert isinstance(obj, ImageTo3DBackend)
    assert isinstance(obj, TextTo3DBackend)


# ---------------- Generation3DBackend union alias ----------------


def test_generation3d_backend_is_a_union_alias():
    """Generation3DBackend is preserved as a union of the two protocols.

    Useful for callers that don't care which direction was used (e.g.,
    consumers of `Generation3DResult`).
    """
    # Either component of the union satisfies the alias for typing
    # purposes; we only verify the alias resolves to the components.
    from typing import get_args, get_origin
    import typing

    origin = get_origin(Generation3DBackend)
    # typing.Union or types.UnionType (3.10+ X|Y syntax)
    assert origin is typing.Union or origin is type(int | str)
    args = set(get_args(Generation3DBackend))
    assert ImageTo3DBackend in args
    assert TextTo3DBackend in args
