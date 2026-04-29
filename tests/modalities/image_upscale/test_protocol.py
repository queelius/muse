"""Tests for image/upscale Protocol + UpscaleResult dataclass."""
from muse.modalities.image_upscale.protocol import (
    ImageUpscaleModel,
    UpscaleResult,
)


def test_upscale_result_required_fields():
    r = UpscaleResult(
        image=object(),
        original_width=128, original_height=128,
        upscaled_width=512, upscaled_height=512,
        scale=4, seed=-1,
    )
    assert r.scale == 4
    assert r.original_width == 128
    assert r.upscaled_width == 512
    assert r.metadata == {}


def test_upscale_result_metadata_default_factory():
    """Default factory must produce a fresh dict per instance."""
    r1 = UpscaleResult(
        image=None, original_width=1, original_height=1,
        upscaled_width=2, upscaled_height=2, scale=2, seed=0,
    )
    r2 = UpscaleResult(
        image=None, original_width=1, original_height=1,
        upscaled_width=2, upscaled_height=2, scale=2, seed=0,
    )
    r1.metadata["x"] = 1
    assert r2.metadata == {}


def test_upscale_result_metadata_can_be_overridden():
    r = UpscaleResult(
        image=None, original_width=1, original_height=1,
        upscaled_width=2, upscaled_height=2, scale=2, seed=0,
        metadata={"prompt": "hi"},
    )
    assert r.metadata == {"prompt": "hi"}


def test_image_upscale_model_protocol_is_runtime_checkable():
    """A duck-typed object satisfying the structural shape passes."""

    class Stub:
        model_id = "x"
        supported_scales = [4]

        def upscale(self, image, **kw):
            return None

    assert isinstance(Stub(), ImageUpscaleModel)


def test_image_upscale_model_protocol_rejects_missing_method():
    """An object without an `upscale` method must NOT satisfy the protocol."""

    class NoUpscale:
        model_id = "x"
        supported_scales = [4]

    assert not isinstance(NoUpscale(), ImageUpscaleModel)
