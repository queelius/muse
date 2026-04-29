"""Tests for image/segmentation Protocol + dataclasses."""
from muse.modalities.image_segmentation.protocol import (
    ImageSegmentationModel,
    MaskRecord,
    SegmentationResult,
)


def test_mask_record_required_fields():
    m = MaskRecord(
        mask=object(), score=0.9, bbox=(10, 20, 30, 40), area=900,
    )
    assert m.score == 0.9
    assert m.bbox == (10, 20, 30, 40)
    assert m.area == 900


def test_segmentation_result_required_fields():
    r = SegmentationResult(
        masks=[], image_size=(1024, 768), mode="auto", seed=-1,
    )
    assert r.image_size == (1024, 768)
    assert r.mode == "auto"
    assert r.seed == -1
    assert r.masks == []
    assert r.metadata == {}


def test_segmentation_result_metadata_default_factory():
    """Default factory must produce a fresh dict per instance."""
    r1 = SegmentationResult(
        masks=[], image_size=(1, 1), mode="auto", seed=0,
    )
    r2 = SegmentationResult(
        masks=[], image_size=(1, 1), mode="auto", seed=0,
    )
    r1.metadata["x"] = 1
    assert r2.metadata == {}


def test_segmentation_result_metadata_can_be_overridden():
    r = SegmentationResult(
        masks=[], image_size=(1, 1), mode="auto", seed=0,
        metadata={"prompt": "hi"},
    )
    assert r.metadata == {"prompt": "hi"}


def test_image_segmentation_model_protocol_is_runtime_checkable():
    """A duck-typed object satisfying the structural shape passes."""

    class Stub:
        model_id = "x"

        def segment(self, image, **kw):
            return None

    assert isinstance(Stub(), ImageSegmentationModel)


def test_image_segmentation_model_protocol_rejects_missing_method():
    """An object without a `segment` method must NOT satisfy the protocol."""

    class NoSegment:
        model_id = "x"

    assert not isinstance(NoSegment(), ImageSegmentationModel)
