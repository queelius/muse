"""Tests for the image_animation protocol surface."""
from unittest.mock import MagicMock

from muse.modalities.image_animation.protocol import (
    AnimationModel,
    AnimationResult,
)


def test_animation_result_dataclass_shape():
    fake_frame = MagicMock()
    r = AnimationResult(
        frames=[fake_frame, fake_frame],
        fps=8,
        width=512, height=512,
        seed=42,
        metadata={"prompt": "x"},
    )
    assert len(r.frames) == 2
    assert r.fps == 8
    assert r.width == 512
    assert r.metadata["prompt"] == "x"


def test_animation_model_protocol_is_runtime_checkable():
    """Plain duck-typed class satisfies the Protocol structurally."""
    class FakeModel:
        model_id = "fake"
        def generate(self, prompt, **kwargs):
            return AnimationResult(frames=[], fps=8, width=0, height=0, seed=-1, metadata={})

    assert isinstance(FakeModel(), AnimationModel)
