"""Tests for the video/generation protocol surface."""
from __future__ import annotations

from PIL import Image

from muse.modalities.video_generation.protocol import (
    VideoGenerationModel,
    VideoResult,
)


def test_video_result_default_metadata():
    """metadata defaults to an empty dict (so callers can `.update(...)` safely)."""
    r = VideoResult(
        frames=[Image.new("RGB", (16, 16))],
        fps=8,
        width=16,
        height=16,
        duration_seconds=0.125,
        seed=42,
    )
    assert r.metadata == {}


def test_video_result_holds_required_fields():
    img = Image.new("RGB", (32, 32))
    r = VideoResult(
        frames=[img, img, img],
        fps=8,
        width=32,
        height=32,
        duration_seconds=0.375,
        seed=7,
        metadata={"prompt": "test"},
    )
    assert len(r.frames) == 3
    assert r.fps == 8
    assert r.width == 32
    assert r.height == 32
    assert r.duration_seconds == 0.375
    assert r.seed == 7
    assert r.metadata["prompt"] == "test"


class _FakeBackend:
    """Tiny stub that should satisfy VideoGenerationModel structurally."""
    model_id = "fake"

    def generate(self, prompt, **kwargs):
        return VideoResult(
            frames=[Image.new("RGB", (8, 8))],
            fps=8,
            width=8,
            height=8,
            duration_seconds=0.125,
            seed=-1,
        )


def test_protocol_runtime_check():
    """Duck-typed: a backend with model_id + generate satisfies the Protocol."""
    backend = _FakeBackend()
    assert isinstance(backend, VideoGenerationModel)


def test_protocol_rejects_missing_attrs():
    class Incomplete:
        # Missing model_id property AND generate method
        pass
    assert not isinstance(Incomplete(), VideoGenerationModel)
