"""Protocol + dataclass shape tests for audio/generation."""
import numpy as np

from muse.modalities.audio_generation import (
    MODALITY,
    AudioGenerationModel,
    AudioGenerationResult,
)


def test_modality_tag_is_mime_shaped():
    assert MODALITY == "audio/generation"


def test_audio_generation_result_minimal():
    audio = np.zeros(44100, dtype=np.float32)
    r = AudioGenerationResult(
        audio=audio,
        sample_rate=44100,
        channels=1,
        duration_seconds=1.0,
    )
    assert r.audio is audio
    assert r.sample_rate == 44100
    assert r.channels == 1
    assert r.duration_seconds == 1.0
    assert r.metadata == {}


def test_audio_generation_result_metadata_default_factory():
    """Each instance gets its own metadata dict (default_factory, not shared)."""
    a = AudioGenerationResult(
        audio=np.zeros(10, dtype=np.float32),
        sample_rate=44100,
        channels=1,
        duration_seconds=0.0,
    )
    b = AudioGenerationResult(
        audio=np.zeros(10, dtype=np.float32),
        sample_rate=44100,
        channels=1,
        duration_seconds=0.0,
    )
    a.metadata["x"] = 1
    assert b.metadata == {}


def test_audio_generation_result_supports_stereo():
    """channels=2 + (samples, 2) shape is a valid combo."""
    audio = np.zeros((44100, 2), dtype=np.float32)
    r = AudioGenerationResult(
        audio=audio, sample_rate=44100, channels=2, duration_seconds=1.0,
    )
    assert r.channels == 2
    assert r.audio.shape == (44100, 2)


def test_audio_generation_protocol_accepts_structural_impl():
    """A class that defines model_id property + generate method passes."""
    class Fake:
        @property
        def model_id(self):
            return "fake"
        def generate(self, prompt, *, duration=None, seed=None,
                     steps=None, guidance=None, negative_prompt=None,
                     **kwargs):
            return AudioGenerationResult(
                audio=np.zeros(10, dtype=np.float32),
                sample_rate=44100, channels=1, duration_seconds=0.0,
            )
    assert isinstance(Fake(), AudioGenerationModel)


def test_audio_generation_protocol_rejects_missing_method():
    class Missing:
        pass
    assert not isinstance(Missing(), AudioGenerationModel)


def test_audio_generation_protocol_rejects_missing_generate():
    """A class with model_id but no generate() must not match."""
    class HalfBaked:
        @property
        def model_id(self):
            return "x"
    assert not isinstance(HalfBaked(), AudioGenerationModel)
