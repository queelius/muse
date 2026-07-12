import numpy as np
import pytest

from muse.modalities.audio_alignment.codec import encode_audio_alignment
from muse.modalities.audio_alignment.protocol import (
    AlignmentWord,
    AudioAlignmentResult,
)


def test_encode_audio_alignment_envelope():
    result = AudioAlignmentResult(
        text="Hello world",
        language="English",
        duration_seconds=np.float32(1.25),
        words=[
            AlignmentWord("Hello", 0.0, 0.4, np.float32(0.9)),
            AlignmentWord("world", 0.5, 1.1),
        ],
        metadata={"family": "qwen3-forced-aligner"},
    )
    body = encode_audio_alignment(result, model_id="qwen-aligner")
    assert body["id"].startswith("audio-alignment-")
    assert body["object"] == "audio.alignment"
    assert body["model"] == "qwen-aligner"
    assert body["text"] == "Hello world"
    assert body["language"] == "English"
    assert body["duration_seconds"] == 1.25
    assert body["words"][0] == {
        "word": "Hello", "start": 0.0, "end": 0.4,
        "confidence": pytest.approx(0.9),
    }
    assert body["words"][1] == {
        "word": "world", "start": 0.5, "end": 1.1,
    }
