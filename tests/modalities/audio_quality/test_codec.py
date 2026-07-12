import numpy as np

from muse.modalities.audio_quality.codec import encode_audio_quality
from muse.modalities.audio_quality.protocol import (
    AudioQualityResult,
    AudioQualityScore,
)


def test_encode_audio_quality_envelope():
    result = AudioQualityResult(
        scores={
            "naturalness": AudioQualityScore(
                value=np.float32(4.25),
                minimum=1,
                maximum=5,
            ),
        },
        primary_score="naturalness",
        metadata={"family": "utmos"},
    )
    body = encode_audio_quality(result, model_id="utmos")
    assert body["id"].startswith("audio-quality-")
    assert body["object"] == "audio.quality"
    assert body["model"] == "utmos"
    assert body["primary_score"] == "naturalness"
    assert body["scores"]["naturalness"] == {
        "value": 4.25,
        "direction": "higher_is_better",
        "minimum": 1.0,
        "maximum": 5.0,
    }
    assert body["metadata"] == {"family": "utmos"}


def test_encode_omits_unknown_range_bounds():
    result = AudioQualityResult(
        scores={
            "raw": AudioQualityScore(
                value=0.3, direction="descriptive",
            ),
        },
        primary_score="raw",
    )
    row = encode_audio_quality(result, model_id="x")["scores"]["raw"]
    assert "minimum" not in row
    assert "maximum" not in row
    assert row["direction"] == "descriptive"
