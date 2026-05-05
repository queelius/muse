"""Codec tests for audio/classification."""
from muse.modalities.audio_classification.codec import (
    encode_audio_classifications,
)
from muse.modalities.audio_classification.protocol import (
    AudioClassificationResult,
)


def test_envelope_shape_and_sort():
    results = [AudioClassificationResult(
        scores={"speech": 0.7, "music": 0.2, "silence": 0.1},
        multi_label=False,
    )]
    body = encode_audio_classifications(results, model_id="ast-test")
    assert body["model"] == "ast-test"
    assert body["id"].startswith("audio-cls-")
    assert len(body["results"]) == 1
    pairs = body["results"][0]
    # Sorted by score desc.
    assert [p["label"] for p in pairs] == ["speech", "music", "silence"]
    assert pairs[0]["score"] == 0.7


def test_top_k_truncates():
    results = [AudioClassificationResult(
        scores={f"l{i}": i * 0.1 for i in range(10)},
        multi_label=False,
    )]
    body = encode_audio_classifications(results, model_id="x", top_k=3)
    assert len(body["results"][0]) == 3


def test_top_k_none_keeps_all():
    results = [AudioClassificationResult(
        scores={f"l{i}": 0.1 for i in range(7)}, multi_label=True,
    )]
    body = encode_audio_classifications(results, model_id="x", top_k=None)
    assert len(body["results"][0]) == 7


def test_id_unique_per_call():
    results = [AudioClassificationResult(scores={"a": 1.0}, multi_label=False)]
    a = encode_audio_classifications(results, model_id="x")
    b = encode_audio_classifications(results, model_id="x")
    assert a["id"] != b["id"]


def test_score_float_coercion():
    """numpy / torch scalar scores get coerced to Python float."""
    import numpy as np
    results = [AudioClassificationResult(
        scores={"a": np.float32(0.9)}, multi_label=False,
    )]
    body = encode_audio_classifications(results, model_id="x")
    assert isinstance(body["results"][0][0]["score"], float)
