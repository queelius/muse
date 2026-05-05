"""Protocol + dataclass tests."""
from muse.modalities.audio_classification.protocol import (
    AudioClassificationResult,
    AudioClassifierModel,
)


def test_result_construction():
    r = AudioClassificationResult(
        scores={"speech": 0.9, "music": 0.1}, multi_label=False,
    )
    assert r.multi_label is False
    assert r.scores["speech"] == 0.9


def test_protocol_accepts_duck_type():
    class _Fake:
        def classify(self, audio_path):
            return [AudioClassificationResult(scores={"x": 1.0}, multi_label=False)]
    assert isinstance(_Fake(), AudioClassifierModel)


def test_protocol_rejects_missing_method():
    class _NoClassify:
        pass
    assert not isinstance(_NoClassify(), AudioClassifierModel)
