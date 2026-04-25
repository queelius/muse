"""Protocol + dataclass shape tests for text/classification."""
from muse.modalities.text_classification import (
    MODALITY,
    ClassificationResult,
    TextClassifierModel,
)


def test_modality_tag_is_mime_shaped():
    assert MODALITY == "text/classification"


def test_classification_result_minimal():
    r = ClassificationResult(
        scores={"H": 0.8, "V": 0.1, "OK": 0.1},
        multi_label=False,
    )
    assert r.scores["H"] == 0.8
    assert r.multi_label is False


def test_classification_result_multi_label():
    r = ClassificationResult(
        scores={"toxic": 0.9, "obscene": 0.7, "threat": 0.05},
        multi_label=True,
    )
    assert r.multi_label is True
    assert len(r.scores) == 3


def test_text_classifier_protocol_accepts_structural_impl():
    """A class that implements `classify(...)` satisfies the protocol."""
    class Fake:
        def classify(self, input):
            return [ClassificationResult(scores={"OK": 1.0}, multi_label=False)]
    assert isinstance(Fake(), TextClassifierModel)


def test_text_classifier_protocol_rejects_missing_method():
    class Missing:
        pass
    assert not isinstance(Missing(), TextClassifierModel)
