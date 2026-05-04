"""Tests for the OcrResult dataclass and OcrModel Protocol."""
from muse.modalities.image_ocr.protocol import OcrModel, OcrResult


def test_result_default_metadata_is_empty_dict():
    """field(default_factory=dict) avoids the mutable-default trap."""
    r = OcrResult(text="x", model_id="m")
    assert r.metadata == {}
    assert r.completion_tokens == 0


def test_result_metadata_independent_per_instance():
    """Mutating one OcrResult's metadata must not affect a sibling."""
    a = OcrResult(text="a", model_id="m")
    a.metadata["k"] = 1
    b = OcrResult(text="b", model_id="m")
    assert b.metadata == {}


def test_result_explicit_completion_tokens():
    r = OcrResult(text="hello", model_id="m", completion_tokens=12)
    assert r.completion_tokens == 12


def test_ocr_model_protocol_runtime_checkable():
    """A duck-typed class with the right ocr() signature satisfies the
    Protocol without inheriting from it."""
    class _Fake:
        def ocr(self, image, *, prompt=None, max_new_tokens=512, num_beams=1):
            return OcrResult(text="x", model_id="fake")

    assert isinstance(_Fake(), OcrModel)


def test_ocr_model_protocol_rejects_missing_method():
    class _NoOcr:
        def something_else(self):
            pass

    assert not isinstance(_NoOcr(), OcrModel)
