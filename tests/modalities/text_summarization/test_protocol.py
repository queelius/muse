"""Protocol + dataclass shape tests for text/summarization."""
from muse.modalities.text_summarization import (
    MODALITY,
    SummarizationModel,
    SummarizationResult,
)


def test_modality_tag_is_mime_shaped():
    assert MODALITY == "text/summarization"


def test_summarization_result_minimal():
    r = SummarizationResult(
        summary="hello",
        length="short",
        format="paragraph",
        model_id="bart-large-cnn",
        prompt_tokens=10,
        completion_tokens=2,
    )
    assert r.summary == "hello"
    assert r.length == "short"
    assert r.format == "paragraph"
    assert r.model_id == "bart-large-cnn"
    assert r.prompt_tokens == 10
    assert r.completion_tokens == 2
    assert r.metadata == {}


def test_summarization_result_carries_metadata():
    r = SummarizationResult(
        summary="x",
        length="medium",
        format="paragraph",
        model_id="m",
        prompt_tokens=1,
        completion_tokens=1,
        metadata={"truncation_warning": True, "language": "en"},
    )
    assert r.metadata["truncation_warning"] is True
    assert r.metadata["language"] == "en"


def test_summarization_protocol_accepts_structural_impl():
    class Fake:
        def summarize(self, text, length="medium", format="paragraph"):
            return SummarizationResult(
                summary="x",
                length=length,
                format=format,
                model_id="m",
                prompt_tokens=0,
                completion_tokens=0,
            )

    assert isinstance(Fake(), SummarizationModel)


def test_summarization_protocol_rejects_missing_method():
    class Missing:
        pass

    assert not isinstance(Missing(), SummarizationModel)


def test_summarization_protocol_rejects_wrong_method_name():
    class WrongName:
        def summary(self, text, length="medium", format="paragraph"):
            return None

    assert not isinstance(WrongName(), SummarizationModel)
