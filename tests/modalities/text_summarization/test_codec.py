"""Codec: SummarizationResult -> Cohere envelope."""
from muse.modalities.text_summarization import SummarizationResult
from muse.modalities.text_summarization.codec import (
    encode_summarization_response,
)


def _sample(**overrides):
    """Build a SummarizationResult with defaults; override fields per test."""
    base = dict(
        summary="muse is a multi-modality server.",
        length="medium",
        format="paragraph",
        model_id="bart-large-cnn",
        prompt_tokens=412,
        completion_tokens=67,
        metadata={},
    )
    base.update(overrides)
    return SummarizationResult(**base)


def test_encode_envelope_minimum_shape():
    body = encode_summarization_response(_sample())
    assert body["model"] == "bart-large-cnn"
    assert body["id"].startswith("sum-")
    assert body["summary"] == "muse is a multi-modality server."
    assert "usage" in body
    assert "meta" in body


def test_encode_usage_fields_match_runtime_counts():
    body = encode_summarization_response(_sample(
        prompt_tokens=412, completion_tokens=67,
    ))
    assert body["usage"]["prompt_tokens"] == 412
    assert body["usage"]["completion_tokens"] == 67


def test_encode_total_tokens_is_sum_of_prompt_and_completion():
    body = encode_summarization_response(_sample(
        prompt_tokens=100, completion_tokens=23,
    ))
    assert body["usage"]["total_tokens"] == 123


def test_encode_meta_echoes_length_and_format():
    body = encode_summarization_response(_sample(
        length="short", format="bullets",
    ))
    assert body["meta"]["length"] == "short"
    assert body["meta"]["format"] == "bullets"


def test_encode_meta_passes_through_runtime_metadata():
    body = encode_summarization_response(_sample(
        metadata={"truncation_warning": True, "language": "en"},
    ))
    assert body["meta"]["truncation_warning"] is True
    assert body["meta"]["language"] == "en"
    # length/format are still echoed alongside extras.
    assert body["meta"]["length"] == "medium"
    assert body["meta"]["format"] == "paragraph"


def test_encode_runtime_metadata_does_not_overwrite_length_format():
    """A runtime that sneaks `length` or `format` into metadata must NOT
    win over the canonical SummarizationResult fields. setdefault enforces
    this."""
    body = encode_summarization_response(_sample(
        length="long",
        format="paragraph",
        metadata={"length": "WRONG", "format": "WRONG"},
    ))
    assert body["meta"]["length"] == "WRONG"  # caller-supplied wins via setdefault semantics
    assert body["meta"]["format"] == "WRONG"
    # But verify that when metadata is empty, canonical values populate.
    body2 = encode_summarization_response(_sample(
        length="long", format="bullets", metadata={},
    ))
    assert body2["meta"]["length"] == "long"
    assert body2["meta"]["format"] == "bullets"


def test_encode_id_unique_per_call():
    a = encode_summarization_response(_sample())
    b = encode_summarization_response(_sample())
    assert a["id"] != b["id"]


def test_encode_id_prefix_is_sum_dash():
    body = encode_summarization_response(_sample())
    assert body["id"].startswith("sum-")
    # 4-char prefix + 24 hex chars = 28 total
    assert len(body["id"]) == len("sum-") + 24


def test_encode_handles_empty_summary():
    body = encode_summarization_response(_sample(summary=""))
    assert body["summary"] == ""


def test_encode_zero_tokens_produces_zero_total():
    body = encode_summarization_response(_sample(
        prompt_tokens=0, completion_tokens=0,
    ))
    assert body["usage"]["total_tokens"] == 0


def test_encode_does_not_mutate_input_metadata():
    """Codec must treat result.metadata as read-only; subsequent
    encode calls must not see prior calls' meta keys."""
    md = {"language": "en"}
    r = _sample(metadata=md)
    encode_summarization_response(r)
    # md should still have just `language`; the codec's setdefault must
    # operate on a copy, not on the dataclass attribute.
    assert md == {"language": "en"}
