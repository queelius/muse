"""Integration tests for /v1/summarize against a live muse server.

Opt-in via MUSE_REMOTE_SERVER. Skipped when the server is unreachable
or no summarization model is loaded.

Targets the configurable summarization model id (default
bart-large-cnn, override via MUSE_SUMMARIZATION_MODEL_ID).
"""
from __future__ import annotations

from muse.modalities.text_summarization import SummarizationClient


_SAMPLE_TEXT = (
    "muse is a model-agnostic multi-modality generation server. It hosts text, "
    "image, audio, and video models behind a unified HTTP API that mirrors OpenAI "
    "where possible. Each modality is a self-contained plugin: it declares its MIME "
    "tag, contributes a build_router function, and the discovery layer wires it in. "
    "Models are pulled into per-model venvs so that conflicting dependencies between "
    "different model families never break each other. The supervisor process runs "
    "one worker subprocess per venv group and proxies requests by the model field."
)


def test_protocol_basic_summarize(remote_url, summarization_model):
    """Hard claim: a summarize call returns the Cohere envelope shape."""
    client = SummarizationClient(remote_url)
    out = client.summarize(text=_SAMPLE_TEXT, model=summarization_model)
    assert "id" in out
    assert out["id"].startswith("sum-")
    assert "summary" in out
    assert isinstance(out["summary"], str)
    assert len(out["summary"]) > 0
    assert "usage" in out
    assert "meta" in out


def test_protocol_usage_fields_present(remote_url, summarization_model):
    """Cohere SDK compatibility: usage must roll up prompt/completion/total."""
    client = SummarizationClient(remote_url)
    out = client.summarize(text=_SAMPLE_TEXT, model=summarization_model)
    usage = out["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage
    assert (
        usage["total_tokens"]
        == usage["prompt_tokens"] + usage["completion_tokens"]
    )


def test_protocol_meta_echoes_length_and_format(remote_url, summarization_model):
    client = SummarizationClient(remote_url)
    out = client.summarize(
        text=_SAMPLE_TEXT,
        length="short",
        format="paragraph",
        model=summarization_model,
    )
    assert out["meta"]["length"] == "short"
    assert out["meta"]["format"] == "paragraph"


def test_protocol_default_length_medium(remote_url, summarization_model):
    client = SummarizationClient(remote_url)
    out = client.summarize(text=_SAMPLE_TEXT, model=summarization_model)
    assert out["meta"]["length"] == "medium"


def test_protocol_default_format_paragraph(remote_url, summarization_model):
    client = SummarizationClient(remote_url)
    out = client.summarize(text=_SAMPLE_TEXT, model=summarization_model)
    assert out["meta"]["format"] == "paragraph"


def test_observe_short_summary_is_shorter_than_long(remote_url, summarization_model):
    """Spot-check: 'long' length should produce more output tokens than 'short'.

    Records what the model actually did. Useful as a watchdog rather
    than a hard claim because exact token counts depend on the model's
    decoding decisions, but short=80 vs long=400 should leave a clear
    signal in the relative completion_tokens.
    """
    client = SummarizationClient(remote_url)
    short = client.summarize(
        text=_SAMPLE_TEXT, length="short", model=summarization_model,
    )
    long_ = client.summarize(
        text=_SAMPLE_TEXT, length="long", model=summarization_model,
    )
    # Long completion_tokens budget is 400, short is 80. Real model
    # output may not always saturate the budget but should generally
    # have long >= short (allowing for rare edge cases where the model
    # naturally stops early on both lengths).
    assert long_["usage"]["completion_tokens"] >= short["usage"]["completion_tokens"]


def test_observe_summary_contains_recognizable_content(remote_url, summarization_model):
    """Sanity: the summary should mention something from the input.

    Loose check: at least one of "muse", "modality", "model", or "server"
    should appear in the summary (case-insensitive). Records what BART
    actually does without claiming a specific output shape.
    """
    client = SummarizationClient(remote_url)
    out = client.summarize(text=_SAMPLE_TEXT, model=summarization_model)
    summary_lower = out["summary"].lower()
    assert any(
        keyword in summary_lower
        for keyword in ("muse", "modality", "model", "server")
    ), f"summary did not mention any expected keyword: {out['summary']!r}"


def test_protocol_response_model_is_catalog_id(remote_url, summarization_model):
    """The response 'model' field should be the catalog id, not the HF
    repo path. Important for canonical id reporting."""
    client = SummarizationClient(remote_url)
    out = client.summarize(text=_SAMPLE_TEXT, model=summarization_model)
    assert out["model"] == summarization_model
