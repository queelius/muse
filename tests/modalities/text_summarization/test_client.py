"""Tests for SummarizationClient (HTTP wrapper)."""
from unittest.mock import MagicMock, patch

from muse.modalities.text_summarization import SummarizationClient


def _fake_response(json_body):
    r = MagicMock()
    r.json.return_value = json_body
    r.raise_for_status.return_value = None
    return r


def test_default_server_url_uses_localhost():
    c = SummarizationClient()
    assert c.server_url == "http://localhost:8000"


def test_server_url_uses_env_when_set(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://gpu-host:8000")
    c = SummarizationClient()
    assert c.server_url == "http://gpu-host:8000"


def test_server_url_arg_beats_env(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://env:8000")
    c = SummarizationClient("http://arg:8000")
    assert c.server_url == "http://arg:8000"


def test_server_url_strips_trailing_slash():
    c = SummarizationClient("http://x:8000/")
    assert c.server_url == "http://x:8000"


def test_summarize_post_minimal_body():
    c = SummarizationClient("http://localhost:8000")
    body = {
        "id": "sum-x", "model": "m", "summary": "...",
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "meta": {"length": "medium", "format": "paragraph"},
    }
    with patch(
        "muse.modalities.text_summarization.client.requests.post",
        return_value=_fake_response(body),
    ) as mock_post:
        out = c.summarize(text="hello world")
    args, kwargs = mock_post.call_args
    assert args[0] == "http://localhost:8000/v1/summarize"
    assert kwargs["json"] == {"text": "hello world"}
    assert out == body


def test_summarize_includes_optional_fields():
    c = SummarizationClient("http://x:8000")
    with patch(
        "muse.modalities.text_summarization.client.requests.post",
        return_value=_fake_response({"summary": "x"}),
    ) as mock_post:
        c.summarize(
            text="hello",
            length="short",
            format="bullets",
            model="bart-large-cnn",
        )
    _, kwargs = mock_post.call_args
    sent = kwargs["json"]
    assert sent["text"] == "hello"
    assert sent["length"] == "short"
    assert sent["format"] == "bullets"
    assert sent["model"] == "bart-large-cnn"


def test_summarize_omits_none_fields():
    """None values for length/format/model should NOT appear in the body."""
    c = SummarizationClient("http://x:8000")
    with patch(
        "muse.modalities.text_summarization.client.requests.post",
        return_value=_fake_response({}),
    ) as mock_post:
        c.summarize(text="hello", length=None, format=None, model=None)
    _, kwargs = mock_post.call_args
    sent = kwargs["json"]
    assert "length" not in sent
    assert "format" not in sent
    assert "model" not in sent


def test_summarize_raises_on_http_error():
    c = SummarizationClient()
    failing = MagicMock()
    failing.raise_for_status.side_effect = RuntimeError("503")
    with patch(
        "muse.modalities.text_summarization.client.requests.post",
        return_value=failing,
    ):
        try:
            c.summarize(text="hello")
        except RuntimeError as e:
            assert "503" in str(e)
        else:
            raise AssertionError("expected RuntimeError")


def test_summarize_returns_response_json_unchanged():
    c = SummarizationClient()
    body = {
        "id": "sum-abc",
        "model": "bart-large-cnn",
        "summary": "Paris is the capital of France.",
        "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
        "meta": {"length": "medium", "format": "paragraph"},
    }
    with patch(
        "muse.modalities.text_summarization.client.requests.post",
        return_value=_fake_response(body),
    ):
        out = c.summarize(text="long text about Paris")
    assert out == body  # full envelope, unchanged


def test_summarize_uses_configured_timeout():
    c = SummarizationClient(timeout=42.0)
    with patch(
        "muse.modalities.text_summarization.client.requests.post",
        return_value=_fake_response({}),
    ) as mock_post:
        c.summarize(text="x")
    _, kwargs = mock_post.call_args
    assert kwargs["timeout"] == 42.0
