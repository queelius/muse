"""Tests for RerankClient (HTTP wrapper)."""
from unittest.mock import MagicMock, patch

from muse.modalities.text_rerank import RerankClient


def _fake_response(json_body):
    r = MagicMock()
    r.json.return_value = json_body
    r.raise_for_status.return_value = None
    return r


def test_default_server_url_uses_localhost(monkeypatch):
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    c = RerankClient()
    assert c.server_url == "http://localhost:8000"


def test_server_url_uses_env_when_set(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://gpu-host:8000")
    c = RerankClient()
    assert c.server_url == "http://gpu-host:8000"


def test_server_url_arg_beats_env(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://env:8000")
    c = RerankClient("http://arg:8000")
    assert c.server_url == "http://arg:8000"


def test_rerank_post_minimal_body():
    c = RerankClient("http://localhost:8000")
    body = {
        "id": "rrk-x", "model": "m", "results": [], "meta": {},
    }
    with patch("muse.modalities.text_rerank.client.requests.post",
               return_value=_fake_response(body)) as mock_post:
        out = c.rerank(query="q", documents=["a", "b"])
    args, kwargs = mock_post.call_args
    assert args[0] == "http://localhost:8000/v1/rerank"
    assert kwargs["json"] == {"query": "q", "documents": ["a", "b"]}
    assert out == body


def test_rerank_includes_optional_fields():
    c = RerankClient("http://x:8000")
    with patch("muse.modalities.text_rerank.client.requests.post",
               return_value=_fake_response({"results": []})) as mock_post:
        c.rerank(
            query="q",
            documents=["a"],
            top_n=5,
            model="bge-reranker-v2-m3",
            return_documents=True,
        )
    _, kwargs = mock_post.call_args
    sent = kwargs["json"]
    assert sent["top_n"] == 5
    assert sent["model"] == "bge-reranker-v2-m3"
    assert sent["return_documents"] is True


def test_rerank_raises_on_http_error():
    c = RerankClient()
    failing = MagicMock()
    failing.raise_for_status.side_effect = RuntimeError("503")
    with patch("muse.modalities.text_rerank.client.requests.post",
               return_value=failing):
        try:
            c.rerank(query="q", documents=["a"])
        except RuntimeError as e:
            assert "503" in str(e)
        else:
            raise AssertionError("expected RuntimeError")
