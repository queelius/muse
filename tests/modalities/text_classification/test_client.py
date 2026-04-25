"""Tests for ModerationsClient HTTP client."""
import json
from unittest.mock import MagicMock, patch


def _make_response(body: dict, status: int = 200):
    mock = MagicMock(
        status_code=status,
        headers={"content-type": "application/json"},
    )
    mock.json = MagicMock(return_value=body)
    mock.text = json.dumps(body)
    mock.raise_for_status = MagicMock()
    return mock


def test_default_server_url():
    from muse.modalities.text_classification import ModerationsClient
    c = ModerationsClient()
    assert c.server_url == "http://localhost:8000"


def test_trailing_slash_stripped():
    from muse.modalities.text_classification import ModerationsClient
    c = ModerationsClient(server_url="http://lan:8000/")
    assert c.server_url == "http://lan:8000"


def test_muse_server_env_fallback(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://custom:9999")
    from muse.modalities.text_classification import ModerationsClient
    c = ModerationsClient()
    assert c.server_url == "http://custom:9999"


def test_classify_scalar_returns_first_result():
    """Scalar input returns dict (the single results[0])."""
    body = {
        "id": "modr-1", "model": "text-moderation",
        "results": [{
            "flagged": True,
            "categories": {"H": True}, "category_scores": {"H": 0.9},
        }],
    }
    with patch("muse.modalities.text_classification.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.text_classification import ModerationsClient
        c = ModerationsClient()
        out = c.classify("I hate everything", model="text-moderation")
    assert isinstance(out, dict)
    assert out["flagged"] is True
    assert out["categories"]["H"] is True


def test_classify_list_returns_list_of_results():
    body = {
        "id": "modr-2", "model": "text-moderation",
        "results": [
            {"flagged": False, "categories": {}, "category_scores": {}},
            {"flagged": True, "categories": {}, "category_scores": {}},
        ],
    }
    with patch("muse.modalities.text_classification.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.text_classification import ModerationsClient
        c = ModerationsClient()
        out = c.classify(["a", "b"], model="text-moderation")
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[1]["flagged"] is True


def test_classify_threshold_forwarded():
    body = {"id": "x", "model": "m", "results": [{"flagged": False, "categories": {}, "category_scores": {}}]}
    with patch("muse.modalities.text_classification.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body)
        from muse.modalities.text_classification import ModerationsClient
        c = ModerationsClient()
        c.classify("x", model="m", threshold=0.7)
        sent = mock_post.call_args.kwargs["json"]
        assert sent["threshold"] == 0.7
        assert sent["input"] == "x"
        assert sent["model"] == "m"


def test_raise_for_status_invoked():
    """4xx propagates as requests.HTTPError."""
    import requests
    with patch("muse.modalities.text_classification.client.requests.post") as mock_post:
        resp = _make_response({"error": {"code": "model_not_found"}}, status=404)
        resp.raise_for_status = MagicMock(
            side_effect=requests.HTTPError("404 model_not_found"),
        )
        mock_post.return_value = resp
        from muse.modalities.text_classification import ModerationsClient
        c = ModerationsClient()
        import pytest
        with pytest.raises(requests.HTTPError):
            c.classify("x", model="no-such-model")
