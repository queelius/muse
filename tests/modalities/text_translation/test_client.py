"""Tests for TranslateClient (HTTP wrapper), mirroring the ChatClient/
SummarizationClient client test style but over httpx."""
from unittest.mock import MagicMock, patch

from muse.modalities.text_translation import TranslateClient


def _fake_response(json_body):
    r = MagicMock()
    r.json.return_value = json_body
    r.raise_for_status.return_value = None
    return r


def test_default_base_url_uses_localhost():
    c = TranslateClient()
    assert c.base_url == "http://localhost:8000"


def test_base_url_uses_env_when_set(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://gpu-host:8000")
    c = TranslateClient()
    assert c.base_url == "http://gpu-host:8000"


def test_base_url_arg_beats_env(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://env:8000")
    c = TranslateClient("http://arg:8000")
    assert c.base_url == "http://arg:8000"


def test_base_url_strips_trailing_slash():
    c = TranslateClient("http://x:8000/")
    assert c.base_url == "http://x:8000"


def test_default_timeout_is_120():
    c = TranslateClient()
    assert c.timeout == 120.0


def test_translate_scalar_returns_str():
    c = TranslateClient("http://localhost:8000")
    with patch(
        "muse.modalities.text_translation.client.httpx.post",
        return_value=_fake_response({"translatedText": "Hola mundo"}),
    ) as mock_post:
        out = c.translate("hello world", source="en", target="es")
    args, kwargs = mock_post.call_args
    assert args[0] == "http://localhost:8000/v1/translate"
    assert kwargs["json"] == {"q": "hello world", "source": "en", "target": "es"}
    assert out == "Hola mundo"


def test_translate_list_returns_list():
    c = TranslateClient("http://localhost:8000")
    with patch(
        "muse.modalities.text_translation.client.httpx.post",
        return_value=_fake_response({"translatedText": ["Hola", "mundo"]}),
    ):
        out = c.translate(["hello", "world"], source="en", target="es")
    assert out == ["Hola", "mundo"]


def test_translate_includes_model_when_given():
    c = TranslateClient("http://x:8000")
    with patch(
        "muse.modalities.text_translation.client.httpx.post",
        return_value=_fake_response({"translatedText": "x"}),
    ) as mock_post:
        c.translate("hello", source="en", target="es", model="m2m100-418m")
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == "m2m100-418m"


def test_translate_omits_model_when_none():
    c = TranslateClient("http://x:8000")
    with patch(
        "muse.modalities.text_translation.client.httpx.post",
        return_value=_fake_response({"translatedText": "x"}),
    ) as mock_post:
        c.translate("hello", source="en", target="es", model=None)
    _, kwargs = mock_post.call_args
    assert "model" not in kwargs["json"]


def test_translate_uses_configured_timeout():
    c = TranslateClient(timeout=42.0)
    with patch(
        "muse.modalities.text_translation.client.httpx.post",
        return_value=_fake_response({"translatedText": "x"}),
    ) as mock_post:
        c.translate("x", source="en", target="es")
    _, kwargs = mock_post.call_args
    assert kwargs["timeout"] == 42.0


def test_translate_raises_on_http_error():
    c = TranslateClient()
    failing = MagicMock()
    failing.raise_for_status.side_effect = RuntimeError("503")
    with patch(
        "muse.modalities.text_translation.client.httpx.post",
        return_value=failing,
    ):
        try:
            c.translate("hello", source="en", target="es")
        except RuntimeError as e:
            assert "503" in str(e)
        else:
            raise AssertionError("expected RuntimeError")


def test_languages_returns_list_of_dicts():
    c = TranslateClient("http://localhost:8000")
    payload = [{"code": "en", "name": "English", "targets": ["es"]}]
    with patch(
        "muse.modalities.text_translation.client.httpx.get",
        return_value=_fake_response(payload),
    ) as mock_get:
        out = c.languages()
    args, kwargs = mock_get.call_args
    assert args[0] == "http://localhost:8000/languages"
    assert out == payload


def test_languages_raises_on_http_error():
    c = TranslateClient()
    failing = MagicMock()
    failing.raise_for_status.side_effect = RuntimeError("503")
    with patch(
        "muse.modalities.text_translation.client.httpx.get",
        return_value=failing,
    ):
        try:
            c.languages()
        except RuntimeError as e:
            assert "503" in str(e)
        else:
            raise AssertionError("expected RuntimeError")
