"""Tests for TranscriptionClient HTTP client."""
from unittest.mock import MagicMock, patch


def _make_response(body: bytes | str, content_type: str, status: int = 200):
    mock = MagicMock(
        status_code=status,
        headers={"content-type": content_type},
    )
    if isinstance(body, bytes):
        mock.content = body
        mock.text = body.decode() if content_type.startswith("text/") or content_type == "application/json" else ""
    else:
        mock.text = body
        mock.content = body.encode()
    if content_type.startswith("application/json"):
        import json
        parsed = json.loads(body if isinstance(body, str) else body.decode())
        mock.json = MagicMock(return_value=parsed)
    else:
        mock.json = MagicMock(side_effect=ValueError("not json"))
    mock.raise_for_status = MagicMock()
    return mock


def test_default_server_url():
    from muse.modalities.audio_transcription import TranscriptionClient
    c = TranscriptionClient()
    assert c.server_url == "http://localhost:8000"


def test_trailing_slash_stripped():
    from muse.modalities.audio_transcription import TranscriptionClient
    c = TranscriptionClient(server_url="http://lan:8000/")
    assert c.server_url == "http://lan:8000"


def test_muse_server_env_fallback(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://custom:9999")
    from muse.modalities.audio_transcription import TranscriptionClient
    c = TranscriptionClient()
    assert c.server_url == "http://custom:9999"


def test_transcribe_default_json_returns_text_string():
    with patch("muse.modalities.audio_transcription.client.requests.post") as mock_post:
        mock_post.return_value = _make_response('{"text":"hello"}', "application/json")
        from muse.modalities.audio_transcription import TranscriptionClient
        c = TranscriptionClient()
        out = c.transcribe(audio=b"FAKEWAV", filename="a.wav", model="whisper-tiny")
        assert out == "hello"
        call = mock_post.call_args
        assert call.args[0].endswith("/v1/audio/transcriptions")
        # Multipart: files contains the wav, data contains the form fields
        files = call.kwargs["files"]
        data_kw = call.kwargs["data"]
        assert "file" in files
        assert files["file"][0] == "a.wav"
        assert ("model", "whisper-tiny") in list(data_kw) or data_kw.get("model") == "whisper-tiny"


def test_transcribe_text_format_returns_raw_string():
    with patch("muse.modalities.audio_transcription.client.requests.post") as mock_post:
        mock_post.return_value = _make_response("plain text transcript", "text/plain")
        from muse.modalities.audio_transcription import TranscriptionClient
        c = TranscriptionClient()
        out = c.transcribe(
            audio=b"x", filename="a.wav", model="whisper-tiny",
            response_format="text",
        )
        assert out == "plain text transcript"


def test_transcribe_srt_format_returns_raw_string():
    srt = "1\n00:00:00,000 --> 00:00:01,000\nhello\n"
    with patch("muse.modalities.audio_transcription.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(srt, "application/x-subrip")
        from muse.modalities.audio_transcription import TranscriptionClient
        c = TranscriptionClient()
        out = c.transcribe(
            audio=b"x", filename="a.wav", model="whisper-tiny",
            response_format="srt",
        )
        assert "hello" in out


def test_transcribe_verbose_json_returns_dict():
    body = '{"task":"transcribe","language":"en","duration":1.0,"text":"hello","segments":[]}'
    with patch("muse.modalities.audio_transcription.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(body, "application/json")
        from muse.modalities.audio_transcription import TranscriptionClient
        c = TranscriptionClient()
        out = c.transcribe(
            audio=b"x", filename="a.wav", model="whisper-tiny",
            response_format="verbose_json",
        )
        assert isinstance(out, dict)
        assert out["language"] == "en"


def test_translate_hits_translations_endpoint():
    with patch("muse.modalities.audio_transcription.client.requests.post") as mock_post:
        mock_post.return_value = _make_response('{"text":"hello"}', "application/json")
        from muse.modalities.audio_transcription import TranscriptionClient
        c = TranscriptionClient()
        c.translate(audio=b"x", filename="a.wav", model="whisper-tiny")
        assert mock_post.call_args.args[0].endswith("/v1/audio/translations")


def test_word_timestamps_sends_bracket_alias():
    """OpenAI SDK sends the alias timestamp_granularities[] with brackets."""
    with patch("muse.modalities.audio_transcription.client.requests.post") as mock_post:
        mock_post.return_value = _make_response(
            '{"task":"transcribe","language":"en","duration":1.0,"text":"hi","segments":[]}',
            "application/json",
        )
        from muse.modalities.audio_transcription import TranscriptionClient
        c = TranscriptionClient()
        c.transcribe(
            audio=b"x", filename="a.wav", model="whisper-tiny",
            response_format="verbose_json",
            word_timestamps=True,
        )
        data_kw = mock_post.call_args.kwargs["data"]
        # data must be a LIST of tuples to carry a bracketed alias
        # (dict keys can't repeat the bracket form cleanly).
        assert isinstance(data_kw, list)
        assert ("timestamp_granularities[]", "word") in data_kw


def test_raise_for_status_invoked():
    """HTTP errors propagate through raise_for_status."""
    with patch("muse.modalities.audio_transcription.client.requests.post") as mock_post:
        resp = _make_response('{"error":{"code":"model_not_found"}}', "application/json", status=404)
        import requests
        resp.raise_for_status = MagicMock(
            side_effect=requests.HTTPError("404 model_not_found")
        )
        mock_post.return_value = resp
        from muse.modalities.audio_transcription import TranscriptionClient
        c = TranscriptionClient()
        import pytest
        with pytest.raises(requests.HTTPError):
            c.transcribe(audio=b"x", filename="a.wav", model="no-such-model")
