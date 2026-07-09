"""Integration tests for /v1/translate + /languages against a live muse
server.

Opt-in via MUSE_REMOTE_SERVER. Skipped when the server is unreachable or
no translation model is loaded.

Targets the configurable translation model id (default m2m100-418m,
override via MUSE_TRANSLATION_MODEL_ID).
"""
from __future__ import annotations

from muse.modalities.text_translation import TranslateClient


def test_protocol_translate_en_es_roundtrip(remote_url, translation_model):
    """Hard claim: an en->es translation returns nonempty text that
    differs from the input (a real translation happened, not an echo)."""
    client = TranslateClient(remote_url)
    source_text = "The weather is nice today."
    out = client.translate(
        source_text, source="en", target="es", model=translation_model,
    )
    assert isinstance(out, str)
    assert len(out) > 0
    assert out != source_text


def test_protocol_translate_batch_list_roundtrip(remote_url, translation_model):
    """List q -> list translatedText, same length, same order."""
    client = TranslateClient(remote_url)
    inputs = ["Good morning.", "How are you?"]
    out = client.translate(inputs, source="en", target="es", model=translation_model)
    assert isinstance(out, list)
    assert len(out) == len(inputs)
    for original, translated in zip(inputs, out):
        assert isinstance(translated, str)
        assert len(translated) > 0
        assert translated != original


def test_protocol_translate_identity_pair_allowed(remote_url, translation_model):
    """source == target is a valid LT request (identity round-trip)."""
    client = TranslateClient(remote_url)
    out = client.translate(
        "hello", source="en", target="en", model=translation_model,
    )
    assert isinstance(out, str)
    assert len(out) > 0


def test_protocol_translate_invalid_language_400(remote_url, translation_model):
    """An unsupported language code returns 400 invalid_language."""
    import httpx

    r = httpx.post(
        f"{remote_url}/v1/translate",
        json={
            "q": "hello", "source": "en", "target": "zz-not-a-real-code",
            "model": translation_model,
        },
        timeout=60.0,
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_language"


def test_protocol_languages_smoke(remote_url, translation_model):
    """GET /languages returns a nonempty LibreTranslate-shape list with
    English present and each row carrying the expected fields."""
    client = TranslateClient(remote_url)
    langs = client.languages()
    assert isinstance(langs, list)
    assert len(langs) > 0
    codes = {row["code"] for row in langs}
    assert "en" in codes
    for row in langs:
        assert "code" in row
        assert "name" in row
        assert "targets" in row
        assert isinstance(row["targets"], list)
