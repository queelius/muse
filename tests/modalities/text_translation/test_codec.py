"""Codec tests: scalar/list symmetry, q normalization, languages payload."""
import pytest

from muse.modalities.text_translation.codec import (
    languages_payload,
    normalize_q,
    shape_response,
)


def test_normalize_q_scalar_string():
    assert normalize_q("hi") == (["hi"], True)


def test_normalize_q_list_of_strings():
    assert normalize_q(["hi", "there"]) == (["hi", "there"], False)


def test_normalize_q_empty_string():
    assert normalize_q("") == ([""], True)


def test_normalize_q_empty_list():
    assert normalize_q([]) == ([], False)


def test_normalize_q_raises_on_non_str_non_list():
    with pytest.raises(ValueError):
        normalize_q(5)


def test_normalize_q_raises_on_none():
    with pytest.raises(ValueError):
        normalize_q(None)


def test_normalize_q_raises_on_list_with_non_str_item():
    with pytest.raises(ValueError):
        normalize_q(["a", 3])


def test_shape_response_scalar():
    assert shape_response(["Hola"], scalar=True) == {"translatedText": "Hola"}


def test_shape_response_list():
    assert shape_response(["Hola", "mundo"], scalar=False) == {
        "translatedText": ["Hola", "mundo"]
    }


def test_shape_response_round_trip_scalar():
    texts, scalar = normalize_q("hi")
    translated = [t.upper() for t in texts]
    assert shape_response(translated, scalar=scalar) == {"translatedText": "HI"}


def test_shape_response_round_trip_list():
    texts, scalar = normalize_q(["hi", "there"])
    translated = [t.upper() for t in texts]
    assert shape_response(translated, scalar=scalar) == {
        "translatedText": ["HI", "THERE"]
    }


def test_languages_payload_basic_shape():
    assert languages_payload({"en": ["es"]}) == [
        {"code": "en", "name": "English", "targets": ["es"]}
    ]


def test_languages_payload_sorted_by_code():
    payload = languages_payload({"es": ["en"], "en": ["es"], "de": ["en"]})
    assert [entry["code"] for entry in payload] == ["de", "en", "es"]


def test_languages_payload_unknown_code_falls_back_to_code_itself():
    payload = languages_payload({"xx": ["en"]})
    assert payload == [{"code": "xx", "name": "xx", "targets": ["en"]}]


def test_languages_payload_multiple_targets_preserved():
    payload = languages_payload({"en": ["es", "de", "fr"]})
    assert payload[0]["targets"] == ["es", "de", "fr"]
