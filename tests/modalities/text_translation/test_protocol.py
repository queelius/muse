"""Protocol + dataclass shape tests for text/translation."""
from muse.modalities.text_translation import (
    MODALITY,
    MODEL_OPTIONAL_PATHS,
    TranslationBackend,
    TranslationResult,
)


def test_modality_tag_is_mime_shaped():
    assert MODALITY == "text/translation"


def test_model_optional_paths():
    assert MODEL_OPTIONAL_PATHS == ("/v1/translate", "/translate", "/languages")


def test_translation_result_holds_texts():
    r = TranslationResult(texts=["Hola", "mundo"])
    assert r.texts == ["Hola", "mundo"]


def test_translation_result_single_text():
    r = TranslationResult(texts=["Hola mundo"])
    assert r.texts == ["Hola mundo"]


def test_translation_protocol_accepts_structural_impl():
    class Fake:
        def translate(self, texts, *, source, target):
            return TranslationResult(texts=[t.upper() for t in texts])

        def supported_languages(self):
            return {"en": ["es", "de"]}

    assert isinstance(Fake(), TranslationBackend)


def test_translation_protocol_rejects_missing_method():
    class Missing:
        def translate(self, texts, *, source, target):
            return TranslationResult(texts=texts)

    assert not isinstance(Missing(), TranslationBackend)


def test_translation_protocol_rejects_wrong_method_name():
    class WrongName:
        def translate_text(self, texts, *, source, target):
            return None

        def supported_languages(self):
            return {}

    assert not isinstance(WrongName(), TranslationBackend)
