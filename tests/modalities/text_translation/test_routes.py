"""Route tests for POST /v1/translate, POST /translate (alias), GET /languages.

FakeModel-pattern backends (MagicMock satisfying TranslationBackend
structurally) registered in a fresh ModalityRegistry -- no real weights.
"""
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.text_translation import (
    MODALITY,
    TranslationResult,
    UnsupportedLanguageError,
    build_router,
)


SUPPORTED = {
    "en": ["es", "de", "fr"],
    "es": ["en", "de", "fr"],
    "de": ["en", "es"],
    "fr": ["en", "es"],
}


def _fake_backend(supported=None):
    backend = MagicMock()
    backend.model_id = "m2m100-418m"
    backend.supported_languages.return_value = (
        dict(supported) if supported is not None else dict(SUPPORTED)
    )
    backend.translate.side_effect = lambda texts, source, target: TranslationResult(
        texts=[f"[{target}]{t}" for t in texts],
    )
    return backend


def _make_client(backend, manifest=None):
    reg = ModalityRegistry()
    if manifest is None:
        manifest = {"model_id": "m2m100-418m"}
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


class TestScalarListSymmetry:
    def test_scalar_q_returns_scalar_translatedText(self):
        client = _make_client(_fake_backend())
        r = client.post(
            "/v1/translate", json={"q": "hello", "source": "en", "target": "es"},
        )
        assert r.status_code == 200
        assert r.json() == {"translatedText": "[es]hello"}

    def test_list_q_returns_list_translatedText(self):
        client = _make_client(_fake_backend())
        r = client.post(
            "/v1/translate",
            json={"q": ["hello", "world"], "source": "en", "target": "es"},
        )
        assert r.status_code == 200
        assert r.json() == {"translatedText": ["[es]hello", "[es]world"]}

    def test_empty_scalar_q_translates_to_empty_string(self):
        client = _make_client(_fake_backend())
        r = client.post(
            "/v1/translate", json={"q": "", "source": "en", "target": "es"},
        )
        assert r.status_code == 200
        assert r.json() == {"translatedText": "[es]"}

    def test_empty_list_q_returns_empty_list(self):
        client = _make_client(_fake_backend())
        r = client.post(
            "/v1/translate", json={"q": [], "source": "en", "target": "es"},
        )
        assert r.status_code == 200
        assert r.json() == {"translatedText": []}


class TestAliasPath:
    def test_bare_translate_matches_v1_translate(self):
        client = _make_client(_fake_backend())
        body = {"q": "hi", "source": "en", "target": "es"}
        r1 = client.post("/v1/translate", json=body)
        r2 = client.post("/translate", json=body)
        assert r1.status_code == r2.status_code == 200
        assert r1.json() == r2.json()


class TestErrors:
    def test_source_auto_returns_400_source_detection_not_supported(self):
        client = _make_client(_fake_backend())
        r = client.post(
            "/v1/translate", json={"q": "hi", "source": "auto", "target": "es"},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "source_detection_not_supported"

    def test_non_text_format_returns_400_unsupported_format(self):
        client = _make_client(_fake_backend())
        r = client.post(
            "/v1/translate",
            json={"q": "hi", "source": "en", "target": "es", "format": "html"},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "unsupported_format"

    def test_unsupported_source_language_returns_400_invalid_language(self):
        client = _make_client(_fake_backend())
        r = client.post(
            "/v1/translate", json={"q": "hi", "source": "xx", "target": "es"},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_language"
        assert "xx" in r.json()["error"]["message"]

    def test_unsupported_target_language_returns_400_invalid_language(self):
        client = _make_client(_fake_backend())
        r = client.post(
            "/v1/translate", json={"q": "hi", "source": "en", "target": "xx"},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_language"
        assert "xx" in r.json()["error"]["message"]

    def test_backend_unsupported_language_error_maps_to_400(self):
        backend = _fake_backend()
        backend.translate.side_effect = UnsupportedLanguageError("es", SUPPORTED)
        client = _make_client(backend)
        r = client.post(
            "/v1/translate", json={"q": "hi", "source": "en", "target": "es"},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_language"

    def test_source_equals_target_is_allowed(self):
        client = _make_client(_fake_backend())
        r = client.post(
            "/v1/translate", json={"q": "hi", "source": "en", "target": "en"},
        )
        assert r.status_code == 200

    def test_missing_q_returns_422(self):
        client = _make_client(_fake_backend())
        r = client.post(
            "/v1/translate", json={"source": "en", "target": "es"},
        )
        assert r.status_code == 422

    def test_wrong_typed_q_returns_422(self):
        client = _make_client(_fake_backend())
        r = client.post(
            "/v1/translate", json={"q": 5, "source": "en", "target": "es"},
        )
        assert r.status_code == 422

    def test_input_too_long_returns_400(self, monkeypatch):
        monkeypatch.setenv("MUSE_TRANSLATE_MAX_CHARS", "5")
        client = _make_client(_fake_backend())
        r = client.post(
            "/v1/translate",
            json={"q": "hello world", "source": "en", "target": "es"},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "input_too_long"

    def test_input_too_long_sums_across_list_items(self, monkeypatch):
        monkeypatch.setenv("MUSE_TRANSLATE_MAX_CHARS", "5")
        client = _make_client(_fake_backend())
        r = client.post(
            "/v1/translate",
            json={"q": ["ab", "cd", "ef"], "source": "en", "target": "es"},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "input_too_long"

    def test_backend_exception_returns_generic_500_without_leaking_details(self):
        backend = _fake_backend()
        backend.translate.side_effect = RuntimeError(
            "CUDA out of memory at /internal/path/model.py:123"
        )
        client = _make_client(backend)
        r = client.post(
            "/v1/translate", json={"q": "hi", "source": "en", "target": "es"},
        )
        assert r.status_code == 500
        body = r.json()
        assert body["error"]["code"] == "internal_error"
        # The raw exception text (which can carry CUDA/filesystem detail)
        # must never reach the client; only a generic message does.
        assert "CUDA out of memory" not in r.text
        assert "/internal/path/model.py" not in r.text
        assert body["error"]["message"] == (
            "translation backend failed; see server logs"
        )

    def test_unknown_model_returns_404(self):
        client = _make_client(_fake_backend())
        r = client.post(
            "/v1/translate",
            json={
                "q": "hi", "source": "en", "target": "es",
                "model": "nonexistent-translator",
            },
        )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "model_not_found"


class TestLanguages:
    def test_languages_returns_libretranslate_shape(self):
        backend = _fake_backend(supported={"en": ["es"], "es": ["en"]})
        client = _make_client(backend)
        r = client.get("/languages")
        assert r.status_code == 200
        assert r.json() == [
            {"code": "en", "name": "English", "targets": ["es"]},
            {"code": "es", "name": "Spanish", "targets": ["en"]},
        ]

    def test_languages_unknown_model_returns_404(self):
        client = _make_client(_fake_backend())
        r = client.get("/languages", params={"model": "nonexistent-translator"})
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "model_not_found"
