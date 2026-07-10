"""Route tests for POST /v1/images/ocr.

FakeModel implements the OcrModel Protocol structurally. Multipart
upload exercises the full path including decode_image_file.
"""
import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_ocr import (
    MODALITY,
    OcrResult,
    build_router,
)


def _png_bytes(width=64, height=32, color=(255, 255, 255)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeOcr:
    """Tracks last call args; returns canned OcrResult."""

    def __init__(self, model_id="trocr-fake", text="hello world",
                 completion_tokens=10):
        self.model_id = model_id
        self._text = text
        self._tokens = completion_tokens
        self.last_kwargs: dict | None = None
        self.last_image = None

    def ocr(self, image, **kwargs):
        self.last_image = image
        self.last_kwargs = kwargs
        return OcrResult(
            text=self._text,
            model_id=self.model_id,
            completion_tokens=self._tokens,
        )


def _client(backend):
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={"model_id": backend.model_id})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


def test_ocr_returns_envelope_for_uploaded_image():
    backend = _FakeOcr(text="extracted text")
    client = _client(backend)
    r = client.post(
        "/v1/images/ocr",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        data={"model": "trocr-fake"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "trocr-fake"
    assert body["id"].startswith("ocr-")
    assert body["text"] == "extracted text"
    assert body["usage"]["completion_tokens"] == 10


def test_ocr_default_model_used_when_unspecified():
    """`model` is optional; the modality default applies."""
    backend = _FakeOcr(model_id="trocr-default")
    client = _client(backend)
    r = client.post(
        "/v1/images/ocr",
        files={"image": ("test.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    assert r.json()["model"] == "trocr-default"


def test_ocr_unknown_model_returns_404():
    backend = _FakeOcr(model_id="real-model")
    client = _client(backend)
    r = client.post(
        "/v1/images/ocr",
        files={"image": ("test.png", _png_bytes(), "image/png")},
        data={"model": "ghost"},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "model_not_found"


def test_ocr_unknown_model_404s_without_decoding_oversized_image(monkeypatch):
    """Model resolution must happen before the image is decoded: an
    oversized upload against an unknown model still 404s
    (model_not_found), rather than 400 from the decode-time size cap
    tripping first."""
    monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "10")
    backend = _FakeOcr(model_id="real-model")
    client = _client(backend)
    oversized = b"\x00" * 100  # > cap; would 400 "exceeds max" if decoded
    r = client.post(
        "/v1/images/ocr",
        files={"image": ("t.png", oversized, "image/png")},
        data={"model": "ghost"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_ocr_missing_image_returns_422():
    """FastAPI raises 422 when the required `image` field is missing."""
    backend = _FakeOcr()
    client = _client(backend)
    r = client.post("/v1/images/ocr", data={"model": "x"})
    assert r.status_code == 422


def test_ocr_corrupt_image_returns_400():
    """A non-image upload fails decode and surfaces as 400."""
    backend = _FakeOcr()
    client = _client(backend)
    r = client.post(
        "/v1/images/ocr",
        files={"image": ("garbage.png", b"not really a png", "image/png")},
    )
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["code"] == "invalid_parameter"


def test_ocr_prompt_forwards_to_backend():
    backend = _FakeOcr()
    client = _client(backend)
    client.post(
        "/v1/images/ocr",
        files={"image": ("t.png", _png_bytes(), "image/png")},
        data={"prompt": "<s_doc>"},
    )
    assert backend.last_kwargs == {"prompt": "<s_doc>"}


def test_ocr_max_new_tokens_forwards_to_backend():
    backend = _FakeOcr()
    client = _client(backend)
    client.post(
        "/v1/images/ocr",
        files={"image": ("t.png", _png_bytes(), "image/png")},
        data={"max_new_tokens": "256"},
    )
    assert backend.last_kwargs == {"max_new_tokens": 256}


def test_ocr_omits_optional_kwargs_when_absent():
    backend = _FakeOcr()
    client = _client(backend)
    client.post(
        "/v1/images/ocr",
        files={"image": ("t.png", _png_bytes(), "image/png")},
    )
    assert backend.last_kwargs == {}


def test_ocr_max_new_tokens_zero_returns_400():
    backend = _FakeOcr()
    client = _client(backend)
    r = client.post(
        "/v1/images/ocr",
        files={"image": ("t.png", _png_bytes(), "image/png")},
        data={"max_new_tokens": "0"},
    )
    assert r.status_code == 400
    assert "in [1, 4096]" in r.json()["error"]["message"]


def test_ocr_max_new_tokens_too_large_returns_400():
    backend = _FakeOcr()
    client = _client(backend)
    r = client.post(
        "/v1/images/ocr",
        files={"image": ("t.png", _png_bytes(), "image/png")},
        data={"max_new_tokens": "9999"},
    )
    assert r.status_code == 400


def test_ocr_runtime_exception_returns_500():
    """A backend.ocr exception surfaces as 500 with the OpenAI envelope,
    not FastAPI's {detail:...} default."""
    backend = MagicMock()
    backend.model_id = "broken"
    backend.ocr = MagicMock(side_effect=RuntimeError("model crashed"))
    reg = ModalityRegistry()
    reg.register(MODALITY, backend, manifest={"model_id": "broken"})
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)

    r = client.post(
        "/v1/images/ocr",
        files={"image": ("t.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["code"] == "internal_error"
    # Finding 1 (v0.58.1 review): the backend exception text must NOT
    # reach the client body; only a generic message does.
    assert "model crashed" not in body["error"]["message"]


def test_ocr_uses_inference_lock_per_backend():
    """The route holds backend._inference_lock during ocr() to serialize
    calls. v0.34.0 attached the lock at registry.register time; this is
    a smoke test that the lock is referenced (not deadlocked)."""
    backend = _FakeOcr()
    client = _client(backend)
    # Two sequential calls succeed (no deadlock).
    for _ in range(2):
        r = client.post(
            "/v1/images/ocr",
            files={"image": ("t.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 200
    # Confirm registry attached the lock (v0.34.0 contract).
    assert hasattr(backend, "_inference_lock")
