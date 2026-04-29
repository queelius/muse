"""End-to-end: multipart image upload flows for /v1/images/edits and
/v1/images/variations through FastAPI + codec correctly.

Uses a fake ImageModel backend; no real diffusion. This catches
integration bugs in the multipart -> UploadFile -> backend -> codec
chain that the per-component unit tests don't see together.

Marked slow because the fast lane shouldn't include this. The other
e2e modules in tests/cli_impl/ also use this convention.
"""
import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_generation import (
    MODALITY,
    ImageResult,
    build_router,
)


pytestmark = pytest.mark.slow


class _FakeEditsBackend:
    """Stand-in ImageModel with inpaint() and vary() methods.

    Records the kwargs passed in so the e2e test can assert that the
    upload bytes really decoded into PIL images and reached the backend.
    """

    def __init__(self):
        self.model_id = "fake-edits"
        self.default_size = (64, 64)
        self.inpaint_calls: list[dict] = []
        self.vary_calls: list[dict] = []

    def generate(self, prompt, **kwargs):  # pragma: no cover - not used here
        raise AssertionError(
            "/v1/images/edits and /variations should NOT route to generate"
        )

    def inpaint(self, prompt, **kwargs):
        self.inpaint_calls.append({"prompt": prompt, **kwargs})
        w = kwargs.get("width", self.default_size[0])
        h = kwargs.get("height", self.default_size[1])
        return ImageResult(
            image=np.zeros((h, w, 3), dtype=np.uint8),
            width=w, height=h, seed=0,
            metadata={"prompt": prompt, "mode": "inpaint"},
        )

    def vary(self, **kwargs):
        self.vary_calls.append(kwargs)
        w = kwargs.get("width", self.default_size[0])
        h = kwargs.get("height", self.default_size[1])
        return ImageResult(
            image=np.zeros((h, w, 3), dtype=np.uint8),
            width=w, height=h, seed=0,
            metadata={"mode": "variations"},
        )


def _png_bytes(width=64, height=64, color=(0, 128, 255)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mask_bytes(width=64, height=64) -> bytes:
    img = Image.new("L", (width, height), 255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_client() -> tuple[TestClient, _FakeEditsBackend]:
    backend = _FakeEditsBackend()
    reg = ModalityRegistry()
    reg.register(
        MODALITY,
        backend,
        manifest={
            "model_id": backend.model_id,
            "modality": MODALITY,
            "capabilities": {
                "supports_inpainting": True,
                "supports_variations": True,
            },
        },
    )
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app), backend


@pytest.mark.timeout(10)
def test_multipart_edits_flow_end_to_end():
    """Full /v1/images/edits round trip with image+mask+prompt."""
    client, backend = _build_client()

    src_png = _png_bytes(64, 64, color=(255, 0, 0))
    msk_png = _mask_bytes(64, 64)

    r = client.post(
        "/v1/images/edits",
        files={
            "image": ("scene.png", src_png, "image/png"),
            "mask": ("mask.png", msk_png, "image/png"),
        },
        data={
            "prompt": "add a moon",
            "model": "fake-edits",
            "n": "1",
            "size": "64x64",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "data" in body and len(body["data"]) == 1
    entry = body["data"][0]
    assert entry["revised_prompt"] == "add a moon"
    decoded = base64.b64decode(entry["b64_json"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    # The backend got two real PIL.Image objects (decoded from the multipart bytes)
    assert len(backend.inpaint_calls) == 1
    call = backend.inpaint_calls[0]
    assert hasattr(call["init_image"], "size")
    assert hasattr(call["mask_image"], "size")
    assert call["init_image"].size == (64, 64)


@pytest.mark.timeout(10)
def test_multipart_variations_flow_end_to_end():
    """Full /v1/images/variations round trip with image only (no prompt)."""
    client, backend = _build_client()

    src_png = _png_bytes(64, 64, color=(0, 255, 0))

    r = client.post(
        "/v1/images/variations",
        files={"image": ("scene.png", src_png, "image/png")},
        data={
            "model": "fake-edits",
            "n": "2",
            "size": "64x64",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "data" in body and len(body["data"]) == 2
    for entry in body["data"]:
        assert "revised_prompt" not in entry
        decoded = base64.b64decode(entry["b64_json"])
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"

    # Backend got two vary() calls; each saw a real PIL.Image
    assert len(backend.vary_calls) == 2
    for call in backend.vary_calls:
        assert hasattr(call["init_image"], "size")
        assert call["init_image"].size == (64, 64)
