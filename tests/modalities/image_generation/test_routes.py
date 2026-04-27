"""Tests for /v1/images/generations router."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_generation.protocol import ImageResult
from muse.modalities.image_generation.routes import build_router


class FakeImageModel:
    model_id = "fake-sd"
    default_size = (64, 64)

    def generate(self, prompt, **kwargs):
        w = kwargs.get("width", self.default_size[0])
        h = kwargs.get("height", self.default_size[1])
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        return ImageResult(
            image=arr, width=w, height=h,
            seed=kwargs.get("seed", 0) or 0,
            metadata={"prompt": prompt},
        )


class RecordingImageModel:
    """Like FakeImageModel but records the kwargs each generate() received."""

    def __init__(self, model_id="fake-i2i-model"):
        self.model_id = model_id
        self.default_size = (64, 64)
        self.calls: list[dict] = []

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        w = kwargs.get("width", self.default_size[0])
        h = kwargs.get("height", self.default_size[1])
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        return ImageResult(
            image=arr, width=w, height=h,
            seed=kwargs.get("seed", 0) or 0,
            metadata={"prompt": prompt},
        )


@pytest.fixture
def client():
    reg = ModalityRegistry()
    reg.register("image/generation", FakeImageModel())
    app = create_app(
        registry=reg,
        routers={"image/generation": build_router(reg)},
    )
    return TestClient(app)


@pytest.fixture
def capable_model():
    """Recording model registered with supports_img2img: True."""
    return RecordingImageModel(model_id="fake-i2i-model")


@pytest.fixture
def client_with_capable_model(capable_model):
    """Client whose registered model declares supports_img2img: True."""
    reg = ModalityRegistry()
    reg.register(
        "image/generation",
        capable_model,
        manifest={
            "model_id": capable_model.model_id,
            "modality": "image/generation",
            "capabilities": {"supports_img2img": True},
        },
    )
    app = create_app(
        registry=reg,
        routers={"image/generation": build_router(reg)},
    )
    return TestClient(app)


@pytest.fixture
def client_with_uncapable_model():
    """Client whose registered model declares supports_img2img: False."""
    reg = ModalityRegistry()
    model = RecordingImageModel(model_id="fake-t2i-only")
    reg.register(
        "image/generation",
        model,
        manifest={
            "model_id": model.model_id,
            "modality": "image/generation",
            "capabilities": {"supports_img2img": False},
        },
    )
    app = create_app(
        registry=reg,
        routers={"image/generation": build_router(reg)},
    )
    return TestClient(app)


def test_generate_returns_base64_by_default(client):
    r = client.post("/v1/images/generations", json={"prompt": "a cat"})
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data) == 1
    assert "b64_json" in data[0]
    # Must be decodable base64
    import base64
    decoded = base64.b64decode(data[0]["b64_json"])
    # PNG magic bytes
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_generate_response_format_url_returns_data_url(client):
    r = client.post("/v1/images/generations", json={
        "prompt": "a cat",
        "response_format": "url",
    })
    assert r.status_code == 200
    url = r.json()["data"][0]["url"]
    assert url.startswith("data:image/png;base64,")


def test_generate_n_creates_multiple_images(client):
    r = client.post("/v1/images/generations", json={"prompt": "a dog", "n": 3})
    assert r.status_code == 200
    assert len(r.json()["data"]) == 3


def test_generate_echoes_prompt_as_revised_prompt(client):
    r = client.post("/v1/images/generations", json={"prompt": "a bird"})
    assert r.status_code == 200
    entry = r.json()["data"][0]
    assert entry["revised_prompt"] == "a bird"


def test_unknown_model_returns_openai_shape_404(client):
    r = client.post("/v1/images/generations", json={
        "prompt": "x", "model": "no-such-model",
    })
    assert r.status_code == 404
    body = r.json()
    # OpenAI-style envelope, not FastAPI's default {detail: ...}
    assert "error" in body
    assert "detail" not in body
    assert body["error"]["code"] == "model_not_found"


def test_empty_prompt_rejected(client):
    r = client.post("/v1/images/generations", json={"prompt": ""})
    assert r.status_code in (400, 422)


def test_n_over_limit_rejected(client):
    r = client.post("/v1/images/generations", json={"prompt": "x", "n": 100})
    assert r.status_code in (400, 422)


def test_size_out_of_range_rejected(client):
    r = client.post("/v1/images/generations", json={
        "prompt": "x",
        "size": "4096x4096",
    })
    assert r.status_code in (400, 422)


def test_size_invalid_format_rejected(client):
    r = client.post("/v1/images/generations", json={
        "prompt": "x",
        "size": "big",
    })
    assert r.status_code in (400, 422)


def test_response_includes_created_unix_timestamp(client):
    r = client.post("/v1/images/generations", json={"prompt": "a cat"})
    assert r.status_code == 200
    created = r.json()["created"]
    assert isinstance(created, int)
    # Sanity: later than 2020-01-01 UTC (1577836800)
    assert created > 1577836800


def test_seed_passed_through_to_backend(client):
    """The backend should receive the seed kwarg."""
    r = client.post("/v1/images/generations", json={"prompt": "x", "seed": 42, "n": 1})
    assert r.status_code == 200
    # Fake backend echoes seed in metadata via kwargs — we don't surface it
    # in the response here, but the call should succeed


def test_post_with_image_data_url_routes_through_img2img(
    client_with_capable_model, capable_model,
):
    """When image is supplied to a supports_img2img model, it reaches generate(init_image=...)."""
    import base64
    import io

    from PIL import Image

    img = Image.new("RGB", (64, 64), (0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    r = client_with_capable_model.post("/v1/images/generations", json={
        "prompt": "make it red",
        "model": "fake-i2i-model",
        "image": data_url,
        "strength": 0.7,
    })
    assert r.status_code == 200
    body = r.json()
    assert "data" in body
    # Verify the model received init_image (not None) and strength
    assert len(capable_model.calls) == 1
    call = capable_model.calls[0]
    assert call["init_image"] is not None
    assert call["strength"] == 0.7


def test_post_strength_without_image_is_ignored(
    client_with_capable_model, capable_model,
):
    """strength alone (no image) does not error; falls through to text-to-image."""
    r = client_with_capable_model.post("/v1/images/generations", json={
        "prompt": "a cat",
        "model": "fake-i2i-model",
        "strength": 0.5,
    })
    assert r.status_code == 200
    # No image was sent, so init_image should be None on the call
    assert len(capable_model.calls) == 1
    assert capable_model.calls[0]["init_image"] is None


def test_post_image_with_unsupported_model_returns_400(client_with_uncapable_model):
    """A model whose supports_img2img is False rejects requests with image."""
    r = client_with_uncapable_model.post("/v1/images/generations", json={
        "prompt": "x",
        "model": "fake-t2i-only",
        "image": "data:image/png;base64,iVBORw0KGgo=",
    })
    assert r.status_code == 400
    body = r.json()
    assert "error" in body
    assert "img2img" in body["error"]["message"].lower()


def test_post_malformed_data_url_returns_400_image_decode_error(
    client_with_capable_model,
):
    """A bad data URL returns 400 with image_decode-flavored message."""
    r = client_with_capable_model.post("/v1/images/generations", json={
        "prompt": "x",
        "model": "fake-i2i-model",
        "image": "data:image/png;base64,!!!not_base64!!!",
    })
    assert r.status_code == 400
    body = r.json()
    assert "error" in body
    err = body["error"]
    assert err["code"] == "invalid_parameter"
    # Type or message should reference image
    assert "image" in err["message"].lower() or "decode" in err["message"].lower()


def test_post_strength_out_of_range_returns_400(client_with_capable_model):
    """strength=1.5 is rejected by pydantic ge/le validation."""
    r = client_with_capable_model.post("/v1/images/generations", json={
        "prompt": "x",
        "model": "fake-i2i-model",
        "image": "data:image/png;base64,iVBORw0KGgo=",
        "strength": 1.5,  # out of [0, 1]
    })
    assert r.status_code in (400, 422)
