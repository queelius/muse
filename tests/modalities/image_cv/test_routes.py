"""Route tests for image/cv (depth, keypoints, detect)."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_cv import (
    DepthResult,
    Keypoint,
    KeypointDetection,
    KeypointResult,
    MODALITY,
    ObjectDetection,
    ObjectDetectionResult,
    build_router,
)


def _png_bytes(width=64, height=32):
    img = Image.new("RGB", (width, height), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------- Fakes per-primitive ----------


class _FakeDepth:
    def __init__(self, model_id="depth-fake", depth=None, metric=False):
        self.model_id = model_id
        self._depth = depth if depth is not None else np.array([[0.5, 0.7], [0.3, 0.9]], dtype="float32")
        self._metric = metric

    def estimate_depth(self, image):
        W, H = image.size
        return DepthResult(
            depth=self._depth,
            model_id=self.model_id,
            image_size=(W, H),
            metric_depth=self._metric,
        )


class _FakeKeypoint:
    def __init__(self, model_id="kp-fake"):
        self.model_id = model_id
        self.last_kwargs: dict | None = None

    def detect_keypoints(self, image, *, threshold=0.3):
        self.last_kwargs = {"threshold": threshold}
        W, H = image.size
        return KeypointResult(
            detections=[KeypointDetection(
                bbox=(0.0, 0.0, float(W), float(H)),
                score=0.95,
                keypoints=[
                    Keypoint(name="nose", x=W / 2, y=H / 2, score=0.99),
                ],
            )],
            model_id=self.model_id,
            image_size=(W, H),
        )


class _FakeDetection:
    def __init__(self, model_id="det-fake"):
        self.model_id = model_id
        self.last_kwargs: dict | None = None

    def detect_objects(self, image, *, threshold=0.5, max_detections=100):
        self.last_kwargs = {"threshold": threshold, "max_detections": max_detections}
        W, H = image.size
        return ObjectDetectionResult(
            detections=[
                ObjectDetection(
                    bbox=(0.0, 0.0, float(W), float(H)),
                    score=0.9,
                    label="cat",
                ),
            ],
            model_id=self.model_id,
            image_size=(W, H),
        )


def _client(backend, capabilities):
    reg = ModalityRegistry()
    manifest = {"model_id": backend.model_id, "capabilities": capabilities}
    reg.register(MODALITY, backend, manifest=manifest)
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    return TestClient(app)


# ---------- /v1/images/depth ----------


class TestDepthRoute:
    def test_returns_envelope(self):
        backend = _FakeDepth()
        client = _client(backend, {"supports_depth": True})
        r = client.post(
            "/v1/images/depth",
            files={"image": ("t.png", _png_bytes(), "image/png")},
            data={"model": backend.model_id},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == backend.model_id
        assert body["format"] == "png16"
        assert body["id"].startswith("depth-")
        assert body["width"] == 64
        assert body["height"] == 32
        assert body["metric_depth"] is False

    def test_response_format_float32(self):
        backend = _FakeDepth()
        client = _client(backend, {"supports_depth": True})
        r = client.post(
            "/v1/images/depth",
            files={"image": ("t.png", _png_bytes(), "image/png")},
            data={"response_format": "float32"},
        )
        assert r.status_code == 200
        assert r.json()["format"] == "float32"

    def test_invalid_response_format_returns_400(self):
        backend = _FakeDepth()
        client = _client(backend, {"supports_depth": True})
        r = client.post(
            "/v1/images/depth",
            files={"image": ("t.png", _png_bytes(), "image/png")},
            data={"response_format": "tiff"},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_parameter"

    def test_capability_gate_blocks_non_depth_model(self):
        """A model without supports_depth=True returns 400 wrong_primitive."""
        backend = _FakeDepth()
        client = _client(backend, {"supports_keypoints": True})
        r = client.post(
            "/v1/images/depth",
            files={"image": ("t.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "wrong_primitive"

    def test_unknown_model_returns_404(self):
        backend = _FakeDepth(model_id="real")
        client = _client(backend, {"supports_depth": True})
        r = client.post(
            "/v1/images/depth",
            files={"image": ("t.png", _png_bytes(), "image/png")},
            data={"model": "ghost"},
        )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "model_not_found"

    def test_unknown_model_404s_without_decoding_oversized_image(self, monkeypatch):
        """Model resolution must happen before the image is decoded: an
        oversized upload against an unknown model still 404s
        (model_not_found), rather than 400 from the decode-time size
        cap tripping first."""
        monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "10")
        backend = _FakeDepth(model_id="real")
        client = _client(backend, {"supports_depth": True})
        oversized = b"\x00" * 100  # > cap; would 400 "exceeds max" if decoded
        r = client.post(
            "/v1/images/depth",
            files={"image": ("t.png", oversized, "image/png")},
            data={"model": "ghost"},
        )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "model_not_found"

    def test_wrong_primitive_400s_without_decoding_oversized_image(self, monkeypatch):
        """The capability gate must run before decode too: an oversized
        upload against a model that doesn't support depth still 400s
        wrong_primitive, not the decode-time size-cap error."""
        monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "10")
        backend = _FakeDepth()
        client = _client(backend, {"supports_keypoints": True})
        oversized = b"\x00" * 100
        r = client.post(
            "/v1/images/depth",
            files={"image": ("t.png", oversized, "image/png")},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "wrong_primitive"

    def test_corrupt_image_returns_400(self):
        backend = _FakeDepth()
        client = _client(backend, {"supports_depth": True})
        r = client.post(
            "/v1/images/depth",
            files={"image": ("garbage.png", b"not really png", "image/png")},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_parameter"

    def test_metric_depth_passes_through(self):
        backend = _FakeDepth(metric=True)
        client = _client(
            backend, {"supports_depth": True, "metric_depth": True},
        )
        r = client.post(
            "/v1/images/depth",
            files={"image": ("t.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 200
        assert r.json()["metric_depth"] is True


# ---------- /v1/images/keypoints ----------


class TestKeypointsRoute:
    def test_returns_envelope(self):
        backend = _FakeKeypoint()
        client = _client(backend, {"supports_keypoints": True})
        r = client.post(
            "/v1/images/keypoints",
            files={"image": ("t.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == backend.model_id
        assert body["id"].startswith("kp-")
        assert body["image_size"] == [64, 32]
        assert len(body["detections"]) == 1
        det = body["detections"][0]
        assert det["score"] == 0.95
        assert len(det["keypoints"]) == 1
        assert det["keypoints"][0]["name"] == "nose"

    def test_threshold_forwards_to_backend(self):
        backend = _FakeKeypoint()
        client = _client(backend, {"supports_keypoints": True})
        client.post(
            "/v1/images/keypoints",
            files={"image": ("t.png", _png_bytes(), "image/png")},
            data={"threshold": "0.6"},
        )
        assert backend.last_kwargs == {"threshold": 0.6}

    def test_invalid_threshold_returns_400(self):
        backend = _FakeKeypoint()
        client = _client(backend, {"supports_keypoints": True})
        r = client.post(
            "/v1/images/keypoints",
            files={"image": ("t.png", _png_bytes(), "image/png")},
            data={"threshold": "1.5"},
        )
        assert r.status_code == 400
        assert "in [0, 1]" in r.json()["error"]["message"]

    def test_capability_gate_blocks_non_keypoint_model(self):
        backend = _FakeKeypoint()
        client = _client(backend, {"supports_depth": True})
        r = client.post(
            "/v1/images/keypoints",
            files={"image": ("t.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "wrong_primitive"

    def test_wrong_primitive_400s_without_decoding_oversized_image(self, monkeypatch):
        """The capability gate must run before decode: an oversized
        upload against a model that doesn't support keypoints still
        400s wrong_primitive, not the decode-time size-cap error."""
        monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "10")
        backend = _FakeKeypoint()
        client = _client(backend, {"supports_depth": True})
        oversized = b"\x00" * 100
        r = client.post(
            "/v1/images/keypoints",
            files={"image": ("t.png", oversized, "image/png")},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "wrong_primitive"

    def test_unknown_model_404s_without_decoding_oversized_image(self, monkeypatch):
        """Model resolution must happen before decode: an oversized
        upload against an unknown model still 404s, rather than 400
        from the decode-time size cap tripping first."""
        monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "10")
        backend = _FakeKeypoint(model_id="real")
        client = _client(backend, {"supports_keypoints": True})
        oversized = b"\x00" * 100
        r = client.post(
            "/v1/images/keypoints",
            files={"image": ("t.png", oversized, "image/png")},
            data={"model": "ghost"},
        )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "model_not_found"

    def test_omits_threshold_when_absent(self):
        backend = _FakeKeypoint()
        client = _client(backend, {"supports_keypoints": True})
        client.post(
            "/v1/images/keypoints",
            files={"image": ("t.png", _png_bytes(), "image/png")},
        )
        assert backend.last_kwargs == {"threshold": 0.3}  # FakeKeypoint default


# ---------- /v1/images/detect ----------


class TestDetectionRoute:
    def test_returns_envelope(self):
        backend = _FakeDetection()
        client = _client(backend, {"supports_detection": True})
        r = client.post(
            "/v1/images/detect",
            files={"image": ("t.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == backend.model_id
        assert body["id"].startswith("det-")
        assert len(body["detections"]) == 1
        assert body["detections"][0]["label"] == "cat"

    def test_threshold_and_max_forward_to_backend(self):
        backend = _FakeDetection()
        client = _client(backend, {"supports_detection": True})
        client.post(
            "/v1/images/detect",
            files={"image": ("t.png", _png_bytes(), "image/png")},
            data={"threshold": "0.7", "max_detections": "20"},
        )
        assert backend.last_kwargs == {"threshold": 0.7, "max_detections": 20}

    def test_invalid_threshold_returns_400(self):
        backend = _FakeDetection()
        client = _client(backend, {"supports_detection": True})
        r = client.post(
            "/v1/images/detect",
            files={"image": ("t.png", _png_bytes(), "image/png")},
            data={"threshold": "-0.1"},
        )
        assert r.status_code == 400

    def test_invalid_max_detections_returns_400(self):
        backend = _FakeDetection()
        client = _client(backend, {"supports_detection": True})
        r = client.post(
            "/v1/images/detect",
            files={"image": ("t.png", _png_bytes(), "image/png")},
            data={"max_detections": "0"},
        )
        assert r.status_code == 400

    def test_max_detections_too_large_returns_400(self):
        backend = _FakeDetection()
        client = _client(backend, {"supports_detection": True})
        r = client.post(
            "/v1/images/detect",
            files={"image": ("t.png", _png_bytes(), "image/png")},
            data={"max_detections": "99999"},
        )
        assert r.status_code == 400

    def test_capability_gate_blocks_non_detection_model(self):
        backend = _FakeDetection()
        client = _client(backend, {"supports_depth": True})
        r = client.post(
            "/v1/images/detect",
            files={"image": ("t.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "wrong_primitive"

    def test_wrong_primitive_400s_without_decoding_oversized_image(self, monkeypatch):
        """The capability gate must run before decode: an oversized
        upload against a model that doesn't support detection still
        400s wrong_primitive, not the decode-time size-cap error."""
        monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "10")
        backend = _FakeDetection()
        client = _client(backend, {"supports_depth": True})
        oversized = b"\x00" * 100
        r = client.post(
            "/v1/images/detect",
            files={"image": ("t.png", oversized, "image/png")},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "wrong_primitive"

    def test_unknown_model_404s_without_decoding_oversized_image(self, monkeypatch):
        """Model resolution must happen before decode: an oversized
        upload against an unknown model still 404s, rather than 400
        from the decode-time size cap tripping first."""
        monkeypatch.setenv("MUSE_IMAGE_INPUT_MAX_BYTES", "10")
        backend = _FakeDetection(model_id="real")
        client = _client(backend, {"supports_detection": True})
        oversized = b"\x00" * 100
        r = client.post(
            "/v1/images/detect",
            files={"image": ("t.png", oversized, "image/png")},
            data={"model": "ghost"},
        )
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "model_not_found"

    def test_omits_optional_when_absent(self):
        backend = _FakeDetection()
        client = _client(backend, {"supports_detection": True})
        client.post(
            "/v1/images/detect",
            files={"image": ("t.png", _png_bytes(), "image/png")},
        )
        # Both threshold and max_detections at FakeDetection's defaults.
        assert backend.last_kwargs == {"threshold": 0.5, "max_detections": 100}


# ---------- shared error handling ----------


def test_runtime_exception_returns_500():
    """Backend exception surfaces as 500 with the OpenAI envelope."""
    from unittest.mock import MagicMock
    backend = MagicMock()
    backend.model_id = "broken"
    backend.estimate_depth = MagicMock(side_effect=RuntimeError("boom"))
    reg = ModalityRegistry()
    reg.register(
        MODALITY, backend,
        manifest={"model_id": "broken", "capabilities": {"supports_depth": True}},
    )
    app = create_app(registry=reg, routers={MODALITY: build_router(reg)})
    client = TestClient(app)
    r = client.post(
        "/v1/images/depth",
        files={"image": ("t.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["code"] == "internal_error"
    assert "boom" in body["error"]["message"]


def test_inference_lock_attached_per_backend():
    """v0.34.0 contract: the registry attaches _inference_lock at
    register() time. Sanity check on this modality."""
    backend = _FakeDepth()
    _client(backend, {"supports_depth": True})
    assert hasattr(backend, "_inference_lock")
