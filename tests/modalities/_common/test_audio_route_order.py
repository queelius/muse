"""Audio upload routes resolve the model before reading the upload."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app


@pytest.mark.parametrize("package, path, extra", [
    ("audio_alignment", "/v1/audio/alignments", {"text": "hello"}),
    ("audio_classification", "/v1/audio/classifications", {}),
    ("audio_quality", "/v1/audio/quality", {}),
    ("audio_embedding", "/v1/audio/embeddings", {}),
])
def test_unknown_model_404s_without_reading_the_upload(package, path, extra):
    import importlib
    mod = importlib.import_module(f"muse.modalities.{package}")
    routes = importlib.import_module(f"muse.modalities.{package}.routes")
    registry = ModalityRegistry()
    app = create_app(registry=registry, routers={mod.MODALITY: mod.build_router(registry)})
    with patch.object(routes, "read_upload") as read_upload:
        r = TestClient(app).post(
            path, data={"model": "nope", **extra},
            files={"file": ("a.wav", b"RIFF", "audio/wav")},
        )
    assert r.status_code == 404
    read_upload.assert_not_called()
