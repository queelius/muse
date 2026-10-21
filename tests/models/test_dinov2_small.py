"""Tests for the bundled dinov2_small script (fully mocked).

The script aliases the generic `ImageEmbeddingRuntime` as its `Model`;
runtime behavior (pooling dispatch, truncation, device moves) is
covered by tests/modalities/image_embedding/runtimes/. These tests pin
the MANIFEST and check that the runtime honors it when constructed the
way `load_backend` does (`model_id` + capabilities splatted as kwargs).

Module-level imports of `muse.models.dinov2_small` are DELIBERATELY
avoided: test_discovery_robust_to_broken_deps pops `muse.models.*` from
sys.modules and re-imports them.
"""
import importlib
from unittest.mock import MagicMock, patch

import numpy as np

from muse.modalities.image_embedding import ImageEmbeddingResult

_RT = "muse.modalities.image_embedding.runtimes.transformers_image"


def _dinov2_script():
    return importlib.import_module("muse.models.dinov2_small")


def _manifest():
    return _dinov2_script().MANIFEST


def test_manifest_required_fields():
    m = _manifest()
    assert m["model_id"] == "dinov2-small"
    assert m["modality"] == "image/embedding"
    assert m["hf_repo"] == "facebook/dinov2-small"
    assert "torch>=2.1.0" in m["pip_extras"]
    assert any("transformers" in x for x in m["pip_extras"])
    assert any("Pillow" in x for x in m["pip_extras"])


def test_manifest_includes_transformers_5_image_processor_dependency():
    assert any(x.startswith("torchvision") for x in _manifest()["pip_extras"])


def test_manifest_capabilities_shape():
    caps = _manifest()["capabilities"]
    assert caps["device"] == "auto"
    assert caps["dimensions"] == 384
    assert caps["image_size"] == 224
    assert caps["supports_text_embeddings_too"] is False
    assert "memory_gb" in caps


def test_manifest_allow_patterns_includes_preprocessor_config():
    """preprocessor_config.json is essential for image processor load."""
    patterns = _manifest()["allow_patterns"]
    assert any("preprocessor_config" in p for p in patterns)
    assert any(".safetensors" in p for p in patterns)


def test_manifest_license_is_apache_2_0():
    assert _manifest()["license"] == "Apache 2.0"


def test_model_is_image_embedding_runtime():
    from muse.modalities.image_embedding.runtimes.transformers_image import (
        ImageEmbeddingRuntime,
    )
    assert _dinov2_script().Model is ImageEmbeddingRuntime


def test_manifest_capabilities_drive_runtime_load_and_embed():
    mod = _dinov2_script()
    manifest = mod.MANIFEST

    fake_inputs = MagicMock()
    fake_inputs.to.return_value = fake_inputs
    fake_processor = MagicMock(return_value=fake_inputs)
    processor_cls = MagicMock()
    processor_cls.from_pretrained.return_value = fake_processor

    cls_token = MagicMock()
    cls_token.detach.return_value = cls_token
    cls_token.to.return_value = cls_token
    cls_token.float.return_value = cls_token
    cls_token.numpy.return_value = np.full((2, 384), 0.1, dtype=np.float32)

    class _Dinov2Outputs:  # Dinov2Model: pooler_output is the CLS token
        pooler_output = cls_token
        last_hidden_state = MagicMock()

    class _Dinov2Model:  # no get_image_features on the class
        config = MagicMock(projection_dim=None, hidden_size=384)

        def __call__(self, **_):
            return _Dinov2Outputs()

        def to(self, _device):
            return self

        def eval(self):
            return self

    model_cls = MagicMock()
    model_cls.from_pretrained.return_value = _Dinov2Model()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False

    kwargs = {"model_id": manifest["model_id"], **manifest["capabilities"]}
    with patch(f"{_RT}.torch", fake_torch), \
            patch(f"{_RT}.AutoModel", model_cls), \
            patch(f"{_RT}.AutoProcessor", processor_cls), \
            patch(f"{_RT}.AutoFeatureExtractor", MagicMock()):
        m = mod.Model(hf_repo=manifest["hf_repo"], local_dir="/tmp/dino", **kwargs)
        out = m.embed(["img-a", "img-b"], dimensions=128)

    assert processor_cls.from_pretrained.call_args.args[0] == "/tmp/dino"
    assert model_cls.from_pretrained.call_args.args[0] == "/tmp/dino"
    assert m.dimensions == 384
    assert isinstance(out, ImageEmbeddingResult)
    assert out.model_id == "dinov2-small"
    assert out.n_images == 2
    assert out.dimensions == 128  # matryoshka truncation + renormalize
    np.testing.assert_allclose(np.linalg.norm(out.embeddings[0]), 1.0, rtol=1e-5)
