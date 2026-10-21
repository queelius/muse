"""Tests for the bundled mert_v1_95m script (fully mocked).

The script aliases the generic `AudioEmbeddingRuntime` as its `Model`;
runtime behavior (decode, pooling dispatch, device moves) is covered by
tests/modalities/audio_embedding/runtimes/test_transformers_audio.py.
These tests pin the MANIFEST and check that the runtime honors it when
constructed the way `load_backend` does: `model_id` + capabilities plus
the download metadata (`revision`) splatted as kwargs.

Module-level imports of `muse.models.mert_v1_95m` are DELIBERATELY
avoided: test_discovery_robust_to_broken_deps pops `muse.models.*` from
sys.modules and re-imports them.
"""
import importlib
from unittest.mock import MagicMock, patch

import numpy as np

from muse.modalities.audio_embedding import AudioEmbeddingResult

_RT = "muse.modalities.audio_embedding.runtimes.transformers_audio"


def _mert_script():
    return importlib.import_module("muse.models.mert_v1_95m")


def _manifest():
    return _mert_script().MANIFEST


def test_manifest_required_fields():
    m = _manifest()
    assert m["model_id"] == "mert-v1-95m"
    assert m["modality"] == "audio/embedding"
    assert m["hf_repo"] == "m-a-p/MERT-v1-95M"
    assert "torch>=2.6.0" in m["pip_extras"]
    assert any("transformers" in x for x in m["pip_extras"])
    assert any("librosa" in x for x in m["pip_extras"])


def test_manifest_capabilities_shape():
    caps = _manifest()["capabilities"]
    assert caps["device"] == "auto"
    assert caps["dimensions"] == 768
    assert caps["sample_rate"] == 24000
    assert caps["max_duration_seconds"] == 60.0
    assert caps["supports_text_embeddings_too"] is False
    assert caps["trust_remote_code"] is True
    assert "memory_gb" in caps


def test_manifest_allow_patterns_includes_required_files():
    """Weights, processor config, and code are all required locally."""
    patterns = _manifest()["allow_patterns"]
    assert any("preprocessor_config" in p for p in patterns)
    assert "pytorch_model.bin" in patterns
    assert any(".py" in p for p in patterns)


def test_manifest_pins_reviewed_remote_code_revision():
    revision = _manifest()["revision"]
    assert len(revision) == 40
    assert all(c in "0123456789abcdef" for c in revision)


def test_manifest_license_is_noncommercial_cc_by_nc_4_0():
    assert _manifest()["license"] == "CC-BY-NC-4.0"


def test_model_is_audio_embedding_runtime():
    from muse.modalities.audio_embedding.runtimes.transformers_audio import (
        AudioEmbeddingRuntime,
    )
    assert _mert_script().Model is AudioEmbeddingRuntime


def _construct(local_dir):
    """Build the bundled Model with mocked deps, load_backend-style."""
    mod = _mert_script()
    manifest = mod.MANIFEST

    fake_inputs = MagicMock()
    fake_inputs.to.return_value = fake_inputs
    fake_processor = MagicMock(return_value=fake_inputs)
    processor_cls = MagicMock()
    processor_cls.from_pretrained.return_value = fake_processor

    pooled = MagicMock()
    pooled.detach.return_value = pooled
    pooled.to.return_value = pooled
    pooled.float.return_value = pooled
    pooled.numpy.return_value = np.full((1, 768), 0.1, dtype=np.float32)
    hidden = MagicMock()
    hidden.mean.return_value = pooled

    class _MertOutputs:  # MERT returns only last_hidden_state
        last_hidden_state = hidden

    fake_model = MagicMock(return_value=_MertOutputs())
    fake_model.to.return_value = fake_model
    model_cls = MagicMock()
    model_cls.from_pretrained.return_value = fake_model

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    fake_librosa = MagicMock()
    fake_librosa.load.return_value = (np.zeros(24000, np.float32), 24000)

    # load_backend splats capabilities plus download metadata (revision).
    kwargs = {
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        **manifest["capabilities"],
    }
    patches = [
        patch(f"{_RT}.torch", fake_torch),
        patch(f"{_RT}.AutoModel", model_cls),
        patch(f"{_RT}.AutoProcessor", processor_cls),
        patch(f"{_RT}.AutoFeatureExtractor", MagicMock()),
        patch(f"{_RT}.librosa", fake_librosa),
    ]
    for p in patches:
        p.start()
    try:
        m = mod.Model(hf_repo=manifest["hf_repo"], local_dir=local_dir, **kwargs)
        out = m.embed([b"fake-audio"])
    finally:
        for p in reversed(patches):
            p.stop()
    return m, out, processor_cls, model_cls, fake_librosa, hidden


def test_manifest_capabilities_drive_runtime_load_and_embed():
    m, out, processor_cls, model_cls, fake_librosa, hidden = _construct("/tmp/mert")

    p_args, p_kwargs = processor_cls.from_pretrained.call_args
    m_args, m_kwargs = model_cls.from_pretrained.call_args
    assert p_args[0] == m_args[0] == "/tmp/mert"
    assert p_kwargs["trust_remote_code"] is True
    assert m_kwargs["trust_remote_code"] is True
    # Local snapshot: the pinned revision is already on disk.
    assert "revision" not in p_kwargs and "revision" not in m_kwargs
    assert fake_librosa.load.call_args.kwargs["sr"] == 24000
    hidden.mean.assert_called_once_with(dim=1)  # mean-pool over time
    assert m.dimensions == 768
    assert isinstance(out, AudioEmbeddingResult)
    assert out.dimensions == 768
    assert out.model_id == "mert-v1-95m"
    assert out.metadata["sample_rate_used"] == 24000


def test_remote_fallback_threads_reviewed_revision():
    """Without a local snapshot the runtime must pin the Hub revision so
    trust_remote_code never executes an unreviewed head."""
    _, _, processor_cls, model_cls, _, _ = _construct(None)
    revision = _manifest()["revision"]
    assert processor_cls.from_pretrained.call_args.args[0] == "m-a-p/MERT-v1-95M"
    assert processor_cls.from_pretrained.call_args.kwargs["revision"] == revision
    assert model_cls.from_pretrained.call_args.kwargs["revision"] == revision
