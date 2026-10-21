"""Tests for the bundled bge_reranker_v2_m3 script.

The script aliases the generic `CrossEncoderRuntime` as its `Model`;
runtime behavior is covered by tests/modalities/text_rerank/runtimes/.
These tests pin the MANIFEST and check that the runtime honors it when
constructed the way `load_backend` does (`model_id` + capabilities
splatted as kwargs).
"""
import importlib
from unittest.mock import MagicMock, patch

from muse.modalities.text_rerank import RerankResult

_RT = "muse.modalities.text_rerank.runtimes.cross_encoder"


def _mod():
    return importlib.import_module("muse.models.bge_reranker_v2_m3")


def test_manifest_required_fields():
    m = _mod().MANIFEST
    assert m["model_id"] == "bge-reranker-v2-m3"
    assert m["modality"] == "text/rerank"
    assert m["hf_repo"] == "BAAI/bge-reranker-v2-m3"
    assert "torch>=2.1.0" in m["pip_extras"]
    assert any("sentence-transformers" in x for x in m["pip_extras"])


def test_manifest_capabilities_shape():
    caps = _mod().MANIFEST["capabilities"]
    assert caps["max_length"] == 8192
    assert caps["device"] == "auto"
    assert "memory_gb" in caps


def test_model_is_cross_encoder_runtime():
    from muse.modalities.text_rerank.runtimes.cross_encoder import (
        CrossEncoderRuntime,
    )
    assert _mod().Model is CrossEncoderRuntime


def test_manifest_capabilities_drive_runtime_construction():
    mod = _mod()
    manifest = mod.MANIFEST
    fake_ce = MagicMock()
    fake_ce.predict.return_value = [0.1, 0.9, 0.5]
    fake_ce_class = MagicMock(return_value=fake_ce)
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False

    kwargs = {"model_id": manifest["model_id"], **manifest["capabilities"]}
    with patch(f"{_RT}.CrossEncoder", fake_ce_class), \
            patch(f"{_RT}.torch", fake_torch):
        m = mod.Model(hf_repo=manifest["hf_repo"], local_dir="/tmp/bge", **kwargs)
        out = m.rerank("q", ["a", "b", "c"], top_n=2)

    args, ce_kwargs = fake_ce_class.call_args
    assert args[0] == "/tmp/bge"
    assert ce_kwargs["max_length"] == 8192
    assert ce_kwargs["device"] == "cpu"  # auto, no accelerator
    assert ce_kwargs["trust_remote_code"] is False
    assert m.model_id == "bge-reranker-v2-m3"
    assert all(isinstance(r, RerankResult) for r in out)
    assert [r.index for r in out] == [1, 2]
