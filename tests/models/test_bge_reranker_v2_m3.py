"""Tests for the bundled bge_reranker_v2_m3 script."""
from unittest.mock import MagicMock, patch

import pytest

import muse.models.bge_reranker_v2_m3 as bge_mod
from muse.modalities.text_rerank import RerankResult


def test_manifest_required_fields():
    m = bge_mod.MANIFEST
    assert m["model_id"] == "bge-reranker-v2-m3"
    assert m["modality"] == "text/rerank"
    assert m["hf_repo"] == "BAAI/bge-reranker-v2-m3"
    assert "pip_extras" in m
    assert "torch>=2.1.0" in m["pip_extras"]
    assert any("sentence-transformers" in x for x in m["pip_extras"])


def test_manifest_capabilities_shape():
    caps = bge_mod.MANIFEST["capabilities"]
    assert caps["max_length"] == 8192
    assert caps["device"] == "auto"
    assert "memory_gb" in caps


def test_model_class_exists():
    assert hasattr(bge_mod, "Model")
    assert bge_mod.Model.model_id == "bge-reranker-v2-m3"


def test_model_construction_lazy_imports():
    """Model.__init__ should call _ensure_deps and read CrossEncoder
    via the module's sentinel so tests can patch it."""
    fake_ce_inst = MagicMock()
    fake_ce_class = MagicMock(return_value=fake_ce_inst)
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    with patch.object(bge_mod, "CrossEncoder", fake_ce_class), \
            patch.object(bge_mod, "torch", fake_torch):
        m = bge_mod.Model(hf_repo="BAAI/bge-reranker-v2-m3", local_dir=None)
    assert m._device == "cpu"
    fake_ce_class.assert_called_once()


def test_model_prefers_local_dir():
    fake_ce_class = MagicMock(return_value=MagicMock())
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    with patch.object(bge_mod, "CrossEncoder", fake_ce_class), \
            patch.object(bge_mod, "torch", fake_torch):
        bge_mod.Model(hf_repo="BAAI/bge-reranker-v2-m3", local_dir="/tmp/bge")
    args, _ = fake_ce_class.call_args
    assert args[0] == "/tmp/bge"


def test_rerank_returns_sorted_results():
    fake_ce = MagicMock()
    fake_ce.predict.return_value = [0.1, 0.9, 0.5]
    fake_ce_class = MagicMock(return_value=fake_ce)
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    with patch.object(bge_mod, "CrossEncoder", fake_ce_class), \
            patch.object(bge_mod, "torch", fake_torch):
        m = bge_mod.Model(hf_repo="BAAI/bge-reranker-v2-m3", local_dir=None)
        out = m.rerank("q", ["a", "b", "c"], top_n=2)
    assert all(isinstance(r, RerankResult) for r in out)
    assert [r.index for r in out] == [1, 2]


def test_rerank_empty_documents():
    fake_ce_class = MagicMock(return_value=MagicMock())
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    with patch.object(bge_mod, "CrossEncoder", fake_ce_class), \
            patch.object(bge_mod, "torch", fake_torch):
        m = bge_mod.Model(hf_repo="BAAI/bge-reranker-v2-m3", local_dir=None)
        out = m.rerank("q", [])
    assert out == []


def test_model_raises_when_sentence_transformers_missing():
    """Stub _ensure_deps so it leaves CrossEncoder as None (simulates
    sentence_transformers not being installed in the venv)."""
    fake_torch = MagicMock()
    with patch.object(bge_mod, "CrossEncoder", None), \
            patch.object(bge_mod, "_ensure_deps", lambda: None), \
            patch.object(bge_mod, "torch", fake_torch):
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            bge_mod.Model(hf_repo="BAAI/bge-reranker-v2-m3", local_dir=None)
