"""Tests for ImageEmbeddingRuntime (transformers AutoModel + AutoProcessor wrapper).

The per-architecture _extract_embeddings dispatch is exercised
explicitly with mocked outputs that match each shape (CLIP-style
image_embeds, SigLIP-style pooler_output, DINOv2-style last_hidden_state
CLS slice).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import muse.modalities.image_embedding.runtimes.transformers_image as ie_mod
from muse.modalities.image_embedding.protocol import ImageEmbeddingResult
from muse.modalities.image_embedding.runtimes.transformers_image import (
    ImageEmbeddingRuntime,
    _detect_dimensions,
    _extract_embeddings,
    _load_processor,
    _resolve_dtype,
    _select_device,
    _set_inference_mode,
    _truncate_and_renormalize,
)


def _np_tensor(arr):
    """Wrap a numpy array in a MagicMock that mimics a torch tensor.

    The runtime calls .detach().to("cpu").float().numpy() to lift values
    back to numpy; chained MagicMocks would all return MagicMock, so we
    set up explicit return_value links.
    """
    arr = np.asarray(arr, dtype=np.float32)
    t = MagicMock()
    t.detach.return_value = t
    t.to.return_value = t
    t.float.return_value = t
    t.numpy.return_value = arr
    t.shape = arr.shape
    return t


def _make_inputs(device="cpu"):
    inputs = MagicMock()
    inputs.to.return_value = inputs
    return inputs


def _patched_runtime(
    *,
    embedding_arr=None,
    arch="dinov2",
    dim_attr="hidden_size",
    dim_val=384,
    device="cpu",
):
    """Build a fully-mocked ImageEmbeddingRuntime ready for embed() calls.

    `arch` selects which output attribute carries embeddings:
      - "clip": outputs.image_embeds set
      - "siglip": outputs.pooler_output set, no image_embeds
      - "dinov2": outputs.last_hidden_state set, no pooler_output, no image_embeds
    """
    if embedding_arr is None:
        embedding_arr = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    # Build outputs object with the right attrs per architecture.
    embeddings_tensor = _np_tensor(embedding_arr)

    class FakeOutputs:
        image_embeds = None
        pooler_output = None
        last_hidden_state = None

    fake_outputs = FakeOutputs()
    if arch == "clip":
        fake_outputs.image_embeds = embeddings_tensor
    elif arch == "siglip":
        fake_outputs.pooler_output = embeddings_tensor
    elif arch == "dinov2":
        # last_hidden_state shape [B, T, H]; the runtime takes [:, 0]
        # which is the CLS token row. Build a 3D tensor whose [:, 0]
        # equals embedding_arr.
        seq_arr = np.array([[
            embedding_arr[0],   # CLS row -> what we'll get extracted
            np.zeros_like(embedding_arr[0]),  # filler row
        ]], dtype=np.float32)
        seq = MagicMock()
        # Slice [:, 0] -> a tensor whose .numpy() returns embedding_arr
        cls_slice = _np_tensor(embedding_arr)
        seq.__getitem__ = lambda self, idx: cls_slice
        fake_outputs.last_hidden_state = seq

    fake_model = MagicMock()
    fake_model.return_value = fake_outputs
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = None
    fake_model.config = SimpleNamespace(**{dim_attr: dim_val})
    fake_model_class = MagicMock()
    fake_model_class.from_pretrained.return_value = fake_model

    fake_processor = MagicMock()
    fake_inputs = _make_inputs(device)
    fake_processor.return_value = fake_inputs
    fake_processor_class = MagicMock()
    fake_processor_class.from_pretrained.return_value = fake_processor

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    fake_torch.float32 = "float32"
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.inference_mode.return_value.__enter__ = MagicMock(return_value=None)
    fake_torch.inference_mode.return_value.__exit__ = MagicMock(return_value=None)

    with patch.object(ie_mod, "AutoModel", fake_model_class), \
            patch.object(ie_mod, "AutoProcessor", fake_processor_class), \
            patch.object(ie_mod, "AutoFeatureExtractor", MagicMock()), \
            patch.object(ie_mod, "torch", fake_torch):
        rt = ImageEmbeddingRuntime(
            model_id="test",
            hf_repo="org/repo",
            local_dir=None,
            device=device,
            dtype="float32",
        )
        # Exit-context: keep torch + processors patched for embed()
        # calls in tests by stuffing them onto the runtime instance.
        rt._fake_torch = fake_torch
        rt._fake_processor = fake_processor
        rt._fake_inputs = fake_inputs
        rt._fake_model = fake_model
    return rt, fake_torch, fake_processor, fake_model


def _patched_embed(rt, *args, **kwargs):
    """Call rt.embed(...) with the same patches the constructor used.

    Tests can't keep the patch context open across the constructor call,
    so we re-patch around .embed() using sentinels stored on the
    runtime by _patched_runtime.
    """
    with patch.object(ie_mod, "torch", rt._fake_torch):
        return rt.embed(*args, **kwargs)


def test_select_device_auto_falls_back_to_cpu_with_no_torch():
    with patch.object(ie_mod, "torch", None):
        assert _select_device("auto") == "cpu"


def test_select_device_auto_picks_cuda():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    with patch.object(ie_mod, "torch", fake_torch):
        assert _select_device("auto") == "cuda"


def test_select_device_auto_picks_mps_when_cuda_absent():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = True
    with patch.object(ie_mod, "torch", fake_torch):
        assert _select_device("auto") == "mps"


def test_select_device_explicit_passes_through():
    with patch.object(ie_mod, "torch", MagicMock()):
        assert _select_device("cuda:1") == "cuda:1"


def test_resolve_dtype_returns_none_without_torch():
    with patch.object(ie_mod, "torch", None):
        assert _resolve_dtype("float16") is None


def test_resolve_dtype_maps_known_strings():
    fake_torch = MagicMock()
    fake_torch.float16 = "FP16"
    fake_torch.bfloat16 = "BF16"
    fake_torch.float32 = "FP32"
    with patch.object(ie_mod, "torch", fake_torch):
        assert _resolve_dtype("float16") == "FP16"
        assert _resolve_dtype("fp16") == "FP16"
        assert _resolve_dtype("bfloat16") == "BF16"
        assert _resolve_dtype("bf16") == "BF16"
        assert _resolve_dtype("float32") == "FP32"
        # Unknown maps to safe default float32.
        assert _resolve_dtype("unknown") == "FP32"


def test_set_inference_mode_invokes_method_when_present():
    """Helper must call the model's inference-switch method when callable."""
    m = MagicMock()
    _set_inference_mode(m)
    # Verify the method was called exactly once.
    assert m.eval.call_count == 1


def test_set_inference_mode_safe_when_method_missing():
    class NoEval:
        pass

    # Must not raise.
    _set_inference_mode(NoEval())


def test_extract_embeddings_clip_branch_returns_image_embeds():
    """CLIP path: outputs.image_embeds is set; pooler_output is also set
    but lower priority. image_embeds wins."""
    expected = MagicMock(name="image_embeds")
    other = MagicMock(name="pooler_output")
    outputs = SimpleNamespace(
        image_embeds=expected,
        pooler_output=other,
        last_hidden_state=MagicMock(),
    )
    assert _extract_embeddings(outputs) is expected


def test_extract_embeddings_siglip_branch_returns_pooler_output():
    """SigLIP path: outputs.pooler_output set, image_embeds absent."""
    expected = MagicMock(name="pooler_output")
    outputs = SimpleNamespace(
        image_embeds=None,
        pooler_output=expected,
        last_hidden_state=MagicMock(),
    )
    assert _extract_embeddings(outputs) is expected


def test_extract_embeddings_dinov2_branch_uses_cls_token():
    """DINOv2 path: only last_hidden_state is set. CLS = [:, 0]."""
    seq = MagicMock(name="last_hidden_state")
    expected = MagicMock(name="cls_token")
    seq.__getitem__ = lambda self, idx: expected
    outputs = SimpleNamespace(
        image_embeds=None,
        pooler_output=None,
        last_hidden_state=seq,
    )
    assert _extract_embeddings(outputs) is expected


def test_extract_embeddings_raises_when_no_path_matches():
    outputs = SimpleNamespace(
        image_embeds=None, pooler_output=None, last_hidden_state=None,
    )
    with pytest.raises(ValueError, match="could not extract embeddings"):
        _extract_embeddings(outputs)


def test_detect_dimensions_uses_projection_dim_first():
    """CLIP-style projection_dim wins over hidden_size."""
    model = SimpleNamespace(
        config=SimpleNamespace(projection_dim=512, hidden_size=768),
    )
    assert _detect_dimensions(model) == 512


def test_detect_dimensions_falls_back_to_hidden_size():
    """When no projection_dim, hidden_size (DINOv2 style) is used."""
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=384),
    )
    assert _detect_dimensions(model) == 384


def test_detect_dimensions_uses_vision_config_hidden_size():
    """Composite CLIP-shaped configs nest hidden_size under vision_config."""
    cfg = SimpleNamespace(
        projection_dim=None,
        hidden_size=None,
        vision_config=SimpleNamespace(hidden_size=768),
    )
    model = SimpleNamespace(config=cfg)
    assert _detect_dimensions(model) == 768


def test_detect_dimensions_returns_minus_one_when_unknowable():
    """Sentinel for the rare repo with no recognizable dim attribute."""
    model = SimpleNamespace(config=None)
    assert _detect_dimensions(model) == -1


def test_truncate_and_renormalize_no_op_when_dimensions_none():
    arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = _truncate_and_renormalize(arr, None)
    np.testing.assert_array_equal(out, arr)


def test_truncate_and_renormalize_no_op_when_dimensions_geq_native():
    arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = _truncate_and_renormalize(arr, 5)  # 5 >= 3
    np.testing.assert_array_equal(out, arr)


def test_truncate_and_renormalize_truncates_and_normalizes():
    arr = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)  # native dim 3
    out = _truncate_and_renormalize(arr, 2)
    # Sliced to [3.0, 4.0], norm = 5.0, normalized to [0.6, 0.8]
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out, [[0.6, 0.8]], rtol=1e-6)


def test_truncate_and_renormalize_handles_zero_norm():
    """All-zero rows get norm clamped to 1 (no division by zero)."""
    arr = np.zeros((1, 3), dtype=np.float32)
    out = _truncate_and_renormalize(arr, 2)
    np.testing.assert_array_equal(out, np.zeros((1, 2), dtype=np.float32))


def test_runtime_constructs_with_local_dir_preference():
    fake_processor_class = MagicMock()
    fake_processor_class.from_pretrained.return_value = MagicMock()
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.config = SimpleNamespace(hidden_size=384)
    fake_model_class = MagicMock()
    fake_model_class.from_pretrained.return_value = fake_model
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False

    with patch.object(ie_mod, "AutoModel", fake_model_class), \
            patch.object(ie_mod, "AutoProcessor", fake_processor_class), \
            patch.object(ie_mod, "torch", fake_torch):
        ImageEmbeddingRuntime(
            model_id="m",
            hf_repo="org/repo",
            local_dir="/tmp/cache/abc",
            device="cpu",
        )
    args, _ = fake_processor_class.from_pretrained.call_args
    assert args[0] == "/tmp/cache/abc"
    args2, _ = fake_model_class.from_pretrained.call_args
    assert args2[0] == "/tmp/cache/abc"


def test_runtime_falls_back_to_hf_repo_when_no_local_dir():
    fake_processor_class = MagicMock()
    fake_processor_class.from_pretrained.return_value = MagicMock()
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.config = SimpleNamespace(hidden_size=384)
    fake_model_class = MagicMock()
    fake_model_class.from_pretrained.return_value = fake_model
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False

    with patch.object(ie_mod, "AutoModel", fake_model_class), \
            patch.object(ie_mod, "AutoProcessor", fake_processor_class), \
            patch.object(ie_mod, "torch", fake_torch):
        ImageEmbeddingRuntime(
            model_id="m",
            hf_repo="facebook/dinov2-small",
            local_dir=None,
            device="cpu",
        )
    args, _ = fake_processor_class.from_pretrained.call_args
    assert args[0] == "facebook/dinov2-small"


def test_runtime_loads_dimensions_from_model_config():
    """ImageEmbeddingRuntime.dimensions comes from model.config detection."""
    rt, *_ = _patched_runtime(dim_attr="hidden_size", dim_val=384)
    assert rt.dimensions == 384


def test_runtime_calls_to_with_device():
    rt, _, _, fake_model = _patched_runtime(device="cpu")
    fake_model.to.assert_called_with("cpu")


def test_runtime_invokes_inference_mode_helper():
    rt, _, _, fake_model = _patched_runtime()
    # Helper calls the model's inference-switch method exactly once.
    assert fake_model.eval.call_count == 1


def test_runtime_raises_when_transformers_missing():
    with patch.object(ie_mod, "AutoModel", None), \
            patch.object(ie_mod, "AutoProcessor", None), \
            patch.object(ie_mod, "_ensure_deps", lambda: None), \
            patch.object(ie_mod, "torch", MagicMock()):
        with pytest.raises(RuntimeError, match="transformers"):
            ImageEmbeddingRuntime(
                model_id="m", hf_repo="org/repo", local_dir=None,
            )


def test_load_processor_falls_back_to_feature_extractor_on_failure():
    """Older repos: AutoProcessor raises; AutoFeatureExtractor takes over."""
    fake_processor_class = MagicMock()
    fake_processor_class.from_pretrained.side_effect = RuntimeError("no proc cfg")
    fake_extractor_class = MagicMock()
    fake_extractor_class.from_pretrained.return_value = "extractor"

    with patch.object(ie_mod, "AutoProcessor", fake_processor_class), \
            patch.object(ie_mod, "AutoFeatureExtractor", fake_extractor_class):
        out = _load_processor("acme/old-vit")
    assert out == "extractor"


def test_load_processor_re_raises_when_both_fail():
    fake_processor_class = MagicMock()
    fake_processor_class.from_pretrained.side_effect = RuntimeError("p")
    fake_extractor_class = MagicMock()
    fake_extractor_class.from_pretrained.side_effect = RuntimeError("e")

    with patch.object(ie_mod, "AutoProcessor", fake_processor_class), \
            patch.object(ie_mod, "AutoFeatureExtractor", fake_extractor_class):
        with pytest.raises(RuntimeError):
            _load_processor("acme/broken")


def test_load_processor_raises_when_extractor_missing():
    """If transformers isn't fully available, both classes may be None;
    AutoProcessor failing without an extractor fallback re-raises."""
    fake_processor_class = MagicMock()
    fake_processor_class.from_pretrained.side_effect = RuntimeError("p")
    with patch.object(ie_mod, "AutoProcessor", fake_processor_class), \
            patch.object(ie_mod, "AutoFeatureExtractor", None):
        with pytest.raises(RuntimeError):
            _load_processor("acme/x")


def test_embed_returns_image_embedding_result_for_dinov2():
    rt, *_ = _patched_runtime(arch="dinov2",
                              embedding_arr=[[0.1, 0.2, 0.3]])
    out = _patched_embed(rt, ["pretend-pil-image"])
    assert isinstance(out, ImageEmbeddingResult)
    assert out.model_id == "test"
    assert out.n_images == 1
    assert out.dimensions == 3
    assert len(out.embeddings) == 1
    assert len(out.embeddings[0]) == 3


def test_embed_clip_path_uses_image_embeds():
    rt, *_ = _patched_runtime(arch="clip",
                              embedding_arr=[[1.0, 2.0]])
    out = _patched_embed(rt, ["x"])
    # Float32 round-trip is exact for these integers.
    assert out.embeddings == [[1.0, 2.0]]


def test_embed_siglip_path_uses_pooler_output():
    rt, *_ = _patched_runtime(arch="siglip",
                              embedding_arr=[[3.0, 4.0]])
    out = _patched_embed(rt, ["x"])
    assert out.embeddings == [[3.0, 4.0]]


def test_embed_dinov2_path_uses_cls_token():
    """The CLS token slice is [:, 0]; verify the runtime extracts it."""
    rt, *_ = _patched_runtime(arch="dinov2",
                              embedding_arr=[[0.5, 0.6, 0.7]])
    out = _patched_embed(rt, ["x"])
    # float32 -> float64 conversion via .tolist() loses some precision;
    # compare element-wise with tolerance.
    assert len(out.embeddings) == 1
    assert len(out.embeddings[0]) == 3
    for got, want in zip(out.embeddings[0], [0.5, 0.6, 0.7]):
        assert got == pytest.approx(want, rel=1e-6)


def test_embed_wraps_single_image_into_list():
    """When the route accidentally passes a non-list image, we wrap it."""
    rt, _, fake_processor, _ = _patched_runtime(arch="siglip",
                                                embedding_arr=[[1.0]])
    _patched_embed(rt, "single-image-not-in-list")
    # Processor was called with images=[<that single image>]
    _, kwargs = fake_processor.call_args
    assert kwargs.get("images") == ["single-image-not-in-list"]


def test_embed_passes_list_to_processor_unchanged():
    rt, _, fake_processor, _ = _patched_runtime(arch="siglip",
                                                embedding_arr=[[1.0]])
    _patched_embed(rt, ["a", "b"])
    _, kwargs = fake_processor.call_args
    assert kwargs.get("images") == ["a", "b"]


def test_embed_passes_return_tensors_pt():
    rt, _, fake_processor, _ = _patched_runtime(arch="siglip",
                                                embedding_arr=[[1.0]])
    _patched_embed(rt, ["x"])
    _, kwargs = fake_processor.call_args
    assert kwargs.get("return_tensors") == "pt"


def test_embed_truncates_to_requested_dimensions():
    """Matryoshka: 384-dim native -> 128 requested -> output dim == 128."""
    raw = np.tile(np.arange(384, dtype=np.float32) / 384.0, (1, 1))
    rt, *_ = _patched_runtime(arch="siglip", embedding_arr=raw)
    out = _patched_embed(rt, ["x"], dimensions=128)
    assert out.dimensions == 128
    assert len(out.embeddings[0]) == 128


def test_embed_does_not_truncate_when_request_dim_geq_native():
    raw = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    rt, *_ = _patched_runtime(arch="siglip", embedding_arr=raw)
    out = _patched_embed(rt, ["x"], dimensions=10)
    assert out.dimensions == 3


def test_embed_metadata_marks_source_as_transformers():
    rt, *_ = _patched_runtime(arch="siglip", embedding_arr=[[1.0]])
    out = _patched_embed(rt, ["x"])
    assert out.metadata.get("source") == "transformers"


def test_embed_n_images_matches_input_length():
    rt, *_ = _patched_runtime(
        arch="siglip", embedding_arr=[[1.0], [2.0], [3.0]],
    )
    out = _patched_embed(rt, ["a", "b", "c"])
    assert out.n_images == 3
