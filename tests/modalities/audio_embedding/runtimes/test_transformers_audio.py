"""Tests for AudioEmbeddingRuntime (transformers AutoModel + librosa wrapper).

The per-architecture _extract_embeddings dispatch is exercised
explicitly with mocked outputs that match each shape (CLAP-style
audio_embeds, pooler_output, MERT-style last_hidden_state mean-pool).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import muse.modalities.audio_embedding.runtimes.transformers_audio as ae_mod
from muse.modalities.audio_embedding.protocol import AudioEmbeddingResult
from muse.modalities.audio_embedding.runtimes.transformers_audio import (
    AudioEmbeddingRuntime,
    _decode_audio,
    _detect_dimensions,
    _extract_embeddings,
    _load_processor,
    _resolve_dtype,
    _select_device,
    _set_inference_mode,
)


def _np_tensor(arr):
    """Wrap a numpy array in a MagicMock that mimics a torch tensor.

    The runtime calls .detach().to("cpu").float().numpy() to lift
    values back to numpy; chained MagicMocks would all return
    MagicMock, so we set up explicit return_value links.
    """
    arr = np.asarray(arr, dtype=np.float32)
    t = MagicMock()
    t.detach.return_value = t
    t.to.return_value = t
    t.float.return_value = t
    t.numpy.return_value = arr
    t.shape = arr.shape
    return t


def _make_inputs():
    inputs = MagicMock()
    inputs.to.return_value = inputs
    return inputs


def _patched_runtime(
    *,
    embedding_arr=None,
    arch="mert",
    dim_attr="hidden_size",
    dim_val=768,
    device="cpu",
    sample_rate=24000,
    trust_remote_code=False,
    dimensions=None,
):
    """Build a fully-mocked AudioEmbeddingRuntime ready for embed() calls.

    `arch` selects which output attribute carries embeddings:
      - "clap": outputs.audio_embeds set
      - "pooler": outputs.pooler_output set, no audio_embeds
      - "mert": outputs.last_hidden_state set, mean-pool over time
    """
    if embedding_arr is None:
        embedding_arr = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    embeddings_tensor = _np_tensor(embedding_arr)

    class FakeOutputs:
        audio_embeds = None
        pooler_output = None
        last_hidden_state = None

    fake_outputs = FakeOutputs()
    if arch == "clap":
        fake_outputs.audio_embeds = embeddings_tensor
    elif arch == "pooler":
        fake_outputs.pooler_output = embeddings_tensor
    elif arch == "mert":
        # last_hidden_state shape [B, T, H]; mean(dim=1) returns the
        # extracted tensor. The mock's .mean() returns the wrapped
        # numpy array.
        seq = MagicMock()
        seq.mean = MagicMock(return_value=_np_tensor(embedding_arr))
        fake_outputs.last_hidden_state = seq

    fake_model = MagicMock()
    fake_model.return_value = fake_outputs
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = None
    fake_model.config = SimpleNamespace(**{dim_attr: dim_val})
    fake_model_class = MagicMock()
    fake_model_class.from_pretrained.return_value = fake_model

    fake_processor = MagicMock()
    fake_inputs = _make_inputs()
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

    fake_librosa = MagicMock()
    # librosa.load(BytesIO, sr=..., mono=True) -> (np.ndarray, sr)
    fake_librosa.load = MagicMock(
        return_value=(np.zeros(sample_rate, dtype=np.float32), sample_rate),
    )

    with patch.object(ae_mod, "AutoModel", fake_model_class), \
            patch.object(ae_mod, "AutoProcessor", fake_processor_class), \
            patch.object(ae_mod, "AutoFeatureExtractor", MagicMock()), \
            patch.object(ae_mod, "torch", fake_torch), \
            patch.object(ae_mod, "librosa", fake_librosa):
        kwargs = dict(
            model_id="test",
            hf_repo="org/repo",
            local_dir=None,
            device=device,
            dtype="float32",
            sample_rate=sample_rate,
            trust_remote_code=trust_remote_code,
        )
        if dimensions is not None:
            kwargs["dimensions"] = dimensions
        rt = AudioEmbeddingRuntime(**kwargs)
        rt._fake_torch = fake_torch
        rt._fake_processor = fake_processor
        rt._fake_inputs = fake_inputs
        rt._fake_model = fake_model
        rt._fake_librosa = fake_librosa
    return rt, fake_torch, fake_processor, fake_model, fake_librosa


def _patched_embed(rt, *args, **kwargs):
    """Call rt.embed(...) with the same patches the constructor used."""
    with patch.object(ae_mod, "torch", rt._fake_torch), \
            patch.object(ae_mod, "librosa", rt._fake_librosa):
        return rt.embed(*args, **kwargs)


def test_select_device_auto_falls_back_to_cpu_with_no_torch():
    with patch.object(ae_mod, "torch", None):
        assert _select_device("auto") == "cpu"


def test_select_device_auto_picks_cuda():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    with patch.object(ae_mod, "torch", fake_torch):
        assert _select_device("auto") == "cuda"


def test_select_device_auto_picks_mps_when_cuda_absent():
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = True
    with patch.object(ae_mod, "torch", fake_torch):
        assert _select_device("auto") == "mps"


def test_select_device_explicit_passes_through():
    with patch.object(ae_mod, "torch", MagicMock()):
        assert _select_device("cuda:1") == "cuda:1"


def test_resolve_dtype_returns_none_without_torch():
    with patch.object(ae_mod, "torch", None):
        assert _resolve_dtype("float16") is None


def test_resolve_dtype_maps_known_strings():
    fake_torch = MagicMock()
    fake_torch.float16 = "FP16"
    fake_torch.bfloat16 = "BF16"
    fake_torch.float32 = "FP32"
    with patch.object(ae_mod, "torch", fake_torch):
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
    assert m.eval.call_count == 1


def test_set_inference_mode_safe_when_method_missing():
    class NoEval:
        pass

    # Must not raise.
    _set_inference_mode(NoEval())


def test_extract_embeddings_clap_branch_returns_audio_embeds():
    """CLAP path: outputs.audio_embeds is set; pooler_output is also
    present but lower priority. audio_embeds wins."""
    expected = MagicMock(name="audio_embeds")
    other = MagicMock(name="pooler_output")
    outputs = SimpleNamespace(
        audio_embeds=expected,
        pooler_output=other,
        last_hidden_state=MagicMock(),
    )
    assert _extract_embeddings(outputs) is expected


def test_extract_embeddings_pooler_branch_returns_pooler_output():
    """Pooler path: outputs.pooler_output set, audio_embeds absent."""
    expected = MagicMock(name="pooler_output")
    outputs = SimpleNamespace(
        audio_embeds=None,
        pooler_output=expected,
        last_hidden_state=MagicMock(),
    )
    assert _extract_embeddings(outputs) is expected


def test_extract_embeddings_mert_branch_uses_mean_pool():
    """MERT path: only last_hidden_state is set. Mean-pool over dim=1."""
    expected = MagicMock(name="mean_pooled")
    seq = MagicMock(name="last_hidden_state")
    seq.mean = MagicMock(return_value=expected)
    outputs = SimpleNamespace(
        audio_embeds=None,
        pooler_output=None,
        last_hidden_state=seq,
    )
    out = _extract_embeddings(outputs)
    assert out is expected
    seq.mean.assert_called_once_with(dim=1)


def test_extract_embeddings_raises_when_no_path_matches():
    outputs = SimpleNamespace(
        audio_embeds=None, pooler_output=None, last_hidden_state=None,
    )
    with pytest.raises(ValueError, match="could not extract embeddings"):
        _extract_embeddings(outputs)


def test_detect_dimensions_uses_projection_dim_first():
    """CLAP-style projection_dim wins over hidden_size."""
    model = SimpleNamespace(
        config=SimpleNamespace(projection_dim=512, hidden_size=768),
    )
    assert _detect_dimensions(model) == 512


def test_detect_dimensions_falls_back_to_hidden_size():
    """When no projection_dim, hidden_size (MERT, wav2vec) is used."""
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=768),
    )
    assert _detect_dimensions(model) == 768


def test_detect_dimensions_uses_audio_config_hidden_size():
    """Composite CLAP-shaped configs nest hidden_size under audio_config."""
    cfg = SimpleNamespace(
        projection_dim=None,
        hidden_size=None,
        audio_config=SimpleNamespace(hidden_size=768),
    )
    model = SimpleNamespace(config=cfg)
    assert _detect_dimensions(model) == 768


def test_detect_dimensions_returns_minus_one_when_unknowable():
    """Sentinel for the rare repo with no recognizable dim attribute."""
    model = SimpleNamespace(config=None)
    assert _detect_dimensions(model) == -1


def test_decode_audio_resamples_to_requested_rate():
    fake_librosa = MagicMock()
    fake_librosa.load = MagicMock(
        return_value=(np.array([0.1, 0.2, 0.3], dtype=np.float32), 24000),
    )
    with patch.object(ae_mod, "librosa", fake_librosa):
        out = _decode_audio(b"raw", sample_rate=24000, max_seconds=60.0)
    fake_librosa.load.assert_called_once()
    _, kwargs = fake_librosa.load.call_args
    assert kwargs["sr"] == 24000
    assert kwargs["mono"] is True
    np.testing.assert_array_equal(out, np.array([0.1, 0.2, 0.3], dtype=np.float32))


def test_decode_audio_truncates_to_max_seconds():
    """Max_seconds * sample_rate samples is the upper bound."""
    long = np.arange(48000, dtype=np.float32)  # 2 seconds at 24kHz
    fake_librosa = MagicMock()
    fake_librosa.load = MagicMock(return_value=(long, 24000))
    with patch.object(ae_mod, "librosa", fake_librosa):
        out = _decode_audio(b"raw", sample_rate=24000, max_seconds=1.0)
    assert len(out) == 24000  # 1 second's worth


def test_decode_audio_returns_float32():
    long = np.arange(100, dtype=np.float64)
    fake_librosa = MagicMock()
    fake_librosa.load = MagicMock(return_value=(long, 16000))
    with patch.object(ae_mod, "librosa", fake_librosa):
        out = _decode_audio(b"raw", sample_rate=16000, max_seconds=60.0)
    assert out.dtype == np.float32


def test_runtime_constructs_with_local_dir_preference():
    fake_processor_class = MagicMock()
    fake_processor_class.from_pretrained.return_value = MagicMock()
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.config = SimpleNamespace(hidden_size=768)
    fake_model_class = MagicMock()
    fake_model_class.from_pretrained.return_value = fake_model
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    fake_librosa = MagicMock()

    with patch.object(ae_mod, "AutoModel", fake_model_class), \
            patch.object(ae_mod, "AutoProcessor", fake_processor_class), \
            patch.object(ae_mod, "torch", fake_torch), \
            patch.object(ae_mod, "librosa", fake_librosa):
        AudioEmbeddingRuntime(
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
    fake_model.config = SimpleNamespace(hidden_size=768)
    fake_model_class = MagicMock()
    fake_model_class.from_pretrained.return_value = fake_model
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    fake_librosa = MagicMock()

    with patch.object(ae_mod, "AutoModel", fake_model_class), \
            patch.object(ae_mod, "AutoProcessor", fake_processor_class), \
            patch.object(ae_mod, "torch", fake_torch), \
            patch.object(ae_mod, "librosa", fake_librosa):
        AudioEmbeddingRuntime(
            model_id="m",
            hf_repo="m-a-p/MERT-v1-95M",
            local_dir=None,
            device="cpu",
        )
    args, _ = fake_processor_class.from_pretrained.call_args
    assert args[0] == "m-a-p/MERT-v1-95M"


def test_runtime_loads_dimensions_from_model_config():
    """AudioEmbeddingRuntime.dimensions comes from model.config detection."""
    rt, *_ = _patched_runtime(dim_attr="hidden_size", dim_val=768)
    assert rt.dimensions == 768


def test_runtime_dimensions_override_via_kwarg():
    """Manifest capabilities can pin dimensions explicitly."""
    rt, *_ = _patched_runtime(dim_attr="hidden_size", dim_val=768, dimensions=512)
    assert rt.dimensions == 512


def test_runtime_calls_to_with_device():
    rt, _, _, fake_model, _ = _patched_runtime(device="cpu")
    fake_model.to.assert_called_with("cpu")


def test_runtime_invokes_inference_mode_helper():
    rt, _, _, fake_model, _ = _patched_runtime()
    # Helper calls the model's inference-switch method exactly once.
    assert fake_model.eval.call_count == 1


def test_runtime_raises_when_transformers_missing():
    with patch.object(ae_mod, "AutoModel", None), \
            patch.object(ae_mod, "AutoProcessor", None), \
            patch.object(ae_mod, "_ensure_deps", lambda: None), \
            patch.object(ae_mod, "torch", MagicMock()), \
            patch.object(ae_mod, "librosa", MagicMock()):
        with pytest.raises(RuntimeError, match="transformers"):
            AudioEmbeddingRuntime(
                model_id="m", hf_repo="org/repo", local_dir=None,
            )


def test_runtime_raises_when_librosa_missing():
    with patch.object(ae_mod, "AutoModel", MagicMock()), \
            patch.object(ae_mod, "AutoProcessor", MagicMock()), \
            patch.object(ae_mod, "_ensure_deps", lambda: None), \
            patch.object(ae_mod, "torch", MagicMock()), \
            patch.object(ae_mod, "librosa", None):
        with pytest.raises(RuntimeError, match="librosa"):
            AudioEmbeddingRuntime(
                model_id="m", hf_repo="org/repo", local_dir=None,
            )


def test_runtime_threads_trust_remote_code_to_processor():
    fake_processor_class = MagicMock()
    fake_processor_class.from_pretrained.return_value = MagicMock()
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.config = SimpleNamespace(hidden_size=768)
    fake_model_class = MagicMock()
    fake_model_class.from_pretrained.return_value = fake_model
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.backends.mps.is_available.return_value = False
    fake_librosa = MagicMock()

    with patch.object(ae_mod, "AutoModel", fake_model_class), \
            patch.object(ae_mod, "AutoProcessor", fake_processor_class), \
            patch.object(ae_mod, "torch", fake_torch), \
            patch.object(ae_mod, "librosa", fake_librosa):
        AudioEmbeddingRuntime(
            model_id="m",
            hf_repo="m-a-p/MERT-v1-95M",
            local_dir=None,
            device="cpu",
            trust_remote_code=True,
        )
    _, kwargs = fake_processor_class.from_pretrained.call_args
    assert kwargs.get("trust_remote_code") is True
    _, mkwargs = fake_model_class.from_pretrained.call_args
    assert mkwargs.get("trust_remote_code") is True


def test_load_processor_falls_back_to_feature_extractor_on_failure():
    """Audio repos: AutoProcessor often raises; AutoFeatureExtractor takes over."""
    fake_processor_class = MagicMock()
    fake_processor_class.from_pretrained.side_effect = RuntimeError("no proc cfg")
    fake_extractor_class = MagicMock()
    fake_extractor_class.from_pretrained.return_value = "extractor"

    with patch.object(ae_mod, "AutoProcessor", fake_processor_class), \
            patch.object(ae_mod, "AutoFeatureExtractor", fake_extractor_class):
        out = _load_processor("acme/audio-encoder")
    assert out == "extractor"


def test_load_processor_re_raises_when_both_fail():
    fake_processor_class = MagicMock()
    fake_processor_class.from_pretrained.side_effect = RuntimeError("p")
    fake_extractor_class = MagicMock()
    fake_extractor_class.from_pretrained.side_effect = RuntimeError("e")

    with patch.object(ae_mod, "AutoProcessor", fake_processor_class), \
            patch.object(ae_mod, "AutoFeatureExtractor", fake_extractor_class):
        with pytest.raises(RuntimeError):
            _load_processor("acme/broken")


def test_load_processor_raises_when_extractor_missing():
    """If transformers isn't fully available, both classes may be None;
    AutoProcessor failing without an extractor fallback re-raises."""
    fake_processor_class = MagicMock()
    fake_processor_class.from_pretrained.side_effect = RuntimeError("p")
    with patch.object(ae_mod, "AutoProcessor", fake_processor_class), \
            patch.object(ae_mod, "AutoFeatureExtractor", None):
        with pytest.raises(RuntimeError):
            _load_processor("acme/x")


def test_embed_returns_audio_embedding_result_for_mert():
    rt, *_ = _patched_runtime(arch="mert", embedding_arr=[[0.1, 0.2, 0.3]])
    out = _patched_embed(rt, [b"FAKEWAV"])
    assert isinstance(out, AudioEmbeddingResult)
    assert out.model_id == "test"
    assert out.n_audio_clips == 1
    assert out.dimensions == 3
    assert len(out.embeddings) == 1
    assert len(out.embeddings[0]) == 3


def test_embed_clap_path_uses_audio_embeds():
    rt, *_ = _patched_runtime(arch="clap", embedding_arr=[[1.0, 2.0]])
    out = _patched_embed(rt, [b"FAKEWAV"])
    assert out.embeddings == [[1.0, 2.0]]


def test_embed_pooler_path_uses_pooler_output():
    rt, *_ = _patched_runtime(arch="pooler", embedding_arr=[[3.0, 4.0]])
    out = _patched_embed(rt, [b"FAKEWAV"])
    assert out.embeddings == [[3.0, 4.0]]


def test_embed_mert_path_uses_mean_pool():
    """MERT mean-pools last_hidden_state over the time dim."""
    rt, *_ = _patched_runtime(arch="mert", embedding_arr=[[0.5, 0.6, 0.7]])
    out = _patched_embed(rt, [b"FAKEWAV"])
    assert len(out.embeddings) == 1
    assert len(out.embeddings[0]) == 3
    for got, want in zip(out.embeddings[0], [0.5, 0.6, 0.7]):
        assert got == pytest.approx(want, rel=1e-6)


def test_embed_wraps_single_input_into_list():
    """When the route accidentally passes a non-list bytes, we wrap it."""
    rt, _, fake_processor, _, _ = _patched_runtime(
        arch="pooler", embedding_arr=[[1.0]],
    )
    _patched_embed(rt, b"single-not-in-list")
    args, kwargs = fake_processor.call_args
    # Processor was called with [<single decoded clip>]
    assert isinstance(args[0], list)
    assert len(args[0]) == 1


def test_embed_passes_list_to_processor_unchanged_length():
    rt, _, fake_processor, _, _ = _patched_runtime(
        arch="pooler", embedding_arr=[[1.0]],
    )
    _patched_embed(rt, [b"a", b"b"])
    args, _ = fake_processor.call_args
    assert isinstance(args[0], list)
    assert len(args[0]) == 2


def test_embed_passes_return_tensors_pt():
    rt, _, fake_processor, _, _ = _patched_runtime(
        arch="pooler", embedding_arr=[[1.0]],
    )
    _patched_embed(rt, [b"FAKEWAV"])
    _, kwargs = fake_processor.call_args
    assert kwargs.get("return_tensors") == "pt"


def test_embed_passes_sampling_rate_to_processor():
    rt, _, fake_processor, _, _ = _patched_runtime(
        arch="pooler", embedding_arr=[[1.0]], sample_rate=24000,
    )
    _patched_embed(rt, [b"FAKEWAV"])
    _, kwargs = fake_processor.call_args
    assert kwargs.get("sampling_rate") == 24000


def test_embed_metadata_marks_source_as_transformers():
    rt, *_ = _patched_runtime(arch="pooler", embedding_arr=[[1.0]])
    out = _patched_embed(rt, [b"FAKEWAV"])
    assert out.metadata.get("source") == "transformers"


def test_embed_metadata_records_sample_rate_used():
    rt, *_ = _patched_runtime(
        arch="pooler", embedding_arr=[[1.0]], sample_rate=48000,
    )
    out = _patched_embed(rt, [b"FAKEWAV"])
    assert out.metadata.get("sample_rate_used") == 48000


def test_embed_n_audio_clips_matches_input_length():
    rt, *_ = _patched_runtime(
        arch="pooler", embedding_arr=[[1.0], [2.0], [3.0]],
    )
    out = _patched_embed(rt, [b"a", b"b", b"c"])
    assert out.n_audio_clips == 3


def test_embed_calls_librosa_load_per_clip():
    """One librosa.load call per input clip."""
    rt, *_ = _patched_runtime(
        arch="pooler", embedding_arr=[[1.0], [2.0]],
    )
    _patched_embed(rt, [b"a", b"b"])
    assert rt._fake_librosa.load.call_count == 2
