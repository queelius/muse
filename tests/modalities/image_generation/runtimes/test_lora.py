"""Tests for the shared LoRA base resolution helper."""
from __future__ import annotations

import sys
import subprocess

import pytest

from muse.modalities.image_generation.runtimes._lora import resolve_lora_base


def test_missing_base_raises_actionable():
    with pytest.raises(RuntimeError) as e:
        resolve_lora_base("my-lora", None, resolve=lambda ref: ref)
    msg = str(e.value)
    assert msg.startswith("my-lora: lora_adapter is set but base_model is missing")
    assert "muse pull <id> --base <muse-id-or-hf-repo>" in msg
    with pytest.raises(RuntimeError, match="base_model is missing"):
        resolve_lora_base("my-lora", "", resolve=lambda ref: ref)


def test_unpulled_muse_id_raises_actionable():
    # resolve_model_source echoes an unresolved muse id back verbatim.
    with pytest.raises(RuntimeError) as e:
        resolve_lora_base("my-lora", "flux2-klein-4b", resolve=lambda ref: ref)
    assert str(e.value) == (
        "my-lora: LoRA base 'flux2-klein-4b' is not pulled (it may have been "
        "removed); run `muse pull flux2-klein-4b` and retry"
    )


def test_hf_repo_id_passes_through():
    assert resolve_lora_base(
        "my-lora", "black-forest-labs/FLUX.2-klein-4B", resolve=lambda ref: ref,
    ) == "black-forest-labs/FLUX.2-klein-4B"


def test_pulled_muse_id_maps_to_local_dir():
    assert resolve_lora_base(
        "my-lora", "sdxl-turbo", resolve={"sdxl-turbo": "/w/sdxl-turbo"}.get,
    ) == "/w/sdxl-turbo"


def test_default_resolver_reads_catalog(tmp_path, monkeypatch):
    from muse.core.catalog import _reset_read_catalog_cache, _write_catalog

    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    _write_catalog({"sdxl-turbo": {"local_dir": "/w/sdxl-turbo", "enabled": True}})
    _reset_read_catalog_cache()
    assert resolve_lora_base("my-lora", "sdxl-turbo") == "/w/sdxl-turbo"
    with pytest.raises(RuntimeError, match="is not pulled"):
        resolve_lora_base("my-lora", "flux2-klein-4b")


def test_module_is_import_light():
    code = (
        "import sys; import muse.modalities.image_generation.runtimes._lora; "
        "bad = [m for m in ('torch', 'diffusers', 'muse.core.catalog') "
        "if m in sys.modules]; print(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "[]"


# ---------------- rsLoRA scale-factor derivation from the adapter header ----


import json
import logging
import math
import struct

from muse.modalities.image_generation.runtimes._lora import (
    adapter_weight_file,
    derive_lora_scale_factor,
    read_safetensors_header,
    resolve_lora_scale_factor,
    rslora_scale_factor,
)


def _write_safetensors(path, tensors, metadata=None):
    """A real (tiny) safetensors file: 8-byte LE header length, JSON header,
    then the tensor bytes. Every tensor is a 1-element F32."""
    header: dict = {}
    offset = 0
    for name in tensors:
        header[name] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        offset += 4
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)
    return path


def _adapter_md(prefix="transformer", **cfg):
    """diffusers' lora_adapter_metadata: one JSON string, keys prefixed by
    the component the config belongs to (save_lora_weights packs them)."""
    return {"format": "pt", "lora_adapter_metadata": json.dumps(
        {f"{prefix}.{k}": v for k, v in cfg.items()}
    )}


_KLEIN_PIXEL_ART_CFG = dict(
    r=64, lora_alpha=128, use_rslora=True, rank_pattern={}, alpha_pattern={},
    target_modules=["to_k", "to_q", "to_v"], lora_dropout=0.0,
)
# The exact `lora_adapter_metadata` of Limbicnation/pixel-art-lora
# (pytorch_lora_weights.safetensors @ 0ac8e5c3, read from its header: 172
# transformer.* tensors, no .alpha tensors). curated klein-pixel-art.
_LIMBICNATION_ADAPTER_METADATA = '{"transformer.alora_invocation_tokens": null, "transformer.alpha_pattern": {}, "transformer.arrow_config": null, "transformer.auto_mapping": null, "transformer.base_model_name_or_path": null, "transformer.bias": "none", "transformer.corda_config": null, "transformer.ensure_weight_tying": false, "transformer.eva_config": null, "transformer.exclude_modules": null, "transformer.fan_in_fan_out": false, "transformer.inference_mode": false, "transformer.init_lora_weights": true, "transformer.layer_replication": null, "transformer.layers_pattern": null, "transformer.layers_to_transform": null, "transformer.loftq_config": {}, "transformer.lora_alpha": 128, "transformer.lora_bias": false, "transformer.lora_dropout": 0.0, "transformer.megatron_config": null, "transformer.megatron_core": "megatron.core", "transformer.modules_to_save": null, "transformer.peft_type": "LORA", "transformer.peft_version": "0.18.1", "transformer.qalora_group_size": 16, "transformer.r": 64, "transformer.rank_pattern": {}, "transformer.revision": null, "transformer.target_modules": ["linear_out", "to_qkv_mlp_proj", "add_q_proj", "context_embedder", "single_stream_modulation.linear", "proj_out", "double_stream_modulation_img.linear", "add_k_proj", "double_stream_modulation_txt.linear", "to_k", "to_q", "to_v", "to_add_out", "linear_in", "to_out.0", "add_v_proj", "x_embedder"], "transformer.target_parameters": null, "transformer.task_type": null, "transformer.trainable_token_indices": null, "transformer.use_dora": false, "transformer.use_qalora": false, "transformer.use_rslora": true}'
_LORA_KEYS = [
    "transformer.single_transformer_blocks.0.attn.to_q.lora_A.weight",
    "transformer.single_transformer_blocks.0.attn.to_q.lora_B.weight",
]


def test_header_reader_reads_metadata_without_tensor_bytes(tmp_path):
    p = _write_safetensors(tmp_path / "a.safetensors", _LORA_KEYS,
                           _adapter_md(**_KLEIN_PIXEL_ART_CFG))
    h = read_safetensors_header(p)
    assert set(h) == {*_LORA_KEYS, "__metadata__"}
    assert h["__metadata__"]["format"] == "pt"


def test_klein_pixel_art_metadata_shape_derives_0_0625(tmp_path):
    # Limbicnation/pixel-art-lora: use_rslora, r=64, alpha=128, no .alpha
    # tensors. diffusers scales by alpha/sqrt(r) = 16; the factor undoes it,
    # and matches the value the curated entry keeps explicit.
    from muse.core.curated import find_curated

    p = _write_safetensors(
        tmp_path / "pytorch_lora_weights.safetensors", _LORA_KEYS,
        {"format": "pt", "lora_adapter_metadata": _LIMBICNATION_ADAPTER_METADATA},
    )
    assert rslora_scale_factor(read_safetensors_header(p)) == 0.0625
    assert derive_lora_scale_factor(tmp_path, model_id="x") == 0.0625
    curated = find_curated("klein-pixel-art").capabilities["lora_scale_factor"]
    assert curated == 0.0625 == derive_lora_scale_factor(tmp_path, model_id="x")


def test_rslora_factor_is_sqrt_r_over_alpha(tmp_path):
    p = _write_safetensors(tmp_path / "a.safetensors", _LORA_KEYS,
                           _adapter_md(r=16, lora_alpha=16, use_rslora=True))
    assert rslora_scale_factor(read_safetensors_header(p)) == pytest.approx(0.25)


def test_non_rslora_metadata_gives_1(tmp_path):
    p = _write_safetensors(tmp_path / "a.safetensors", _LORA_KEYS,
                           _adapter_md(r=64, lora_alpha=128, use_rslora=False))
    assert rslora_scale_factor(read_safetensors_header(p)) == 1.0


def test_no_adapter_metadata_gives_1(tmp_path):
    p = _write_safetensors(tmp_path / "a.safetensors", _LORA_KEYS, {"format": "pt"})
    assert rslora_scale_factor(read_safetensors_header(p)) == 1.0
    p2 = _write_safetensors(tmp_path / "b.safetensors", _LORA_KEYS)
    assert rslora_scale_factor(read_safetensors_header(p2)) == 1.0


def test_alpha_tensors_give_1(tmp_path):
    # kohya-style .alpha tensors: the alpha is per module in the weights,
    # not the metadata's rsLoRA scaling, so no derived correction.
    p = _write_safetensors(
        tmp_path / "a.safetensors",
        [*_LORA_KEYS, "transformer.single_transformer_blocks.0.attn.to_q.alpha"],
        _adapter_md(**_KLEIN_PIXEL_ART_CFG),
    )
    assert rslora_scale_factor(read_safetensors_header(p)) == 1.0


def test_unprefixed_metadata_keys_give_1(tmp_path):
    # diffusers keeps only `<component>.` keys, so bare keys never reach peft.
    md = {"lora_adapter_metadata": json.dumps(
        {"r": 64, "lora_alpha": 128, "use_rslora": True})}
    p = _write_safetensors(tmp_path / "a.safetensors", _LORA_KEYS, md)
    assert rslora_scale_factor(read_safetensors_header(p)) == 1.0


def test_components_that_agree_share_the_factor(tmp_path):
    cfg = {"r": 64, "lora_alpha": 128, "use_rslora": True}
    md = {"lora_adapter_metadata": json.dumps({
        **{f"unet.{k}": v for k, v in cfg.items()},
        **{f"text_encoder.{k}": v for k, v in cfg.items()},
    })}
    p = _write_safetensors(tmp_path / "a.safetensors", _LORA_KEYS, md)
    assert rslora_scale_factor(read_safetensors_header(p)) == 0.0625


@pytest.mark.parametrize("md", [
    # Components disagree: one scalar cannot normalize both.
    {"lora_adapter_metadata": json.dumps({
        "unet.r": 64, "unet.lora_alpha": 128, "unet.use_rslora": True,
        "text_encoder.r": 8, "text_encoder.lora_alpha": 8,
        "text_encoder.use_rslora": False,
    })},
    # Per-module rank/alpha patterns: likewise no single factor.
    _adapter_md(r=64, lora_alpha=128, use_rslora=True,
                alpha_pattern={"blocks.0.to_q": 32}),
    # Unusable numbers.
    _adapter_md(r=0, lora_alpha=128, use_rslora=True),
    _adapter_md(r=64, lora_alpha="x", use_rslora=True),
    _adapter_md(r=64, lora_alpha=True, use_rslora=True),
    _adapter_md(r=64, lora_alpha=float("inf"), use_rslora=True),
    # Not JSON / not an object.
    {"lora_adapter_metadata": "{not json"},
    {"lora_adapter_metadata": "[1, 2]"},
], ids=["disagree", "pattern", "r0", "alpha-str", "alpha-bool", "alpha-inf",
        "bad-json", "not-object"])
def test_ambiguous_or_bad_metadata_gives_1_with_warning(tmp_path, caplog, md):
    p = _write_safetensors(tmp_path / "pytorch_lora_weights.safetensors", _LORA_KEYS, md)
    with caplog.at_level(logging.WARNING):
        assert derive_lora_scale_factor(tmp_path, model_id="m") == 1.0
    assert any(r.levelno == logging.WARNING and "m:" in r.getMessage()
               for r in caplog.records)


@pytest.mark.parametrize("payload", [
    b"",                                      # empty file
    b"\x01\x02",                              # shorter than the length prefix
    struct.pack("<Q", 10**12) + b"{}",        # length past EOF
    struct.pack("<Q", 5) + b"{oops",          # not JSON
    struct.pack("<Q", 2) + b"[]",             # JSON, not an object
    struct.pack("<Q", 4) + b"\xff\xfe{}",     # not UTF-8
    struct.pack("<Q", 21) + b'{"__metadata__": [1]}',  # metadata not a dict
], ids=["empty", "short", "past-eof", "not-json", "array", "not-utf8", "md-list"])
def test_corrupt_header_gives_1_with_warning_never_raises(tmp_path, caplog, payload):
    (tmp_path / "pytorch_lora_weights.safetensors").write_bytes(payload)
    with caplog.at_level(logging.WARNING):
        assert derive_lora_scale_factor(tmp_path, model_id="m") == 1.0
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_header_reader_raises_valueerror_on_corruption(tmp_path):
    p = tmp_path / "x.safetensors"
    p.write_bytes(struct.pack("<Q", 5) + b"{oops")
    with pytest.raises(ValueError):
        read_safetensors_header(p)


def test_header_reader_refuses_oversized_header(tmp_path):
    # An adapter header is tens of KB (Limbicnation's is 25 KB); a length
    # prefix over 16 MB is corruption, rejected before reading it.
    p = tmp_path / "x.safetensors"
    with open(p, "wb") as f:
        f.write(struct.pack("<Q", 16 * 1024 * 1024 + 1))
        f.seek(16 * 1024 * 1024 + 16)
        f.write(b"\0")
    with pytest.raises(ValueError, match="exceeds 16 MB"):
        read_safetensors_header(p)


def test_adapter_weight_file_mirrors_diffusers_choice(tmp_path):
    assert adapter_weight_file(tmp_path) is None
    one = _write_safetensors(tmp_path / "lora.safetensors", _LORA_KEYS)
    assert adapter_weight_file(tmp_path) == one
    # diffusers skips optimizer/scheduler/checkpoint files.
    _write_safetensors(tmp_path / "optimizer.safetensors", _LORA_KEYS)
    assert adapter_weight_file(tmp_path) == one
    # ... and prefers its own weight name over any other file.
    pref = _write_safetensors(tmp_path / "pytorch_lora_weights.safetensors", _LORA_KEYS)
    assert adapter_weight_file(tmp_path) == pref
    # A direct file path is used as is.
    assert adapter_weight_file(one) == one


def test_adapter_weight_file_ambiguous_dir_is_none(tmp_path):
    _write_safetensors(tmp_path / "a.safetensors", _LORA_KEYS)
    _write_safetensors(tmp_path / "b.safetensors", _LORA_KEYS)
    assert adapter_weight_file(tmp_path) is None


def test_derive_without_local_file_gives_1(tmp_path):
    # A bare HF repo id (not a local path) is never fetched to derive.
    assert derive_lora_scale_factor("org/some-lora", model_id="m") == 1.0
    assert derive_lora_scale_factor(None, model_id="m") == 1.0
    assert derive_lora_scale_factor(tmp_path, model_id="m") == 1.0


def test_derive_logs_derived_factor_at_info(tmp_path, caplog):
    _write_safetensors(tmp_path / "pytorch_lora_weights.safetensors",
                       _LORA_KEYS, _adapter_md(**_KLEIN_PIXEL_ART_CFG))
    with caplog.at_level(logging.INFO, logger="muse.modalities.image_generation.runtimes._lora"):
        derive_lora_scale_factor(tmp_path, model_id="klein-lora")
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(msgs) == 1
    assert "klein-lora" in msgs[0] and "0.0625" in msgs[0]


def test_lora_scale_factor_explicit_wins(tmp_path):
    _write_safetensors(tmp_path / "pytorch_lora_weights.safetensors",
                       _LORA_KEYS, _adapter_md(**_KLEIN_PIXEL_ART_CFG))
    assert resolve_lora_scale_factor("m", tmp_path, explicit=0.5) == 0.5
    assert resolve_lora_scale_factor("m", tmp_path, explicit=1) == 1.0
    assert resolve_lora_scale_factor("m", tmp_path, explicit=None) == 0.0625


@pytest.mark.parametrize("bad", [0, -0.0625, float("nan"), float("inf"), "x", True])
def test_lora_scale_factor_explicit_must_be_finite_positive(tmp_path, bad):
    with pytest.raises(ValueError, match="lora_scale_factor"):
        resolve_lora_scale_factor("m", tmp_path, explicit=bad)
