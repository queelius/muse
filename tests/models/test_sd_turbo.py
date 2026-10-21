"""Tests for the SD-Turbo model script (fully mocked; no weights loaded).

The script aliases the generic `DiffusersText2ImageModel` as its
`Model`; runtime behavior (img2img, inpaint, variations, step bumping,
from_pipe VRAM sharing) is covered by tests/modalities/image_generation/
runtimes/test_diffusers.py. These tests pin the MANIFEST and check that
the runtime honors it when constructed the way `load_backend` does
(`model_id` + capabilities splatted as kwargs), plus the lazy-build
thread safety and pipeline retention the runtime inherited from this
script.
"""
from unittest.mock import MagicMock, patch

from muse.modalities.image_generation.protocol import ImageResult

_RT = "muse.modalities.image_generation.runtimes.diffusers"


def _patched_pipe():
    fake_pipe = MagicMock()
    fake_image = MagicMock()
    fake_image.size = (512, 512)
    fake_pipe.return_value.images = [fake_image]
    return fake_pipe


def _make(local_dir="/fake", **overrides):
    """Construct the bundled Model the way load_backend does."""
    from muse.models.sd_turbo import MANIFEST, Model
    kwargs = {
        "model_id": MANIFEST["model_id"],
        **MANIFEST["capabilities"],
        **overrides,
    }
    return Model(hf_repo=MANIFEST["hf_repo"], local_dir=local_dir, **kwargs)


def test_manifest_has_required_fields():
    from muse.models.sd_turbo import MANIFEST
    assert MANIFEST["model_id"] == "sd-turbo"
    assert MANIFEST["modality"] == "image/generation"
    assert "hf_repo" in MANIFEST
    assert "pip_extras" in MANIFEST


def test_manifest_pins_cuda_device_not_auto():
    """sd-turbo's allow_patterns download ONLY *.fp16.safetensors -- no
    fp32 weights exist on disk. A device:auto resolution on a CPU-only
    host would load an fp16 pipeline that errors at inference. Pin
    device to 'cuda' so a CPU-only host reports the model unservable
    instead of silently 500ing on first request."""
    from muse.models.sd_turbo import MANIFEST
    assert MANIFEST["capabilities"]["device"] == "cuda"


def test_manifest_pip_extras_declares_transformers_and_torch():
    """Regression (v0.12.1): a venv without transformers exited at load
    time because StableDiffusionPipeline needs the CLIP text encoder."""
    from muse.models.sd_turbo import MANIFEST
    extras_str = " ".join(MANIFEST["pip_extras"])
    assert "transformers" in extras_str
    assert "torch" in extras_str


def test_manifest_advertises_image_capabilities():
    from muse.models.sd_turbo import MANIFEST
    caps = MANIFEST["capabilities"]
    assert caps["supports_img2img"] is True
    assert caps["supports_inpainting"] is True
    assert caps["supports_variations"] is True
    assert caps["default_steps"] == 1
    assert caps["default_guidance"] == 0.0


def test_model_is_diffusers_runtime():
    from muse.models.sd_turbo import Model
    from muse.modalities.image_generation.runtimes.diffusers import (
        DiffusersText2ImageModel,
    )
    assert Model is DiffusersText2ImageModel


def test_manifest_drives_construction_and_one_step_defaults(tmp_path):
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"")
    fake_t2i = MagicMock()
    pipe = _patched_pipe()
    fake_t2i.from_pretrained.return_value = pipe

    with patch(f"{_RT}.AutoPipelineForText2Image", fake_t2i), \
            patch(f"{_RT}.torch", MagicMock()):
        m = _make(local_dir=str(tmp_path))
        result = m.generate("a cat on mars", seed=42)

    assert m.model_id == "sd-turbo"
    assert m.default_size == (512, 512)
    load = fake_t2i.from_pretrained.call_args
    assert load.args[0] == str(tmp_path)
    assert load.kwargs["variant"] == "fp16"  # snapshot holds only fp16
    pipe.to.assert_called_once_with("cuda")
    call = pipe.call_args.kwargs
    assert call["num_inference_steps"] == 1
    assert call["guidance_scale"] == 0.0
    assert (call["width"], call["height"]) == (512, 512)
    assert isinstance(result, ImageResult)
    assert result.seed == 42


def test_retains_pipeline_when_device_move_returns_other_object():
    fake_t2i = MagicMock()
    pipe = _patched_pipe()
    pipe.to.return_value = MagicMock(name="unrelated-fluent-result")
    fake_t2i.from_pretrained.return_value = pipe

    with patch(f"{_RT}.AutoPipelineForText2Image", fake_t2i), \
            patch(f"{_RT}.torch", MagicMock()):
        m = _make()

    assert m._pipe is pipe
    pipe.to.assert_called_once_with("cuda")


def _race_lazy_build(pipeline_attr, call):
    """Run two concurrent first-access calls with a slow from_pipe and
    return how many times the lazy pipeline was built."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    fake_t2i = MagicMock()
    fake_t2i.from_pretrained.return_value = _patched_pipe()
    fake_lazy = MagicMock()

    def _slow_from_pipe(*_a, **_kw):
        time.sleep(0.05)
        return _patched_pipe()

    fake_lazy.from_pipe.side_effect = _slow_from_pipe

    with patch(f"{_RT}.AutoPipelineForText2Image", fake_t2i), \
            patch(f"{_RT}.{pipeline_attr}", fake_lazy), \
            patch(f"{_RT}.torch", MagicMock()):
        m = _make(device="cpu")
        with ThreadPoolExecutor(max_workers=2) as ex:
            for f in [ex.submit(call, m) for _ in range(2)]:
                f.result()
    return fake_lazy.from_pipe.call_count


def test_img2img_lazy_build_is_thread_safe():
    """Two concurrent first img2img calls build the pipeline once."""
    from PIL import Image
    init_img = Image.new("RGB", (64, 64))
    count = _race_lazy_build(
        "AutoPipelineForImage2Image",
        lambda m: m.generate("repaint", init_image=init_img, strength=0.6),
    )
    assert count == 1


def test_inpaint_lazy_build_is_thread_safe():
    """Same double-build race as img2img, guarded by self._inp_lock."""
    from PIL import Image
    init_img = Image.new("RGB", (64, 64))
    mask = Image.new("L", (64, 64), 255)
    count = _race_lazy_build(
        "AutoPipelineForInpainting",
        lambda m: m.inpaint("a", init_image=init_img, mask_image=mask),
    )
    assert count == 1
