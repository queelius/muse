"""Tests for DiffusersText2ImageModel generic runtime.

The runtime wraps diffusers.AutoPipelineForText2Image; tests stub it
out so no real diffusion happens. Mirrors the patching pattern used
in tests/models/test_sd_turbo.py.
"""
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.image_generation.protocol import ImageResult
from muse.modalities.image_generation.runtimes.diffusers import (
    DiffusersText2ImageModel,
)


def _patched_pipe():
    """Return a fake pipeline whose .from_pretrained yields a mock that
    returns one PIL-shaped image when called."""
    fake_pipe = MagicMock()
    fake_image = MagicMock()
    fake_image.size = (512, 512)
    fake_pipe.return_value.images = [fake_image]
    return fake_pipe


def test_construction_uses_local_dir_when_provided(tmp_path):
    """Constructor passes local_dir as the first positional to from_pretrained."""
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo",
            local_dir=str(tmp_path),
            device="cpu",
            dtype="float32",
            model_id="org-repo",
        )
    fake_class.from_pretrained.assert_called_once()
    assert fake_class.from_pretrained.call_args.args[0] == str(tmp_path)
    assert m.model_id == "org-repo"


def test_construction_falls_back_to_hf_repo_when_no_local_dir():
    """When local_dir is None, from_pretrained gets the hf_repo as first arg."""
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        DiffusersText2ImageModel(
            hf_repo="org/repo",
            local_dir=None,
            device="cpu",
            dtype="float32",
            model_id="m",
        )
    assert fake_class.from_pretrained.call_args.args[0] == "org/repo"


def test_default_size_steps_guidance_from_kwargs():
    """Constructor reads default_size/steps/guidance from kwargs (manifest capabilities)."""
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo",
            local_dir="/tmp/fake",
            device="cpu",
            model_id="m",
            default_size=(1024, 1024),
            default_steps=4,
            default_guidance=3.5,
        )
    assert m.default_size == (1024, 1024)
    assert m._default_steps == 4
    assert m._default_guidance == 3.5


def test_generate_uses_request_overrides_when_provided():
    """When generate() is called with steps/guidance/size, those override defaults."""
    pipe = _patched_pipe()
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = pipe
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m", default_steps=1, default_guidance=0.0,
        )
        result = m.generate(
            "a fox", steps=25, guidance=7.5, width=768, height=768,
        )
    call_kwargs = pipe.call_args.kwargs
    assert call_kwargs["num_inference_steps"] == 25
    assert call_kwargs["guidance_scale"] == 7.5
    assert call_kwargs["width"] == 768
    assert call_kwargs["height"] == 768
    assert isinstance(result, ImageResult)


def test_generate_uses_defaults_when_request_omits_them():
    """When generate() omits steps/guidance, defaults from constructor are used."""
    pipe = _patched_pipe()
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = pipe
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m", default_size=(1024, 1024),
            default_steps=4, default_guidance=0.0,
        )
        m.generate("a fox")
    call_kwargs = pipe.call_args.kwargs
    assert call_kwargs["num_inference_steps"] == 4
    assert call_kwargs["guidance_scale"] == 0.0
    assert call_kwargs["width"] == 1024
    assert call_kwargs["height"] == 1024


def test_generate_passes_negative_prompt_when_set():
    pipe = _patched_pipe()
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = pipe
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu", model_id="m",
        )
        m.generate("a fox", negative_prompt="blurry, ugly")
    assert pipe.call_args.kwargs.get("negative_prompt") == "blurry, ugly"


def test_generate_omits_negative_prompt_when_none():
    """negative_prompt=None should NOT be passed to the pipe (some models reject it)."""
    pipe = _patched_pipe()
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = pipe
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu", model_id="m",
        )
        m.generate("a fox")
    assert "negative_prompt" not in pipe.call_args.kwargs


def test_generate_returns_image_result_with_seed():
    pipe = _patched_pipe()
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = pipe
    fake_torch = MagicMock()
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        fake_torch,
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu", model_id="m",
        )
        result = m.generate("a fox", seed=42)
    assert result.seed == 42
    assert result.metadata["model"] == "m"


def test_construction_absorbs_unknown_kwargs():
    """Future capability keys must not crash the constructor.

    Catalog's load_backend splats manifest.capabilities into the constructor
    call. As muse adds new capability flags (e.g., supports_negative_prompt,
    supports_seeded_generation, supports_img2img), older runtimes must accept
    them gracefully via **_.
    """
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        DiffusersText2ImageModel(
            hf_repo="org/repo",
            local_dir="/tmp/fake",
            device="cpu",
            model_id="m",
            supports_negative_prompt=True,
            supports_seeded_generation=True,
            future_unrecognized_flag="whatever",
        )
    fake_class.from_pretrained.assert_called_once()


def test_default_size_normalized_to_tuple_when_passed_list():
    """JSON-loaded manifests produce default_size as list[int]; coerce to tuple."""
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m", default_size=[1024, 1024],  # list, not tuple
        )
    assert m.default_size == (1024, 1024)
    assert isinstance(m.default_size, tuple)


def _variant_kwarg(fake_class):
    return fake_class.from_pretrained.call_args.kwargs.get("variant")


def test_no_fp16_variant_requested_when_snapshot_lacks_fp16_weights(tmp_path):
    """H6: repos like flux-schnell ship no .fp16. weights, so the downloader
    fetches only bare weights. Requesting variant='fp16' against that local
    snapshot makes from_pretrained raise -> worker crashes on cold load. The
    runtime must request variant=None when no fp16 files are present, even at
    the default dtype='float16'."""
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir=str(tmp_path), device="cpu",
            dtype="float16", model_id="m",
        )
    assert _variant_kwarg(fake_class) is None


def test_fp16_variant_requested_when_snapshot_has_fp16_weights(tmp_path):
    """When the snapshot holds .fp16. weights (sd-turbo/sdxl-turbo, where the
    downloader fetched ONLY the fp16 files), the runtime must request
    variant='fp16' so from_pretrained finds them."""
    (tmp_path / "unet").mkdir()
    (tmp_path / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir=str(tmp_path), device="cpu",
            dtype="float16", model_id="m",
        )
    assert _variant_kwarg(fake_class) == "fp16"


def test_generate_with_init_image_uses_img2img_pipeline():
    """When init_image is set, runtime calls AutoPipelineForImage2Image."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()
    fake_i2i_class.from_pipe.return_value = _patched_pipe()

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForImage2Image",
        fake_i2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m",
        )
        init_img = Image.new("RGB", (64, 64))
        m.generate("repaint", init_image=init_img, strength=0.6)

    fake_i2i_class.from_pipe.assert_called_once()
    # The img2img pipeline (not the t2i one) was called for inference
    fake_i2i_class.from_pipe.return_value.assert_called()


def test_generate_without_init_image_uses_text2image_pipeline():
    """init_image=None keeps the existing text-to-image path (no regression)."""
    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForImage2Image",
        fake_i2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m",
        )
        m.generate("a fox")

    # img2img was NEVER loaded
    fake_i2i_class.from_pipe.assert_not_called()


def test_generate_img2img_default_strength_when_omitted():
    """When strength is None on an img2img call, defaults to 0.5."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()
    fake_i2i_pipe = _patched_pipe()
    fake_i2i_class.from_pipe.return_value = fake_i2i_pipe

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForImage2Image",
        fake_i2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m",
        )
        init_img = Image.new("RGB", (64, 64))
        m.generate("repaint", init_image=init_img)  # no strength

    assert fake_i2i_pipe.call_args.kwargs["strength"] == 0.5


def test_generate_img2img_caches_pipeline():
    """Second img2img call reuses the cached pipeline (no second from_pipe)."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()
    fake_i2i_class.from_pipe.return_value = _patched_pipe()

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForImage2Image",
        fake_i2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m",
        )
        init_img = Image.new("RGB", (64, 64))
        m.generate("a", init_image=init_img)
        m.generate("b", init_image=init_img)

    assert fake_i2i_class.from_pipe.call_count == 1


def test_img2img_bumps_steps_to_satisfy_strength_contract():
    """num_inference_steps * strength must be >= 1 for diffusers img2img.

    With num_inference_steps=1 and strength=0.4, naive math gives 0.4
    effective denoise steps (rounds to 0) and the VAE crashes with an
    empty tensor. Runtime must bump steps to ceil(1/0.4) = 3.
    """
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()
    fake_i2i_pipe = _patched_pipe()
    fake_i2i_class.from_pipe.return_value = fake_i2i_pipe

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForImage2Image",
        fake_i2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m", default_steps=1,  # turbo-style 1-step model
        )
        init_img = Image.new("RGB", (64, 64))
        m.generate("repaint", init_image=init_img, strength=0.4)

    # With strength=0.4, ceil(1/0.4) = 3 steps minimum
    assert fake_i2i_pipe.call_args.kwargs["num_inference_steps"] == 3
    assert fake_i2i_pipe.call_args.kwargs["strength"] == 0.4


def test_img2img_floors_zero_strength_to_keep_effective_steps_nonzero():
    """L12: strength=0.0 is schema-valid (ge=0.0) but yields zero effective
    denoise steps (int(steps*0.0)=0), emptying the timestep schedule and
    crashing the VAE. Bumping only num_inference_steps does not help because
    steps*0.0 is still 0. The runtime must floor the strength VALUE passed
    to the pipeline so steps*strength >= 1 is actually reachable.
    """
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()
    fake_i2i_pipe = _patched_pipe()
    fake_i2i_class.from_pipe.return_value = fake_i2i_pipe

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForImage2Image",
        fake_i2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m", default_steps=1,
        )
        init_img = Image.new("RGB", (64, 64))
        m.generate("repaint", init_image=init_img, strength=0.0)

    call = fake_i2i_pipe.call_args.kwargs
    strength = call["strength"]
    steps = call["num_inference_steps"]
    assert strength > 0.0, "zero strength must be floored above 0"
    # The whole point of the floor: effective denoise steps stay >= 1.
    assert int(steps * strength) >= 1


def test_img2img_does_not_bump_when_steps_already_sufficient():
    """If user explicitly requests enough steps, don't bump."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()
    fake_i2i_pipe = _patched_pipe()
    fake_i2i_class.from_pipe.return_value = fake_i2i_pipe

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForImage2Image",
        fake_i2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m", default_steps=25,
        )
        init_img = Image.new("RGB", (64, 64))
        m.generate("repaint", init_image=init_img, strength=0.4)

    # 25 * 0.4 = 10 effective steps, well above the >= 1 contract
    assert fake_i2i_pipe.call_args.kwargs["num_inference_steps"] == 25


def test_img2img_uses_from_pipe_not_from_pretrained_to_share_vram():
    """from_pipe shares weights; from_pretrained would OOM on small GPUs.

    Regression for v0.17.2: SDXL-Turbo on a 12GB GPU crashed loading a
    second pipeline because from_pretrained allocated a fresh copy of
    all weights. from_pipe reuses the loaded UNet/VAE/text-encoders.
    """
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_pipe = _patched_pipe()
    fake_t2i_class.from_pretrained.return_value = fake_t2i_pipe
    fake_i2i_class = MagicMock()
    fake_i2i_class.from_pipe.return_value = _patched_pipe()

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForImage2Image",
        fake_i2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m",
        )
        init_img = Image.new("RGB", (64, 64))
        m.generate("repaint", init_image=init_img, strength=0.4)

    # Critical: from_pipe was called (shares weights). from_pretrained on
    # the i2i class was NOT called (would have allocated a fresh copy).
    fake_i2i_class.from_pipe.assert_called_once_with(fake_t2i_pipe)
    fake_i2i_class.from_pretrained.assert_not_called()


# ---------------- inpaint() / vary() (#100, v0.21.0) ----------------


def _patched_t2i_with_inp(fake_inp_class):
    """Build a t2i + inp class pair, patched into the runtime module."""
    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    return fake_t2i_class, fake_inp_class


def test_inpaint_uses_inpainting_pipeline():
    """When inpaint() is called, runtime calls AutoPipelineForInpainting.from_pipe."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_inp_class = MagicMock()
    fake_inp_pipe = _patched_pipe()
    fake_inp_class.from_pipe.return_value = fake_inp_pipe

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForInpainting",
        fake_inp_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m",
        )
        init_img = Image.new("RGB", (64, 64))
        mask = Image.new("L", (64, 64), 255)
        m.inpaint(
            "add a moon", init_image=init_img, mask_image=mask, strength=0.9,
        )

    fake_inp_class.from_pipe.assert_called_once()
    fake_inp_pipe.assert_called()
    # The pipe call carries image + mask_image kwargs.
    call_kwargs = fake_inp_pipe.call_args.kwargs
    assert call_kwargs["image"] is init_img
    assert call_kwargs["mask_image"] is mask


def test_inpaint_caches_pipeline():
    """Second inpaint call reuses the cached pipeline (no second from_pipe)."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_inp_class = MagicMock()
    fake_inp_class.from_pipe.return_value = _patched_pipe()

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForInpainting",
        fake_inp_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m",
        )
        init_img = Image.new("RGB", (64, 64))
        mask = Image.new("L", (64, 64), 255)
        m.inpaint("a", init_image=init_img, mask_image=mask)
        m.inpaint("b", init_image=init_img, mask_image=mask)

    assert fake_inp_class.from_pipe.call_count == 1


def test_inpaint_uses_from_pipe_not_from_pretrained_to_share_vram():
    """from_pipe shares weights; from_pretrained would OOM on small GPUs.

    Same VRAM-sharing reasoning as the v0.17.2 img2img regression. The
    inpaint pipeline must be loaded via from_pipe(self._pipe) so it
    reuses the already-resident UNet/VAE/text-encoders.
    """
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_pipe = _patched_pipe()
    fake_t2i_class.from_pretrained.return_value = fake_t2i_pipe
    fake_inp_class = MagicMock()
    fake_inp_class.from_pipe.return_value = _patched_pipe()

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForInpainting",
        fake_inp_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m",
        )
        init_img = Image.new("RGB", (64, 64))
        mask = Image.new("L", (64, 64), 255)
        m.inpaint("repaint", init_image=init_img, mask_image=mask)

    fake_inp_class.from_pipe.assert_called_once_with(fake_t2i_pipe)
    fake_inp_class.from_pretrained.assert_not_called()


def test_inpaint_normalizes_rgba_mask_to_grayscale():
    """A non-L mask gets converted before being passed to the pipe."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_inp_class = MagicMock()
    fake_inp_pipe = _patched_pipe()
    fake_inp_class.from_pipe.return_value = fake_inp_pipe

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForInpainting",
        fake_inp_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m",
        )
        init_img = Image.new("RGB", (64, 64))
        rgba_mask = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
        m.inpaint("repaint", init_image=init_img, mask_image=rgba_mask)

    passed_mask = fake_inp_pipe.call_args.kwargs["mask_image"]
    assert passed_mask.mode == "L"


def test_inpaint_bumps_steps_to_satisfy_strength_contract():
    """num_inference_steps * strength must be >= 1 for diffusers pipelines.

    With default_steps=1 (turbo-style) and strength=0.4 the naive math
    gives 0.4 effective denoise steps and the VAE crashes. The runtime
    must bump steps to ceil(1/0.4) = 3.
    """
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_inp_class = MagicMock()
    fake_inp_pipe = _patched_pipe()
    fake_inp_class.from_pipe.return_value = fake_inp_pipe

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForInpainting",
        fake_inp_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m", default_steps=1,
        )
        init_img = Image.new("RGB", (64, 64))
        mask = Image.new("L", (64, 64), 255)
        m.inpaint("repaint", init_image=init_img, mask_image=mask, strength=0.4)

    assert fake_inp_pipe.call_args.kwargs["num_inference_steps"] == 3
    assert fake_inp_pipe.call_args.kwargs["strength"] == 0.4


def test_inpaint_returns_imageresult_with_mode_inpaint():
    """ImageResult.metadata.mode should be 'inpaint' for the inpaint path."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_inp_class = MagicMock()
    fake_inp_class.from_pipe.return_value = _patched_pipe()

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForInpainting",
        fake_inp_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m",
        )
        init_img = Image.new("RGB", (64, 64))
        mask = Image.new("L", (64, 64), 255)
        result = m.inpaint("paint", init_image=init_img, mask_image=mask)

    assert isinstance(result, ImageResult)
    assert result.metadata["mode"] == "inpaint"


def test_vary_delegates_to_img2img_with_empty_prompt_and_default_strength():
    """vary() reuses the img2img pipeline; default strength is 0.85."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()
    fake_i2i_pipe = _patched_pipe()
    fake_i2i_class.from_pipe.return_value = fake_i2i_pipe

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForImage2Image",
        fake_i2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m", default_steps=4,
        )
        init_img = Image.new("RGB", (64, 64))
        m.vary(init_image=init_img)

    call_kwargs = fake_i2i_pipe.call_args.kwargs
    assert call_kwargs["prompt"] == ""
    assert call_kwargs["strength"] == 0.85


def test_vary_returns_imageresult_with_mode_variations():
    """vary() overrides the underlying img2img mode label with 'variations'."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()
    fake_i2i_class.from_pipe.return_value = _patched_pipe()

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForImage2Image",
        fake_i2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m",
        )
        init_img = Image.new("RGB", (64, 64))
        result = m.vary(init_image=init_img)

    assert isinstance(result, ImageResult)
    assert result.metadata["mode"] == "variations"


def test_vary_honors_user_strength_override():
    """User-supplied strength wins over the 0.85 default."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()
    fake_i2i_pipe = _patched_pipe()
    fake_i2i_class.from_pipe.return_value = fake_i2i_pipe

    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_t2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForImage2Image",
        fake_i2i_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        m = DiffusersText2ImageModel(
            hf_repo="org/repo", local_dir="/tmp/fake", device="cpu",
            model_id="m", default_steps=4,
        )
        init_img = Image.new("RGB", (64, 64))
        m.vary(init_image=init_img, strength=0.6)

    assert fake_i2i_pipe.call_args.kwargs["strength"] == 0.6


def _lora_model(fake_class, tmp_path, monkeypatch, **extra):
    """Construct a LoRA-configured runtime with mocked diffusers + a
    resolve_model_source that maps the muse id to a fake dir."""
    monkeypatch.setattr(
        "muse.modalities.image_generation.runtimes.diffusers.resolve_model_source",
        lambda ref: "/weights/sdxl-turbo" if ref == "sdxl-turbo" else ref,
    )
    with patch(
        "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
        fake_class,
    ), patch(
        "muse.modalities.image_generation.runtimes.diffusers.torch",
        MagicMock(),
    ):
        return DiffusersText2ImageModel(
            hf_repo="nerijs/pixel-art-xl",
            local_dir=str(tmp_path),
            device="cpu",
            dtype="float32",
            model_id="pixel-art-xl",
            lora_adapter=True,
            base_model="sdxl-turbo",
            **extra,
        )


class TestLoraLoading:
    def test_pipeline_loads_from_base_not_adapter(self, tmp_path, monkeypatch):
        fake_class = MagicMock()
        fake_class.from_pretrained.return_value = _patched_pipe()
        _lora_model(fake_class, tmp_path, monkeypatch)
        assert fake_class.from_pretrained.call_args.args[0] == "/weights/sdxl-turbo"

    def test_load_lora_weights_called_with_adapter_dir(self, tmp_path, monkeypatch):
        fake_class = MagicMock()
        pipe = _patched_pipe()
        fake_class.from_pretrained.return_value = pipe
        _lora_model(fake_class, tmp_path, monkeypatch)
        pipe.load_lora_weights.assert_called_once_with(str(tmp_path))

    def test_fuse_lora_never_called(self, tmp_path, monkeypatch):
        fake_class = MagicMock()
        pipe = _patched_pipe()
        fake_class.from_pretrained.return_value = pipe
        _lora_model(fake_class, tmp_path, monkeypatch)
        pipe.fuse_lora.assert_not_called()

    def test_hf_repo_base_passes_through_verbatim(self, tmp_path, monkeypatch):
        fake_class = MagicMock()
        fake_class.from_pretrained.return_value = _patched_pipe()
        monkeypatch.setattr(
            "muse.modalities.image_generation.runtimes.diffusers.resolve_model_source",
            lambda ref: ref,
        )
        with patch(
            "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
            fake_class,
        ), patch(
            "muse.modalities.image_generation.runtimes.diffusers.torch",
            MagicMock(),
        ):
            DiffusersText2ImageModel(
                hf_repo="nerijs/pixel-art-xl",
                local_dir=str(tmp_path),
                device="cpu",
                dtype="float32",
                model_id="pixel-art-xl",
                lora_adapter=True,
                base_model="stabilityai/stable-diffusion-xl-base-1.0",
            )
        assert fake_class.from_pretrained.call_args.args[0] == (
            "stabilityai/stable-diffusion-xl-base-1.0"
        )

    def test_lora_without_base_model_raises_actionable(self, tmp_path, monkeypatch):
        fake_class = MagicMock()
        fake_class.from_pretrained.return_value = _patched_pipe()
        with patch(
            "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
            fake_class,
        ), patch(
            "muse.modalities.image_generation.runtimes.diffusers.torch",
            MagicMock(),
        ):
            with pytest.raises(RuntimeError, match="base_model"):
                DiffusersText2ImageModel(
                    hf_repo="nerijs/pixel-art-xl",
                    local_dir=str(tmp_path),
                    device="cpu",
                    dtype="float32",
                    model_id="pixel-art-xl",
                    lora_adapter=True,
                )

    def test_unpulled_muse_id_base_raises_actionable_muse_pull_hint(
        self, tmp_path, monkeypatch,
    ):
        """I1: when base_model is a muse-id (no '/') that resolve_model_source
        cannot map to a local_dir (base removed after pull, or never
        pulled), the runtime must raise a clear `muse pull <base>` hint
        instead of letting from_pretrained raise a cryptic HF
        RepositoryNotFoundError for a bare id it can't resolve."""
        fake_class = MagicMock()
        fake_class.from_pretrained.return_value = _patched_pipe()
        # Identity: resolve_model_source didn't find a local_dir for the
        # muse id, so it just echoed the id back verbatim (the documented
        # unknown-id passthrough behavior).
        monkeypatch.setattr(
            "muse.modalities.image_generation.runtimes.diffusers.resolve_model_source",
            lambda ref: ref,
        )
        with patch(
            "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
            fake_class,
        ), patch(
            "muse.modalities.image_generation.runtimes.diffusers.torch",
            MagicMock(),
        ):
            with pytest.raises(RuntimeError, match=r"muse pull sdxl-turbo"):
                DiffusersText2ImageModel(
                    hf_repo="nerijs/pixel-art-xl",
                    local_dir=str(tmp_path),
                    device="cpu",
                    dtype="float32",
                    model_id="pixel-art-xl",
                    lora_adapter=True,
                    base_model="sdxl-turbo",
                )
        # from_pretrained must never be reached with the bare unresolved id.
        fake_class.from_pretrained.assert_not_called()


class TestLoraScale:
    def test_request_scale_passes_cross_attention_kwargs(self, tmp_path, monkeypatch):
        fake_class = MagicMock()
        pipe = _patched_pipe()
        fake_class.from_pretrained.return_value = pipe
        m = _lora_model(fake_class, tmp_path, monkeypatch)
        m.generate("pixel art, a knight", lora_scale=0.7)
        kwargs = pipe.call_args.kwargs
        assert kwargs["cross_attention_kwargs"] == {"scale": 0.7}

    def test_default_scale_used_when_request_omits(self, tmp_path, monkeypatch):
        fake_class = MagicMock()
        pipe = _patched_pipe()
        fake_class.from_pretrained.return_value = pipe
        m = _lora_model(fake_class, tmp_path, monkeypatch, lora_scale=0.5)
        m.generate("pixel art, a knight")
        assert pipe.call_args.kwargs["cross_attention_kwargs"] == {"scale": 0.5}

    def test_non_lora_model_sends_no_cross_attention_kwargs(self):
        fake_class = MagicMock()
        pipe = _patched_pipe()
        fake_class.from_pretrained.return_value = pipe
        with patch(
            "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
            fake_class,
        ), patch(
            "muse.modalities.image_generation.runtimes.diffusers.torch",
            MagicMock(),
        ):
            m = DiffusersText2ImageModel(
                hf_repo="org/repo", local_dir=None, device="cpu",
                dtype="float32", model_id="m",
            )
        m.generate("a cat")
        assert "cross_attention_kwargs" not in pipe.call_args.kwargs


class TestLoraScaleImg2Img:
    """I3: per-request lora_scale must reach the img2img path too, not
    just t2i. Before the fix, generate(..., lora_scale=...) dropped the
    request value when init_image was set (delegating to
    _generate_img2img without it; the configured default was injected
    instead)."""

    def _lora_model_with_i2i(self, tmp_path, monkeypatch, fake_i2i_pipe, **extra):
        fake_t2i_class = MagicMock()
        fake_t2i_class.from_pretrained.return_value = _patched_pipe()
        fake_i2i_class = MagicMock()
        fake_i2i_class.from_pipe.return_value = fake_i2i_pipe
        monkeypatch.setattr(
            "muse.modalities.image_generation.runtimes.diffusers.resolve_model_source",
            lambda ref: "/weights/sdxl-turbo" if ref == "sdxl-turbo" else ref,
        )
        # monkeypatch (not `with patch(...)`) so the substitution stays
        # active for the test's later m.generate(...) call, not just
        # during construction.
        monkeypatch.setattr(
            "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForText2Image",
            fake_t2i_class,
        )
        monkeypatch.setattr(
            "muse.modalities.image_generation.runtimes.diffusers.AutoPipelineForImage2Image",
            fake_i2i_class,
        )
        monkeypatch.setattr(
            "muse.modalities.image_generation.runtimes.diffusers.torch",
            MagicMock(),
        )
        m = DiffusersText2ImageModel(
            hf_repo="nerijs/pixel-art-xl",
            local_dir=str(tmp_path),
            device="cpu",
            dtype="float32",
            model_id="pixel-art-xl",
            lora_adapter=True,
            base_model="sdxl-turbo",
            **extra,
        )
        return m

    def test_request_lora_scale_forwarded_on_img2img(self, tmp_path, monkeypatch):
        from PIL import Image

        fake_i2i_pipe = _patched_pipe()
        m = self._lora_model_with_i2i(tmp_path, monkeypatch, fake_i2i_pipe)
        init_img = Image.new("RGB", (64, 64))
        m.generate("pixel art, a knight", init_image=init_img, lora_scale=0.4)
        assert fake_i2i_pipe.call_args.kwargs["cross_attention_kwargs"] == {
            "scale": 0.4,
        }

    def test_default_lora_scale_used_on_img2img_when_request_omits(
        self, tmp_path, monkeypatch,
    ):
        from PIL import Image

        fake_i2i_pipe = _patched_pipe()
        m = self._lora_model_with_i2i(
            tmp_path, monkeypatch, fake_i2i_pipe, lora_scale=0.5,
        )
        init_img = Image.new("RGB", (64, 64))
        m.generate("pixel art, a knight", init_image=init_img)
        assert fake_i2i_pipe.call_args.kwargs["cross_attention_kwargs"] == {
            "scale": 0.5,
        }
