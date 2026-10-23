"""Tests for DiffusersReferenceModel (Flux2Klein-style reference-image runtime)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from muse.modalities.image_generation.runtimes.reference import (
    DiffusersReferenceModel,
)

MOD = "muse.modalities.image_generation.runtimes.reference"


class RecordingPipe:
    def __init__(self):
        self.calls = []
        self.model_offload = []
        self.to_calls = []

    def __call__(self, prompt, image=None, width=None, height=None,
                 num_inference_steps=None, guidance_scale=None,
                 generator=None, negative_prompt=None):
        self.calls.append({
            "prompt": prompt, "image": image, "width": width,
            "height": height, "steps": num_inference_steps,
            "guidance": guidance_scale, "generator": generator,
            "negative_prompt": negative_prompt,
        })
        return SimpleNamespace(images=[Image.new("RGB", (width or 8, height or 8))])

    def enable_model_cpu_offload(self, device=None):
        self.model_offload.append(device)

    def to(self, device):
        self.to_calls.append(device)
        return self


class NoNegPipe(RecordingPipe):
    def __call__(self, prompt, image=None, width=None, height=None,
                 num_inference_steps=None, guidance_scale=None, generator=None):
        return super().__call__(
            prompt, image=image, width=width, height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, generator=generator,
        )


_patches: list = []


@pytest.fixture(autouse=True)
def _stop_patches():
    yield
    while _patches:
        _patches.pop().stop()


def _build(pipe=None, **kw):
    pipe = pipe or RecordingPipe()
    cls = MagicMock()
    cls.from_pretrained.return_value = pipe
    fake_diffusers = SimpleNamespace(Flux2KleinPipeline=cls)
    fake_torch = MagicMock()
    fake_torch.bfloat16 = "BF16"
    # Patches stay active for the test: generate() reads the torch sentinel.
    for name, obj in (("diffusers", fake_diffusers), ("torch", fake_torch)):
        p = patch(f"{MOD}.{name}", obj)
        p.start()
        _patches.append(p)
    kw.setdefault("device", "cuda")
    m = DiffusersReferenceModel(hf_repo="org/klein", model_id="klein", **kw)
    return m, pipe, cls, fake_torch


def test_loads_named_pipeline_class_with_dtype():
    m, pipe, cls, _ = _build(local_dir="/w/klein")
    args, kwargs = cls.from_pretrained.call_args
    assert args[0] == "/w/klein"
    assert kwargs["torch_dtype"] == "BF16"
    assert m.default_size == (1024, 1024)
    assert m.supports_reference_images is True
    assert m.max_reference_images == 4


def test_model_offload_used_not_to(monkeypatch):
    monkeypatch.setenv("MUSE_VIDEO_CPU_OFFLOAD", "off")
    m, pipe, _, _ = _build(cpu_offload="model")
    assert pipe.model_offload == ["cuda"]
    assert pipe.to_calls == []


def test_generate_passes_size_steps_and_cpu_generator():
    m, pipe, _, torch_ = _build()
    res = m.generate("a cat", width=512, height=768, steps=6, guidance=2.5, seed=7)
    c = pipe.calls[0]
    assert (c["width"], c["height"], c["steps"], c["guidance"]) == (512, 768, 6, 2.5)
    torch_.Generator.assert_called_once_with(device="cpu")
    torch_.Generator.return_value.manual_seed.assert_called_once_with(7)
    assert c["generator"] is torch_.Generator.return_value.manual_seed.return_value
    assert c["image"] is None
    assert res.seed == 7 and (res.width, res.height) == (512, 768)


def test_generate_defaults():
    m, pipe, _, _ = _build()
    m.generate("x")
    c = pipe.calls[0]
    assert (c["width"], c["height"], c["steps"], c["guidance"]) == (1024, 1024, 4, 1.0)
    assert c["generator"] is None


def test_edit_passes_image_list_in_order():
    m, pipe, _, _ = _build()
    a, b = Image.new("RGB", (4, 4), "red"), Image.new("RGB", (4, 4), "blue")
    m.edit("combine", images=(a, b))
    assert pipe.calls[0]["image"] == [a, b]
    assert pipe.calls[0]["image"][0] is a


def test_generate_rejects_init_image():
    m, _, _, _ = _build()
    with pytest.raises(ValueError):
        m.generate("x", init_image=Image.new("RGB", (4, 4)))


def test_negative_prompt_dropped_when_pipeline_lacks_it():
    m, pipe, _, _ = _build(pipe=NoNegPipe())
    m.generate("x", negative_prompt="bad")
    assert "negative_prompt" not in pipe.calls[0] or pipe.calls[0]["negative_prompt"] is None
    m2, pipe2, _, _ = _build()
    m2.generate("x", negative_prompt="bad")
    assert pipe2.calls[0]["negative_prompt"] == "bad"


def test_metadata_counts_references():
    m, _, _, _ = _build()
    img = Image.new("RGB", (4, 4))
    assert m.edit("p", images=[img, img]).metadata["reference_images"] == 2
    md = m.generate("p").metadata
    assert md["reference_images"] == 0
    assert set(md) >= {"prompt", "steps", "guidance", "reference_images"}
