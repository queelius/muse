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
    """A pipeline without a negative_prompt parameter. `**extra` records any
    other keyword the runtime sends, so a dropped field is visibly absent
    (not passed as None)."""

    def __init__(self):
        super().__init__()
        self.extra = []

    def __call__(self, prompt, image=None, width=None, height=None,
                 num_inference_steps=None, guidance_scale=None, generator=None,
                 **extra):
        self.extra.append(extra)
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
    m.edit("x", images=[Image.new("RGB", (4, 4))], negative_prompt="bad")
    # Not passed at all (neither as a value nor as None).
    assert pipe.extra == [{}, {}]
    m2, pipe2, _, _ = _build()
    m2.generate("x", negative_prompt="bad")
    assert pipe2.calls[0]["negative_prompt"] == "bad"


def test_negative_prompt_reported_ignored_through_the_route():
    """Behind the real route, with the klein capabilities the HF plugin
    synthesizes (supports_negative_prompt False): the pipeline never sees
    negative_prompt and the response lists it in metadata.ignored."""
    from fastapi.testclient import TestClient

    from muse.core.registry import ModalityRegistry
    from muse.core.server import create_app
    from muse.modalities.image_generation.hf import _klein_capabilities
    from muse.modalities.image_generation.routes import build_router

    m, pipe, _, _ = _build(pipe=NoNegPipe(), default_size=(256, 256))
    caps = _klein_capabilities()
    reg = ModalityRegistry()
    reg.register("image/generation", m, manifest={
        "model_id": "klein", "modality": "image/generation", "capabilities": caps,
    })
    app = create_app(registry=reg, routers={"image/generation": build_router(reg)})
    r = TestClient(app).post("/v1/images/generations", json={
        "model": "klein", "prompt": "a fox", "negative_prompt": "blurry",
        "size": "256x256",
    })
    assert r.status_code == 200, r.text
    assert "negative_prompt" in r.json()["metadata"]["ignored"]
    assert pipe.extra == [{}]
    assert pipe.calls[0]["prompt"] == "a fox"


def test_explicit_device_kwarg_is_used(monkeypatch):
    # An explicit device (e.g. a set-device pin) goes through select_device
    # verbatim and reaches placement; "auto" would consult torch instead.
    seen = []
    monkeypatch.setattr(
        f"{MOD}.select_device",
        lambda device, torch_module=None: seen.append(device) or device,
    )
    monkeypatch.setenv("MUSE_VIDEO_CPU_OFFLOAD", "off")
    m, pipe, _, _ = _build(device="cpu")
    assert seen == ["cpu"]
    assert m._device == "cpu"
    assert pipe.to_calls == [] and pipe.model_offload == []
    m2, pipe2, _, _ = _build(device="cuda", cpu_offload="model")
    assert seen[-1] == "cuda"
    assert pipe2.model_offload == ["cuda"]


def test_metadata_counts_references():
    m, _, _, _ = _build()
    img = Image.new("RGB", (4, 4))
    assert m.edit("p", images=[img, img]).metadata["reference_images"] == 2
    md = m.generate("p").metadata
    assert md["reference_images"] == 0
    assert set(md) >= {"prompt", "steps", "guidance", "reference_images"}


# ---------------- LoRA adapters on the reference runtime ----------------


class LoraPipe(RecordingPipe):
    """RecordingPipe plus the diffusers LoRA mixin surface, logging call order."""

    def __init__(self, events):
        super().__init__()
        self.events = events
        self.lora_loads = []
        self.adapter_sets = []

    def load_lora_weights(self, src, **kw):
        self.events.append("load_lora_weights")
        self.lora_loads.append((src, kw))

    def set_adapters(self, names, adapter_weights=None):
        self.adapter_sets.append((list(names), list(adapter_weights)))

    def enable_model_cpu_offload(self, device=None):
        self.events.append("place")
        super().enable_model_cpu_offload(device=device)


def _build_lora(monkeypatch, *, base_dir="/w/klein-base", local_dir="/w/lora", **kw):
    events: list = []
    pipe = LoraPipe(events)
    monkeypatch.setattr(f"{MOD}.resolve_model_source", lambda ref: base_dir)
    cls = MagicMock()

    def _from_pretrained(src, **k):
        events.append(("from_pretrained", src))
        return pipe

    cls.from_pretrained.side_effect = _from_pretrained
    fake_diffusers = SimpleNamespace(Flux2KleinPipeline=cls)
    fake_torch = MagicMock()
    for name, obj in (("diffusers", fake_diffusers), ("torch", fake_torch)):
        p = patch(f"{MOD}.{name}", obj)
        p.start()
        _patches.append(p)
    kw.setdefault("device", "cuda")
    kw.setdefault("cpu_offload", "model")
    m = DiffusersReferenceModel(
        hf_repo="Limbicnation/pixel-art-lora", local_dir=local_dir,
        model_id="klein-pixel-art", lora_adapter=True,
        base_model="flux2-klein-4b", **kw,
    )
    return m, pipe, events


def test_reference_lora_loads_adapter_before_offload(monkeypatch):
    m, pipe, events = _build_lora(monkeypatch)
    assert events == [
        ("from_pretrained", "/w/klein-base"), "load_lora_weights", "place",
    ]
    assert pipe.lora_loads == [("/w/lora", {"adapter_name": "lora"})]


def test_reference_lora_missing_base_raises(monkeypatch):
    # resolve_model_source echoes an unpulled muse id back verbatim.
    with pytest.raises(RuntimeError, match="is not pulled"):
        _build_lora(monkeypatch, base_dir="flux2-klein-4b")


def test_reference_lora_without_base_model_raises(monkeypatch):
    cls = MagicMock()
    for name, obj in (
        ("diffusers", SimpleNamespace(Flux2KleinPipeline=cls)),
        ("torch", MagicMock()),
    ):
        p = patch(f"{MOD}.{name}", obj)
        p.start()
        _patches.append(p)
    with pytest.raises(RuntimeError, match="base_model is missing"):
        DiffusersReferenceModel(
            hf_repo="org/lora", model_id="x", lora_adapter=True, device="cuda",
        )
    cls.from_pretrained.assert_not_called()


def test_reference_lora_hf_repo_base_passes_through(monkeypatch):
    m, pipe, events = _build_lora(
        monkeypatch, base_dir="black-forest-labs/FLUX.2-klein-4B",
    )
    assert events[0] == ("from_pretrained", "black-forest-labs/FLUX.2-klein-4B")


def test_reference_lora_scale_per_request(monkeypatch):
    m, pipe, _ = _build_lora(monkeypatch)
    m.generate("a knight", lora_scale=0.5)
    m.generate("a knight", lora_scale=0.5)
    assert pipe.adapter_sets == [(["lora"], [0.5])]
    # Omitted scale falls back to the manifest default (1.0): applied once.
    m.generate("a knight")
    m.edit("a knight", images=[Image.new("RGB", (4, 4))])
    assert pipe.adapter_sets == [(["lora"], [0.5]), (["lora"], [1.0])]
    m.edit("a knight", images=[Image.new("RGB", (4, 4))], lora_scale=1.3)
    assert pipe.adapter_sets[-1] == (["lora"], [1.3])
    assert len(pipe.calls) == 5


def test_reference_lora_default_scale_from_manifest(monkeypatch):
    m, pipe, _ = _build_lora(monkeypatch, lora_scale=0.85)
    m.generate("x")
    assert pipe.adapter_sets == [(["lora"], [0.85])]


def test_reference_lora_scale_factor_scales_adapter_weight(monkeypatch):
    # An rsLoRA adapter whose metadata makes diffusers apply alpha/sqrt(r)
    # (16 for r=64, alpha=128) needs lora_scale 1.0 to mean adapter weight
    # 1/16; the factor maps the user scale onto the set_adapters weight.
    m, pipe, _ = _build_lora(monkeypatch, lora_scale_factor=0.0625)
    md = m.generate("x").metadata
    assert pipe.adapter_sets == [(["lora"], [0.0625])]
    assert md["lora_scale"] == 1.0
    md = m.edit("x", images=[Image.new("RGB", (4, 4))], lora_scale=1.4).metadata
    assert pipe.adapter_sets[-1] == (["lora"], [pytest.approx(0.0875)])
    assert md["lora_scale"] == 1.4
    m.generate("x", lora_scale=0.0)
    assert pipe.adapter_sets[-1] == (["lora"], [0.0])


@pytest.mark.parametrize("factor", [0, -0.0625, float("nan"), float("inf"), "x", True])
def test_reference_lora_scale_factor_must_be_finite_and_positive(monkeypatch, factor):
    events: list = []
    monkeypatch.setattr(f"{MOD}.resolve_model_source", lambda ref: "/w/klein-base")
    cls = MagicMock()
    for name, obj in (
        ("diffusers", SimpleNamespace(Flux2KleinPipeline=cls)),
        ("torch", MagicMock()),
    ):
        p = patch(f"{MOD}.{name}", obj)
        p.start()
        _patches.append(p)
    with pytest.raises(ValueError, match="lora_scale_factor"):
        DiffusersReferenceModel(
            hf_repo="org/lora", model_id="x", lora_adapter=True,
            base_model="flux2-klein-4b", device="cuda", lora_scale_factor=factor,
        )
    cls.from_pretrained.assert_not_called()


def test_reference_without_lora_unchanged():
    pipe = LoraPipe([])
    m, _, cls, _ = _build(pipe=pipe, local_dir="/w/klein")
    assert cls.from_pretrained.call_args.args[0] == "/w/klein"
    m.generate("x", lora_scale=0.5)
    m.edit("x", images=[Image.new("RGB", (4, 4))], lora_scale=0.5)
    assert pipe.lora_loads == []
    assert pipe.adapter_sets == []


# ---------------- derived lora_scale_factor (rsLoRA header) ----------------

from tests.modalities.image_generation.runtimes.test_lora import (  # noqa: E402
    _KLEIN_PIXEL_ART_CFG, _LORA_KEYS, _adapter_md, _write_safetensors,
)


def _rslora_dir(tmp_path):
    _write_safetensors(tmp_path / "pytorch_lora_weights.safetensors",
                       _LORA_KEYS, _adapter_md(**_KLEIN_PIXEL_ART_CFG))
    return str(tmp_path)


def test_reference_lora_derives_factor_from_rslora_header(monkeypatch, tmp_path):
    # No lora_scale_factor capability (a raw hf:// klein LoRA): r=64,
    # alpha=128 rsLoRA metadata gives 1/16, so lora_scale 1.0 is not 16x.
    m, pipe, _ = _build_lora(monkeypatch, local_dir=_rslora_dir(tmp_path))
    m.generate("x")
    assert pipe.adapter_sets == [(["lora"], [0.0625])]
    m.edit("x", images=[Image.new("RGB", (4, 4))], lora_scale=2.0)
    assert pipe.adapter_sets[-1] == (["lora"], [0.125])


def test_reference_lora_explicit_factor_wins_over_header(monkeypatch, tmp_path):
    m, pipe, _ = _build_lora(
        monkeypatch, local_dir=_rslora_dir(tmp_path), lora_scale_factor=0.5,
    )
    m.generate("x")
    assert pipe.adapter_sets == [(["lora"], [0.5])]


def test_reference_lora_plain_adapter_keeps_factor_1(monkeypatch, tmp_path):
    _write_safetensors(tmp_path / "lora.safetensors", _LORA_KEYS)
    m, pipe, _ = _build_lora(monkeypatch, local_dir=str(tmp_path))
    m.generate("x")
    assert pipe.adapter_sets == [(["lora"], [1.0])]


def test_reference_lora_corrupt_header_keeps_factor_1(monkeypatch, tmp_path):
    (tmp_path / "pytorch_lora_weights.safetensors").write_bytes(b"garbage")
    m, pipe, _ = _build_lora(monkeypatch, local_dir=str(tmp_path))
    m.generate("x")
    assert pipe.adapter_sets == [(["lora"], [1.0])]


def test_reference_without_lora_never_reads_a_header(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        f"{MOD}.resolve_lora_scale_factor",
        lambda *a, **k: calls.append(a) or 1.0,
    )
    _build(local_dir=_rslora_dir(tmp_path))
    assert calls == []
