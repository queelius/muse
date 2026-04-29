# `image/upscale` Modality Implementation Plan (#147)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add the 13th muse modality, `image/upscale`, exposing `POST /v1/images/upscale` for diffusion-based super-resolution. Bundle `stabilityai/stable-diffusion-x4-upscaler` as the default model.

**Architecture:** new `image_upscale` modality directory; new `DiffusersUpscaleRuntime` generic runtime; new bundled `stable_diffusion_x4_upscaler.py`; new HF plugin (priority 105); new multipart route + client. AuraSR and Real-ESRGAN deferred to v1.next.

**Spec:** `docs/superpowers/specs/2026-04-28-image-upscale-modality-design.md`

**Target version:** v0.25.0

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/muse/modalities/image_upscale/__init__.py` | create | MODALITY, build_router, exports, PROBE_DEFAULTS |
| `src/muse/modalities/image_upscale/protocol.py` | create | UpscaleResult dataclass + ImageUpscaleModel Protocol |
| `src/muse/modalities/image_upscale/codec.py` | create | re-export to_bytes, to_data_url from image_generation |
| `src/muse/modalities/image_upscale/routes.py` | create | POST /v1/images/upscale (multipart) |
| `src/muse/modalities/image_upscale/client.py` | create | ImageUpscaleClient (multipart upload) |
| `src/muse/modalities/image_upscale/hf.py` | create | HF plugin (priority 105) |
| `src/muse/modalities/image_upscale/runtimes/__init__.py` | create | empty (package marker) |
| `src/muse/modalities/image_upscale/runtimes/diffusers_upscaler.py` | create | DiffusersUpscaleRuntime |
| `src/muse/models/stable_diffusion_x4_upscaler.py` | create | bundled Model + MANIFEST |
| `src/muse/curated.yaml` | modify | add stable-diffusion-x4-upscaler entry |
| `tests/modalities/image_upscale/__init__.py` | create | empty |
| `tests/modalities/image_upscale/test_protocol.py` | create | dataclass + Protocol tests |
| `tests/modalities/image_upscale/test_codec.py` | create | re-export tests |
| `tests/modalities/image_upscale/test_routes.py` | create | route tests (happy + 4xx) |
| `tests/modalities/image_upscale/test_client.py` | create | client tests |
| `tests/modalities/image_upscale/test_hf_plugin.py` | create | HF plugin tests |
| `tests/modalities/image_upscale/runtimes/__init__.py` | create | empty |
| `tests/modalities/image_upscale/runtimes/test_diffusers_upscaler.py` | create | runtime tests |
| `tests/models/test_stable_diffusion_x4_upscaler.py` | create | bundled-script tests |
| `tests/cli_impl/test_e2e_image_upscale.py` | create | slow e2e in-process |
| `tests/integration/test_remote_image_upscale.py` | create | opt-in remote |
| `pyproject.toml` | modify | bump 0.24.0 -> 0.25.0 |
| `src/muse/__init__.py` | modify | docstring v0.25.0 with image/upscale |
| `CLAUDE.md` | modify | document image/upscale modality |
| `README.md` | modify | document image/upscale modality |

---

## Task A: Protocol + codec + skeleton modality package

Pure data shapes, no heavy deps. Establishes the modality directory.

**Files:**
- Create: `src/muse/modalities/image_upscale/__init__.py`
- Create: `src/muse/modalities/image_upscale/protocol.py`
- Create: `src/muse/modalities/image_upscale/codec.py`
- Create: `tests/modalities/image_upscale/__init__.py`
- Create: `tests/modalities/image_upscale/test_protocol.py`
- Create: `tests/modalities/image_upscale/test_codec.py`

- [ ] **Step 1: Write failing tests for protocol**

`tests/modalities/image_upscale/test_protocol.py`:

```python
from muse.modalities.image_upscale.protocol import (
    ImageUpscaleModel,
    UpscaleResult,
)


def test_upscale_result_required_fields():
    r = UpscaleResult(
        image=object(),
        original_width=128, original_height=128,
        upscaled_width=512, upscaled_height=512,
        scale=4, seed=-1,
    )
    assert r.scale == 4
    assert r.metadata == {}


def test_upscale_result_metadata_default_factory():
    r1 = UpscaleResult(image=None, original_width=1, original_height=1,
                       upscaled_width=2, upscaled_height=2, scale=2, seed=0)
    r2 = UpscaleResult(image=None, original_width=1, original_height=1,
                       upscaled_width=2, upscaled_height=2, scale=2, seed=0)
    r1.metadata["x"] = 1
    assert r2.metadata == {}


def test_image_upscale_model_protocol_is_runtime_checkable():
    # Anything with model_id, supported_scales, upscale satisfies it
    class Stub:
        model_id = "x"
        supported_scales = [4]
        def upscale(self, image, **kw):
            return None
    assert isinstance(Stub(), ImageUpscaleModel)
```

- [ ] **Step 2: Write protocol.py**

`src/muse/modalities/image_upscale/protocol.py`:

```python
"""Protocol + dataclasses for the image/upscale modality."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class UpscaleResult:
    image: Any
    original_width: int
    original_height: int
    upscaled_width: int
    upscaled_height: int
    scale: int
    seed: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ImageUpscaleModel(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def supported_scales(self) -> list[int]: ...

    def upscale(
        self,
        image: Any,
        *,
        scale: int | None = None,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> UpscaleResult: ...
```

- [ ] **Step 3: Write failing codec tests**

`tests/modalities/image_upscale/test_codec.py`:

```python
import io
from PIL import Image

from muse.modalities.image_upscale.codec import to_bytes, to_data_url


def test_codec_to_bytes_roundtrip():
    img = Image.new("RGB", (16, 16), (10, 20, 30))
    raw = to_bytes(img, fmt="png")
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_codec_to_data_url_prefixes_png():
    img = Image.new("RGB", (8, 8), (0, 0, 0))
    url = to_data_url(img, fmt="png")
    assert url.startswith("data:image/png;base64,")
```

- [ ] **Step 4: Write codec.py (re-export)**

`src/muse/modalities/image_upscale/codec.py`:

```python
"""Image encoding for the image/upscale modality.

Re-exports the same to_bytes / to_data_url helpers used by
image/generation. Kept as a thin module rather than a duplicate so the
modality is self-describing without code duplication.
"""
from muse.modalities.image_generation.codec import to_bytes, to_data_url

__all__ = ["to_bytes", "to_data_url"]
```

- [ ] **Step 5: Write modality `__init__.py` skeleton**

`src/muse/modalities/image_upscale/__init__.py` (build_router stub raises NotImplementedError until Task C):

```python
"""Image upscale modality: image-to-image super-resolution.

Wire contract: POST /v1/images/upscale (multipart/form-data) with
{image, model?, scale?, prompt?, negative_prompt?, steps?, guidance?,
seed?, n?, response_format?} returns list of upscaled images in the
OpenAI-compatible envelope (b64_json bytes or data URL).

Bundled model: stable-diffusion-x4-upscaler. The HF resolver also
synthesizes manifests for any diffusers-shape upscaler (model_index.json
+ image-to-image tag + upscaler-name allowlist).
"""
from muse.modalities.image_upscale.client import ImageUpscaleClient
from muse.modalities.image_upscale.protocol import (
    ImageUpscaleModel,
    UpscaleResult,
)
from muse.modalities.image_upscale.routes import build_router

MODALITY = "image/upscale"


def _make_probe_image():
    """Generate a small synthetic test image for `muse models probe`.

    PIL is imported here lazily so the modality package loads without
    PIL on the host python (per the muse `--help` should not need ML
    deps contract).
    """
    from PIL import Image
    return Image.new("RGB", (128, 128), (128, 128, 128))


PROBE_DEFAULTS = {
    "shape": "128x128 -> 512x512 (4x), 20 steps",
    "call": lambda m: m.upscale(_make_probe_image(), scale=4),
}

__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "ImageUpscaleClient",
    "ImageUpscaleModel",
    "UpscaleResult",
]
```

NOTE: this Step's `__init__.py` won't import successfully until Task B (runtime) and Task C (routes/client) are in place. We'll commit the protocol + codec first; the `__init__.py` import chain is satisfied in Task C.

Actually do a **partial init** here: only export what exists at this step. Update again in Task C. Skip the `client` and `routes` imports for now:

```python
"""Image upscale modality (skeleton; routes and client land in Task C)."""
from muse.modalities.image_upscale.protocol import (
    ImageUpscaleModel,
    UpscaleResult,
)

MODALITY = "image/upscale"

__all__ = ["MODALITY", "ImageUpscaleModel", "UpscaleResult"]
```

- [ ] **Step 6: Run targeted + fast lane**

```bash
pytest tests/modalities/image_upscale/ -v
pytest tests/ -q -m "not slow"
```

- [ ] **Step 7: Commit (Task A)**

```
feat(image-upscale): protocol + codec + skeleton modality package (#147)
```

---

## Task B: `DiffusersUpscaleRuntime`

Generic runtime; lazy imports torch + diffusers.StableDiffusionUpscalePipeline. Mirrors DiffusersText2ImageModel structure.

**Files:**
- Create: `src/muse/modalities/image_upscale/runtimes/__init__.py`
- Create: `src/muse/modalities/image_upscale/runtimes/diffusers_upscaler.py`
- Create: `tests/modalities/image_upscale/runtimes/__init__.py`
- Create: `tests/modalities/image_upscale/runtimes/test_diffusers_upscaler.py`

- [ ] **Step 1: Write failing runtime tests**

`tests/modalities/image_upscale/runtimes/test_diffusers_upscaler.py`:

```python
"""Tests for DiffusersUpscaleRuntime (fully patched; no real weights)."""
from unittest.mock import MagicMock, patch

from PIL import Image

from muse.modalities.image_upscale.protocol import UpscaleResult


def _patched_pipe(out_size=(512, 512)):
    """Fake StableDiffusionUpscalePipeline whose call returns one image."""
    fake_img = Image.new("RGB", out_size, (40, 40, 40))
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [fake_img]
    fake_pipe.to = MagicMock(return_value=fake_pipe)
    return fake_pipe


def test_runtime_constructor_loads_pipeline():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="stabilityai/stable-diffusion-x4-upscaler",
            local_dir="/fake",
            device="cpu",
            model_id="stable-diffusion-x4-upscaler",
        )
        assert m.model_id == "stable-diffusion-x4-upscaler"
        assert m.supported_scales == [4]
    fake_cls.from_pretrained.assert_called_once()


def test_runtime_uses_local_dir_over_hf_repo():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        DiffusersUpscaleRuntime(
            hf_repo="stabilityai/stable-diffusion-x4-upscaler",
            local_dir="/real/local",
            device="cpu",
            model_id="x",
        )
    assert fake_cls.from_pretrained.call_args.args[0] == "/real/local"


def test_runtime_upscale_returns_upscale_result():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe(out_size=(512, 512))
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="stabilityai/stable-diffusion-x4-upscaler",
            local_dir="/fake",
            device="cpu",
            model_id="stable-diffusion-x4-upscaler",
        )
        src = Image.new("RGB", (128, 128), (10, 10, 10))
        result = m.upscale(src, scale=4)
        assert isinstance(result, UpscaleResult)
        assert result.original_width == 128
        assert result.original_height == 128
        assert result.upscaled_width == 512
        assert result.upscaled_height == 512
        assert result.scale == 4


def test_runtime_passes_prompt_to_pipeline():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
        )
        src = Image.new("RGB", (128, 128), (0, 0, 0))
        m.upscale(src, scale=4, prompt="sharper than the original")
    assert fake_pipe.call_args.kwargs["prompt"] == "sharper than the original"


def test_runtime_defaults_empty_prompt():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
        )
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4)
    assert fake_pipe.call_args.kwargs["prompt"] == ""


def test_runtime_uses_seeded_generator():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    fake_torch = MagicMock()
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        fake_torch,
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
        )
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4, seed=42)
    fake_torch.Generator.return_value.manual_seed.assert_called_with(42)


def test_runtime_honors_custom_steps_and_guidance():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
            default_steps=20, default_guidance=9.0,
        )
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4, steps=5, guidance=3.0)
    assert fake_pipe.call_args.kwargs["num_inference_steps"] == 5
    assert fake_pipe.call_args.kwargs["guidance_scale"] == 3.0


def test_runtime_supported_scales_from_capability():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        m = DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
            supported_scales=[2, 4],
        )
    assert m.supported_scales == [2, 4]


def test_runtime_accepts_unknown_kwargs():
    """Future catalog kwargs must be absorbed by **_."""
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    with patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ".StableDiffusionUpscalePipeline",
        fake_cls,
    ), patch(
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler.torch",
        MagicMock(),
    ):
        from muse.modalities.image_upscale.runtimes.diffusers_upscaler import (
            DiffusersUpscaleRuntime,
        )
        # Should not TypeError on unknown future kwarg
        DiffusersUpscaleRuntime(
            hf_repo="x", local_dir="/fake", device="cpu", model_id="x",
            future_param="ignored",
        )
```

- [ ] **Step 2: Write `runtimes/__init__.py` (empty)**

```python
"""Generic runtime classes for the image/upscale modality."""
```

- [ ] **Step 3: Write `runtimes/diffusers_upscaler.py`**

Mirror the DiffusersText2ImageModel pattern:
- module-level sentinels: `torch`, `StableDiffusionUpscalePipeline`
- `_ensure_deps()` lazy-imports both, short-circuiting when sentinels are non-None (mocks)
- `_select_device(device)` standard helper
- `class DiffusersUpscaleRuntime` with constructor + `supported_scales` property + `upscale()` method
- `upscale()` calls `self._pipe(image=image, prompt=prompt or "", num_inference_steps=..., guidance_scale=..., generator=...)` and reads `out.images[0]` for the upscaled image
- returns `UpscaleResult(image, original_width, original_height, upscaled_width, upscaled_height, scale, seed, metadata)`

- [ ] **Step 4: Run targeted + fast lane**

```bash
pytest tests/modalities/image_upscale/runtimes/ -v
pytest tests/ -q -m "not slow"
```

- [ ] **Step 5: Commit (Task B)**

```
feat(image-upscale): DiffusersUpscaleRuntime (#147)
```

---

## Task C: Routes (multipart) + ImageUpscaleClient + modality `__init__.py`

Mount POST /v1/images/upscale; FastAPI multipart parsing; ImageUpscaleClient. Update `__init__.py` to export the full surface.

**Files:**
- Create: `src/muse/modalities/image_upscale/routes.py`
- Create: `src/muse/modalities/image_upscale/client.py`
- Modify: `src/muse/modalities/image_upscale/__init__.py`
- Create: `tests/modalities/image_upscale/test_routes.py`
- Create: `tests/modalities/image_upscale/test_client.py`

- [ ] **Step 1: Write failing route tests**

`tests/modalities/image_upscale/test_routes.py`:

```python
"""Tests for /v1/images/upscale router."""
import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_upscale.protocol import UpscaleResult
from muse.modalities.image_upscale.routes import build_router


class RecordingUpscaleModel:
    def __init__(self, model_id="fake-upscale", supported_scales=(4,)):
        self.model_id = model_id
        self.supported_scales = list(supported_scales)
        self.calls: list[dict] = []

    def upscale(self, image, *, scale=None, prompt=None,
                negative_prompt=None, steps=None, guidance=None,
                seed=None, **kwargs):
        self.calls.append({
            "image": image, "scale": scale, "prompt": prompt,
            "negative_prompt": negative_prompt, "steps": steps,
            "guidance": guidance, "seed": seed, **kwargs,
        })
        ow, oh = image.size
        scl = scale or 4
        out = Image.new("RGB", (ow * scl, oh * scl), (60, 60, 60))
        return UpscaleResult(
            image=out,
            original_width=ow, original_height=oh,
            upscaled_width=ow * scl, upscaled_height=oh * scl,
            scale=scl, seed=seed if seed is not None else -1,
            metadata={"prompt": prompt or ""},
        )


def _png_bytes(width=64, height=64, color=(0, 128, 255)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def model():
    return RecordingUpscaleModel(supported_scales=(4,))


@pytest.fixture
def client(model):
    reg = ModalityRegistry()
    reg.register(
        "image/upscale", model,
        manifest={
            "model_id": model.model_id,
            "modality": "image/upscale",
            "capabilities": {
                "supported_scales": [4],
                "default_scale": 4,
            },
        },
    )
    app = create_app(
        registry=reg,
        routers={"image/upscale": build_router(reg)},
    )
    return TestClient(app)


def test_post_upscale_returns_envelope(client, model):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(64, 64), "image/png")},
        data={"model": "fake-upscale", "scale": "4"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "data" in body
    assert len(body["data"]) == 1
    entry = body["data"][0]
    assert "b64_json" in entry
    import base64
    decoded = base64.b64decode(entry["b64_json"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(model.calls) == 1


def test_post_upscale_response_format_url(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-upscale", "scale": "4", "response_format": "url"},
    )
    assert r.status_code == 200
    url = r.json()["data"][0]["url"]
    assert url.startswith("data:image/png;base64,")


def test_post_upscale_n_creates_multiple_entries(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-upscale", "scale": "4", "n": "3"},
    )
    assert r.status_code == 200
    assert len(r.json()["data"]) == 3


def test_post_upscale_revised_prompt_echoes(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-upscale", "scale": "4",
            "prompt": "make it crisp",
        },
    )
    assert r.status_code == 200
    assert r.json()["data"][0]["revised_prompt"] == "make it crisp"


def test_post_upscale_unknown_model_returns_404(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "no-such", "scale": "4"},
    )
    assert r.status_code == 404
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == "model_not_found"


def test_post_upscale_unsupported_scale_returns_400(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-upscale", "scale": "8"},
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_parameter"
    assert "supported scales" in err["message"].lower() or "scale" in err["message"].lower()


def test_post_upscale_empty_image_returns_400(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", b"", "image/png")},
        data={"model": "fake-upscale", "scale": "4"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_parameter"


def test_post_upscale_malformed_image_returns_400(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", b"not an image", "image/png")},
        data={"model": "fake-upscale", "scale": "4"},
    )
    assert r.status_code == 400


def test_post_upscale_oversize_input_returns_400(client, monkeypatch):
    monkeypatch.setenv("MUSE_UPSCALE_MAX_INPUT_SIDE", "32")
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(64, 64), "image/png")},
        data={"model": "fake-upscale", "scale": "4"},
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_parameter"
    assert "too large" in err["message"].lower() or "max" in err["message"].lower()


def test_post_upscale_n_over_limit_rejected(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-upscale", "scale": "4", "n": "100"},
    )
    assert r.status_code in (400, 422)


def test_post_upscale_passes_prompt_steps_guidance_seed_to_backend(client, model):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={
            "model": "fake-upscale", "scale": "4",
            "prompt": "high detail",
            "negative_prompt": "blurry",
            "steps": "15", "guidance": "7.5", "seed": "42",
        },
    )
    assert r.status_code == 200
    call = model.calls[0]
    assert call["prompt"] == "high detail"
    assert call["negative_prompt"] == "blurry"
    assert call["steps"] == 15
    assert call["guidance"] == 7.5
    assert call["seed"] == 42


def test_post_upscale_response_includes_created_unix_timestamp(client):
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(), "image/png")},
        data={"model": "fake-upscale", "scale": "4"},
    )
    assert r.status_code == 200
    created = r.json()["created"]
    assert isinstance(created, int)
    assert created > 1577836800  # 2020-01-01 UTC
```

- [ ] **Step 2: Write `routes.py`**

```python
"""FastAPI router for /v1/images/upscale (multipart)."""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from threading import Lock

from fastapi import APIRouter, File, Form, UploadFile

from muse.core.errors import ModelNotFoundError, error_response
from muse.core.registry import ModalityRegistry
from muse.modalities.image_generation.image_input import decode_image_file
from muse.modalities.image_upscale.codec import to_bytes, to_data_url

logger = logging.getLogger(__name__)

MODALITY = "image/upscale"
_inference_lock = Lock()


def _max_input_side() -> int:
    """Read the per-request input-side cap from the environment.

    Default 1024. Tunable via MUSE_UPSCALE_MAX_INPUT_SIDE so users with
    more VRAM can lift the cap without a code change.
    """
    raw = os.environ.get("MUSE_UPSCALE_MAX_INPUT_SIDE", "1024")
    try:
        v = int(raw)
        return v if v > 0 else 1024
    except ValueError:
        return 1024


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=["image/upscale"])

    @router.post("/upscale")
    async def upscale(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        scale: int = Form(4),
        prompt: str = Form(""),
        negative_prompt: str | None = Form(None),
        steps: int | None = Form(None),
        guidance: float | None = Form(None),
        seed: int | None = Form(None),
        n: int = Form(1),
        response_format: str = Form("b64_json"),
    ):
        # Validate Form fields. Pydantic can't validate a multipart body
        # directly; we mirror the route hand-validation pattern used by
        # /v1/images/edits.
        if not (1 <= n <= 4):
            return error_response(
                400, "invalid_parameter", "n must be in [1, 4]",
            )
        if response_format not in ("b64_json", "url"):
            return error_response(
                400, "invalid_parameter",
                "response_format must be 'b64_json' or 'url'",
            )
        if len(prompt) > 4000:
            return error_response(
                400, "invalid_parameter",
                "prompt must be 0 to 4000 characters",
            )
        if negative_prompt is not None and len(negative_prompt) > 4000:
            return error_response(
                400, "invalid_parameter",
                "negative_prompt must be 0 to 4000 characters",
            )
        if steps is not None and not (1 <= steps <= 100):
            return error_response(
                400, "invalid_parameter",
                "steps must be in [1, 100]",
            )
        if guidance is not None and not (0.0 <= guidance <= 20.0):
            return error_response(
                400, "invalid_parameter",
                "guidance must be in [0.0, 20.0]",
            )

        try:
            backend = registry.get(MODALITY, model)
        except KeyError:
            raise ModelNotFoundError(model_id=model or "<default>", modality=MODALITY)

        effective_id = getattr(backend, "model_id", None) or (model or "<default>")
        manifest = registry.manifest(MODALITY, effective_id) or {}
        capabilities = manifest.get("capabilities", {}) or {}
        supported = list(capabilities.get("supported_scales") or [4])
        if scale not in supported:
            return error_response(
                400, "invalid_parameter",
                f"model {effective_id!r} only supports scales: {supported}",
            )

        try:
            init_image = await decode_image_file(image)
        except ValueError as e:
            return error_response(
                400, "invalid_parameter", f"image decode failed: {e}",
            )

        max_side = _max_input_side()
        ow, oh = init_image.size
        if ow > max_side or oh > max_side:
            return error_response(
                400, "invalid_parameter",
                f"image too large: {ow}x{oh} exceeds max input side "
                f"{max_side} (set MUSE_UPSCALE_MAX_INPUT_SIDE to raise)",
            )

        def _call_one(seed_offset: int):
            kwargs: dict = {
                "scale": scale,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance": guidance,
            }
            if seed is not None:
                kwargs["seed"] = seed + seed_offset
            with _inference_lock:
                return backend.upscale(init_image, **kwargs)

        results = []
        for i in range(n):
            results.append(await asyncio.to_thread(_call_one, i))

        data = []
        for r in results:
            entry: dict = {"revised_prompt": prompt or None}
            if response_format == "url":
                entry["url"] = to_data_url(r.image, fmt="png")
            else:
                entry["b64_json"] = base64.b64encode(to_bytes(r.image, fmt="png")).decode()
            data.append(entry)

        return {"created": int(time.time()), "data": data}

    return router
```

- [ ] **Step 3: Write failing client tests**

`tests/modalities/image_upscale/test_client.py`:

```python
"""Tests for ImageUpscaleClient (HTTP, multipart)."""
import base64
import io
import os
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from muse.modalities.image_upscale.client import ImageUpscaleClient


def _png_bytes(width=64, height=64, color=(10, 20, 30)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _server_response(n=1):
    out = _png_bytes(256, 256)
    return {
        "created": 1730000000,
        "data": [
            {"b64_json": base64.b64encode(out).decode(), "revised_prompt": "x"}
            for _ in range(n)
        ],
    }


def test_default_base_url():
    c = ImageUpscaleClient()
    assert c.base_url == "http://localhost:8000"


def test_muse_server_env_var(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://example.com:9000/")
    c = ImageUpscaleClient()
    assert c.base_url == "http://example.com:9000"


def test_explicit_base_url_overrides_env(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://ignored")
    c = ImageUpscaleClient(base_url="http://explicit.example/")
    assert c.base_url == "http://explicit.example"


def test_upscale_posts_multipart_and_returns_bytes():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = _server_response()
    with patch("muse.modalities.image_upscale.client.requests.post",
               return_value=fake_resp) as mock_post:
        c = ImageUpscaleClient(base_url="http://localhost:8000")
        result = c.upscale(image=_png_bytes(), model="x", scale=4, prompt="sharp")
    assert isinstance(result, list)
    assert isinstance(result[0], (bytes, bytearray))
    args, kwargs = mock_post.call_args
    assert args[0] == "http://localhost:8000/v1/images/upscale"
    assert "files" in kwargs
    assert "image" in kwargs["files"]
    assert "data" in kwargs


def test_upscale_data_carries_scale_and_prompt():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = _server_response()
    with patch("muse.modalities.image_upscale.client.requests.post",
               return_value=fake_resp) as mock_post:
        c = ImageUpscaleClient(base_url="http://localhost:8000")
        c.upscale(image=_png_bytes(), model="m1", scale=4, prompt="hi", n=2)
    data = dict(mock_post.call_args.kwargs["data"])
    assert data["scale"] == "4"
    assert data["prompt"] == "hi"
    assert data["model"] == "m1"
    assert data["n"] == "2"


def test_upscale_url_response_format_decodes_back_to_bytes():
    out = _png_bytes(256, 256)
    body = {
        "created": 1730000000,
        "data": [{
            "url": "data:image/png;base64," + base64.b64encode(out).decode(),
            "revised_prompt": None,
        }],
    }
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = body
    with patch("muse.modalities.image_upscale.client.requests.post",
               return_value=fake_resp):
        c = ImageUpscaleClient(base_url="http://localhost:8000")
        result = c.upscale(image=_png_bytes(), model="x", scale=4,
                           response_format="url")
    assert result[0][:8] == b"\x89PNG\r\n\x1a\n"


def test_upscale_non_200_raises():
    fake_resp = MagicMock(status_code=500, text="boom")
    with patch("muse.modalities.image_upscale.client.requests.post",
               return_value=fake_resp):
        c = ImageUpscaleClient(base_url="http://localhost:8000")
        with pytest.raises(RuntimeError, match="500"):
            c.upscale(image=_png_bytes(), model="x", scale=4)
```

- [ ] **Step 4: Write `client.py`**

```python
"""HTTP client for /v1/images/upscale.

Mirrors the multipart shape used by ImageEditsClient and
ImageVariationsClient. Default response_format is b64_json so
upscale() returns raw PNG bytes.
"""
from __future__ import annotations

import base64
import os
from typing import Any

import requests


def _resolve_base_url(base_url: str | None) -> str:
    base = base_url or os.environ.get("MUSE_SERVER", "http://localhost:8000")
    return base.rstrip("/")


class ImageUpscaleClient:
    """Thin HTTP client against the muse images.upscale endpoint."""

    def __init__(self, base_url: str | None = None, timeout: float = 600.0) -> None:
        self.base_url = _resolve_base_url(base_url)
        self.timeout = timeout

    def upscale(
        self,
        *,
        image: bytes,
        model: str | None = None,
        scale: int = 4,
        prompt: str = "",
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        n: int = 1,
        response_format: str = "b64_json",
    ) -> list[bytes]:
        """Upscale `image` by `scale`. Returns a list of raw PNG bytes."""
        files = {"image": ("image.png", image, "image/png")}
        data: list[tuple[str, str]] = [
            ("scale", str(scale)),
            ("prompt", prompt),
            ("n", str(n)),
            ("response_format", response_format),
        ]
        if model is not None:
            data.append(("model", model))
        if negative_prompt is not None:
            data.append(("negative_prompt", negative_prompt))
        if steps is not None:
            data.append(("steps", str(steps)))
        if guidance is not None:
            data.append(("guidance", str(guidance)))
        if seed is not None:
            data.append(("seed", str(seed)))

        r = requests.post(
            f"{self.base_url}/v1/images/upscale",
            files=files, data=data, timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"server returned {r.status_code}: {r.text[:500]}")

        entries = r.json()["data"]
        if response_format == "b64_json":
            return [base64.b64decode(e["b64_json"]) for e in entries]
        out: list[bytes] = []
        for e in entries:
            url = e["url"]
            comma = url.index(",")
            out.append(base64.b64decode(url[comma + 1:]))
        return out
```

- [ ] **Step 5: Update modality `__init__.py` (full surface)**

```python
"""Image upscale modality: image-to-image super-resolution.
... (full docstring as in spec)
"""
from muse.modalities.image_upscale.client import ImageUpscaleClient
from muse.modalities.image_upscale.protocol import (
    ImageUpscaleModel,
    UpscaleResult,
)
from muse.modalities.image_upscale.routes import build_router

MODALITY = "image/upscale"


def _make_probe_image():
    from PIL import Image
    return Image.new("RGB", (128, 128), (128, 128, 128))


PROBE_DEFAULTS = {
    "shape": "128x128 -> 512x512 (4x), 20 steps",
    "call": lambda m: m.upscale(_make_probe_image(), scale=4),
}

__all__ = [
    "MODALITY",
    "PROBE_DEFAULTS",
    "build_router",
    "ImageUpscaleClient",
    "ImageUpscaleModel",
    "UpscaleResult",
]
```

- [ ] **Step 6: Run targeted + fast lane**

```bash
pytest tests/modalities/image_upscale/test_routes.py tests/modalities/image_upscale/test_client.py -v
pytest tests/ -q -m "not slow"
```

- [ ] **Step 7: Commit (Task C)**

```
feat(image-upscale): /v1/images/upscale routes + ImageUpscaleClient (#147)
```

---

## Task D: Bundled `stable_diffusion_x4_upscaler.py`

The bundled SD x4 upscaler script. Lazy imports; mirrors sd_turbo.py structure.

**Files:**
- Create: `src/muse/models/stable_diffusion_x4_upscaler.py`
- Create: `tests/models/test_stable_diffusion_x4_upscaler.py`

- [ ] **Step 1: Write failing tests**

`tests/models/test_stable_diffusion_x4_upscaler.py`:

```python
"""Tests for the SD x4 Upscaler model script (fully mocked)."""
import importlib
from unittest.mock import MagicMock, patch

from PIL import Image


def _patched_pipe(out_size=(512, 512)):
    fake_img = Image.new("RGB", out_size, (40, 40, 40))
    fake_pipe = MagicMock()
    fake_pipe.return_value.images = [fake_img]
    fake_pipe.to = MagicMock(return_value=fake_pipe)
    return fake_pipe


def test_manifest_required_fields():
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    M = mod.MANIFEST
    assert M["model_id"] == "stable-diffusion-x4-upscaler"
    assert M["modality"] == "image/upscale"
    assert "hf_repo" in M
    assert "pip_extras" in M


def test_manifest_pip_extras_declares_torch_diffusers_transformers():
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    extras_str = " ".join(mod.MANIFEST["pip_extras"])
    assert "torch" in extras_str
    assert "diffusers" in extras_str
    assert "transformers" in extras_str


def test_manifest_advertises_supported_scales_4():
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    caps = mod.MANIFEST["capabilities"]
    assert caps["supported_scales"] == [4]
    assert caps["default_scale"] == 4


def test_model_loads_via_patched_pipeline():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        m = mod.Model(
            hf_repo="stabilityai/stable-diffusion-x4-upscaler",
            local_dir="/fake",
            device="cpu",
        )
    assert m.model_id == "stable-diffusion-x4-upscaler"
    assert m.supported_scales == [4]
    fake_cls.from_pretrained.assert_called_once()


def test_model_uses_local_dir_over_hf_repo():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        mod.Model(
            hf_repo="stabilityai/stable-diffusion-x4-upscaler",
            local_dir="/real/local",
            device="cpu",
        )
    assert fake_cls.from_pretrained.call_args.args[0] == "/real/local"


def test_model_upscale_returns_upscale_result():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe(out_size=(512, 512))
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        m = mod.Model(
            hf_repo="stabilityai/stable-diffusion-x4-upscaler",
            local_dir="/fake",
            device="cpu",
        )
        src = Image.new("RGB", (128, 128))
        result = m.upscale(src, scale=4)
    from muse.modalities.image_upscale.protocol import UpscaleResult
    assert isinstance(result, UpscaleResult)
    assert result.original_width == 128
    assert result.upscaled_width == 512
    assert result.scale == 4


def test_model_passes_prompt_to_pipeline():
    fake_pipe = _patched_pipe()
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = fake_pipe
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        m = mod.Model(
            hf_repo="x", local_dir="/fake", device="cpu",
        )
        src = Image.new("RGB", (128, 128))
        m.upscale(src, scale=4, prompt="razor sharp")
    assert fake_pipe.call_args.kwargs["prompt"] == "razor sharp"


def test_model_accepts_unknown_kwargs():
    fake_cls = MagicMock()
    fake_cls.from_pretrained.return_value = _patched_pipe()
    mod = importlib.import_module("muse.models.stable_diffusion_x4_upscaler")
    with patch.object(mod, "StableDiffusionUpscalePipeline", fake_cls), \
         patch.object(mod, "torch", MagicMock()):
        # Should not TypeError
        mod.Model(
            hf_repo="x", local_dir="/fake", device="cpu",
            extra_future_param="ignored",
        )
```

- [ ] **Step 2: Write `stable_diffusion_x4_upscaler.py`**

```python
"""stabilityai/stable-diffusion-x4-upscaler bundled model script.

4x latent diffusion super-resolution. Apache 2.0. ~3GB on disk.
Uses diffusers.StableDiffusionUpscalePipeline. Roughly 30-60s per
512->2048 image at 20 steps on a 12GB GPU.

Lazy imports torch + diffusers; mirrors sd_turbo.py.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.image_upscale.protocol import UpscaleResult


logger = logging.getLogger(__name__)


# Sentinels patched by tests; populated by _ensure_deps at runtime.
torch: Any = None
StableDiffusionUpscalePipeline: Any = None


def _ensure_deps() -> None:
    global torch, StableDiffusionUpscalePipeline
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("sd-x4-upscaler: torch unavailable: %s", e)
    if StableDiffusionUpscalePipeline is None:
        try:
            from diffusers import StableDiffusionUpscalePipeline as _p
            StableDiffusionUpscalePipeline = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("sd-x4-upscaler: diffusers unavailable: %s", e)


MANIFEST = {
    "model_id": "stable-diffusion-x4-upscaler",
    "modality": "image/upscale",
    "hf_repo": "stabilityai/stable-diffusion-x4-upscaler",
    "description": "SD x4 upscaler: 4x super-resolution via latent diffusion, Apache 2.0",
    "license": "Apache 2.0",
    "pip_extras": (
        "torch>=2.1.0",
        "diffusers>=0.27.0",
        "transformers>=4.36.0",
        "accelerate",
        "Pillow",
        "safetensors",
    ),
    "system_packages": (),
    "capabilities": {
        "device": "cuda",
        "default_scale": 4,
        "supported_scales": [4],
        "default_steps": 20,
        "default_guidance": 9.0,
        "memory_gb": 6.0,
    },
    "allow_patterns": [
        "*.safetensors", "*.json", "*.txt", "*.md",
        "feature_extractor/*.json",
        "scheduler/*.json",
        "text_encoder/*.fp16.safetensors", "text_encoder/*.json",
        "tokenizer/*",
        "unet/*.fp16.safetensors", "unet/*.json",
        "vae/*.fp16.safetensors", "vae/*.json",
    ],
}


def _select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Model:
    """SD x4 upscaler backend."""

    model_id = MANIFEST["model_id"]
    supported_scales = list(MANIFEST["capabilities"]["supported_scales"])

    def __init__(
        self,
        *,
        hf_repo: str = MANIFEST["hf_repo"],
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        default_steps: int = 20,
        default_guidance: float = 9.0,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if StableDiffusionUpscalePipeline is None:
            raise RuntimeError(
                "diffusers is not installed; run `muse pull stable-diffusion-x4-upscaler`"
            )
        self._device = _select_device(device)
        import muse.models.stable_diffusion_x4_upscaler as _self_mod
        _torch = _self_mod.torch
        torch_dtype = None
        if _torch is not None:
            torch_dtype = {
                "float16": _torch.float16,
                "float32": _torch.float32,
                "bfloat16": _torch.bfloat16,
            }[dtype]
        self._src = local_dir or hf_repo
        self._dtype = dtype
        self._torch_dtype = torch_dtype
        self._default_steps = default_steps
        self._default_guidance = default_guidance
        logger.info(
            "loading SD x4 upscaler from %s (device=%s, dtype=%s)",
            self._src, self._device, dtype,
        )
        self._pipe = StableDiffusionUpscalePipeline.from_pretrained(
            self._src,
            torch_dtype=torch_dtype,
        )
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

    def upscale(
        self,
        image: Any,
        *,
        scale: int | None = None,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        seed: int | None = None,
        **_: Any,
    ) -> UpscaleResult:
        ow, oh = image.size
        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance
        prompt_str = prompt if prompt is not None else ""
        actual_scale = scale if scale is not None else 4

        gen = None
        if seed is not None:
            import muse.models.stable_diffusion_x4_upscaler as _self_mod
            _torch = _self_mod.torch
            if _torch is not None:
                gen = _torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict = {
            "prompt": prompt_str,
            "image": image,
            "num_inference_steps": n_steps,
            "guidance_scale": cfg,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if gen is not None:
            call_kwargs["generator"] = gen

        out = self._pipe(**call_kwargs)
        upscaled = out.images[0]
        uw, uh = upscaled.size
        return UpscaleResult(
            image=upscaled,
            original_width=ow,
            original_height=oh,
            upscaled_width=uw,
            upscaled_height=uh,
            scale=actual_scale,
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt_str,
                "steps": n_steps,
                "guidance": cfg,
                "model": self.model_id,
            },
        )
```

- [ ] **Step 3: Run targeted + fast lane**

```bash
pytest tests/models/test_stable_diffusion_x4_upscaler.py -v
pytest tests/ -q -m "not slow"
```

- [ ] **Step 4: Commit (Task D)**

```
feat(image-upscale): bundled stable_diffusion_x4_upscaler script (#147)
```

---

## Task E: HF resolver plugin

Sniff diffusers-shape upscalers; priority 105.

**Files:**
- Create: `src/muse/modalities/image_upscale/hf.py`
- Create: `tests/modalities/image_upscale/test_hf_plugin.py`

- [ ] **Step 1: Write failing plugin tests**

```python
"""Tests for image_upscale HF plugin."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from muse.modalities.image_upscale.hf import HF_PLUGIN


def _fake_info(*, repo_id="x", siblings=(), tags=()):
    return SimpleNamespace(
        id=repo_id,
        siblings=[SimpleNamespace(rfilename=f) for f in siblings],
        tags=list(tags),
        card_data=None,
    )


def test_priority_is_105():
    assert HF_PLUGIN["priority"] == 105


def test_modality_is_image_upscale():
    assert HF_PLUGIN["modality"] == "image/upscale"


def test_sniff_x4_upscaler():
    info = _fake_info(
        repo_id="stabilityai/stable-diffusion-x4-upscaler",
        siblings=["model_index.json", "unet/diffusion_pytorch_model.safetensors"],
        tags=["image-to-image", "diffusers"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_rejects_non_upscaler_diffusers_repo():
    info = _fake_info(
        repo_id="runwayml/stable-diffusion-v1-5",
        siblings=["model_index.json"],
        tags=["image-to-image", "diffusers"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_rejects_repo_without_model_index():
    info = _fake_info(
        repo_id="user/some-upscaler",
        siblings=["other.txt"],
        tags=["image-to-image"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_rejects_repo_without_i2i_tag():
    info = _fake_info(
        repo_id="stabilityai/stable-diffusion-x4-upscaler",
        siblings=["model_index.json"],
        tags=["text-to-image"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_aura_currently_excluded_until_runtime_lands():
    """AuraSR repos don't ship model_index.json; if one did, the plugin
    still wouldn't claim it because no Aura runtime exists in v0.25.0."""
    info = _fake_info(
        repo_id="fal/AuraSR-v2",
        siblings=["aura_sr.py", "model.safetensors"],
        tags=["image-to-image"],
    )
    assert HF_PLUGIN["sniff"](info) is False  # missing model_index.json


def test_resolve_x4_upscaler_capabilities():
    info = _fake_info(
        repo_id="stabilityai/stable-diffusion-x4-upscaler",
        siblings=["model_index.json"],
        tags=["image-to-image"],
    )
    resolved = HF_PLUGIN["resolve"](
        "stabilityai/stable-diffusion-x4-upscaler", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["default_scale"] == 4
    assert caps["supported_scales"] == [4]
    assert caps["default_steps"] == 20
    assert caps["default_guidance"] == 9.0


def test_resolve_fallback_capabilities():
    info = _fake_info(
        repo_id="user/some-other-upscaler",
        siblings=["model_index.json"],
        tags=["image-to-image"],
    )
    resolved = HF_PLUGIN["resolve"](
        "user/some-other-upscaler", None, info,
    )
    caps = resolved.manifest["capabilities"]
    assert caps["default_scale"] == 4
    assert caps["supported_scales"] == [4]


def test_resolve_runtime_path_points_at_diffusers_upscaler():
    info = _fake_info(
        repo_id="stabilityai/stable-diffusion-x4-upscaler",
        siblings=["model_index.json"],
        tags=["image-to-image"],
    )
    resolved = HF_PLUGIN["resolve"](
        "stabilityai/stable-diffusion-x4-upscaler", None, info,
    )
    assert resolved.backend_path == (
        "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
        ":DiffusersUpscaleRuntime"
    )


def test_resolve_pip_extras_includes_diffusers_and_transformers():
    info = _fake_info(
        repo_id="stabilityai/stable-diffusion-x4-upscaler",
        siblings=["model_index.json"],
        tags=["image-to-image"],
    )
    resolved = HF_PLUGIN["resolve"](
        "stabilityai/stable-diffusion-x4-upscaler", None, info,
    )
    extras = " ".join(resolved.manifest["pip_extras"])
    assert "diffusers" in extras
    assert "transformers" in extras
    assert "torch" in extras


def test_search_yields_results():
    api = MagicMock()
    api.list_models.return_value = [
        SimpleNamespace(id="stabilityai/stable-diffusion-x4-upscaler", downloads=1000),
        SimpleNamespace(id="runwayml/stable-diffusion-v1-5", downloads=2000),  # not an upscaler
    ]
    results = list(HF_PLUGIN["search"](api, "upscaler", sort="downloads", limit=10))
    # Only the upscaler is yielded; the SD v1.5 repo is filtered out
    assert any(r.uri.endswith("stable-diffusion-x4-upscaler") for r in results)
    assert not any(r.uri.endswith("stable-diffusion-v1-5") for r in results)
```

- [ ] **Step 2: Write `hf.py`**

```python
"""HF resolver plugin for diffusers-shape image upscalers.

Sniffs HF repos for `model_index.json` (the diffusers pipeline config)
plus the `image-to-image` tag plus an upscaler-name allowlist
(upscaler / super-resolution / esrgan / upscale / x4-upscaler /
ldm-super). Synthesizes a manifest with capabilities inferred from
repo name (x4 -> [4]; fallback -> [4]).

Priority 105: between image/animation (110) and image/generation (100),
so upscaler repos get correctly classified even though they share the
diffusers `model_index.json` shape and the `image-to-image` tag with
regular i2i checkpoints.

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult


_RUNTIME_PATH = (
    "muse.modalities.image_upscale.runtimes.diffusers_upscaler"
    ":DiffusersUpscaleRuntime"
)
_PIP_EXTRAS = (
    "torch>=2.1.0",
    "diffusers>=0.27.0",
    "transformers>=4.36.0",
    "accelerate",
    "Pillow",
    "safetensors",
)
_UPSCALER_NAMES = (
    "upscaler", "super-resolution", "esrgan", "upscale",
    "x4-upscaler", "ldm-super",
)


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _is_upscaler_name(repo_id: str) -> bool:
    rid = (repo_id or "").lower()
    return any(s in rid for s in _UPSCALER_NAMES)


def _infer_defaults(repo_id: str) -> dict[str, Any]:
    """Sensible per-pattern defaults so each upscaler model lands with
    reasonable scale / steps / guidance defaults out of the box."""
    rid = repo_id.lower()
    if "x4-upscaler" in rid or "x4" in rid:
        return {
            "default_scale": 4,
            "supported_scales": [4],
            "default_steps": 20,
            "default_guidance": 9.0,
        }
    return {
        "default_scale": 4,
        "supported_scales": [4],
        "default_steps": 20,
        "default_guidance": 9.0,
    }


def _sniff(info) -> bool:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    has_pipeline_config = any(
        Path(f).name == "model_index.json" for f in siblings
    )
    has_i2i_tag = "image-to-image" in tags
    return has_pipeline_config and has_i2i_tag and _is_upscaler_name(
        getattr(info, "id", "") or ""
    )


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    defaults = _infer_defaults(repo_id)
    capabilities = {
        **defaults,
        "device": "cuda",
        "memory_gb": 6.0,
    }
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "image/upscale",
        "hf_repo": repo_id,
        "description": f"Diffusers super-resolution upscaler: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }

    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    has_fp16_variants = any(".fp16." in name for name in siblings)

    def _download(cache_root: Path) -> Path:
        allow_patterns = [
            "model_index.json",
            "*/*.json",
            "*/*.txt",
        ]
        if has_fp16_variants:
            allow_patterns.extend([
                "*/*.fp16.safetensors",
                "*/*.fp16.bin",
            ])
        else:
            allow_patterns.extend([
                "*/*.safetensors",
                "*/*.bin",
            ])
        return Path(snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_root) if cache_root else None,
            allow_patterns=allow_patterns,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    repos = api.list_models(
        search=query, filter="image-to-image",
        sort=sort, limit=limit,
    )
    for repo in repos:
        if not _is_upscaler_name(getattr(repo, "id", "") or ""):
            continue
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="image/upscale",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


HF_PLUGIN = {
    "modality": "image/upscale",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 105,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
```

- [ ] **Step 3: Run targeted + fast lane**

```bash
pytest tests/modalities/image_upscale/test_hf_plugin.py -v
pytest tests/ -q -m "not slow"
```

- [ ] **Step 4: Commit (Task E)**

```
feat(image-upscale): HF resolver plugin (priority 105) (#147)
```

---

## Task F: Curated entry

Add `stable-diffusion-x4-upscaler` to `curated.yaml`.

**Files:**
- Modify: `src/muse/curated.yaml`

- [ ] **Step 1: Append entry**

Add a new section at the bottom of `curated.yaml`:

```yaml
# ---------- image/upscale (super-resolution) ----------

- id: stable-diffusion-x4-upscaler
  bundled: true
```

- [ ] **Step 2: Run fast lane**

```bash
pytest tests/ -q -m "not slow"
```

The existing curated tests should pick up the entry. If a test asserts a specific count of curated entries, update it.

- [ ] **Step 3: Commit (Task F)**

```
feat(image-upscale): curate stable-diffusion-x4-upscaler (#147)
```

---

## Task G: Slow e2e + integration tests

Slow e2e exercises the full multipart -> FastAPI -> codec chain in-process with a fake backend. Integration tests hit a real muse server (opt-in via MUSE_REMOTE_SERVER).

**Files:**
- Create: `tests/cli_impl/test_e2e_image_upscale.py`
- Create: `tests/integration/test_remote_image_upscale.py`

- [ ] **Step 1: Write slow e2e**

```python
"""Slow e2e: full multipart -> FastAPI -> codec round-trip."""
import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from muse.core.registry import ModalityRegistry
from muse.core.server import create_app
from muse.modalities.image_upscale.protocol import UpscaleResult
from muse.modalities.image_upscale.routes import build_router


class FakeUpscaler:
    model_id = "fake-upscaler"
    supported_scales = [4]

    def upscale(self, image, *, scale=None, **kwargs):
        ow, oh = image.size
        scl = scale or 4
        out = Image.new("RGB", (ow * scl, oh * scl), (10, 200, 50))
        return UpscaleResult(
            image=out,
            original_width=ow, original_height=oh,
            upscaled_width=ow * scl, upscaled_height=oh * scl,
            scale=scl, seed=-1,
            metadata={"prompt": kwargs.get("prompt") or ""},
        )


def _png_bytes(width=32, height=32):
    img = Image.new("RGB", (width, height), (5, 10, 15))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.slow
def test_upscale_e2e_multipart_b64_json():
    reg = ModalityRegistry()
    reg.register(
        "image/upscale", FakeUpscaler(),
        manifest={
            "model_id": "fake-upscaler",
            "modality": "image/upscale",
            "capabilities": {"supported_scales": [4], "default_scale": 4},
        },
    )
    app = create_app(
        registry=reg,
        routers={"image/upscale": build_router(reg)},
    )
    client = TestClient(app)
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(64, 64), "image/png")},
        data={"model": "fake-upscaler", "scale": "4", "prompt": "neat"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "data" in body
    decoded = base64.b64decode(body["data"][0]["b64_json"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
    # Result should be 4x source dimensions
    out_img = Image.open(io.BytesIO(decoded))
    assert out_img.size == (256, 256)


@pytest.mark.slow
def test_upscale_e2e_multipart_url_response_format():
    reg = ModalityRegistry()
    reg.register(
        "image/upscale", FakeUpscaler(),
        manifest={
            "model_id": "fake-upscaler",
            "modality": "image/upscale",
            "capabilities": {"supported_scales": [4], "default_scale": 4},
        },
    )
    app = create_app(
        registry=reg,
        routers={"image/upscale": build_router(reg)},
    )
    client = TestClient(app)
    r = client.post(
        "/v1/images/upscale",
        files={"image": ("src.png", _png_bytes(32, 32), "image/png")},
        data={"model": "fake-upscaler", "scale": "4", "response_format": "url"},
    )
    assert r.status_code == 200
    url = r.json()["data"][0]["url"]
    assert url.startswith("data:image/png;base64,")
```

- [ ] **Step 2: Write integration test**

```python
"""Integration tests for /v1/images/upscale (opt-in via MUSE_REMOTE_SERVER)."""
import base64
import io
import os

import pytest
import httpx
from PIL import Image


pytestmark = pytest.mark.slow


@pytest.fixture(scope="session")
def remote_url():
    url = os.environ.get("MUSE_REMOTE_SERVER")
    if not url:
        pytest.skip("MUSE_REMOTE_SERVER not set")
    return url.rstrip("/")


@pytest.fixture(scope="session")
def remote_health(remote_url):
    try:
        r = httpx.get(f"{remote_url}/health", timeout=5.0)
        r.raise_for_status()
    except Exception:
        pytest.skip("muse server not reachable")
    return r.json()


@pytest.fixture(scope="session")
def upscale_model(remote_health):
    model_id = os.environ.get(
        "MUSE_UPSCALE_MODEL_ID", "stable-diffusion-x4-upscaler",
    )
    loaded = remote_health.get("models") or []
    if model_id not in loaded:
        pytest.skip(f"model {model_id!r} not loaded on the remote server")
    return model_id


def _png_bytes(width=128, height=128, color=(40, 40, 100)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_protocol_upscale_returns_envelope(remote_url, upscale_model):
    src = _png_bytes(128, 128)
    r = httpx.post(
        f"{remote_url}/v1/images/upscale",
        files={"image": ("src.png", src, "image/png")},
        data={"model": upscale_model, "scale": "4", "prompt": ""},
        timeout=600.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "data" in body
    entry = body["data"][0]
    assert "b64_json" in entry
    decoded = base64.b64decode(entry["b64_json"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_observe_upscale_increases_resolution(remote_url, upscale_model):
    """4x upscale should yield a 4x larger image."""
    src = _png_bytes(128, 128)
    r = httpx.post(
        f"{remote_url}/v1/images/upscale",
        files={"image": ("src.png", src, "image/png")},
        data={"model": upscale_model, "scale": "4"},
        timeout=600.0,
    )
    assert r.status_code == 200
    decoded = base64.b64decode(r.json()["data"][0]["b64_json"])
    out_img = Image.open(io.BytesIO(decoded))
    assert out_img.size == (512, 512)
```

- [ ] **Step 3: Run fast + slow lanes**

```bash
pytest tests/ -q -m "not slow"
pytest tests/cli_impl/test_e2e_image_upscale.py -v
```

- [ ] **Step 4: Commit (Task G)**

```
test(image-upscale): slow e2e + opt-in integration (#147)
```

---

## Task H: Documentation + v0.25.0 release

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `pyproject.toml` (0.24.0 -> 0.25.0)
- Modify: `src/muse/__init__.py` (docstring v0.25.0 + image/upscale)

- [ ] **Step 1: Update CLAUDE.md**

Add image/upscale to the modality list:

```
- **image/upscale**: image-to-image super-resolution via `/v1/images/upscale` (Stable Diffusion x4 Upscaler; multipart upload, OpenAI-shape envelope; 4x diffusion-based upscaling)
```

Update the modality count (12 -> 13). Add a note in the Modality conventions section about `image_upscale/` being the third multipart consumer (after audio_transcription, image_generation).

- [ ] **Step 2: Update README.md**

Add image/upscale to the modality MIME tag list and add a usage example:

```bash
curl -s -X POST http://localhost:8000/v1/images/upscale \
  -F "image=@source.png" \
  -F "model=stable-diffusion-x4-upscaler" \
  -F "scale=4" \
  -F "prompt=high detail" \
  | jq -r '.data[0].b64_json' \
  | base64 -d > upscaled.png
```

```python
from muse.modalities.image_upscale import ImageUpscaleClient
from pathlib import Path

with open("source.png", "rb") as f:
    src = f.read()
result = ImageUpscaleClient().upscale(
    image=src,
    model="stable-diffusion-x4-upscaler",
    scale=4,
    prompt="razor sharp detail",
)
Path("upscaled.png").write_bytes(result[0])
```

- [ ] **Step 3: Bump version**

`pyproject.toml`: `version = "0.25.0"`. `src/muse/__init__.py` docstring: bump to v0.25.0 and add the `image/upscale` line.

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -q --timeout=300
```

- [ ] **Step 5: Em-dash + literal-eval check**

```bash
python - <<'PY'
import sys, pathlib
em = chr(0x2014)
banned = "ev" + "al"  # split to avoid the literal token in this script
hits = []
for p in pathlib.Path(".").rglob("*"):
    if not p.is_file():
        continue
    if any(seg in p.parts for seg in (".git", "node_modules", "__pycache__", ".venv", "venv")):
        continue
    if p.suffix not in (".py", ".md", ".yaml", ".yml", ".toml"):
        continue
    try:
        text = p.read_text()
    except Exception:
        continue
    # Only check files I added in this task
    if "image_upscale" in str(p) or "image-upscale" in str(p):
        if em in text:
            hits.append((str(p), "EM-DASH"))
        if banned in text:
            hits.append((str(p), "EVAL-TOKEN"))
sys.exit(1 if hits else 0)
PY
```

- [ ] **Step 6: Commit + tag + push + GitHub release**

```bash
git add CLAUDE.md README.md pyproject.toml src/muse/__init__.py
git commit -m "$(cat <<'EOF'
chore(release): v0.25.0

Adds the image/upscale modality, muse's 13th. Mounts POST
/v1/images/upscale (multipart/form-data) for diffusion-based
super-resolution. Bundles stabilityai/stable-diffusion-x4-upscaler
as the default model. The HF resolver plugin (priority 105) sniffs
diffusers-shape upscalers and synthesizes manifests with default_scale,
supported_scales, default_steps, and default_guidance per pattern.

The supported_scales capability gates the request scale parameter;
unsupported scales return 400. An env-tunable input-side cap
(MUSE_UPSCALE_MAX_INPUT_SIDE, default 1024) prevents runaway VRAM
usage.

GAN-based upscalers (AuraSR, Real-ESRGAN) are deferred to v1.next.

Closes #147.
EOF
)"

git tag -a v0.25.0 -m "v0.25.0: image/upscale modality (SD x4 upscaler)"
git push origin main
git push origin v0.25.0
gh release create v0.25.0 \
  --title "v0.25.0: image/upscale (SD x4 upscaler)" \
  --notes "..."
```

---

## Self-review checklist

1. **13 modalities total.** Discovery picks up `image_upscale/` because the directory contains an `__init__.py` exporting `MODALITY` + `build_router`.
2. **HF plugin coverage.** 12 of 13 modalities have an HF plugin (audio/speech bundles only Soprano/Kokoro/Bark which lack a clean HF sniff signal).
3. **Endpoint lives.** POST /v1/images/upscale, multipart, returns OpenAI-shape envelope.
4. **Bundled + curated.** stable-diffusion-x4-upscaler is bundled; curated.yaml aliases it.
5. **Capability gating.** supported_scales rejects unsupported scales with 400.
6. **Input cap.** MUSE_UPSCALE_MAX_INPUT_SIDE rejects oversized inputs with 400.
7. **No em-dashes.** Em-dash audit script exits zero.
8. **No literal eval token.** No file in image_upscale/ contains the bare 4-letter Python builtin token.
9. **Multipart inline.** Routes use FastAPI UploadFile + Form; same pattern as image_generation /edits.
10. **Per-task commits.** One commit per task A-H. Push only at v0.25.0.
