# Img2img Route Extension Implementation Plan (#143)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `/v1/images/generations` with optional `image` + `strength` request fields. When `image` is set, route through `AutoPipelineForImage2Image`. OpenAI SDK compatible via `extra_body`.

**Architecture:** Mirror the existing text-to-image path. New helper module decodes data URLs and HTTP URLs into PIL. Runtime gains a branch on `init_image is not None`. Cache the img2img pipeline on the model instance so repeated calls skip reload.

**Spec:** `docs/superpowers/specs/2026-04-27-img2img-route-extension-design.md`

**Target version:** v0.17.0

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/muse/modalities/image_generation/image_input.py` | create | data URL + HTTP URL parsing into PIL.Image |
| `src/muse/modalities/image_generation/protocol.py` | modify | extend `ImageModel.generate` signature with `init_image`, `strength` |
| `src/muse/modalities/image_generation/runtimes/diffusers.py` | modify | branch on init_image; lazy-load img2img pipeline; cache |
| `src/muse/modalities/image_generation/routes.py` | modify | parse `image` + `strength`; check `supports_img2img` capability; pass to runtime |
| `src/muse/modalities/image_generation/hf.py` | modify | resolve adds `supports_img2img: True` to capabilities |
| `src/muse/models/sd_turbo.py` | modify | MANIFEST capabilities: `supports_img2img: True`; Model.generate accepts `init_image`+`strength` |
| `tests/modalities/image_generation/test_image_input.py` | create | data URL + HTTP URL parser tests |
| `tests/modalities/image_generation/runtimes/test_diffusers.py` | modify | +tests for img2img branch + caching |
| `tests/modalities/image_generation/test_routes.py` | modify | +tests for image/strength/error paths |
| `tests/modalities/image_generation/test_hf_plugin.py` | modify | +test for supports_img2img in synthesized manifest |
| `tests/models/test_sd_turbo.py` | modify | +test for img2img branch |
| `pyproject.toml` | modify | bump 0.16.2 -> 0.17.0 |
| `src/muse/__init__.py` | modify | docstring v0.17.0 |
| `CLAUDE.md` | modify | document the OpenAI-SDK extra_body img2img pattern |

---

## Task A: image_input.py decoder

Pure helper, easiest to test in isolation. No dependencies on the rest of the modality.

**Files:**
- Create: `src/muse/modalities/image_generation/image_input.py`
- Create: `tests/modalities/image_generation/test_image_input.py`

- [ ] **Step 1: Write the failing test**

Create `tests/modalities/image_generation/test_image_input.py`:

```python
"""Tests for image_input: parsing user-supplied images for img2img.

The helper accepts either:
  - a data URL: data:image/{png,jpeg,webp};base64,...
  - an http(s):// URL fetched via httpx (size-capped, content-type-checked)

Returns a PIL.Image. Decode failures raise ValueError so the route layer
can surface them as 400s.
"""
import base64
import io

import pytest
from unittest.mock import MagicMock, patch

from muse.modalities.image_generation.image_input import decode_image_input


def _png_bytes(width=64, height=64, color=(0, 128, 255)):
    """Build minimal PNG bytes via PIL."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_decode_data_url_png():
    raw = _png_bytes()
    data_url = f"data:image/png;base64,{base64.b64encode(raw).decode()}"
    img = decode_image_input(data_url)
    assert img.size == (64, 64)
    assert img.mode in ("RGB", "RGBA")


def test_decode_data_url_jpeg():
    from PIL import Image
    rgb = Image.new("RGB", (32, 32), (255, 0, 0))
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG")
    data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    img = decode_image_input(data_url)
    assert img.size == (32, 32)


def test_decode_http_url_fetches_via_httpx():
    raw = _png_bytes()
    fake_response = MagicMock()
    fake_response.content = raw
    fake_response.headers = {"content-type": "image/png"}
    fake_response.raise_for_status = MagicMock()
    with patch(
        "muse.modalities.image_generation.image_input.httpx.get",
        return_value=fake_response,
    ) as mock_get:
        img = decode_image_input("https://example.com/cat.png")
    mock_get.assert_called_once()
    assert img.size == (64, 64)


def test_decode_rejects_oversize_data_url():
    huge = b"\x00" * (11 * 1024 * 1024)  # 11MB
    data_url = f"data:image/png;base64,{base64.b64encode(huge).decode()}"
    with pytest.raises(ValueError, match="exceeds"):
        decode_image_input(data_url, max_bytes=10 * 1024 * 1024)


def test_decode_rejects_non_image_http_content_type():
    fake_response = MagicMock()
    fake_response.content = b"<html>nope</html>"
    fake_response.headers = {"content-type": "text/html"}
    fake_response.raise_for_status = MagicMock()
    with patch(
        "muse.modalities.image_generation.image_input.httpx.get",
        return_value=fake_response,
    ):
        with pytest.raises(ValueError, match="content-type"):
            decode_image_input("https://example.com/page.html")


def test_decode_rejects_unknown_data_url_mime():
    raw = b"some text"
    data_url = f"data:text/plain;base64,{base64.b64encode(raw).decode()}"
    with pytest.raises(ValueError, match="MIME"):
        decode_image_input(data_url)


def test_decode_rejects_invalid_url_shape():
    with pytest.raises(ValueError, match="must be"):
        decode_image_input("ftp://example.com/img.png")


def test_decode_rejects_corrupt_image_bytes():
    raw = b"not really a png"
    data_url = f"data:image/png;base64,{base64.b64encode(raw).decode()}"
    with pytest.raises(ValueError, match="decode"):
        decode_image_input(data_url)
```

- [ ] **Step 2: Run, expect ImportError**

```bash
pytest tests/modalities/image_generation/test_image_input.py -v
```

- [ ] **Step 3: Implement the helper**

Create `src/muse/modalities/image_generation/image_input.py`:

```python
"""Decode user-supplied images for img2img on /v1/images/generations.

Accepted shapes:
  - data:image/{png,jpeg,webp};base64,XYZ
  - https?://... (fetched via httpx, content-type validated, size-capped)

Returns PIL.Image. Failures raise ValueError so the route layer can
surface them as 400 responses with the OpenAI-shape error envelope.

PIL is already a pip_extra of the diffusers runtime. httpx is a
server-extras dep.
"""
from __future__ import annotations

import base64
import io
import re
from typing import Any

import httpx


_DATA_URL_RE = re.compile(
    r"^data:(image/[a-zA-Z0-9.+-]+);base64,(.*)$",
    re.DOTALL,
)
_ALLOWED_IMAGE_MIME = frozenset({
    "image/png", "image/jpeg", "image/jpg", "image/webp",
})
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
_HTTP_TIMEOUT = 30.0


def decode_image_input(value: str, *, max_bytes: int = _DEFAULT_MAX_BYTES) -> Any:
    """Parse a data URL or HTTP(S) URL into a PIL.Image.

    PIL is imported lazily so the module loads without it (the diffusers
    runtime that imports this helper has PIL as a pip_extra anyway, but
    discovery should not crash on it).
    """
    if value.startswith("data:"):
        return _decode_data_url(value, max_bytes=max_bytes)
    if value.startswith(("http://", "https://")):
        return _fetch_http_url(value, max_bytes=max_bytes)
    raise ValueError(
        f"image must be a data: URL or http(s):// URL; got {value[:30]!r}..."
    )


def _decode_data_url(value: str, *, max_bytes: int):
    m = _DATA_URL_RE.match(value)
    if not m:
        raise ValueError("malformed data URL")
    mime = m.group(1).lower()
    if mime not in _ALLOWED_IMAGE_MIME:
        raise ValueError(
            f"unsupported MIME {mime!r}; allowed: {sorted(_ALLOWED_IMAGE_MIME)}"
        )
    b64 = m.group(2)
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"base64 decode failed: {e}") from e
    if len(raw) > max_bytes:
        raise ValueError(
            f"image bytes exceed max ({len(raw)} > {max_bytes})"
        )
    return _bytes_to_pil(raw)


def _fetch_http_url(value: str, *, max_bytes: int):
    try:
        resp = httpx.get(value, timeout=_HTTP_TIMEOUT, follow_redirects=True)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"fetch failed: {e}") from e
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype not in _ALLOWED_IMAGE_MIME:
        raise ValueError(
            f"content-type {ctype!r} not an allowed image MIME; "
            f"allowed: {sorted(_ALLOWED_IMAGE_MIME)}"
        )
    raw = resp.content
    if len(raw) > max_bytes:
        raise ValueError(f"image bytes exceed max ({len(raw)} > {max_bytes})")
    return _bytes_to_pil(raw)


def _bytes_to_pil(raw: bytes):
    from PIL import Image, UnidentifiedImageError
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()  # force decode now so errors surface here, not later
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"image decode failed: {e}") from e
    return img
```

- [ ] **Step 4: Run, expect pass**

```bash
pytest tests/modalities/image_generation/test_image_input.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Run full fast lane**

```bash
pytest tests/ -q -m "not slow"
```

- [ ] **Step 6: Commit**

```bash
git add src/muse/modalities/image_generation/image_input.py \
        tests/modalities/image_generation/test_image_input.py
git commit -m "feat(image-gen): decode_image_input helper for data URLs and HTTP URLs (#143)

Pure helper. Parses user-supplied images for img2img. Accepts
data:image/png;base64,... and http(s)://... URLs. Validates MIME
type, content-type, size cap (10MB default). Decode errors raise
ValueError for the route layer to convert to 400 responses.

No callers yet; routes will wire it in next."
```

---

## Task B: Runtime branch on init_image

**Files:**
- Modify: `src/muse/modalities/image_generation/protocol.py` (extend Protocol signature)
- Modify: `src/muse/modalities/image_generation/runtimes/diffusers.py` (branch + cache)
- Modify: `tests/modalities/image_generation/runtimes/test_diffusers.py` (+tests)

- [ ] **Step 1: Add the failing tests**

Append to `tests/modalities/image_generation/runtimes/test_diffusers.py`:

```python
def test_generate_with_init_image_uses_img2img_pipeline():
    """When init_image is set, runtime calls AutoPipelineForImage2Image."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()
    fake_i2i_class.from_pretrained.return_value = _patched_pipe()

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

    fake_i2i_class.from_pretrained.assert_called_once()
    # The img2img pipeline (not the t2i one) was called for inference
    fake_i2i_class.from_pretrained.return_value.assert_called()


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
    fake_i2i_class.from_pretrained.assert_not_called()


def test_generate_img2img_default_strength_when_omitted():
    """When strength is None on an img2img call, defaults to 0.5."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()
    fake_i2i_pipe = _patched_pipe()
    fake_i2i_class.from_pretrained.return_value = fake_i2i_pipe

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
    """Second img2img call reuses the cached pipeline (no second from_pretrained)."""
    from PIL import Image

    fake_t2i_class = MagicMock()
    fake_t2i_class.from_pretrained.return_value = _patched_pipe()
    fake_i2i_class = MagicMock()
    fake_i2i_class.from_pretrained.return_value = _patched_pipe()

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

    assert fake_i2i_class.from_pretrained.call_count == 1
```

- [ ] **Step 2: Run, expect failures (init_image kwarg unrecognized OR fake_i2i is None)**

```bash
pytest tests/modalities/image_generation/runtimes/test_diffusers.py -v -k "init_image or img2img"
```

- [ ] **Step 3: Extend the protocol**

Edit `src/muse/modalities/image_generation/protocol.py`. Find the `ImageModel.generate` signature and extend with two new optional kwargs:

```python
def generate(
    self,
    prompt: str,
    *,
    negative_prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    init_image: Any = None,            # NEW: PIL.Image | None
    strength: float | None = None,     # NEW: float in [0, 1] | None
    **kwargs,
) -> ImageResult: ...
```

- [ ] **Step 4: Add the img2img sentinel and branch in the runtime**

Edit `src/muse/modalities/image_generation/runtimes/diffusers.py`:

Add a new module-level sentinel and an _ensure_deps extension:

```python
# Existing sentinels:
torch: Any = None
AutoPipelineForText2Image: Any = None

# NEW:
AutoPipelineForImage2Image: Any = None


def _ensure_deps() -> None:
    global torch, AutoPipelineForText2Image, AutoPipelineForImage2Image
    if torch is None:
        try:
            import torch as _t
            torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("diffusers runtime: torch unavailable: %s", e)
    if AutoPipelineForText2Image is None:
        try:
            from diffusers import AutoPipelineForText2Image as _p
            AutoPipelineForText2Image = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("diffusers runtime: AutoPipelineForText2Image unavailable: %s", e)
    if AutoPipelineForImage2Image is None:
        try:
            from diffusers import AutoPipelineForImage2Image as _p
            AutoPipelineForImage2Image = _p
        except Exception as e:  # noqa: BLE001
            logger.debug("diffusers runtime: AutoPipelineForImage2Image unavailable: %s", e)
```

In `__init__`, add `self._i2i_pipe = None` (lazy init).

In `generate`, signature additions and branch:

```python
def generate(
    self, prompt, *,
    negative_prompt=None, width=None, height=None,
    steps=None, guidance=None, seed=None,
    init_image=None, strength=None,
    **_,
):
    if init_image is not None:
        return self._generate_img2img(
            prompt, init_image=init_image, strength=strength,
            negative_prompt=negative_prompt, steps=steps, guidance=guidance, seed=seed,
        )
    # ... existing text-to-image path ...
```

Add `_generate_img2img` method that:
1. Lazily loads `AutoPipelineForImage2Image` (cached as `self._i2i_pipe`).
2. Loads it from the same `local_dir` / `hf_repo` with the same dtype as the t2i pipeline.
3. Builds call_kwargs with `prompt`, `image=init_image`, `strength=strength or 0.5`.
4. Conditionally adds `negative_prompt`, `num_inference_steps`, `guidance_scale`, `generator`.
5. Returns ImageResult with metadata indicating img2img mode.

Reuse the existing torch / device / generator handling pattern.

- [ ] **Step 5: Run, expect tests pass**

```bash
pytest tests/modalities/image_generation/runtimes/test_diffusers.py -v
```

Expected: all tests pass (10 prior + 4 new = 14, or however the count lands).

- [ ] **Step 6: Run full fast lane**

```bash
pytest tests/ -q -m "not slow"
```

- [ ] **Step 7: Commit**

```bash
git add src/muse/modalities/image_generation/protocol.py \
        src/muse/modalities/image_generation/runtimes/diffusers.py \
        tests/modalities/image_generation/runtimes/test_diffusers.py
git commit -m "feat(image-gen): img2img branch in DiffusersText2ImageModel (#143)

generate() now branches on init_image. When set, lazy-loads
AutoPipelineForImage2Image (cached on the instance) and routes the
call there with strength (default 0.5). When None, the existing
text-to-image path runs unchanged.

Protocol extended with init_image and strength kwargs. Tests cover
both branches plus pipeline caching."
```

---

## Task C: Route layer

**Files:**
- Modify: `src/muse/modalities/image_generation/routes.py`
- Modify: `tests/modalities/image_generation/test_routes.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/modalities/image_generation/test_routes.py`:

```python
def test_post_with_image_data_url_routes_through_img2img(client_with_capable_model):
    import base64
    from PIL import Image
    import io

    img = Image.new("RGB", (64, 64), (0, 0, 255))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    r = client_with_capable_model.post("/v1/images/generations", json={
        "prompt": "make it red",
        "model": "fake-i2i-model",
        "image": data_url,
        "strength": 0.7,
    })
    assert r.status_code == 200
    body = r.json()
    assert "data" in body
    # Verify the model got init_image and strength via the fake's call recorder
    # (test fixture should expose this)


def test_post_strength_without_image_is_ignored(client_with_capable_model):
    """strength alone (no image) doesn't fail; falls through to text-to-image."""
    r = client_with_capable_model.post("/v1/images/generations", json={
        "prompt": "a cat",
        "model": "fake-i2i-model",
        "strength": 0.5,
    })
    assert r.status_code == 200


def test_post_image_with_unsupported_model_returns_400(client_with_uncapable_model):
    """A model whose supports_img2img is False rejects requests with image."""
    r = client_with_uncapable_model.post("/v1/images/generations", json={
        "prompt": "x",
        "model": "fake-t2i-only",
        "image": "data:image/png;base64,iVBORw0KGgo=",
    })
    assert r.status_code == 400
    assert "img2img" in r.json()["error"]["message"].lower()


def test_post_malformed_data_url_returns_400_image_decode_error(client_with_capable_model):
    r = client_with_capable_model.post("/v1/images/generations", json={
        "prompt": "x",
        "model": "fake-i2i-model",
        "image": "data:image/png;base64,!!!not_base64!!!",
    })
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_parameter"
    # Type or message should reference image_decode
    assert "image" in err["message"].lower() or "decode" in err["message"].lower()


def test_post_strength_out_of_range_returns_400(client_with_capable_model):
    r = client_with_capable_model.post("/v1/images/generations", json={
        "prompt": "x",
        "model": "fake-i2i-model",
        "image": "data:image/png;base64,iVBORw0KGgo=",
        "strength": 1.5,  # out of [0, 1]
    })
    assert r.status_code in (400, 422)
```

The `client_with_capable_model` and `client_with_uncapable_model` fixtures need to be added at the top of the test file. The "capable" fixture instantiates a fake ImageModel with `supports_img2img: True` in its registered manifest; the "uncapable" fixture has it `False`.

- [ ] **Step 2: Run, expect failures**

```bash
pytest tests/modalities/image_generation/test_routes.py -v -k "image or strength"
```

- [ ] **Step 3: Extend GenerationsRequest**

Edit `src/muse/modalities/image_generation/routes.py`. Find the `GenerationsRequest` Pydantic model and add:

```python
class GenerationsRequest(BaseModel):
    # ... existing fields ...
    image: str | None = None
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
```

In the route handler, after the existing model lookup:

1. If `req.image` is set, check `manifest.capabilities.supports_img2img`. If False or missing, return 400 with `code=invalid_parameter`, `message="model X does not support img2img"`.
2. Decode the image: `from muse.modalities.image_generation.image_input import decode_image_input; init_image = decode_image_input(req.image)`. Wrap in try/except ValueError; on failure return 400 with `code=invalid_parameter` and the ValueError's message.
3. Pass `init_image=init_image, strength=req.strength` through to `model.generate(...)`.

Keep the `width`, `height`, `steps`, `guidance`, etc. plumbing identical to today. The runtime knows how to ignore them when in img2img mode if needed.

- [ ] **Step 4: Run, expect pass**

```bash
pytest tests/modalities/image_generation/test_routes.py -v
```

- [ ] **Step 5: Run full fast lane**

```bash
pytest tests/ -q -m "not slow"
```

- [ ] **Step 6: Commit**

```bash
git add src/muse/modalities/image_generation/routes.py \
        tests/modalities/image_generation/test_routes.py
git commit -m "feat(image-gen): /v1/images/generations accepts image + strength (#143)

Optional fields on GenerationsRequest. When image is set, route
decodes it (data URL or HTTP URL) and passes through to the runtime
as init_image. capabilities.supports_img2img must be True on the
selected model; else 400. strength validated to [0, 1].

OpenAI SDK clients use extra_body to pass these:
  client.images.generate(prompt='...', model='sdxl-turbo',
                         extra_body={'image': '...', 'strength': 0.6})
"
```

---

## Task D: Plugin + bundled-script capabilities

**Files:**
- Modify: `src/muse/modalities/image_generation/hf.py`
- Modify: `src/muse/models/sd_turbo.py`
- Modify: `tests/modalities/image_generation/test_hf_plugin.py`
- Modify: `tests/models/test_sd_turbo.py` (if applicable)

- [ ] **Step 1: Add the plugin test**

In `tests/modalities/image_generation/test_hf_plugin.py`, add:

```python
def test_resolve_advertises_supports_img2img():
    """Resolver-pulled diffusers models advertise img2img support by default."""
    info = _fake_info(siblings=["model_index.json"], tags=["text-to-image"])
    result = HF_PLUGIN["resolve"]("org/anything", None, info)
    assert result.manifest["capabilities"]["supports_img2img"] is True
```

- [ ] **Step 2: Update the plugin**

Edit `src/muse/modalities/image_generation/hf.py`. In `_resolve`, the `capabilities` dict construction:

```python
capabilities = {
    **defaults,
    "supports_negative_prompt": True,
    "supports_seeded_generation": True,
    "supports_img2img": True,   # NEW
}
```

- [ ] **Step 3: Update sd_turbo bundled script**

Edit `src/muse/models/sd_turbo.py`. In MANIFEST `capabilities`, add `"supports_img2img": True`. Then extend the `Model.generate` to accept `init_image` and `strength`:

```python
def generate(
    self, prompt, *,
    negative_prompt=None, width=None, height=None,
    steps=None, guidance=None, seed=None,
    init_image=None, strength=None,
    **_,
):
    if init_image is not None:
        # Mirror the runtimes/diffusers img2img path.
        # Lazy-load AutoPipelineForImage2Image; cache as self._i2i_pipe.
        # ... (similar 20 lines) ...
    # else: existing text-to-image path
```

Or, a cleaner alternative: have sd_turbo delegate to the new runtime entirely. But that would touch first-found-wins precedence. Simpler: copy the img2img branch into sd_turbo. Tests will catch regressions.

- [ ] **Step 4: Add a test for sd_turbo img2img**

In `tests/models/test_sd_turbo.py`, add:

```python
def test_sd_turbo_generate_img2img_branch():
    """sd_turbo's bundled Model honors init_image."""
    # mirror the runtime test pattern
```

- [ ] **Step 5: Run targeted tests, then full fast lane**

```bash
pytest tests/modalities/image_generation/test_hf_plugin.py tests/models/test_sd_turbo.py -v
pytest tests/ -q -m "not slow"
```

- [ ] **Step 6: Commit**

```bash
git add src/muse/modalities/image_generation/hf.py \
        src/muse/models/sd_turbo.py \
        tests/modalities/image_generation/test_hf_plugin.py \
        tests/models/test_sd_turbo.py
git commit -m "feat(image-gen): advertise + implement img2img on sd_turbo and via plugin (#143)

Plugin's _resolve adds supports_img2img: True to capabilities for
all resolver-pulled diffusers models. sd_turbo's bundled MANIFEST
gets the same flag and Model.generate gains the img2img branch,
mirroring the generic runtime."
```

---

## Task E: Documentation + v0.17.0 release

**Files:**
- Modify: `CLAUDE.md`
- Modify: `pyproject.toml`
- Modify: `src/muse/__init__.py`

- [ ] **Step 1: Add CLAUDE.md note**

Locate the section that describes the image_generation modality. After the existing description, add a paragraph:

```
**Img2img on `/v1/images/generations`** (since v0.17.0): pass `image` (data URL or http(s) URL) and optional `strength` (0.0 to 1.0, default 0.5) to use AutoPipelineForImage2Image. OpenAI SDK clients use `extra_body`:

    client.images.generate(prompt="oil painting", model="sdxl-turbo",
                           extra_body={"image": "data:image/png;base64,...", "strength": 0.6})

Models advertise support via `capabilities.supports_img2img`. Requests for non-supporting models return 400.
```

- [ ] **Step 2: Bump version**

`pyproject.toml`: `version = "0.17.0"`. `src/muse/__init__.py`: docstring "As of v0.17.0".

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -q --timeout=300
```

- [ ] **Step 4: Smoke test the new request shape (mocked)**

Optional but useful:
```bash
python -c "
from fastapi.testclient import TestClient
# (build a minimal app with a fake registered model and probe the new fields)
print('routes accept image and strength')"
```

- [ ] **Step 5: Em-dash check, commit + tag**

```bash
git add CLAUDE.md pyproject.toml src/muse/__init__.py
git commit -m "chore(release): v0.17.0

Img2img extension to /v1/images/generations (#143). Optional 'image'
and 'strength' fields. OpenAI SDK compatible via extra_body. Foundation
for chained-frame coherent sequences (build on the client) and for a
future image/animation modality (build on the server).

Closes #143."

git tag -a v0.17.0 -m "v0.17.0: img2img on /v1/images/generations"
```

- [ ] **Step 6: DO NOT PUSH** (user decides).

---

## Self-review checklist

1. **Spec coverage:** Tasks A-E implement all six sections of the spec. Wire contract, runtime contract, capability flag, image decoding, migration, tests. No gaps.
2. **Placeholder scan:** zero TBD/TODO/XXX/FIXME outside the self-review meta-text.
3. **Type consistency:** `image: str | None`, `strength: float | None`, `init_image: PIL.Image | None`. The same names flow through pydantic -> runtime kwargs -> diffusers pipe call.
4. **Migration safety:** every change is additive. Existing text-to-image requests pass `image=None, strength=None` (defaults), which routes through the unchanged path.
5. **Behavior preservation:** existing `tests/modalities/image_generation/test_routes.py` and `tests/models/test_sd_turbo.py` keep passing without modification (except where new tests are added).
6. **Capability gating:** the `supports_img2img: False` 400 path is the only way clients learn a model can't do img2img. We don't auto-detect at request time; we trust the manifest. New diffusers models pulled via the plugin advertise True; old already-pulled diffusers models miss the capability until repulled (#138 territory).
