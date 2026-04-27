# Image Generation HF Plugin Implementation Plan (#141)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `image_generation/hf.py` + `DiffusersText2ImageModel` runtime + 2 curated entries so `muse pull hf://stabilityai/sdxl-turbo` works.

**Architecture:** Mirror Task #129's plugin shape. New generic runtime parameterized via manifest capabilities. Sniff requires `model_index.json` sibling AND `text-to-image` tag.

**Spec:** `docs/superpowers/specs/2026-04-26-image-generation-hf-plugin-design.md`

**Target version:** v0.16.0

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/muse/modalities/image_generation/runtimes/__init__.py` | create | empty package marker |
| `src/muse/modalities/image_generation/runtimes/diffusers.py` | create | `DiffusersText2ImageModel` generic runtime |
| `src/muse/modalities/image_generation/hf.py` | create | `HF_PLUGIN` for diffusers text-to-image repos |
| `src/muse/curated.yaml` | modify | +2 entries: sdxl-turbo, flux-schnell |
| `tests/modalities/image_generation/runtimes/__init__.py` | create | test package marker |
| `tests/modalities/image_generation/runtimes/test_diffusers.py` | create | runtime tests (mocked) |
| `tests/modalities/image_generation/test_hf_plugin.py` | create | plugin tests |
| `tests/core/test_curated.py` | modify | +1 test asserting both new entries |
| `CLAUDE.md` | modify | mention image/generation now has HF resolver support |
| `pyproject.toml` | modify | bump to 0.16.0 |
| `src/muse/__init__.py` | modify | docstring "as of v0.15.0" -> "as of v0.16.0" |

---

## Task A: DiffusersText2ImageModel runtime

**Files:**
- Create: `src/muse/modalities/image_generation/runtimes/__init__.py` (empty)
- Create: `src/muse/modalities/image_generation/runtimes/diffusers.py`
- Create: `tests/modalities/image_generation/runtimes/__init__.py` (empty)
- Create: `tests/modalities/image_generation/runtimes/test_diffusers.py`

- [ ] **Step 1: Write the failing test**

Create `tests/modalities/image_generation/runtimes/test_diffusers.py`:

```python
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


def test_construction_loads_from_local_dir(tmp_path):
    """Constructor passes local_dir to AutoPipelineForText2Image.from_pretrained."""
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
    assert m.model_id == "org-repo"


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
```

- [ ] **Step 2: Run, expect ImportError**

```bash
pytest tests/modalities/image_generation/runtimes/test_diffusers.py -v
```

Expected: ModuleNotFoundError for `muse.modalities.image_generation.runtimes.diffusers`.

- [ ] **Step 3: Implement the runtime**

Create `src/muse/modalities/image_generation/runtimes/__init__.py` (empty file).

Create `src/muse/modalities/image_generation/runtimes/diffusers.py`. Mirror `src/muse/models/sd_turbo.py` but parameterize defaults via constructor kwargs:

```python
"""Generic text-to-image runtime via diffusers.AutoPipelineForText2Image.

One runtime serves many models. The model_id, default_size, default_steps,
and default_guidance are injected at construction time from manifest
capabilities. Heavy imports (torch, diffusers) are lazy: discovery must
work without them on the host python.

Mirrors the lazy-import pattern from src/muse/models/sd_turbo.py: tests
patch the module-level `torch` and `AutoPipelineForText2Image` sentinels
directly; `_ensure_deps()` short-circuits when sentinels are non-None
(mocked) so the patches survive.
"""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.image_generation.protocol import ImageResult


logger = logging.getLogger(__name__)


# Sentinels patched by tests; populated by _ensure_deps at runtime.
torch: Any = None
AutoPipelineForText2Image: Any = None


def _ensure_deps() -> None:
    global torch, AutoPipelineForText2Image
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
            logger.debug("diffusers runtime: diffusers unavailable: %s", e)


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


class DiffusersText2ImageModel:
    """Text-to-image runtime backed by diffusers.AutoPipelineForText2Image.

    Construction kwargs (set by catalog at load_backend time, sourced from
    manifest fields and capabilities):
      - hf_repo, local_dir, device, dtype: standard
      - model_id: catalog id (response envelope echoes this)
      - default_size: (width, height) when request omits size
      - default_steps: num_inference_steps default
      - default_guidance: guidance_scale default
    """

    def __init__(
        self,
        *,
        hf_repo: str,
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        model_id: str,
        default_size: tuple[int, int] = (512, 512),
        default_steps: int = 1,
        default_guidance: float = 0.0,
        **_: Any,
    ) -> None:
        _ensure_deps()
        if AutoPipelineForText2Image is None:
            raise RuntimeError(
                "diffusers is not installed; ensure muse[images] extras are "
                "installed in the per-model venv"
            )
        self.model_id = model_id
        self.default_size = default_size
        self._default_steps = default_steps
        self._default_guidance = default_guidance
        self._device = _select_device(device)

        # Access torch through this module so tests' patches survive.
        import muse.modalities.image_generation.runtimes.diffusers as _mod
        _torch = _mod.torch
        torch_dtype = None
        if _torch is not None:
            torch_dtype = {
                "float16": _torch.float16,
                "float32": _torch.float32,
                "bfloat16": _torch.bfloat16,
            }[dtype]

        src = local_dir or hf_repo
        logger.info(
            "loading diffusers pipeline from %s (model_id=%s, device=%s, dtype=%s)",
            src, model_id, self._device, dtype,
        )
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            src,
            torch_dtype=torch_dtype,
            variant="fp16" if dtype == "float16" else None,
        )
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)

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
        **_: Any,
    ) -> ImageResult:
        w = width or self.default_size[0]
        h = height or self.default_size[1]
        n_steps = steps if steps is not None else self._default_steps
        cfg = guidance if guidance is not None else self._default_guidance

        gen = None
        if seed is not None:
            import muse.modalities.image_generation.runtimes.diffusers as _mod
            _torch = _mod.torch
            if _torch is not None:
                gen = _torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict = {
            "prompt": prompt,
            "width": w,
            "height": h,
            "num_inference_steps": n_steps,
            "guidance_scale": cfg,
        }
        if negative_prompt is not None:
            call_kwargs["negative_prompt"] = negative_prompt
        if gen is not None:
            call_kwargs["generator"] = gen

        out = self._pipe(**call_kwargs)
        img = out.images[0]
        return ImageResult(
            image=img,
            width=img.size[0],
            height=img.size[1],
            seed=seed if seed is not None else -1,
            metadata={
                "prompt": prompt,
                "steps": n_steps,
                "guidance": cfg,
                "model": self.model_id,
            },
        )
```

- [ ] **Step 4: Run, expect pass**

```bash
pytest tests/modalities/image_generation/runtimes/test_diffusers.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Run full fast lane**

```bash
pytest tests/ -q -m "not slow"
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/muse/modalities/image_generation/runtimes/ \
        tests/modalities/image_generation/runtimes/
git commit -m "feat(image-gen): DiffusersText2ImageModel generic runtime (#141)

Wraps diffusers.AutoPipelineForText2Image with manifest-driven
defaults (default_size, default_steps, default_guidance). Mirrors
the lazy-import pattern from sd_turbo.py so discovery and CLI work
without diffusers/torch installed on the host python.

No callers yet; HF plugin wires it in next."
```

---

## Task B: image_generation/hf.py plugin

**Files:**
- Create: `src/muse/modalities/image_generation/hf.py`
- Create: `tests/modalities/image_generation/test_hf_plugin.py`

- [ ] **Step 1: Write the failing test**

Create `tests/modalities/image_generation/test_hf_plugin.py`:

```python
"""Tests for the image_generation HF plugin (diffusers text-to-image)."""
from unittest.mock import MagicMock

from muse.modalities.image_generation.hf import HF_PLUGIN
from muse.core.discovery import REQUIRED_HF_PLUGIN_KEYS
from muse.core.resolvers import ResolvedModel


def _fake_info(siblings=None, tags=None):
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (siblings or [])]
    info.tags = tags or []
    info.card_data = MagicMock(license=None)
    return info


def test_plugin_has_all_required_keys():
    for key in REQUIRED_HF_PLUGIN_KEYS:
        assert key in HF_PLUGIN


def test_plugin_metadata_correct():
    assert HF_PLUGIN["modality"] == "image/generation"
    assert HF_PLUGIN["runtime_path"].endswith(":DiffusersText2ImageModel")
    assert HF_PLUGIN["priority"] == 100


def test_sniff_true_on_diffusers_text_to_image_repo():
    info = _fake_info(
        siblings=["model_index.json", "unet/diffusion_pytorch_model.safetensors"],
        tags=["text-to-image", "diffusers"],
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_without_text_to_image_tag():
    info = _fake_info(
        siblings=["model_index.json"],
        tags=["image-to-image"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_without_model_index_json():
    info = _fake_info(
        siblings=["model.safetensors"],
        tags=["text-to-image"],
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_resolve_returns_resolved_model_with_turbo_defaults():
    """Repo name containing 'turbo' should default steps=1, guidance=0."""
    info = _fake_info(
        siblings=["model_index.json"],
        tags=["text-to-image"],
    )
    result = HF_PLUGIN["resolve"]("stabilityai/sdxl-turbo", None, info)
    assert isinstance(result, ResolvedModel)
    caps = result.manifest["capabilities"]
    assert caps["default_steps"] == 1
    assert caps["default_guidance"] == 0.0


def test_resolve_flux_schnell_defaults():
    info = _fake_info(siblings=["model_index.json"], tags=["text-to-image"])
    result = HF_PLUGIN["resolve"]("black-forest-labs/FLUX.1-schnell", None, info)
    caps = result.manifest["capabilities"]
    assert caps["default_steps"] == 4
    assert caps["default_size"] == [1024, 1024]


def test_resolve_default_fallback():
    """Non-turbo, non-flux gets the conservative fallback."""
    info = _fake_info(siblings=["model_index.json"], tags=["text-to-image"])
    result = HF_PLUGIN["resolve"]("random-org/random-sd", None, info)
    caps = result.manifest["capabilities"]
    assert caps["default_steps"] == 25
    assert caps["default_guidance"] == 7.5


def test_search_yields_results_with_modality_tag():
    fake_api = MagicMock()
    fake_repo = MagicMock(id="org/repo", downloads=42)
    fake_api.list_models.return_value = [fake_repo]
    rows = list(HF_PLUGIN["search"](fake_api, "sdxl", sort="downloads", limit=20))
    assert len(rows) == 1
    assert rows[0].modality == "image/generation"
```

- [ ] **Step 2: Run, expect ImportError**

```bash
pytest tests/modalities/image_generation/test_hf_plugin.py -v
```

- [ ] **Step 3: Implement the plugin**

Create `src/muse/modalities/image_generation/hf.py`:

```python
"""HF resolver plugin for diffusers text-to-image models.

Sniffs HF repos for `model_index.json` (the diffusers pipeline config)
plus the `text-to-image` tag. Synthesizes a manifest with capabilities
inferred from repo name (turbo/flux/sdxl/sd3 patterns set sensible
default steps/guidance/size).

Loaded via single-file import; no relative imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.resolvers import ResolvedModel, SearchResult


_RUNTIME_PATH = (
    "muse.modalities.image_generation.runtimes.diffusers"
    ":DiffusersText2ImageModel"
)
_PIP_EXTRAS = (
    "torch>=2.1.0",
    "diffusers>=0.27.0",
    "transformers>=4.36.0",
    "accelerate",
    "Pillow",
    "safetensors",
)


def _model_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].lower()


def _repo_license(info) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    return getattr(card, "license", None)


def _infer_defaults(repo_id: str) -> dict[str, Any]:
    """Sensible per-pattern defaults so each model lands with reasonable
    steps/guidance/size without users having to tweak. Override per-call
    via request fields or per-model via curated capabilities overlay."""
    rid = repo_id.lower()
    if "flux" in rid and "schnell" in rid:
        return {
            "default_size": [1024, 1024],
            "default_steps": 4,
            "default_guidance": 0.0,
        }
    if "flux" in rid and "dev" in rid:
        return {
            "default_size": [1024, 1024],
            "default_steps": 28,
            "default_guidance": 3.5,
        }
    if "turbo" in rid:
        return {
            "default_size": [512, 512],
            "default_steps": 1,
            "default_guidance": 0.0,
        }
    if "sdxl" in rid or "stable-diffusion-xl" in rid:
        return {
            "default_size": [1024, 1024],
            "default_steps": 25,
            "default_guidance": 7.5,
        }
    if "stable-diffusion-3" in rid or "sd3" in rid:
        return {
            "default_size": [1024, 1024],
            "default_steps": 28,
            "default_guidance": 4.5,
        }
    return {
        "default_size": [512, 512],
        "default_steps": 25,
        "default_guidance": 7.5,
    }


def _sniff(info) -> bool:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    has_pipeline_config = any(
        Path(f).name == "model_index.json" for f in siblings
    )
    has_t2i_tag = "text-to-image" in tags
    return has_pipeline_config and has_t2i_tag


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    defaults = _infer_defaults(repo_id)
    capabilities = {
        **defaults,
        "supports_negative_prompt": True,
        "supports_seeded_generation": True,
    }
    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "image/generation",
        "hf_repo": repo_id,
        "description": f"Diffusers text-to-image: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }

    def _download(cache_root: Path) -> Path:
        return Path(snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_root) if cache_root else None,
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    repos = api.list_models(
        search=query, filter="text-to-image",
        sort=sort, limit=limit,
    )
    for repo in repos:
        yield SearchResult(
            uri=f"hf://{repo.id}",
            model_id=_model_id(repo.id),
            modality="image/generation",
            size_gb=None,
            downloads=getattr(repo, "downloads", None),
            license=None,
            description=repo.id,
        )


HF_PLUGIN = {
    "modality": "image/generation",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 100,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
```

- [ ] **Step 4: Run, expect pass**

```bash
pytest tests/modalities/image_generation/test_hf_plugin.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Smoke test plugin discovery**

```bash
python -c "from muse.core.resolvers_hf import HFResolver; print([(p['priority'], p['modality']) for p in HFResolver()._plugins])"
```

Expected: 5 plugins now: `[(100, 'audio/transcription'), (100, 'chat/completion'), (100, 'image/generation'), (110, 'embedding/text'), (200, 'text/classification')]`.

- [ ] **Step 6: Run full fast lane**

```bash
pytest tests/ -q -m "not slow"
```

- [ ] **Step 7: Commit**

```bash
git add src/muse/modalities/image_generation/hf.py \
        tests/modalities/image_generation/test_hf_plugin.py
git commit -m "feat(image-gen): HF resolver plugin for diffusers text-to-image (#141)

modalities/image_generation/hf.py exports HF_PLUGIN with sniff
(model_index.json sibling AND text-to-image tag), resolve (defaults
inferred per-pattern: turbo/flux/sdxl/sd3), and search (HF
list_models filter). Priority 100 (file-pattern + tag specific).

DiffusersText2ImageModel runtime (Task A) handles the inference.
Bundled sd-turbo.py script is unaffected; first-found-wins discovery
keeps it working for the curated 'sd-turbo' id."
```

---

## Task C: Curated entries for sdxl-turbo and flux-schnell

**Files:**
- Modify: `src/muse/curated.yaml`
- Modify: `tests/core/test_curated.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/core/test_curated.py`:

```python
def test_load_curated_includes_sdxl_turbo_and_flux_schnell():
    """v0.16.0 adds two image/generation curated aliases via the new HF plugin."""
    entries = load_curated()
    by_id = {e.id: e for e in entries}

    assert "sdxl-turbo" in by_id
    e = by_id["sdxl-turbo"]
    assert e.modality == "image/generation"
    assert e.uri == "hf://stabilityai/sdxl-turbo"

    assert "flux-schnell" in by_id
    e = by_id["flux-schnell"]
    assert e.modality == "image/generation"
    assert e.uri == "hf://black-forest-labs/FLUX.1-schnell"
```

- [ ] **Step 2: Run, expect failure**

```bash
pytest tests/core/test_curated.py::test_load_curated_includes_sdxl_turbo_and_flux_schnell -v
```

Expected: FAIL (entries don't exist).

- [ ] **Step 3: Add curated entries**

Edit `src/muse/curated.yaml`. Find the `# ---------- image/generation ----------` section. Add after the existing `sd-turbo` entry:

```yaml
- id: sdxl-turbo
  uri: hf://stabilityai/sdxl-turbo
  modality: image/generation
  size_gb: 7.0
  description: "SDXL Turbo: 1-step distilled SDXL, 512x512, fast"

- id: flux-schnell
  uri: hf://black-forest-labs/FLUX.1-schnell
  modality: image/generation
  size_gb: 24.0
  description: "FLUX.1 Schnell: 4-step distilled, 1024x1024, Apache 2.0"
```

- [ ] **Step 4: Run, expect pass**

```bash
pytest tests/core/test_curated.py -v -k sdxl_turbo
```

Expected: PASS.

- [ ] **Step 5: Run full fast lane**

```bash
pytest tests/ -q -m "not slow"
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/muse/curated.yaml tests/core/test_curated.py
git commit -m "feat(curated): sdxl-turbo and flux-schnell aliases (#141)

Both route through the new image_generation/hf.py plugin, which
synthesizes a manifest pointing at DiffusersText2ImageModel.

\`muse pull sdxl-turbo\` and \`muse pull flux-schnell\` now work."
```

---

## Task D: Documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md modality table**

In CLAUDE.md, locate the modality list near the top. The line about image/generation likely reads:

```
- **image/generation**: text-to-image via `/v1/images/generations` (SD-Turbo)
```

Change to:

```
- **image/generation**: text-to-image via `/v1/images/generations` (SD-Turbo, SDXL-Turbo, FLUX.1-schnell, any diffusers HF repo)
```

Also locate the section that says "image/generation has no HF resolver plugin" or similar, and remove that caveat. Verify the docs match the new state where 5 modalities have HF plugins (chat_completion, embedding_text, audio_transcription, text_classification, image_generation).

- [ ] **Step 2: Verify no em-dashes**

```bash
python -c "print(open('CLAUDE.md').read().count(chr(0x2014)))"
```

Expected: 0.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(image-gen): note HF plugin support in CLAUDE.md (#141)"
```

---

## Task E: v0.16.0 release

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/muse/__init__.py` (docstring "as of v0.15.0" -> "as of v0.16.0")

- [ ] **Step 1: Bump version**

```toml
version = "0.16.0"
```

Update `src/muse/__init__.py` docstring from "As of v0.15.0" to "As of v0.16.0".

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -q --timeout=300
```

Expected: all green (slow lane included).

- [ ] **Step 3: Smoke test plugin discovery**

```bash
python -c "
from muse.core.discovery import discover_hf_plugins, _default_hf_plugin_dirs
plugins = discover_hf_plugins(_default_hf_plugin_dirs())
print(f'Discovered {len(plugins)} HF plugins:')
for p in plugins:
    print(f'  {p[\"priority\"]:3d}  {p[\"modality\"]}  -> {p[\"runtime_path\"]}')
"
```

Expected: 5 plugins, including the new `image/generation` at priority 100.

- [ ] **Step 4: Commit + tag**

```bash
git add pyproject.toml src/muse/__init__.py
git commit -m "chore(release): v0.16.0

image_generation HF plugin + DiffusersText2ImageModel runtime (#141).

\`muse pull hf://stabilityai/sdxl-turbo\`, \`muse pull flux-schnell\`,
and any diffusers text-to-image repo on HF now resolve and serve via
the same plugin pattern as the four other modalities. The bundled
sd_turbo.py script is unchanged.

Closes #141."

git tag -a v0.16.0 -m "v0.16.0: image_generation HF plugin"
```

- [ ] **Step 5: DO NOT PUSH**

The user decides on push.

---

## Self-review checklist

1. **Spec coverage:** Tasks A-E implement every section of the spec. Plugin contract, runtime interface, capability defaults, curated entries, docs, release. No spec sections orphaned.
2. **Placeholder scan:** zero TBD/TODO/XXX/FIXME (other than self-review meta-text).
3. **Type consistency:** `DiffusersText2ImageModel` constructor kwargs match what `catalog.load_backend` will pass (manifest fields + capabilities). The plugin's manifest emits `model_id` (top-level) and `default_size`/`default_steps`/`default_guidance` (under `capabilities`). The catalog's load_backend already merges manifest fields into runtime kwargs.
4. **Migration safety:** purely additive. Bundled `sd_turbo.py` untouched; first-found-wins discovery keeps the curated `sd-turbo` id pointing at the bundled script.
5. **Behavior preservation:** existing `muse pull sd-turbo` and `muse search ... --modality image/generation` work unchanged. New paths are bonus.
