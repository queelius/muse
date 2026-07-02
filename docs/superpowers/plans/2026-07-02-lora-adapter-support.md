# LoRA Adapter Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `muse pull hf://nerijs/pixel-art-xl` turns a LoRA adapter repo into a servable image/generation model paired with a base diffusers pipeline, served through `/v1/images/generations` with per-request `lora_scale`.

**Architecture:** Approach A from the spec (`docs/superpowers/specs/2026-07-02-lora-adapter-design.md`): extend the existing image_generation HF plugin (adapter-shape sniff + LoRA resolve) and the existing `DiffusersText2ImageModel` runtime (load base pipeline, then unfused `load_lora_weights`). `capabilities.base_model` holds either a muse catalog id (resolved to its `local_dir`) or an HF repo id (downloaded to the HF cache at first load). No new modality, no new runtime class.

**Tech Stack:** Python 3.10+, diffusers >= 0.27 (+ new `peft` pip_extra for LoRA entries), FastAPI, pydantic, pytest with FakeModel/MagicMock patterns.

## Global Constraints

- TDD for every task: write the failing test, watch it fail, then implement (red-green).
- Fast lane must stay green after every task: `python -m pytest tests/ -q -m "not slow"`.
- Commits are LOCAL ONLY until the user explicitly says "go" for push/release. Exception: Task 9's deploy-to-.204 step is pre-authorized (push main + ssh git pull is the established deploy path).
- Commit messages: ASCII only, conventional-commit style, and MUST end with BOTH trailers:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01J2SDRmdTMP3sVoBpAqZ3VV`
- Never echo the admin token (`MUSE_ADMIN_TOKEN`) into logs, errors, or command output; redact with sed when curling admin endpoints.
- No em-dash characters in any written file (a soul-voice hook rejects them); use colons, commas, or parentheses.
- Version bump / tag / PyPI publish are GATED on an explicit user "go" (target: v0.50.0).
- New capability keys used throughout: `lora_adapter: bool`, `base_model: str`, `lora_scale: float`. A muse catalog id contains no `/`; an HF repo id contains exactly one.

## File Map

| File | Change |
|---|---|
| `src/muse/core/runtime_helpers.py` | add `resolve_model_source(ref)` |
| `src/muse/modalities/image_generation/hf.py` | adapter sniff branch, `_resolve_lora`, base-tag parsing, base-size estimate, `peft` extra |
| `src/muse/core/catalog.py` | `pull(..., base_override=)`, `_validate_lora_capabilities` in `_pull_via_resolver` |
| `src/muse/cli.py` | `muse pull --base` option |
| `src/muse/cli_impl/pull_errors.py` | render `ResolverError` cleanly (no traceback) |
| `src/muse/modalities/image_generation/runtimes/diffusers.py` | LoRA load branch + per-call scale |
| `src/muse/modalities/image_generation/routes.py` | `lora_scale` request field + 400 gate |
| `src/muse/cli_impl/supervisor.py` | `backfill_manifest_memory` chases `base_model` |
| `src/muse/curated.yaml` | `pixel-art-xl` entry |
| `CLAUDE.md`, `README.md` | docs |
| Tests | `tests/core/test_runtime_helpers.py`, `tests/modalities/image_generation/test_hf_plugin.py`, `tests/modalities/image_generation/runtimes/test_diffusers.py`, `tests/modalities/image_generation/test_routes.py`, `tests/core/test_catalog.py`, `tests/cli_impl/test_pull_errors.py`, `tests/cli_impl/test_supervisor_memory.py` (or the file that already tests `backfill_manifest_memory`; locate with `grep -rln backfill_manifest_memory tests/`) |

---

### Task 1: `resolve_model_source` helper

**Files:**
- Modify: `src/muse/core/runtime_helpers.py`
- Test: `tests/core/test_runtime_helpers.py` (append; create the class if the file lacks one for this helper)

**Interfaces:**
- Produces: `resolve_model_source(ref: str) -> str`. Pulled muse id with a `local_dir` -> that path (str). Anything else (HF repo id, unknown id, entry without local_dir) -> `ref` verbatim. Task 5 imports it from `muse.core.runtime_helpers`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_runtime_helpers.py`:

```python
class TestResolveModelSource:
    """resolve_model_source maps a muse catalog id to its local weights dir
    and passes anything else through verbatim (HF repo ids, unknown ids)."""

    def _write_catalog(self, tmp_path, entries):
        import json
        (tmp_path / "catalog.json").write_text(json.dumps(entries))

    def test_pulled_id_resolves_to_local_dir(self, tmp_path, monkeypatch):
        from muse.core.catalog import _reset_read_catalog_cache
        from muse.core.runtime_helpers import resolve_model_source

        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        self._write_catalog(tmp_path, {
            "sdxl-turbo": {"local_dir": "/weights/sdxl-turbo", "enabled": True},
        })
        _reset_read_catalog_cache()
        assert resolve_model_source("sdxl-turbo") == "/weights/sdxl-turbo"

    def test_unknown_ref_passes_through(self, tmp_path, monkeypatch):
        from muse.core.catalog import _reset_read_catalog_cache
        from muse.core.runtime_helpers import resolve_model_source

        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        self._write_catalog(tmp_path, {})
        _reset_read_catalog_cache()
        assert resolve_model_source(
            "stabilityai/stable-diffusion-xl-base-1.0"
        ) == "stabilityai/stable-diffusion-xl-base-1.0"

    def test_entry_without_local_dir_passes_through(self, tmp_path, monkeypatch):
        from muse.core.catalog import _reset_read_catalog_cache
        from muse.core.runtime_helpers import resolve_model_source

        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        self._write_catalog(tmp_path, {"half-entry": {"enabled": True}})
        _reset_read_catalog_cache()
        assert resolve_model_source("half-entry") == "half-entry"
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest tests/core/test_runtime_helpers.py::TestResolveModelSource -v`
Expected: 3x FAIL/ERROR with `ImportError: cannot import name 'resolve_model_source'`

- [ ] **Step 3: Implement**

Append to `src/muse/core/runtime_helpers.py` (and add
`resolve_model_source(ref) -> str` to the "Public surface" list in the module
docstring):

```python
def resolve_model_source(ref: str) -> str:
    """Map a muse catalog id to its local weights dir; pass others through.

    LoRA runtimes receive ``capabilities.base_model`` that is either a
    pulled muse model id (e.g. ``sdxl-turbo``, resolved here to the
    snapshot dir recorded at pull time) or an HF repo id (returned
    verbatim; ``from_pretrained`` downloads it into the HF cache at
    first load, the AnimateDiff precedent). Unknown ids and entries
    without a ``local_dir`` also pass through verbatim so the caller's
    ``from_pretrained`` raises its own (informative) error.

    Imports the catalog lazily: runtime modules must stay importable
    without dragging the catalog machinery in at module import time.
    """
    from muse.core.catalog import _read_catalog

    entry = _read_catalog().get(ref)
    if entry:
        local_dir = entry.get("local_dir")
        if local_dir:
            return str(local_dir)
    return ref
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/core/test_runtime_helpers.py -v`
Expected: all PASS (new 3 + existing)

- [ ] **Step 5: Commit**

```bash
git add src/muse/core/runtime_helpers.py tests/core/test_runtime_helpers.py
git commit -m "feat(runtime_helpers): resolve_model_source maps muse ids to local dirs

First piece of LoRA adapter support (spec:
docs/superpowers/specs/2026-07-02-lora-adapter-design.md). LoRA
runtimes receive base_model as either a pulled muse id or an HF repo;
this helper resolves the former to its snapshot dir and passes the
latter through for from_pretrained to fetch.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J2SDRmdTMP3sVoBpAqZ3VV"
```

---

### Task 2: Plugin sniff accepts the adapter shape

**Files:**
- Modify: `src/muse/modalities/image_generation/hf.py` (function `_sniff`, currently ~lines 87-102)
- Test: `tests/modalities/image_generation/test_hf_plugin.py`

**Interfaces:**
- Produces: `_is_lora_adapter(siblings: list[str], tags: list[str]) -> bool` (module-level in hf.py; Task 3's resolve dispatch reuses it). `_sniff` returns True for BOTH the diffusers-t2i shape and the adapter shape.

- [ ] **Step 1: Write the failing tests**

Append to `tests/modalities/image_generation/test_hf_plugin.py` (the file's
`_fake_info(siblings=, tags=, pipeline_tag=)` helper already exists):

```python
def test_sniff_true_on_lora_adapter_repo():
    """Real shape of nerijs/pixel-art-xl (verified 2026-07-02): no
    model_index.json, one top-level safetensors, lora + base_model tags."""
    info = _fake_info(
        siblings=[".gitattributes", "README.md", "pixel-art-xl.safetensors"],
        tags=[
            "diffusers", "text-to-image", "stable-diffusion", "lora",
            "base_model:stabilityai/stable-diffusion-xl-base-1.0",
            "base_model:adapter:stabilityai/stable-diffusion-xl-base-1.0",
        ],
        pipeline_tag="text-to-image",
    )
    assert HF_PLUGIN["sniff"](info) is True


def test_sniff_false_on_safetensors_without_lora_signal():
    """A bare safetensors artifact repo (no lora tag, no adapter tag) is
    NOT claimed, even when tagged text-to-image."""
    info = _fake_info(
        siblings=["model.safetensors"],
        tags=["text-to-image", "diffusers"],
        pipeline_tag="text-to-image",
    )
    assert HF_PLUGIN["sniff"](info) is False


def test_sniff_false_on_lora_repo_without_text_to_image_signal():
    """A LoRA for some non-t2i pipeline is not ours."""
    info = _fake_info(
        siblings=["adapter.safetensors"],
        tags=["lora", "base_model:adapter:some/llm"],
        pipeline_tag="text-generation",
    )
    assert HF_PLUGIN["sniff"](info) is False
```

- [ ] **Step 2: Run tests, verify the first fails**

Run: `python -m pytest tests/modalities/image_generation/test_hf_plugin.py -q -k "lora"`
Expected: `test_sniff_true_on_lora_adapter_repo` FAILS (False is not True); the two negative tests PASS already.

- [ ] **Step 3: Implement**

In `src/muse/modalities/image_generation/hf.py`, add above `_sniff`:

```python
def _is_lora_adapter(siblings: list[str], tags: list[str]) -> bool:
    """Adapter-only repo shape: top-level safetensors weights plus an
    explicit LoRA signal in the tags. Callers have already established
    the repo is text-to-image and has NO model_index.json."""
    has_weights = any(
        f.endswith(".safetensors") and "/" not in f for f in siblings
    )
    has_lora_tag = "lora" in tags or any(
        t.startswith("base_model:adapter:") for t in tags
    )
    return has_weights and has_lora_tag
```

Replace the body of `_sniff` with:

```python
def _sniff(info) -> bool:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    # `pipeline_tag` is HF's canonical single-value task field and is the
    # authoritative text-to-image signal. Many community SD checkpoints set
    # it but do NOT mirror "text-to-image" into the loose `tags` bag (which
    # may only carry "diffusers:StableDiffusionPipeline"). Read the structured
    # field first, falling back to the tags mirror for repos that only tag.
    is_text_to_image = (
        getattr(info, "pipeline_tag", None) == "text-to-image"
        or "text-to-image" in tags
    )
    if not is_text_to_image:
        return False
    has_pipeline_config = any(
        Path(f).name == "model_index.json" for f in siblings
    )
    if has_pipeline_config:
        return True
    # Second accepted shape: a LoRA adapter repo (weights only, no
    # pipeline config). Resolved via _resolve_lora and served by pairing
    # with a base pipeline at load time.
    return _is_lora_adapter(siblings, tags)
```

- [ ] **Step 4: Run the full plugin test file**

Run: `python -m pytest tests/modalities/image_generation/test_hf_plugin.py -q`
Expected: all PASS (18 = 15 existing + 3 new)

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/image_generation/hf.py tests/modalities/image_generation/test_hf_plugin.py
git commit -m "feat(image_generation): sniff LoRA adapter repos (no model_index.json)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J2SDRmdTMP3sVoBpAqZ3VV"
```

---

### Task 3: Plugin resolve for adapters

**Files:**
- Modify: `src/muse/modalities/image_generation/hf.py`
- Test: `tests/modalities/image_generation/test_hf_plugin.py`

**Interfaces:**
- Consumes: `_is_lora_adapter` (Task 2), existing `_infer_defaults`, `_model_id`, `_repo_license`, `_RUNTIME_PATH`, `_PIP_EXTRAS`.
- Produces: manifests whose capabilities carry `lora_adapter: True`, `base_model: <repo>` (omitted when no tag; Task 4 validates post-overlay), `lora_scale: 1.0`, defaults derived from the base id, and `pip_extras` including `"peft"`. Raises `ResolverError` on multiple top-level safetensors.

- [ ] **Step 1: Write the failing tests**

Append to `tests/modalities/image_generation/test_hf_plugin.py`:

```python
def _lora_info(siblings=None, tags=None):
    return _fake_info(
        siblings=siblings or ["README.md", "pixel-art-xl.safetensors"],
        tags=tags or [
            "diffusers", "text-to-image", "lora",
            "base_model:stabilityai/stable-diffusion-xl-base-1.0",
            "base_model:adapter:stabilityai/stable-diffusion-xl-base-1.0",
        ],
        pipeline_tag="text-to-image",
    )


def test_resolve_lora_extracts_base_from_adapter_tag():
    from unittest.mock import patch
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=None,
    ):
        result = HF_PLUGIN["resolve"]("nerijs/pixel-art-xl", None, _lora_info())
    caps = result.manifest["capabilities"]
    assert caps["lora_adapter"] is True
    assert caps["base_model"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert caps["lora_scale"] == 1.0
    assert result.manifest["model_id"] == "pixel-art-xl"
    assert result.manifest["modality"] == "image/generation"
    assert "peft" in result.manifest["pip_extras"]


def test_resolve_lora_falls_back_to_plain_base_model_tag():
    from unittest.mock import patch
    info = _lora_info(tags=[
        "text-to-image", "lora",
        "base_model:runwayml/stable-diffusion-v1-5",
    ])
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=None,
    ):
        result = HF_PLUGIN["resolve"]("org/some-lora", None, info)
    assert result.manifest["capabilities"]["base_model"] == (
        "runwayml/stable-diffusion-v1-5"
    )


def test_resolve_lora_without_base_tag_omits_base_model():
    """No base tag: resolve succeeds WITHOUT base_model; the post-overlay
    validation in catalog.pull rejects it unless --base supplied it."""
    from unittest.mock import patch
    info = _lora_info(tags=["text-to-image", "lora"])
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=None,
    ):
        result = HF_PLUGIN["resolve"]("org/tagless-lora", None, info)
    caps = result.manifest["capabilities"]
    assert caps["lora_adapter"] is True
    assert "base_model" not in caps


def test_resolve_lora_defaults_derive_from_base_id():
    """An SDXL base yields the sdxl defaults (25 steps, 1024), proving
    _infer_defaults ran against the BASE, not the adapter repo name."""
    from unittest.mock import patch
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=None,
    ):
        result = HF_PLUGIN["resolve"]("nerijs/pixel-art-xl", None, _lora_info())
    caps = result.manifest["capabilities"]
    assert caps["default_steps"] == 25
    assert caps["default_size"] == [1024, 1024]


def test_resolve_lora_multiple_safetensors_fails_actionably():
    import pytest
    from muse.core.resolvers import ResolverError
    info = _lora_info(
        siblings=["a.safetensors", "b.safetensors", "README.md"],
    )
    with pytest.raises(ResolverError, match="a.safetensors"):
        HF_PLUGIN["resolve"]("org/multi-lora", None, info)


def test_resolve_lora_memory_estimate_from_base_weights():
    from unittest.mock import patch
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=6.9,
    ) as est:
        result = HF_PLUGIN["resolve"]("nerijs/pixel-art-xl", None, _lora_info())
    est.assert_called_once_with("stabilityai/stable-diffusion-xl-base-1.0")
    assert result.manifest["capabilities"]["memory_gb"] == 7.2  # 6.9 + 0.3


def test_resolve_lora_download_patterns():
    from pathlib import Path
    from unittest.mock import patch
    with patch(
        "muse.modalities.image_generation.hf._estimate_repo_weights_gb",
        return_value=None,
    ), patch(
        "muse.modalities.image_generation.hf.snapshot_download",
        return_value="/tmp/fake",
    ) as snap:
        result = HF_PLUGIN["resolve"]("nerijs/pixel-art-xl", None, _lora_info())
        result.download(Path("/tmp/cache"))
    patterns = snap.call_args.kwargs["allow_patterns"]
    assert "*.safetensors" in patterns
    assert "*.json" in patterns
    # No subfolder pipeline patterns for an adapter-only repo.
    assert "*/*.safetensors" not in patterns
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest tests/modalities/image_generation/test_hf_plugin.py -q -k "resolve_lora"`
Expected: 7x FAIL/ERROR (`_estimate_repo_weights_gb` does not exist / adapter repos fall into the diffusers-shape resolve)

- [ ] **Step 3: Implement**

In `src/muse/modalities/image_generation/hf.py`:

Add imports at the top (`logging` is new; `ResolverError` joins the existing
`from muse.core.resolvers import ...` line):

```python
import logging

from muse.core.resolvers import ResolvedModel, ResolverError, SearchResult

logger = logging.getLogger(__name__)
```

Add module-level helpers (below `_is_lora_adapter`):

```python
_LORA_PIP_EXTRAS = _PIP_EXTRAS + ("peft",)

# base_model:<qualifier>:<repo> forms that do NOT name a usable base.
_NON_BASE_QUALIFIERS = (
    "base_model:finetune:", "base_model:quantized:", "base_model:merge:",
)


def _lora_base_from_tags(tags: list[str]) -> str | None:
    """Extract the base repo from HF's base_model tag convention.

    `base_model:adapter:<repo>` is the explicit adapter-relationship tag;
    prefer it. Fall back to a plain `base_model:<repo>` tag that is not a
    qualified non-base form. Returns None when nothing matches (the
    post-overlay validation in catalog.pull handles that case).
    """
    for t in tags:
        if t.startswith("base_model:adapter:"):
            return t[len("base_model:adapter:"):]
    for t in tags:
        if (
            t.startswith("base_model:")
            and not t.startswith("base_model:adapter:")
            and not t.startswith(_NON_BASE_QUALIFIERS)
        ):
            return t[len("base_model:"):]
    return None


def _estimate_repo_weights_gb(repo_id: str) -> float | None:
    """Sum an HF repo's weight-file sizes (GB) for a memory estimate.

    Used to size a LoRA entry from its BASE repo, because the adapter's
    own on-disk footprint (tens of MB) would grossly undersize the load.
    Mirrors the fp16-preference of the downloader: when fp16 variants
    exist, only they are fetched, so only they should be summed.
    Returns None on any failure; the post-pull probe measures the real
    peak and self-heals sizing regardless.
    """
    if "/" not in repo_id:
        return None  # muse catalog id: sizing derives from that entry
    try:
        api = HfApi()
        info = api.model_info(repo_id, files_metadata=True)
        files = [
            (s.rfilename, s.size or 0)
            for s in getattr(info, "siblings", [])
            if s.rfilename.endswith((".safetensors", ".bin"))
        ]
        if not files:
            return None
        fp16 = [(n, sz) for n, sz in files if ".fp16." in n]
        total = sum(sz for _, sz in (fp16 or files))
        return total / 1e9 if total else None
    except Exception as e:  # noqa: BLE001
        logger.debug("base weight-size estimate failed for %s: %s", repo_id, e)
        return None


def _resolve_lora(repo_id: str, info) -> ResolvedModel:
    """Synthesize a manifest for an adapter-only repo.

    The manifest reuses the standard diffusers runtime; the runtime's
    lora_adapter branch loads the BASE pipeline and layers the adapter
    on top (unfused). base_model may be absent here (tagless repo with
    a --base override coming via the curated/CLI capabilities overlay);
    catalog.pull validates the merged result.
    """
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []

    weights = [
        f for f in siblings if f.endswith(".safetensors") and "/" not in f
    ]
    if len(weights) != 1:
        raise ResolverError(
            f"LoRA repo {repo_id!r} has {len(weights)} top-level .safetensors "
            f"files ({sorted(weights)}); muse supports exactly one adapter "
            f"weight file per entry"
        )

    base = _lora_base_from_tags(tags)
    capabilities: dict[str, Any] = {
        # Generation defaults follow the BASE the adapter was declared
        # against (turbo bases get 1-step/no-guidance automatically).
        **_infer_defaults(base if base else repo_id),
        "lora_adapter": True,
        "lora_scale": 1.0,
        "supports_negative_prompt": True,
        "supports_seeded_generation": True,
        "supports_img2img": True,
        "supports_inpainting": True,
        "supports_variations": True,
    }
    if base:
        capabilities["base_model"] = base
        est = _estimate_repo_weights_gb(base)
        if est:
            # Adapter + runtime overhead margin on top of base weights.
            capabilities["memory_gb"] = round(est + 0.3, 1)

    manifest = {
        "model_id": _model_id(repo_id),
        "modality": "image/generation",
        "hf_repo": repo_id,
        "description": f"Diffusers LoRA adapter: {repo_id}",
        "license": _repo_license(info),
        "pip_extras": list(_LORA_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": capabilities,
    }

    def _download(cache_root: Path) -> Path:
        # Adapter repos are flat: weights + configs at the top level,
        # tens of MB total. No subfolder pipeline tree to filter.
        return Path(snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_root) if cache_root else None,
            allow_patterns=["*.safetensors", "*.json", "*.txt"],
        ))

    return ResolvedModel(
        manifest=manifest,
        backend_path=_RUNTIME_PATH,
        download=_download,
    )
```

Dispatch at the top of the existing `_resolve`:

```python
def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    siblings = [s.rfilename for s in getattr(info, "siblings", [])]
    tags = getattr(info, "tags", None) or []
    has_pipeline_config = any(
        Path(f).name == "model_index.json" for f in siblings
    )
    if not has_pipeline_config and _is_lora_adapter(siblings, tags):
        return _resolve_lora(repo_id, info)
    # ... existing body unchanged from here ...
```

(`from typing import Any` is already imported in hf.py.)

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/modalities/image_generation/test_hf_plugin.py -q`
Expected: all PASS (25)

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/image_generation/hf.py tests/modalities/image_generation/test_hf_plugin.py
git commit -m "feat(image_generation): resolve LoRA adapter repos with base pairing

Adapter-shape repos synthesize a manifest on the standard diffusers
runtime with lora_adapter/base_model/lora_scale capabilities. Base
extracted from base_model:adapter: tags (plain base_model: fallback);
defaults derive from the base id; memory estimated from the base
repo's weight sizes via one HF API call; peft added to pip_extras
(modern diffusers requires the PEFT backend for load_lora_weights).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J2SDRmdTMP3sVoBpAqZ3VV"
```

---

### Task 4: Pull threading (`--base`) + post-overlay validation

**Files:**
- Modify: `src/muse/core/catalog.py` (`pull`, `_pull_via_resolver`)
- Modify: `src/muse/cli.py` (pull command)
- Modify: `src/muse/cli_impl/pull_errors.py`
- Test: `tests/core/test_catalog.py`, `tests/cli_impl/test_pull_errors.py`

**Interfaces:**
- Consumes: manifests from Task 3 (capabilities may lack `base_model`).
- Produces: `pull(identifier: str, *, base_override: str | None = None)`. `_validate_lora_capabilities(manifest)` raises `ResolverError` when a lora_adapter manifest has no `base_model` post-overlay, or when its muse-id base is not pulled. `friendly_pull_error` renders `ResolverError` as a clean one-liner.

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_catalog.py`:

```python
class TestLoraPullValidation:
    def _lora_manifest(self, caps):
        return {
            "model_id": "some-lora",
            "modality": "image/generation",
            "hf_repo": "org/some-lora",
            "backend_path": "muse.modalities.image_generation.runtimes.diffusers:DiffusersText2ImageModel",
            "capabilities": caps,
        }

    def test_lora_without_base_model_raises_actionable(self, tmp_catalog):
        import pytest
        from muse.core.catalog import _validate_lora_capabilities
        from muse.core.resolvers import ResolverError

        with pytest.raises(ResolverError, match="--base"):
            _validate_lora_capabilities(
                self._lora_manifest({"lora_adapter": True})
            )

    def test_lora_with_unpulled_muse_base_raises_actionable(self, tmp_catalog):
        import pytest
        from muse.core.catalog import _validate_lora_capabilities
        from muse.core.resolvers import ResolverError

        with pytest.raises(ResolverError, match="muse pull sdxl-turbo"):
            _validate_lora_capabilities(self._lora_manifest(
                {"lora_adapter": True, "base_model": "sdxl-turbo"}
            ))

    def test_lora_with_pulled_muse_base_passes(self, tmp_catalog):
        import json
        from muse.core.catalog import (
            _catalog_path, _reset_read_catalog_cache,
            _validate_lora_capabilities,
        )

        p = _catalog_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "sdxl-turbo": {"local_dir": "/w/sdxl-turbo", "enabled": True},
        }))
        _reset_read_catalog_cache()
        _validate_lora_capabilities(self._lora_manifest(
            {"lora_adapter": True, "base_model": "sdxl-turbo"}
        ))  # no raise

    def test_lora_with_hf_repo_base_passes_without_catalog(self, tmp_catalog):
        from muse.core.catalog import _validate_lora_capabilities

        _validate_lora_capabilities(self._lora_manifest({
            "lora_adapter": True,
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        }))  # no raise: HF-repo bases download at load time

    def test_non_lora_manifest_is_ignored(self, tmp_catalog):
        from muse.core.catalog import _validate_lora_capabilities

        _validate_lora_capabilities(self._lora_manifest({}))  # no raise
```

Append to `tests/cli_impl/test_pull_errors.py` (inside or after
`TestOtherAccessErrors`):

```python
class TestResolverErrors:
    def test_resolver_error_renders_message_without_traceback(self):
        from muse.core.resolvers import ResolverError

        exc = ResolverError(
            "LoRA adapter 'x' declares no base model; re-run with --base"
        )
        msg = friendly_pull_error("x", exc)
        assert msg is not None
        assert "--base" in msg
        assert "Traceback" not in msg
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest tests/core/test_catalog.py::TestLoraPullValidation tests/cli_impl/test_pull_errors.py::TestResolverErrors -v`
Expected: ImportError on `_validate_lora_capabilities`; the pull_errors test FAILS (returns None)

- [ ] **Step 3: Implement**

In `src/muse/core/catalog.py`, add near `_pull_via_resolver`:

```python
def _validate_lora_capabilities(manifest: dict) -> None:
    """Reject unservable LoRA manifests at pull time, post-overlay.

    A lora_adapter entry without a base_model can never load; a
    muse-id base that is not pulled would fail at first request with a
    from_pretrained error. Both fail here, BEFORE the expensive venv
    creation and download, with the fix in the message. Runs after the
    curated/--base capabilities overlay merge so a --base override can
    satisfy a tagless adapter repo.
    """
    from muse.core.resolvers import ResolverError

    caps = manifest.get("capabilities") or {}
    if not caps.get("lora_adapter"):
        return
    model_id = manifest.get("model_id", "<unknown>")
    base = caps.get("base_model")
    if not base:
        raise ResolverError(
            f"LoRA adapter {model_id!r} declares no base model and none was "
            f"given; re-run with: muse pull <identifier> --base "
            f"<muse-id-or-hf-repo>"
        )
    if "/" not in base:
        entry = _read_catalog().get(base)
        if not entry or not entry.get("local_dir"):
            raise ResolverError(
                f"LoRA base {base!r} is not pulled; run `muse pull {base}` "
                f"first, then retry"
            )
```

In `_pull_via_resolver`, immediately AFTER the `capabilities_overlay` merge
block and the `model_id_override` assignment (before `venvs_root = ...`):

```python
    _validate_lora_capabilities(manifest)
```

Change `pull`'s signature and the three dispatch branches:

```python
def pull(identifier: str, *, base_override: str | None = None) -> None:
```

Curated-with-uri branch (replace the existing `_pull_via_resolver(...)` call):

```python
            overlay = dict(curated.capabilities or {})
            if base_override:
                overlay["base_model"] = base_override
            _pull_via_resolver(
                curated.uri,
                model_id_override=curated.id,
                capabilities_overlay=overlay or None,
                modality_override=curated.modality,
            )
```

Bare `://` URI branch:

```python
            _pull_via_resolver(
                identifier,
                capabilities_overlay=(
                    {"base_model": base_override} if base_override else None
                ),
            )
```

Bare-id branch: add before the existing logic:

```python
    if base_override:
        logger.warning(
            "--base only applies to resolver-pulled LoRA adapters; "
            "ignored for %s", identifier,
        )
```

In `src/muse/cli.py` pull command, add the option after `identifier` and
thread it into the `_pull` call inside the try block:

```python
    base: Annotated[
        str | None,
        typer.Option(
            "--base",
            help=(
                "(LoRA pulls) pair the adapter with this base model: a "
                "pulled muse id (e.g. sdxl-turbo) or an HF repo "
                "(org/name). Overrides the base the repo declares."
            ),
        ),
    ] = None,
```

```python
            _pull(identifier, base_override=base)
```

In `src/muse/cli_impl/pull_errors.py`, add as the FIRST classification in
`friendly_pull_error` (before the huggingface_hub import):

```python
    try:
        from muse.core.resolvers import ResolverError
    except Exception:  # noqa: BLE001
        ResolverError = None  # type: ignore[assignment]
    if ResolverError is not None and isinstance(exc, ResolverError):
        # Muse's own resolver errors are written to be actionable
        # one-liners; show them without the traceback.
        return f"error: {exc}"
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/core/test_catalog.py tests/cli_impl/test_pull_errors.py tests/test_cli.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/muse/core/catalog.py src/muse/cli.py src/muse/cli_impl/pull_errors.py tests/core/test_catalog.py tests/cli_impl/test_pull_errors.py
git commit -m "feat(pull): --base override + pull-time LoRA validation

--base threads through pull() as a capabilities overlay (base_model),
so curated pairings and tagless adapter repos share one mechanism.
_validate_lora_capabilities rejects baseless adapters and unpulled
muse-id bases before venv/download work, with the fix in the message.
friendly_pull_error renders ResolverError without a traceback.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J2SDRmdTMP3sVoBpAqZ3VV"
```

---

### Task 5: Runtime LoRA branch

**Files:**
- Modify: `src/muse/modalities/image_generation/runtimes/diffusers.py`
- Test: `tests/modalities/image_generation/runtimes/test_diffusers.py`

**Interfaces:**
- Consumes: `resolve_model_source` (Task 1); capability kwargs `lora_adapter`, `base_model`, `lora_scale` (Task 3) arriving via the existing `**` capabilities splat in `load_backend`.
- Produces: `generate(..., lora_scale: float | None = None)`; LoRA pipelines loaded from the base source with `load_lora_weights(adapter_src)`, never `fuse_lora`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/modalities/image_generation/runtimes/test_diffusers.py`:

```python
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest tests/modalities/image_generation/runtimes/test_diffusers.py -q -k "Lora"`
Expected: FAIL/ERROR across the board (`resolve_model_source` not importable from the runtime module; constructor loads adapter dir; no cross_attention_kwargs)

- [ ] **Step 3: Implement**

In `src/muse/modalities/image_generation/runtimes/diffusers.py`:

Extend the runtime_helpers import:

```python
from muse.core.runtime_helpers import (
    dtype_for_name,
    resolve_model_source,
    select_device,
)
```

Constructor: add kwargs after `default_guidance`:

```python
        lora_adapter: bool = False,
        base_model: str | None = None,
        lora_scale: float = 1.0,
```

Replace the load block (from the `self._src = local_dir or hf_repo` stash
through the `from_pretrained` call) with:

```python
        # Stash for lazy img2img / inpaint pipeline loads (same checkpoint + dtype).
        # For LoRA entries self._src is the ADAPTER source; the pipeline
        # itself loads from the resolved BASE below.
        self._src = local_dir or hf_repo
        self._dtype = dtype
        self._i2i_pipe = None
        self._inp_pipe = None
        self._lora_adapter = bool(lora_adapter)
        self._default_lora_scale = float(lora_scale)

        if self._lora_adapter:
            if not base_model:
                raise RuntimeError(
                    f"{model_id}: lora_adapter is set but base_model is "
                    f"missing; re-pull with `muse pull <id> --base "
                    f"<muse-id-or-hf-repo>`"
                )
            # muse catalog id -> that entry's snapshot dir; HF repo id
            # passes through for from_pretrained to fetch (HF cache).
            load_src = resolve_model_source(base_model)
        else:
            load_src = self._src

        # Access torch through this module so tests' patches survive.
        import muse.modalities.image_generation.runtimes.diffusers as _mod
        _torch = _mod.torch
        torch_dtype = dtype_for_name(dtype, _torch)
        self._torch_dtype = torch_dtype

        logger.info(
            "loading diffusers pipeline from %s (model_id=%s, device=%s, dtype=%s)",
            load_src, model_id, self._device, dtype,
        )
        # Request the fp16 variant only when the local snapshot actually
        # holds .fp16. weights; otherwise from_pretrained raises for repos
        # that ship no fp16 files (e.g. flux-schnell). torch_dtype still
        # governs the compute dtype independently of variant (H6).
        use_fp16_variant = dtype == "float16" and _local_has_fp16_variant(load_src)
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            load_src,
            torch_dtype=torch_dtype,
            variant="fp16" if use_fp16_variant else None,
        )
        if self._lora_adapter:
            # Unfused: fuse_lora would bake the adapter into the base
            # weights and make per-request lora_scale impossible.
            logger.info(
                "loading LoRA adapter %s onto base %s", self._src, base_model,
            )
            self._pipe.load_lora_weights(self._src)
        if self._device != "cpu":
            self._pipe = self._pipe.to(self._device)
```

`generate` signature: add `lora_scale: float | None = None,` after
`strength`. In the t2i path, after the `if gen is not None:` block:

```python
        if self._lora_adapter:
            s_lora = (
                lora_scale if lora_scale is not None
                else self._default_lora_scale
            )
            call_kwargs["cross_attention_kwargs"] = {"scale": s_lora}
```

In `_generate_img2img` and `inpaint`, after their `if gen is not None:`
blocks (v1: edit routes use the configured default scale):

```python
        if self._lora_adapter:
            call_kwargs["cross_attention_kwargs"] = {
                "scale": self._default_lora_scale,
            }
```

Update the class docstring's construction-kwargs list with the three new
keys.

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/modalities/image_generation/runtimes/test_diffusers.py -q`
Expected: all PASS (existing + 8 new)

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/image_generation/runtimes/diffusers.py tests/modalities/image_generation/runtimes/test_diffusers.py
git commit -m "feat(image_generation): LoRA branch in the diffusers runtime

lora_adapter entries load the pipeline from the resolved base source
(muse id -> local_dir via resolve_model_source; HF repo verbatim),
then layer the adapter with load_lora_weights, unfused. Per-request
lora_scale flows into cross_attention_kwargs on the t2i path; edit
and variation paths use the configured default scale.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J2SDRmdTMP3sVoBpAqZ3VV"
```

---

### Task 6: Route `lora_scale` field + capability gate

**Files:**
- Modify: `src/muse/modalities/image_generation/routes.py`
- Test: `tests/modalities/image_generation/test_routes.py`

**Interfaces:**
- Consumes: `registry.manifest(MODALITY, effective_id)` (existing, see the img2img gate at routes.py:78-85); runtime `generate(..., lora_scale=...)` from Task 5.
- Produces: `GenerationRequest.lora_scale: float | None` (ge=0.0, le=2.0); 400 `lora_not_supported` for non-LoRA models.

- [ ] **Step 1: Write the failing tests**

Append to `tests/modalities/image_generation/test_routes.py`, matching the
file's existing FakeModel + TestClient conventions (read the top of the file
first and reuse its fixture/helper names; the sketch below assumes a helper
that builds an app from a registry with a fake model and a manifest):

```python
def test_lora_scale_on_non_lora_model_returns_400():
    """lora_scale against a model without capabilities.lora_adapter is a
    capability mismatch, mirroring the img2img gate."""
    # Register a fake model WITHOUT lora_adapter in its manifest
    # (follow the file's existing registry+manifest setup pattern).
    resp = client.post("/v1/images/generations", json={
        "prompt": "a cat", "lora_scale": 0.8,
    })
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "lora_not_supported"


def test_lora_scale_forwarded_to_lora_model():
    """A model whose manifest declares lora_adapter receives the value."""
    # Register a fake model WITH {"capabilities": {"lora_adapter": True}}.
    resp = client.post("/v1/images/generations", json={
        "prompt": "pixel art, a knight", "lora_scale": 0.7,
    })
    assert resp.status_code == 200
    assert fake_model.last_kwargs["lora_scale"] == 0.7


def test_lora_scale_out_of_range_is_422():
    resp = client.post("/v1/images/generations", json={
        "prompt": "a cat", "lora_scale": 3.5,
    })
    assert resp.status_code == 422
```

(The exact fixture wiring must copy the pattern already used by the img2img
gate tests in that file: same registry construction, same fake model class
extended with a `last_kwargs` capture in its `generate`.)

- [ ] **Step 2: Run tests, verify they fail**

Run: `python -m pytest tests/modalities/image_generation/test_routes.py -q -k "lora"`
Expected: FAIL (unknown field is ignored by pydantic, no 400/422 emitted)

- [ ] **Step 3: Implement**

In `src/muse/modalities/image_generation/routes.py`:

Add to `GenerationRequest` after `strength`:

```python
    lora_scale: float | None = Field(default=None, ge=0.0, le=2.0)
```

In the `generations` handler, after the `effective_id` resolution and BEFORE
the img2img block, add the gate:

```python
        # lora_scale only applies to LoRA adapter models; reject early
        # for others (mirrors the supports_img2img capability gate).
        if req.lora_scale is not None:
            manifest = registry.manifest(MODALITY, effective_id) or {}
            if not manifest.get("capabilities", {}).get("lora_adapter"):
                return error_response(
                    400,
                    "lora_not_supported",
                    f"model {effective_id!r} is not a LoRA adapter model; "
                    f"lora_scale only applies to LoRA models",
                )
```

Add to the `_call_one` kwargs dict:

```python
                "lora_scale": req.lora_scale,
```

(Non-LoRA models never see a non-None value because of the gate; the runtime
signature default absorbs the None.)

Update the module docstring's field list to mention `lora_scale`.

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest tests/modalities/image_generation/test_routes.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/muse/modalities/image_generation/routes.py tests/modalities/image_generation/test_routes.py
git commit -m "feat(image_generation): per-request lora_scale on /v1/images/generations

Optional field, [0.0, 2.0]; 400 lora_not_supported when the target
model lacks capabilities.lora_adapter, mirroring the img2img gate.
OpenAI SDK callers pass it via extra_body.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J2SDRmdTMP3sVoBpAqZ3VV"
```

---

### Task 7: Sizing: `backfill_manifest_memory` chases the base

**Files:**
- Modify: `src/muse/cli_impl/supervisor.py` (`backfill_manifest_memory`, ~line 820)
- Test: locate with `grep -rln "backfill_manifest_memory" tests/` and append there (create `tests/cli_impl/test_backfill_memory.py` if none exists)

**Interfaces:**
- Consumes: `_has_memory_data(entry) -> (bool, float, str)` (existing, same module); capabilities `lora_adapter` / `base_model` (Task 3).
- Produces: a LoRA entry with no own probe measurement is sized from its muse-id base entry instead of its (tiny) adapter dir.

- [ ] **Step 1: Write the failing tests**

Append (adapting to the located test file's existing fixtures for
`MUSE_CATALOG_DIR` + catalog writing; the shape below is self-contained):

```python
class TestBackfillLoraChase:
    def _write(self, tmp_path, entries):
        import json
        (tmp_path / "catalog.json").write_text(json.dumps(entries))

    def _lora_manifest(self, base="sdxl-turbo"):
        return {
            "model_id": "pixel-art-xl",
            "modality": "image/generation",
            "capabilities": {"lora_adapter": True, "base_model": base},
        }

    def test_unprobed_lora_sizes_from_base_measurement(self, tmp_path, monkeypatch):
        from muse.cli_impl.supervisor import backfill_manifest_memory
        from muse.core.catalog import _reset_read_catalog_cache

        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        self._write(tmp_path, {
            "pixel-art-xl": {"local_dir": str(tmp_path), "enabled": True},
            "sdxl-turbo": {
                "local_dir": "/w/sdxl-turbo", "enabled": True,
                "measurements": {"cuda": {"peak_bytes": 8_000_000_000}},
            },
        })
        _reset_read_catalog_cache()
        out = backfill_manifest_memory(self._lora_manifest(), "pixel-art-xl")
        assert out["capabilities"]["memory_gb"] == pytest.approx(8.0, rel=0.1)

    def test_probed_lora_uses_own_measurement_not_base(self, tmp_path, monkeypatch):
        from muse.cli_impl.supervisor import backfill_manifest_memory
        from muse.core.catalog import _reset_read_catalog_cache

        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        self._write(tmp_path, {
            "pixel-art-xl": {
                "local_dir": str(tmp_path), "enabled": True,
                "measurements": {"cuda": {"peak_bytes": 9_000_000_000}},
            },
            "sdxl-turbo": {
                "local_dir": "/w/sdxl-turbo", "enabled": True,
                "measurements": {"cuda": {"peak_bytes": 8_000_000_000}},
            },
        })
        _reset_read_catalog_cache()
        out = backfill_manifest_memory(self._lora_manifest(), "pixel-art-xl")
        assert out["capabilities"]["memory_gb"] == pytest.approx(9.0, rel=0.1)

    def test_hf_repo_base_lora_keeps_existing_behavior(self, tmp_path, monkeypatch):
        """HF-repo base (contains /): no catalog entry to chase; the
        entry's own ladder result is used (resolve-time estimate covers
        the honest number in practice)."""
        from muse.cli_impl.supervisor import backfill_manifest_memory
        from muse.core.catalog import _reset_read_catalog_cache

        monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
        self._write(tmp_path, {
            "pixel-art-xl": {"local_dir": str(tmp_path), "enabled": True},
        })
        _reset_read_catalog_cache()
        out = backfill_manifest_memory(
            self._lora_manifest(base="stabilityai/stable-diffusion-xl-base-1.0"),
            "pixel-art-xl",
        )
        # No base entry, no own measurement, empty local_dir: nothing
        # sensible to backfill; capabilities stay untouched.
        assert out["capabilities"].get("memory_gb") is None
```

(Exact `gb` values depend on `_has_memory_data`'s bytes-to-GB conversion;
read that function first and pin the expected numbers to its arithmetic,
using `pytest.approx` as above.)

- [ ] **Step 2: Run tests, verify the first fails**

Run: `python -m pytest <located-test-file> -q -k "BackfillLoraChase"`
Expected: first test FAILS (memory_gb is None or a tiny weights-derived number, not ~8.0)

- [ ] **Step 3: Implement**

In `backfill_manifest_memory`, replace the memory-backfill block:

```python
    catalog = _read_catalog()
    entry = catalog.get(model_id)
    out = manifest
    caps = manifest.get("capabilities", {}) or {}

    if entry is not None and caps.get("memory_gb") is None:
        gb: float | None = None
        # A LoRA entry's own dir holds only the adapter (tens of MB), so
        # the weights-on-disk fallback would grossly undersize the load.
        # When it has no probe measurement of its own, size it from its
        # muse-id base entry instead. A probed LoRA entry measured the
        # real base+adapter peak; prefer that.
        if caps.get("lora_adapter") and not (entry.get("measurements") or {}):
            base = caps.get("base_model")
            base_entry = (
                catalog.get(base) if base and "/" not in base else None
            )
            if base_entry is not None:
                has_b, gb_b, _d = _has_memory_data(base_entry)
                if has_b and gb_b > 0:
                    gb = gb_b
        if gb is None:
            has_data, gb_own, _device = _has_memory_data(entry)
            if has_data and gb_own > 0:
                gb = gb_own
        if gb is not None:
            out = dict(out)
            out_caps = dict(caps)
            out_caps["memory_gb"] = gb
            out["capabilities"] = out_caps
```

(The function previously called `_read_catalog().get(model_id)` once; keep a
single `catalog` read so entry and base come from the same snapshot. The
`device_override` block below it is unchanged.)

Note for the HF-repo-base test: `_has_memory_data` on an entry whose
`local_dir` is an empty tmp dir may return a tiny positive weights size; if
it does, adjust the test to assert `memory_gb` is either None or < 0.01
rather than strictly None. Verify against `_has_memory_data`'s actual
behavior when running the test.

- [ ] **Step 4: Run tests, verify they pass**

Run: `python -m pytest <located-test-file> -q` then `python -m pytest tests/cli_impl -q -m "not slow"`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/muse/cli_impl/supervisor.py <located-test-file>
git commit -m "feat(sizing): size unprobed LoRA entries from their base entry

A LoRA entry's on-disk dir holds only the adapter, so the
weights-on-disk fallback undersized the load by ~100x. When a
lora_adapter entry has no probe measurement, backfill_manifest_memory
now chases capabilities.base_model (muse-id bases) and uses that
entry's sizing. Probed LoRA entries keep their own measured peak.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J2SDRmdTMP3sVoBpAqZ3VV"
```

---

### Task 8: Curated entry + docs

**Files:**
- Modify: `src/muse/curated.yaml` (after the `flux-schnell` entry in the image/generation block)
- Modify: `CLAUDE.md` (image/generation bullet in the modality list), `README.md` (models/features mention)

**Interfaces:**
- Consumes: everything above. The curated overlay mechanism (existing) merges these capabilities into the resolved manifest.

- [ ] **Step 1: Add the curated entry**

```yaml
- id: pixel-art-xl
  uri: hf://nerijs/pixel-art-xl
  modality: image/generation
  size_gb: 0.05
  description: "Pixel-art LoRA for SDXL (trigger phrase: 'pixel art'); pre-paired with sdxl-turbo (pull sdxl-turbo first) for fast 4-step generation"
  capabilities:
    lora_adapter: true
    base_model: sdxl-turbo
    lora_scale: 1.0
    default_size: [1024, 1024]
    default_steps: 4
    default_guidance: 0.0
    memory_gb: 8.0
```

Rationale pinned here so the implementer does not second-guess: defaults are
set explicitly because the resolver derives them from the DECLARED base
(SDXL-base: 25 steps), while this pairing runs on sdxl-turbo (guidance must
be 0.0; 4 steps gives the adapter room to act; pixel-art-xl was trained at
1024). memory_gb matches the sdxl-turbo curated entry. Verify the trigger
phrase against the repo README during Task 9 and correct the description if
it differs.

- [ ] **Step 2: Run the curated validation tests**

Run: `python -m pytest tests/core/ -q -k "curated"`
Expected: all PASS (the yaml loads, entry shape valid)

- [ ] **Step 3: Docs**

CLAUDE.md: in the modality list bullet for **image/generation**, extend the
parenthetical with: `; LoRA adapters via lora_adapter/base_model
capabilities (pixel-art-xl curated, pre-paired with sdxl-turbo; per-request
lora_scale; muse pull --base overrides the declared base)`.

README.md: add one row/line in the image generation feature area:
`LoRA adapters: muse pull hf://nerijs/pixel-art-xl (or curated pixel-art-xl),
optional --base <muse-id-or-hf-repo>, per-request lora_scale via extra_body.`

- [ ] **Step 4: Full fast lane**

Run: `python -m pytest tests/ -q -m "not slow"`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/muse/curated.yaml CLAUDE.md README.md
git commit -m "feat(curated): pixel-art-xl LoRA entry pre-paired with sdxl-turbo

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01J2SDRmdTMP3sVoBpAqZ3VV"
```

---

### Task 9: Deploy + real-API verification on .204 (Step B1)

**Files:** none (operational). Release bump is a separate, user-gated step.

**Interfaces:**
- Consumes: everything above, deployed to the .204 GPU box.

- [ ] **Step 1: Push main and pull on .204** (pre-authorized deploy path)

```bash
git push origin main
ssh spinoza@192.168.0.204 'cd /home/spinoza/github/muse && git pull --ff-only origin main && git log --oneline -1'
```

Expected: .204 HEAD at the Task 8 commit.

- [ ] **Step 2: Refresh the sdxl-turbo venv is NOT needed; pull the LoRA**

The LoRA gets its own venv with peft. On .204 (or via admin API with the
token from the scratchpad `.admin_token`, redacting it from output):

```bash
ssh spinoza@192.168.0.204 '/home/spinoza/miniforge3/bin/muse pull pixel-art-xl'
```

Expected: sniff resolves the adapter, downloads ~50 MB, venv installs
peft, post-pull probe loads sdxl-turbo + adapter on cuda and records a
measurement. If the probe fails with a PEFT/diffusers version error,
capture the exact message; that is the Step B1 signal the pinned versions
need adjusting in `_LORA_PIP_EXTRAS`.

- [ ] **Step 3: Enable + generate**

```bash
TOK=$(cat <scratchpad>/.admin_token)
curl -s -X POST http://192.168.0.204:8000/v1/admin/models/pixel-art-xl/enable -H "Authorization: Bearer $TOK" | sed "s/$TOK/<REDACTED>/g"
# then two generations, same seed, different scales:
curl -s -X POST http://192.168.0.204:8000/v1/images/generations -H "Content-Type: application/json" -d '{"model": "pixel-art-xl", "prompt": "pixel art, a knight in a forest, crisp pixels", "seed": 42, "steps": 4, "guidance": 0.0, "lora_scale": 1.0}'
curl -s -X POST http://192.168.0.204:8000/v1/images/generations -H "Content-Type: application/json" -d '{"model": "pixel-art-xl", "prompt": "pixel art, a knight in a forest, crisp pixels", "seed": 42, "steps": 4, "guidance": 0.0, "lora_scale": 0.2}'
```

Verify: (a) both return 200 with images; (b) pixel-art style is visible at
scale 1.0 and visibly weaker at 0.2 (send both images to the user);
(c) latency is turbo-class (a few seconds, not 15+); (d) a request with
`"model": "sdxl-turbo", "lora_scale": 0.5` returns 400 `lora_not_supported`.

- [ ] **Step 4: Report + gate**

Report results (images, latency, any B1 corrections applied). The v0.50.0
version bump, tag, push, and PyPI publish wait for the user's explicit "go".

---

## Self-Review (performed at plan-writing time)

- Spec coverage: sniff (T2), resolve incl. tag parsing / weight-file rule /
  peft / estimate (T3), --base + validation + clean errors (T4), runtime
  branch + unfused + scale (T5), wire field + 400 (T6), sizing chase (T7),
  curated + docs (T8), Step B1 (T9). Spec's "defaults derived from base"
  covered for declared bases (T3) and pinned explicitly for the curated
  turbo pairing (T8, with rationale).
- Placeholders: Task 6 tests and Task 7 test-file location intentionally
  defer to in-repo conventions the implementer must read first; each such
  step says exactly what to read and mirrors a named existing pattern
  (img2img gate tests; backfill test location via grep). No TBDs.
- Type consistency: `resolve_model_source(ref: str) -> str` (T1) matches the
  T5 import and monkeypatch path; capability keys `lora_adapter` /
  `base_model` / `lora_scale` are spelled identically in T3, T4, T5, T6, T7,
  T8; `pull(identifier, *, base_override=None)` (T4) matches the cli.py call.
