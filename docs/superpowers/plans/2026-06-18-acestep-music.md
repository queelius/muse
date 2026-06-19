# ACE-Step 1.5 Music Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **RUN THIS ON A GPU HOST.** ACE-Step 1.5 needs an NVIDIA GPU, a git-installed SDK, and (per current docs) vllm. The current authoring host is CPU-only. Task 1 (B1) installs the real SDK on the GPU and CONFIRMS the specifics the later tasks depend on; correct the marked spots in Tasks 2/4 from Task 1's findings before relying on them.

**Goal:** Add `ACE-Step/Ace-Step1.5` (full song generation: vocals + lyrics + style) as a curated GPU runtime in muse's existing `audio/generation` modality (`POST /v1/audio/music`), shipped as v0.48.0.

**Architecture:** A new `ACEStepRuntime` in `audio_generation/runtimes/`, satisfying the existing `AudioGenerationModel` protocol (`generate(prompt, **kwargs) -> AudioGenerationResult`). ACE-Step 1.5 is a two-handler system (DIT `AceStepHandler` + LM `LLMHandler`) driven by a functional `generate_music(...)`. The single audio/generation HF plugin is extended to dispatch by repo name (stable-audio -> StableAudioRuntime, ace-step -> ACEStepRuntime). The wire gains one optional `lyrics` field.

**Tech Stack:** Python, the `acestep` SDK (git-installed from github.com/ace-step/ACE-Step-1.5), vllm, torch; pytest with the SDK mocked.

## Global Constraints

- **ASCII only in committed files.** A pre-commit hook rejects em-dashes/non-ASCII. Use `-`, `:`, `,`, `()`, `->`, `...`. If a commit is blocked, strip the char and retry; never `--no-verify`.
- **GPU-only.** `device: cuda`. Curated entry, never bundled. NOT added to `.github/workflows/fresh-venv-smoke.yml` (the free-tier CI matrix).
- **Audio is float32** at the protocol boundary, shape `(samples,)` mono or `(samples, channels)`; the codec converts to PCM/compressed downstream.
- **Deferred imports**: torch + the `acestep` handlers + `generate_music`/`GenerationParams`/`GenerationConfig` stay as module-top sentinels populated by `_ensure_deps()`; tests patch the sentinels. `muse --help`/discovery must work without the SDK.
- **Use `muse.core.runtime_helpers`** (`select_device`, `dtype_for_name`, `LoadTimer`); the meta-test flags re-implementations.
- **text2music only.** No editing / cover / audio2audio. `supports_sfx: false`.
- **B1-first**: do Task 1 before writing the Task 2 runtime body; the runtime code below is best-evidence from the GitHub `docs/en/INFERENCE.md` and the marked spots may need correction.
- **Release**: bump, full fast lane, build + wheel smoke-install, plus a REAL GPU generate before publishing. Direct-to-main. PyPI token in `~/.pypirc`.

Verified API facts (from ACE-Step-1.5 docs/en/INFERENCE.md; CONFIRM at B1):
```python
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import generate_music, GenerationParams, GenerationConfig

dit = AceStepHandler(); dit.initialize_service(project_root=..., config_path="acestep-v15-turbo", device="cuda")
llm = LLMHandler();    llm.initialize(checkpoint_dir=..., lm_model_path="acestep-5Hz-lm-1.7B", backend="vllm", device="cuda")
params = GenerationParams(task_type="text2music", caption=<style>, lyrics=<lyrics>,
                          duration=-1.0, inference_steps=8, guidance_scale=7.0, seed=-1)
config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True)
result = generate_music(dit, llm, params, config, save_dir=None)
# result.audios[0] = {"tensor": Tensor[channels, samples] float32 CPU, "sample_rate": 48000, ...}
```

---

### Task 1: B1 verification on the GPU host

**Files:** record findings in `docs/superpowers/specs/2026-06-18-acestep-music-design.md` (append a "B1 findings" section). No muse code.

This gates Tasks 2 and 4. Run it on the GPU box.

- [ ] **Step 1: Install the SDK**

```bash
git clone https://github.com/ace-step/ACE-Step-1.5.git /tmp/acestep15
cd /tmp/acestep15
python -m venv /tmp/acestep-b1 && . /tmp/acestep-b1/bin/activate
pip install -e .            # if this fails, follow the repo's uv sync path; record what worked
python -c "import acestep, vllm; print('acestep + vllm import OK')"
```

- [ ] **Step 2: Confirm the API + run one real generation**

Download the HF repo (`huggingface-cli download ACE-Step/Ace-Step1.5 --local-dir /tmp/acestep15-weights`), then introspect + generate:
```python
import inspect
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import generate_music, GenerationParams, GenerationConfig
print("AceStepHandler.initialize_service:", inspect.signature(AceStepHandler().initialize_service))
print("LLMHandler.initialize:", inspect.signature(LLMHandler().initialize))
print("GenerationParams fields:", GenerationParams.__dataclass_fields__.keys() if hasattr(GenerationParams,'__dataclass_fields__') else inspect.signature(GenerationParams))
print("GenerationConfig fields:", inspect.signature(GenerationConfig))
# then initialize both handlers against /tmp/acestep15-weights and run a 10s generate;
# inspect result.audios[0]: tensor.shape, tensor.dtype, sample_rate.
```

Record and answer:
1. The exact install recipe that worked (pip -e vs uv sync); confirm `vllm` is mandatory or whether `LLMHandler.initialize(backend=...)` accepts `"transformers"`/`"hf"`.
2. The arg names of `initialize_service` / `initialize`, and how the downloaded HF `Ace-Step1.5` dir maps to `project_root` / `checkpoint_dir` / `config_path` / `lm_model_path` (do the sub-model names `acestep-v15-turbo` / `acestep-5Hz-lm-1.7B` match subdirs in the snapshot?).
3. `GenerationParams` and `GenerationConfig` exact field names + defaults (caption, lyrics, duration, inference_steps, guidance_scale, seed).
4. `result.audios[0]["tensor"]` orientation (`[channels, samples]` vs `[samples, channels]`), dtype, range, and `sample_rate` (expected 48000).
5. Peak VRAM (DIT turbo + LM 1.7B + VAE) for `memory_gb`; on-disk size for `size_gb`.

- [ ] **Step 3: Record findings + commit**

```bash
# append the answers to the spec
git add docs/superpowers/specs/2026-06-18-acestep-music-design.md
git commit -m "docs(spec): record ACE-Step 1.5 B1 SDK-verification findings"
```

---

### Task 2: ACEStepRuntime + unit tests

**Files:**
- Create: `src/muse/modalities/audio_generation/runtimes/acestep.py`
- Test: `tests/modalities/audio_generation/runtimes/test_acestep.py`

**Interfaces:**
- Consumes: `muse.modalities.audio_generation.protocol.AudioGenerationResult`; `muse.core.runtime_helpers.select_device`.
- Produces: `ACEStepRuntime(model_id=, hf_repo=, local_dir=, device="cuda", default_steps=8, default_guidance=7.0, default_sample_rate=48000, **_)` with `model_id` attr + `generate(prompt, *, duration=None, seed=None, steps=None, guidance=None, negative_prompt=None, **kwargs) -> AudioGenerationResult`. A module-level `_normalize_acestep_output(tensor) -> (np.ndarray, int)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/modalities/audio_generation/runtimes/test_acestep.py`:

```python
"""ACEStepRuntime: mocked-SDK tests (no GPU, no real acestep).
Mocks reflect the ACE-Step 1.5 API from docs/en/INFERENCE.md."""
from __future__ import annotations
from unittest.mock import MagicMock
import numpy as np
import pytest

import muse.modalities.audio_generation.runtimes.acestep as mod
from muse.modalities.audio_generation.protocol import AudioGenerationResult, AudioGenerationModel


@pytest.fixture(autouse=True)
def _reset_sentinels():
    orig = (mod.torch, mod.AceStepHandler, mod.LLMHandler,
            mod.generate_music, mod.GenerationParams, mod.GenerationConfig)
    yield
    (mod.torch, mod.AceStepHandler, mod.LLMHandler,
     mod.generate_music, mod.GenerationParams, mod.GenerationConfig) = orig


def _wire(tensor=None):
    mod.torch = MagicMock()
    mod.torch.cuda.is_available.return_value = True
    mod.AceStepHandler = MagicMock()
    mod.LLMHandler = MagicMock()
    # GenerationParams/Config: record kwargs by returning a namespace-like mock.
    mod.GenerationParams = MagicMock(side_effect=lambda **kw: ("PARAMS", kw))
    mod.GenerationConfig = MagicMock(side_effect=lambda **kw: ("CONFIG", kw))
    # tensor: [channels, samples] float32 (real ACE-Step orientation)
    if tensor is None:
        tensor = np.random.rand(2, 48000).astype(np.float32) * 2 - 1
    t = MagicMock()
    t.detach.return_value = t; t.cpu.return_value = t; t.float.return_value = t
    t.numpy.return_value = tensor
    result = MagicMock()
    result.audios = [{"tensor": t, "sample_rate": 48000}]
    mod.generate_music = MagicMock(return_value=result)
    return tensor


def _runtime():
    return mod.ACEStepRuntime(model_id="ace-step-1.5", hf_repo="ACE-Step/Ace-Step1.5", device="cuda")


def test_dep_missing_raises():
    mod.AceStepHandler = None
    mod.torch = MagicMock()
    with pytest.raises(RuntimeError, match="ACE-Step"):
        mod.ACEStepRuntime(model_id="m", hf_repo="x", device="cuda")


def test_protocol_conformance():
    _wire(); assert isinstance(_runtime(), AudioGenerationModel)


def test_generate_returns_audiogenerationresult_48k_float32():
    _wire(np.random.rand(2, 24000).astype(np.float32) * 2 - 1)
    r = _runtime().generate("uplifting synthwave", lyrics="la la la", steps=8, seed=7)
    assert isinstance(r, AudioGenerationResult)
    assert r.sample_rate == 48000
    assert r.audio.dtype == np.float32
    assert r.audio.shape == (24000, 2)      # transposed (channels, samples) -> (samples, channels)
    assert r.channels == 2
    assert abs(r.duration_seconds - 0.5) < 1e-6


def test_generate_maps_caption_lyrics_steps_guidance_seed_to_params():
    _wire()
    _runtime().generate("a jazzy tune", lyrics="hello world", steps=12, guidance=5.0, seed=42)
    _, kw = mod.GenerationParams.call_args
    assert kw["task_type"] == "text2music"
    assert kw["caption"] == "a jazzy tune"
    assert kw["lyrics"] == "hello world"
    assert kw["inference_steps"] == 12
    assert kw["guidance_scale"] == 5.0
    assert kw["seed"] == 42


def test_generate_defaults_when_unset():
    _wire()
    _runtime().generate("ambient", )
    _, kw = mod.GenerationParams.call_args
    assert kw["lyrics"] == ""           # no lyrics -> empty string
    assert kw["inference_steps"] == 8   # turbo default
    assert kw["guidance_scale"] == 7.0
    assert kw["seed"] == -1             # no seed -> -1 (random)
    _, cfgkw = mod.GenerationConfig.call_args
    assert cfgkw["use_random_seed"] is True


def test_mono_output_stays_1d():
    _wire(np.random.rand(1, 12000).astype(np.float32) * 2 - 1)  # [1, samples]
    r = _runtime().generate("solo piano")
    assert r.audio.ndim == 1 and r.channels == 1 and r.audio.shape[0] == 12000
```

- [ ] **Step 2: Run to verify failure** -> `pytest tests/modalities/audio_generation/runtimes/test_acestep.py -v` FAILS (module missing).

- [ ] **Step 3: Implement the runtime** (mirrors `stable_audio.py` structure; the marked lines are B1-confirmable)

Create `src/muse/modalities/audio_generation/runtimes/acestep.py`:

```python
"""ACEStepRuntime: GPU music generation via the ACE-Step 1.5 SDK.

ACE-Step 1.5 is a two-stage system (NOT a single diffusers pipeline):
  - AceStepHandler (DIT/diffusion) + LLMHandler (Qwen3-based planner, vllm).
  - module-level generate_music(dit, llm, params, config) -> GenerationResult.
Installed from github.com/ace-step/ACE-Step-1.5 (not the ace-step PyPI v0.1.0).

API (docs/en/INFERENCE.md; verified at B1 on a GPU host - see the spec's
B1 findings note; CONFIRM the marked items below).

Deferred-imports sentinel pattern; tests patch the sentinels.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from muse.core.runtime_helpers import LoadTimer, select_device
from muse.modalities.audio_generation.protocol import AudioGenerationResult

logger = logging.getLogger(__name__)

torch: Any = None
AceStepHandler: Any = None
LLMHandler: Any = None
generate_music: Any = None
GenerationParams: Any = None
GenerationConfig: Any = None

# B1-confirmable: the DIT config + LM sub-model the Ace-Step1.5 repo bundles.
_DIT_CONFIG = "acestep-v15-turbo"
_LM_MODEL = "acestep-5Hz-lm-1.7B"
_LM_BACKEND = "vllm"            # B1: confirm vllm mandatory or transformers backend exists


def _ensure_deps() -> None:
    global torch, AceStepHandler, LLMHandler, generate_music, GenerationParams, GenerationConfig
    if torch is None:
        try:
            import torch as _t; torch = _t
        except Exception as e:  # noqa: BLE001
            logger.debug("ACEStepRuntime torch unavailable: %s", e)
    if AceStepHandler is None:
        try:
            from acestep.handler import AceStepHandler as _h
            from acestep.llm_inference import LLMHandler as _l
            from acestep.inference import (
                generate_music as _g, GenerationParams as _p, GenerationConfig as _c,
            )
            AceStepHandler, LLMHandler, generate_music, GenerationParams, GenerationConfig = _h, _l, _g, _p, _c
        except Exception as e:  # noqa: BLE001
            logger.debug("ACEStepRuntime acestep SDK unavailable: %s", e)


def _normalize_acestep_output(tensor: Any) -> tuple[np.ndarray, int]:
    """ACE-Step returns a torch tensor [channels, samples] (B1-confirm orientation).
    muse wants (samples,) mono or (samples, channels). Returns (audio, channels)."""
    if torch is not None and hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor)
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 1:
        return arr, 1
    if arr.ndim == 2:
        # [channels, samples] (channels small) -> transpose to (samples, channels)
        if arr.shape[0] <= 2 and arr.shape[1] > 2:
            ch = arr.shape[0]
            return (arr[0].copy(), 1) if ch == 1 else (arr.T.copy(), ch)
        return arr, arr.shape[1]
    raise ValueError(f"unsupported ACE-Step output shape: {arr.shape}")


class ACEStepRuntime:
    model_id: str

    def __init__(
        self, *, model_id: str, hf_repo: str, local_dir: str | None = None,
        device: str = "cuda", default_steps: int = 8, default_guidance: float = 7.0,
        default_sample_rate: int = 48000, **_: Any,
    ) -> None:
        _ensure_deps()
        if AceStepHandler is None or generate_music is None:
            raise RuntimeError(
                "ACE-Step SDK not available: install from "
                "git+https://github.com/ace-step/ACE-Step-1.5 (GPU + vllm). "
                f"Run `muse models refresh {model_id}`."
            )
        self.model_id = model_id
        self._device = select_device(device, torch_module=torch)
        self._default_steps = default_steps
        self._default_guidance = default_guidance
        self._default_sample_rate = default_sample_rate
        src = local_dir or hf_repo
        # B1-confirm: exact init arg names + how `src` maps to project_root /
        # checkpoint_dir / config_path / lm_model_path.
        with LoadTimer(f"loading ACE-Step from {src}", logger):
            self._dit = AceStepHandler()
            self._dit.initialize_service(project_root=src, config_path=_DIT_CONFIG, device=self._device)
            self._llm = LLMHandler()
            self._llm.initialize(checkpoint_dir=src, lm_model_path=_LM_MODEL,
                                 backend=_LM_BACKEND, device=self._device)

    def generate(
        self, prompt: str, *, duration: float | None = None, seed: int | None = None,
        steps: int | None = None, guidance: float | None = None,
        negative_prompt: str | None = None, **kwargs: Any,
    ) -> AudioGenerationResult:
        params = GenerationParams(
            task_type="text2music",
            caption=prompt,
            lyrics=kwargs.get("lyrics") or "",
            duration=float(duration) if duration is not None else -1.0,
            inference_steps=int(steps) if steps is not None else self._default_steps,
            guidance_scale=float(guidance) if guidance is not None else self._default_guidance,
            seed=int(seed) if seed is not None else -1,
        )
        config = GenerationConfig(batch_size=1, use_random_seed=(seed is None))
        result = generate_music(self._dit, self._llm, params, config, save_dir=None)
        entry = result.audios[0]
        audio, channels = _normalize_acestep_output(entry["tensor"])
        sr = int(entry.get("sample_rate", self._default_sample_rate))
        return AudioGenerationResult(
            audio=audio, sample_rate=sr, channels=channels,
            duration_seconds=audio.shape[0] / float(sr),
            metadata={"model": self.model_id, "caption": prompt,
                      "lyrics": bool(params.lyrics if not isinstance(params, tuple) else params[1]["lyrics"]),
                      "seed": seed},
        )
```
(NOTE: the `metadata["lyrics"]` line above handles the test's tuple-mock; in the real SDK `params` is a GenerationParams object, so simplify to `bool(kwargs.get("lyrics"))` once B1 confirms - keep it `bool(kwargs.get("lyrics"))` for clarity and update the test mock to match.)

- [ ] **Step 4: Run tests** -> all pass. Then full fast lane: `pytest tests/ -m "not slow" -q` (new runtime module imports cleanly via sentinels).
- [ ] **Step 5: Commit** -> `git commit -m "feat(audio/generation): ACEStepRuntime (ACE-Step 1.5 two-handler music gen)"`

(If B1 found a different tensor orientation, init arg names, or that vllm is optional, correct the runtime + the test mock to match before committing.)

---

### Task 3: Wire - optional lyrics field

**Files:**
- Modify: `src/muse/modalities/audio_generation/routes.py` (the `AudioGenerationRequest` model + the `_handle` kwargs dict)
- Test: `tests/modalities/audio_generation/test_routes.py`

- [ ] **Step 1: Failing test** - add to the audio_generation routes test:

```python
def test_music_forwards_lyrics_to_backend(<existing fixtures/client>):
    # register a fake backend with supports_music=True whose generate() records kwargs
    # POST /v1/audio/music {"prompt": "synthwave", "lyrics": "neon nights"}
    # assert backend.generate received lyrics="neon nights"
    ...
```
Use the existing audio_generation route-test harness (fake backend recording `generate` kwargs). Confirm it fails (lyrics not forwarded).

- [ ] **Step 2: Add the field + forward it**

In `routes.py`, add to `AudioGenerationRequest` (after `negative_prompt`):
```python
    lyrics: str | None = Field(default=None, max_length=8000)
```
In `_handle`, add `lyrics` to the kwargs dict passed to `model.generate(req.prompt, **kwargs)`:
```python
                "lyrics": req.lyrics,
```
(Backends that ignore `lyrics` are unaffected - `generate(**kwargs)` absorbs it; Stable Audio's `generate` has `**_`.)

- [ ] **Step 3: Run** -> route test passes; `pytest tests/modalities/audio_generation/ -q` green.
- [ ] **Step 4: Commit** -> `git commit -m "feat(audio/generation): optional lyrics field on /v1/audio/music"`

---

### Task 4: HF plugin dispatch + curated entry

**Files:**
- Modify: `src/muse/modalities/audio_generation/hf.py` (dispatch stable-audio vs ace-step)
- Modify: `src/muse/curated.yaml` (add `ace-step-1.5`)
- Test: `tests/modalities/audio_generation/` (plugin) + `tests/core/test_curated.py`

**Interfaces:**
- Consumes: the existing `_sniff`/`_resolve`/`HF_PLUGIN` in audio_generation/hf.py; `ACEStepRuntime` from Task 2.

- [ ] **Step 1: Failing tests**

Add a plugin-resolve test (mirror the existing audio_generation hf test): a fake `info` for `ACE-Step/Ace-Step1.5` (tag `text-to-audio`, name contains `ace-step`) resolves to `backend_path` ending `:ACEStepRuntime`, modality `audio/generation`, capabilities `supports_music=True`, `device=cuda`, `trust_remote_code=True`. And in `tests/core/test_curated.py`:
```python
def test_load_curated_includes_ace_step_1_5():
    by_id = {e.id: e for e in load_curated()}
    assert "ace-step-1.5" in by_id
    e = by_id["ace-step-1.5"]
    assert e.modality == "audio/generation"
    assert e.uri == "hf://ACE-Step/Ace-Step1.5"
    assert (e.capabilities or {}).get("device") == "cuda"
    assert (e.capabilities or {}).get("supports_music") is True
```

- [ ] **Step 2: Extend the plugin to dispatch by repo name**

In `audio_generation/hf.py`: add the ACE-Step runtime path + pip_extras + capabilities, widen `_sniff`, and branch `_resolve`:
```python
_ACESTEP_RUNTIME_PATH = "muse.modalities.audio_generation.runtimes.acestep:ACEStepRuntime"
_ACESTEP_PIP_EXTRAS = (
    "torch>=2.1.0",
    "ace-step @ git+https://github.com/ace-step/ACE-Step-1.5.git",  # B1: confirm install recipe
    "vllm",                                                          # B1: confirm mandatory
    "soundfile",
)

def _is_acestep(repo_id: str) -> bool:
    r = (repo_id or "").lower()
    return "ace-step" in r or "acestep" in r

def _acestep_capabilities() -> dict:
    return {
        "device": "cuda",
        "supports_music": True,
        "supports_sfx": False,
        "trust_remote_code": True,
        "default_steps": 8,
        "default_guidance": 7.0,
        "default_sample_rate": 48000,
        "memory_gb": 16.0,   # B1/probe: DIT turbo + LM 1.7B + VAE
    }
```
In `_sniff`, after the existing stable-audio check, also return True for `text-to-audio`-tagged repos where `_is_acestep(info.id)`. In `_resolve`, branch: if `_is_acestep(repo_id)` -> manifest with `backend_path=_ACESTEP_RUNTIME_PATH`, `pip_extras=_ACESTEP_PIP_EXTRAS`, `capabilities=_acestep_capabilities()`, description "ACE-Step 1.5: ...", and a `_download` that snapshots the whole `Ace-Step1.5` repo (no fp16-subfolder restriction - it is not a diffusers layout). Else keep the existing Stable Audio path. (Also extend `_search`'s post-filter to allow ace-step names, or leave search stable-audio-only and rely on the curated id - note which in the code comment.)

- [ ] **Step 3: Add the curated entry**

In `curated.yaml`, in the audio/generation section after `stable-audio-open-1.0`:
```yaml
- id: ace-step-1.5
  uri: hf://ACE-Step/Ace-Step1.5
  modality: audio/generation
  size_gb: 16.0          # B1/probe
  description: "ACE-Step 1.5: full song generation (vocals + lyrics + style), GPU-only, git-install SDK"
  capabilities:
    device: cuda
    supports_music: true
    supports_sfx: false
    trust_remote_code: true
    memory_gb: 16.0      # B1/probe
```

- [ ] **Step 4: Run** -> plugin + curated tests pass; `pytest tests/modalities/audio_generation/ tests/core/test_curated.py -q` green. Then verify resolution end to end:
```bash
python -c "
import muse.core.resolvers_hf as rhf
r = rhf.HFResolver()
m = r.resolve_via_modality('hf://ACE-Step/Ace-Step1.5','audio/generation')
print(m.manifest['modality'], m.backend_path)
assert m.backend_path.endswith(':ACEStepRuntime')
print('OK')
"
```
- [ ] **Step 5: Commit** -> `git commit -m "feat(audio/generation): ace-step plugin dispatch + curated ace-step-1.5"`

---

### Task 5: Docs + v0.48.0 release (GPU)

**Files:** `CLAUDE.md` (audio/generation line), `pyproject.toml` (version).

- [ ] **Step 1: Docs** - in CLAUDE.md's `audio/generation` bullet, note ACE-Step 1.5 as a curated GPU music engine alongside Stable Audio Open. ASCII only.
- [ ] **Step 2: Bump** -> `sed -i 's/^version = "0.47.0"/version = "0.48.0"/' pyproject.toml` (confirm current version first; bump from whatever HEAD is).
- [ ] **Step 3: Full fast lane** -> `pytest tests/ -m "not slow" -q` green.
- [ ] **Step 4: Build + wheel smoke + REAL GPU generate**
```bash
rm -rf dist/ build/ src/museq.egg-info src/muse.egg-info && python -m build && twine check dist/*
python -m venv /tmp/museq-v0480 && /tmp/museq-v0480/bin/pip install -q "dist/museq-0.48.0-py3-none-any.whl"
/tmp/museq-v0480/bin/python -c "
from muse import __version__; from muse.core import curated
print(__version__, 'ace-step-1.5' in [e.id for e in curated.load_curated()])"
# GOLD-STANDARD: in the GPU venv that has the acestep SDK installed, run a real generate
# through ACEStepRuntime and assert float32 (samples[,channels]) at 48000.
```
- [ ] **Step 5: Commit, tag, push, publish, GitHub release**
```bash
git add CLAUDE.md pyproject.toml && git commit -m "chore(release): v0.48.0 (ACE-Step 1.5 music runtime)"
twine upload dist/* && git tag -a v0.48.0 -m "v0.48.0: ACE-Step 1.5 music generation"
git push origin main && git push origin v0.48.0
gh release create v0.48.0 --title "v0.48.0: ACE-Step 1.5 music" --notes "<summary: curated GPU music runtime; full song gen with lyrics; two-handler ACE-Step 1.5 SDK; /v1/audio/music + optional lyrics field>"
```
- [ ] **Step 6: Verify** -> `curl -s https://pypi.org/pypi/museq/0.48.0/json | python -c "import json,sys;print(json.load(sys.stdin)['info']['version'])"` and `gh release view v0.48.0`.

---

## Self-Review

**Spec coverage:** two-handler runtime (Task 2), generate mapping + 48k transpose (Task 2 + tests), lyrics-only wire (Task 3), plugin dispatch + curated GPU entry (Task 4), B1-on-GPU (Task 1), docs + release + wheel smoke + real GPU generate (Task 5). text2music-only + supports_sfx false (Task 2/4 caps). Not-in-CI-smoke (Global Constraints + Task 4 omission). All covered.

**Placeholder scan:** the `B1` markers (`_DIT_CONFIG`/`_LM_MODEL`/`_LM_BACKEND` values, init arg mapping, tensor orientation, vllm necessity, size/memory) are produced by Task 1's exact commands and corrected in Tasks 2/4 - concrete starting code is given (from INFERENCE.md), not a TBD. The `metadata["lyrics"]` tuple-mock wart is called out with the real-SDK simplification. No lazy placeholders.

**Type consistency:** `ACEStepRuntime`, `generate(prompt, *, duration, seed, steps, guidance, negative_prompt, **kwargs) -> AudioGenerationResult`, `_normalize_acestep_output(tensor) -> (np.ndarray, int)`, sentinels `torch/AceStepHandler/LLMHandler/generate_music/GenerationParams/GenerationConfig`, runtime path `:ACEStepRuntime`, curated id `ace-step-1.5` - consistent across Tasks 2-5 and tests.
