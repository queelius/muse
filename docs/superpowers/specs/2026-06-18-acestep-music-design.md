# ACE-Step 1.5 music runtime - design (v0.48.0)

**Status:** approved (brainstorm), pending spec review -> writing-plans.
**Where it runs:** designed on a CPU host; the real-SDK B1 verification + build
+ test + release execute on the remote GPU host (see
`docs/superpowers/HANDOFF-2026-06-18-gpu-runtime-upgrades.md`).

**Goal:** Add `ACE-Step/Ace-Step1.5` (full song generation: vocals + lyrics +
style) as a curated GPU runtime in muse's existing `audio/generation` modality
(`POST /v1/audio/music`). Second runtime-upgrade build after Supertonic-3.

## Why a new runtime (and a scope correction)

ACE-Step 1.5 is NOT a single diffusers pipeline like the existing Stable Audio
runtime. It is a two-stage system with a functional API:

- `AceStepHandler` (DIT / diffusion synthesis) + `LLMHandler` (a Qwen3-based
  "planner" LM, default `backend="vllm"`). Both must be initialized.
- A module-level `generate_music(dit_handler, llm_handler, params, config)`
  drives inference (not a `pipeline.generate(...)` method).
- Installs from GitHub (`git clone ... && uv sync` or `pip install -e .`); the
  `ace-step` PyPI v0.1.0 is the OLD v1 package, NOT v1.5. Pulls vllm + multiple
  sub-models (DIT turbo/base, LM 0.6B/1.7B/4B, VAE, Qwen3-Embedding).

So effort/risk is closer to the standalone-SDK 3D runtimes (TRELLIS/Hunyuan3D)
than to Stable Audio: heavier git install, two-handler init, vllm. It still
fits the existing `audio/generation` protocol and wire.

## ACE-Step 1.5 API (researched from docs/en/INFERENCE.md; confirm exact args at B1)

```python
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import generate_music, GenerationParams, GenerationConfig

dit = AceStepHandler()
dit.initialize_service(project_root=..., config_path="acestep-v15-turbo", device="cuda")
llm = LLMHandler()
llm.initialize(checkpoint_dir=..., lm_model_path="acestep-5Hz-lm-1.7B", backend="vllm", device="cuda")

params = GenerationParams(
    task_type="text2music", caption=<style/tags>, lyrics=<lyrics>,
    duration=-1.0, inference_steps=8, guidance_scale=7.0, seed=-1,
)  # also: bpm, shift, thinking (defaults kept)
config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=True, seeds=None)
result = generate_music(dit, llm, params, config, save_dir=None)  # -> GenerationResult
# result.audios[0] = {"tensor": Tensor[channels, samples] float32 CPU,
#                     "sample_rate": 48000, "path": ..., "key": ..., "params": ...}
```

## Modality fit (existing)

`muse.modalities.audio_generation.protocol`:
- `AudioGenerationModel.generate(prompt, *, duration, seed, steps, guidance, negative_prompt, **kwargs) -> AudioGenerationResult`
- `AudioGenerationResult(audio: np.float32 (samples,) or (samples,channels), sample_rate, channels, duration_seconds, metadata)`
- Routes: `POST /v1/audio/music` (gated on `capabilities.supports_music`),
  `_handle` calls `model.generate(req.prompt, **kwargs)` with duration / seed /
  steps / guidance / negative_prompt.

## Architecture

### 1. Runtime `src/muse/modalities/audio_generation/runtimes/acestep.py`

`ACEStepRuntime` (satisfies `AudioGenerationModel`):
- Deferred-import sentinels for torch + `acestep` handlers + `generate_music` /
  `GenerationParams` / `GenerationConfig`; `_ensure_deps()`. `muse --help` /
  discovery work without the SDK; tests mock the sentinels.
- `__init__(*, model_id, hf_repo, local_dir=None, device="cuda", **_)`: init
  BOTH handlers. Map the `Ace-Step1.5` HF repo layout (the repo bundles
  `acestep-v15-turbo`, `acestep-5Hz-lm-1.7B`, `vae`, `Qwen3-Embedding-0.6B`) to
  `project_root` / `checkpoint_dir` / `config_path` / `lm_model_path` (exact
  mapping is a B1 item). Use `muse.core.runtime_helpers.select_device` /
  `LoadTimer`. cuda-only.
- `model_id` property -> the catalog id.
- `generate(prompt, *, duration=None, seed=None, steps=None, guidance=None, negative_prompt=None, **kwargs) -> AudioGenerationResult`:
  ```python
  params = GenerationParams(
      task_type="text2music",
      caption=prompt,
      lyrics=kwargs.get("lyrics", "") or "",
      duration=float(duration) if duration is not None else -1.0,
      inference_steps=int(steps) if steps is not None else 8,
      guidance_scale=float(guidance) if guidance is not None else 7.0,
      seed=int(seed) if seed is not None else -1,
  )
  config = GenerationConfig(batch_size=1, use_random_seed=(seed is None))
  result = generate_music(self._dit, self._llm, params, config, save_dir=None)
  a = result.audios[0]
  tensor = a["tensor"]            # [channels, samples], float32, CPU
  arr = tensor.numpy()
  audio = arr.T if arr.ndim == 2 else arr   # -> (samples, channels) or (samples,)
  sr = int(a.get("sample_rate", 48000))
  channels = audio.shape[1] if audio.ndim == 2 else 1
  return AudioGenerationResult(audio=audio.astype("float32"), sample_rate=sr,
      channels=channels, duration_seconds=audio.shape[0] / sr,
      metadata={"model_id": self.model_id, "caption": prompt,
                "lyrics": bool(params.lyrics), "seed": params.seed})
  ```
  (The exact tensor orientation/transpose is a B1 item; verify `[channels,
  samples]` vs `[samples, channels]` against the real output.)

### 2. Wire (small addition) - lyrics only

Add to `AudioGenerationRequest` (routes.py):
```python
lyrics: str | None = Field(default=None, max_length=8000)
```
and forward it in `_handle`'s kwargs dict (`"lyrics": req.lyrics`). `prompt`
stays the style/tags caption. No `bpm` for v1 (YAGNI; ACE-Step default). Only
`text2music`; editing / cover / audio2audio out of scope.

### 3. HF plugin dispatch (`audio_generation/hf.py`)

Sniff `text-to-audio`-tagged repos whose name matches `ace-step` / `acestep`
-> `ACEStepRuntime`, `trust_remote_code: true`, `supports_music: true`,
`supports_sfx: false`, `device: cuda`. (Confirm the existing audio_generation
hf.py plugin shape when implementing.)

### 4. Curated GPU entry (`curated.yaml`)

```yaml
- id: ace-step-1.5
  uri: hf://ACE-Step/Ace-Step1.5
  modality: audio/generation
  size_gb: <B1 / probe>          # multi-model; expect large
  description: "ACE-Step 1.5: full song generation (vocals + lyrics + style), GPU-only, git-install SDK"
  capabilities:
    device: cuda
    supports_music: true
    supports_sfx: false
    trust_remote_code: true
    memory_gb: <B1 / probe>      # DIT + LM(vllm) + VAE; expect 12-24GB+
```
pip_extras (synthesized by the plugin): `ace-step @
git+https://github.com/ace-step/ACE-Step-1.5.git`, plus vllm / torch / diffusers
/ accelerate as B1 confirms, with the "may need manual setup; GPU-only" caveat
the 3D SDK runtimes carry. NOT bundled; NOT in the CI smoke matrix.

## B1 verification (on the GPU box, before writing the runtime body)

1. Real install recipe (git URL; does `pip install git+...` work or is `uv sync`
   required; is vllm mandatory or is there a `transformers`/`hf` LM backend?).
2. Exact `initialize_service` / `initialize` arg names + how the downloaded
   `Ace-Step1.5` HF snapshot dir maps to project_root / checkpoint_dir /
   config_path / lm_model_path.
3. `GenerationParams` / `GenerationConfig` field names + defaults (confirm
   caption/lyrics/duration/inference_steps/guidance_scale/seed).
4. `result.audios[0]["tensor"]` orientation ([channels, samples] vs
   [samples, channels]) + dtype/range + `sample_rate` (expected 48000).
5. VRAM footprint (DIT turbo + LM 1.7B + VAE) for the `memory_gb` estimate.

Record findings in this spec's "B1 findings" note and the runtime docstring.

## Testing

- Unit (`tests/modalities/audio_generation/runtimes/test_acestep.py`): mock the
  `acestep` handlers + `generate_music` (return a fake GenerationResult with a
  `[channels, samples]` tensor at 48000); assert `generate` returns an
  `AudioGenerationResult` with float32 audio, sr 48000, the transpose applied,
  and that `caption`/`lyrics`/`inference_steps`/`guidance_scale`/`seed` are
  passed through to `GenerationParams`. Mirror the existing Stable Audio runtime
  tests.
- Route test: `POST /v1/audio/music` with a `lyrics` field forwards it to
  `generate`; music gating (`supports_music`) unchanged.
- Opt-in GPU integration test (real generate), gated like the other heavy
  models. NOT added to the free-tier CI smoke matrix.

## Release

v0.48.0 (on GPU): bump, full fast lane, build + wheel smoke-install (script /
plugin discoverable + curated loads), and a REAL generate on the GPU as
pre-release verification. twine upload, tag, push, GitHub release.

## Out of scope

- `bpm` and other GenerationParams knobs (defaults only).
- Editing / cover generation / audio2audio / understand_music.
- SFX via ACE-Step (`supports_sfx: false`).
- The non-turbo DIT variants and the 0.6B / 4B LMs (ship the turbo + 1.7B the
  HF repo bundles; others can be curated later).
- Wan2.2 image-to-video - the separate next sub-project.

## Open items resolved at B1 (GPU)

- Install recipe + vllm necessity; handler init arg mapping to the HF repo
  layout; GenerationParams/Config exact fields; tensor orientation + sample
  rate; VRAM / size_gb / memory_gb.
