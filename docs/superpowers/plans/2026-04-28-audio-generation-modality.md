# `audio/generation` Modality Implementation Plan (#96)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `audio/generation` modality with two routes
(`/v1/audio/music` and `/v1/audio/sfx`) on one modality. Bundled
`stable-audio-open-1.0` (Stable Audio Open 1.0) and HF plugin sniffing
stable-audio repos. Generic `StableAudioRuntime` over
`diffusers.StableAudioPipeline`.

**Architecture:** Mirror existing modalities. New `audio_generation/`
package with protocol, codec, routes, two clients (Music + SFX),
`hf.py`, and `runtimes/stable_audio.py`. New bundled
`stable_audio_open_1_0.py`. Plugin priority 105 (specific to
stable-audio-shaped repos; loses to file-pattern plugins at 100, wins
over embedding/text at 110).

**Spec:** `docs/superpowers/specs/2026-04-28-audio-generation-modality-design.md`

**Target version:** v0.20.0

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/muse/modalities/audio_generation/__init__.py` | create | exports `MODALITY`, `build_router`, Protocol, Result, Clients, PROBE_DEFAULTS |
| `src/muse/modalities/audio_generation/protocol.py` | create | `AudioGenerationResult` dataclass, `AudioGenerationModel` Protocol |
| `src/muse/modalities/audio_generation/codec.py` | create | `encode_wav/mp3/opus/flac` + `UnsupportedFormatError` |
| `src/muse/modalities/audio_generation/routes.py` | create | `POST /v1/audio/music` + `POST /v1/audio/sfx`; capability gates; content-type |
| `src/muse/modalities/audio_generation/client.py` | create | `MusicClient` + `SFXClient` (HTTP) |
| `src/muse/modalities/audio_generation/runtimes/__init__.py` | create | empty marker |
| `src/muse/modalities/audio_generation/runtimes/stable_audio.py` | create | `StableAudioRuntime` generic runtime |
| `src/muse/modalities/audio_generation/hf.py` | create | HF plugin for Stable Audio Open shapes (priority 105) |
| `src/muse/models/stable_audio_open_1_0.py` | create | bundled script (stabilityai/stable-audio-open-1.0) |
| `src/muse/curated.yaml` | modify | +1 entry: `stable-audio-open-1.0` (bundled) |
| `pyproject.toml` | modify | bump 0.19.0 to 0.20.0 |
| `src/muse/__init__.py` | modify | docstring v0.20.0; add `audio/generation` to bundled modalities list |
| `CLAUDE.md` | modify | document new modality (two-routes-one-modality note) |
| `README.md` | modify | add `audio/generation` to route list + curl examples |
| `tests/modalities/audio_generation/` (full tree) | create | protocol, codec, routes, client, hf_plugin, runtime |
| `tests/models/test_stable_audio_open_1_0.py` | create | bundled-script tests |
| `tests/cli_impl/test_e2e_supervisor.py` | modify | (optional) extend slow e2e to cover the new modality |
| `tests/integration/test_remote_audio_generation.py` | create | opt-in integration tests |
| `tests/integration/conftest.py` | modify | `audio_generation_model` fixture |

---

## Task A: Protocol + Codec

Smallest, most isolated. No callers. Foundation for everything else.

**Files:**
- Create: `src/muse/modalities/audio_generation/__init__.py` (skeleton with re-exports; build_router stubbed in routes)
- Create: `src/muse/modalities/audio_generation/protocol.py`
- Create: `src/muse/modalities/audio_generation/codec.py`
- Create: `src/muse/modalities/audio_generation/routes.py` (stub returning empty APIRouter; replaced in Task C)
- Create: `tests/modalities/audio_generation/__init__.py` (empty)
- Create: `tests/modalities/audio_generation/test_protocol.py`
- Create: `tests/modalities/audio_generation/test_codec.py`

- [ ] **Step 1: Write failing protocol tests**

Cover: MODALITY tag is `"audio/generation"`; AudioGenerationResult dataclass shape; AudioGenerationModel structural protocol acceptance/rejection.

- [ ] **Step 2: Write failing codec tests**

Cover: wav round-trip (mono + stereo); wav header sample rate honored; flac via soundfile (skip-if-missing); mp3 raises UnsupportedFormatError when pydub absent (mocked); opus raises likewise; UnsupportedFormatError is an Exception subclass (mirrors v0.18.0 image_animation).

- [ ] **Step 3: Run, expect ImportError**

```bash
pytest tests/modalities/audio_generation/ -v
```

- [ ] **Step 4: Implement protocol**

`src/muse/modalities/audio_generation/protocol.py`: AudioGenerationResult dataclass (audio, sample_rate, channels, duration_seconds, metadata) + AudioGenerationModel Protocol (model_id property + generate method).

- [ ] **Step 5: Implement codec**

`src/muse/modalities/audio_generation/codec.py`:
- `UnsupportedFormatError` exception
- `encode_wav(audio, sample_rate, channels)` via stdlib `wave`
- `encode_flac(audio, sample_rate, channels)` via soundfile (lazy)
- `encode_mp3(audio, sample_rate, channels)` via WAV-then-pydub
- `encode_opus(audio, sample_rate, channels)` via WAV-then-pydub

Audio array shape: `(samples,)` for mono, `(samples, channels)` for multi-channel; codec normalizes for `wave` PCM packing.

- [ ] **Step 6: Stub routes + __init__**

`routes.py` exports `build_router(registry)` returning empty APIRouter (Task C replaces). `__init__.py` exports MODALITY, Protocol, Result, build_router, PROBE_DEFAULTS.

- [ ] **Step 7: Verify tests pass**

```bash
pytest tests/modalities/audio_generation/ -v
pytest tests/ -q -m "not slow"
```

**Commit:** `feat(audio-generation): protocol + codec for audio/generation modality (#96, A/J)`

---

## Task B: StableAudioRuntime

Generic runtime wrapping diffusers.StableAudioPipeline. Tests mock the pipeline.

**Files:**
- Create: `src/muse/modalities/audio_generation/runtimes/__init__.py` (empty)
- Create: `src/muse/modalities/audio_generation/runtimes/stable_audio.py`
- Create: `tests/modalities/audio_generation/runtimes/__init__.py` (empty)
- Create: `tests/modalities/audio_generation/runtimes/test_stable_audio.py`

- [ ] **Step 1: Write failing runtime tests**

Cover:
- Lazy imports: torch + StableAudioPipeline sentinels start None; `_ensure_deps()` populates them; tests patch sentinels directly.
- Construction: instantiates with model_id, hf_repo, capabilities-derived defaults; calls StableAudioPipeline.from_pretrained.
- generate(): returns AudioGenerationResult; honors capability defaults when call omits fields; max_duration clamping; seed passed through; channels detected from output array shape.
- Device selection: auto, cpu, cuda, mps; mirrors siblings.
- Audio array normalization: mono `(samples,)` and stereo `(samples, channels)` outputs both round-trip.

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement StableAudioRuntime**

Lazy-import pattern (sentinel pattern from sd_turbo and cross_encoder). torch_dtype dict per dtype. `_select_device` mirrors siblings.

generate() flow:
1. Compute effective duration, steps, guidance, seed (defaults from manifest capabilities).
2. Clamp duration to `[min_duration, max_duration]`.
3. Call `self._pipe(prompt=prompt, audio_end_in_s=duration, num_inference_steps=steps, guidance_scale=guidance, generator=gen, negative_prompt=negative_prompt)`.
4. Extract `out.audios[0]`; normalize from `(channels, samples)` to `(samples, channels)` (or keep `(samples,)` for mono).
5. Return AudioGenerationResult.

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/modalities/audio_generation/runtimes/ -v
pytest tests/ -q -m "not slow"
```

**Commit:** `feat(audio-generation): StableAudioRuntime over diffusers StableAudioPipeline (#96, B/J)`

---

## Task C: Routes (music + sfx) + modality __init__ wiring

Two routes on one modality, capability-gated. `__init__.py` upgraded with full PROBE_DEFAULTS.

**Files:**
- Modify: `src/muse/modalities/audio_generation/routes.py`
- Modify: `src/muse/modalities/audio_generation/__init__.py`
- Create: `tests/modalities/audio_generation/test_routes.py`

- [ ] **Step 1: Write failing route tests**

Cover:
- 200 on `/v1/audio/music` with happy-path body; returns audio bytes; content-type `audio/wav`.
- 200 on `/v1/audio/sfx` with happy-path body; returns audio bytes; content-type `audio/wav`.
- 400 on `/v1/audio/music` when model declares `supports_music: False`.
- 400 on `/v1/audio/sfx` when model declares `supports_sfx: False`.
- 200 when capability key missing entirely (defaults True).
- 404 (`{"error":...}` envelope) when `model` unknown.
- 400 on invalid `response_format` (Pydantic).
- 400 on missing/empty `prompt` (Pydantic).
- 400 on `duration` out of range (Pydantic).
- Content-type header per format: `wav -> audio/wav`, `flac -> audio/flac`, `mp3 -> audio/mpeg`, `opus -> audio/ogg`.
- 400 on unsupported response_format (UnsupportedFormatError mocked).
- Streaming negative: bytes returned in one Response (no SSE).

Use a `FakeAudioGenerationModel` that returns a deterministic 1-second sine wave mono numpy array.

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement routes**

```python
class AudioGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str | None = None
    duration: float | None = Field(default=None, ge=0.5, le=120.0)
    seed: int | None = None
    response_format: str = Field(default="wav", pattern="^(wav|mp3|opus|flac)$")
    steps: int | None = Field(default=None, ge=1, le=200)
    guidance: float | None = Field(default=None, ge=0.0, le=20.0)
    negative_prompt: str | None = None


def build_router(registry):
    router = APIRouter(prefix="/v1/audio", tags=["audio/generation"])

    @router.post("/music")
    async def music(req: AudioGenerationRequest):
        return await _handle(registry, req, kind="music")

    @router.post("/sfx")
    async def sfx(req: AudioGenerationRequest):
        return await _handle(registry, req, kind="sfx")

    return router


async def _handle(registry, req, *, kind):
    try:
        model = registry.get(MODALITY, req.model)
    except KeyError:
        raise ModelNotFoundError(model_id=req.model or "<default>", modality=MODALITY)

    manifest = registry.manifest(MODALITY, model.model_id) or {}
    capabilities = manifest.get("capabilities") or {}
    cap_key = "supports_music" if kind == "music" else "supports_sfx"
    if not capabilities.get(cap_key, True):
        return error_response(
            400, "invalid_parameter",
            f"model {model.model_id!r} does not support {kind} generation",
        )

    def _call():
        return model.generate(
            req.prompt,
            duration=req.duration,
            seed=req.seed,
            steps=req.steps,
            guidance=req.guidance,
            negative_prompt=req.negative_prompt,
        )

    result = await asyncio.to_thread(_call)
    try:
        body = _encode(req.response_format, result)
    except UnsupportedFormatError as e:
        return error_response(400, "invalid_parameter", str(e))
    return Response(content=body, media_type=_content_type(req.response_format))
```

`_content_type` returns mapping for {wav,flac,mp3,opus}.

- [ ] **Step 4: Update __init__.py**

Full re-exports: MODALITY, build_router, AudioGenerationModel, AudioGenerationResult, MusicClient, SFXClient (from client.py written in Task D; placeholder import is fine), PROBE_DEFAULTS.

- [ ] **Step 5: Verify tests pass**

```bash
pytest tests/modalities/audio_generation/ -v
pytest tests/ -q -m "not slow"
```

**Commit:** `feat(audio-generation): routes for /v1/audio/music + /v1/audio/sfx with capability gates (#96, C/J)`

---

## Task D: MusicClient + SFXClient

Two minimal HTTP clients. Same shape as other muse clients (server_url, MUSE_SERVER fallback, requests, raise_for_status).

**Files:**
- Create: `src/muse/modalities/audio_generation/client.py`
- Create: `tests/modalities/audio_generation/test_client.py`

- [ ] **Step 1: Write failing client tests**

Cover:
- `MusicClient` and `SFXClient` instantiate with no args (default server URL).
- `MUSE_SERVER` env honored.
- `generate()` POSTs to `/v1/audio/music` (Music) or `/v1/audio/sfx` (SFX) with the right body shape.
- Returns raw bytes (response.content).
- Optional kwargs (model, duration, seed, response_format, steps, guidance, negative_prompt) all passed through.
- raise_for_status called (4xx/5xx propagates).

Use `requests.post` mock.

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement clients**

```python
class _AudioGenerationClient:
    def __init__(self, server_url=None, *, timeout=300.0):
        self.server_url = (server_url or os.environ.get("MUSE_SERVER") or "http://localhost:8000").rstrip("/")
        self._timeout = timeout

    def _post(self, path, prompt, **kwargs):
        body = {"prompt": prompt}
        for k, v in kwargs.items():
            if v is not None:
                body[k] = v
        r = requests.post(f"{self.server_url}{path}", json=body, timeout=self._timeout)
        r.raise_for_status()
        return r.content


class MusicClient(_AudioGenerationClient):
    def generate(self, prompt, *, model=None, duration=None, seed=None,
                 response_format="wav", steps=None, guidance=None,
                 negative_prompt=None) -> bytes:
        return self._post("/v1/audio/music", prompt, model=model,
                          duration=duration, seed=seed,
                          response_format=response_format, steps=steps,
                          guidance=guidance, negative_prompt=negative_prompt)


class SFXClient(_AudioGenerationClient):
    def generate(self, prompt, *, model=None, duration=None, seed=None,
                 response_format="wav", steps=None, guidance=None,
                 negative_prompt=None) -> bytes:
        return self._post("/v1/audio/sfx", prompt, model=model,
                          duration=duration, seed=seed,
                          response_format=response_format, steps=steps,
                          guidance=guidance, negative_prompt=negative_prompt)
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/modalities/audio_generation/ -v
pytest tests/ -q -m "not slow"
```

**Commit:** `feat(audio-generation): MusicClient + SFXClient (#96, D/J)`

---

## Task E: Bundled stable_audio_open_1_0.py

Bundled script + tests. Mirrors sd_turbo.py structure.

**Files:**
- Create: `src/muse/models/stable_audio_open_1_0.py`
- Create: `tests/models/test_stable_audio_open_1_0.py`

- [ ] **Step 1: Write failing tests**

Cover:
- MANIFEST shape: model_id, modality (`audio/generation`), hf_repo (stabilityai/stable-audio-open-1.0), license (Apache 2.0), pip_extras, system_packages (ffmpeg), capabilities (supports_music, supports_sfx, default_duration etc.), allow_patterns.
- Model class construction: lazy imports + sentinels patched.
- generate() returns AudioGenerationResult; defaults applied; uses StableAudioPipeline mock.
- Backend protocol satisfaction (structural).

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement script**

Mirror sd_turbo.py:
- Sentinels: `torch`, `StableAudioPipeline = None` at module top.
- `_ensure_deps()` lazy-imports.
- MANIFEST as in spec.
- `_select_device(device)` helper.
- `Model` class with `model_id = MANIFEST["model_id"]`. Constructor accepts hf_repo/local_dir/device/dtype/**_ ; lazy-loads pipeline; calls `.to(device)` if not cpu.
- `generate(prompt, *, duration=None, seed=None, steps=None, guidance=None, negative_prompt=None, **_)` returns AudioGenerationResult.

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/models/test_stable_audio_open_1_0.py -v
pytest tests/ -q -m "not slow"
```

**Commit:** `feat(audio-generation): bundled stable-audio-open-1.0 script (#96, E/J)`

---

## Task F: HF plugin (priority 105, narrow sniff)

Plugin sniffs Stable Audio Open-shaped repos.

**Files:**
- Create: `src/muse/modalities/audio_generation/hf.py`
- Create: `tests/modalities/audio_generation/test_hf_plugin.py`

- [ ] **Step 1: Write failing tests**

Cover:
- HF_PLUGIN keys: modality, runtime_path, pip_extras, system_packages, priority (105), sniff, resolve, search.
- Sniff positive: text-to-audio tag + stable-audio in repo id + model_index.json.
- Sniff negative: missing tag; missing model_index.json; missing stable-audio in repo id.
- Resolve: synthesizes manifest with default capabilities (Stable Audio Open lineage).
- Search: filters by text-to-audio tag, returns SearchResult rows.

- [ ] **Step 2: Run, expect ImportError**

- [ ] **Step 3: Implement plugin**

Mirror text_rerank/hf.py structure. Single-file import (stdlib + huggingface_hub + muse.core.resolvers only).

```python
HF_PLUGIN = {
    "modality": "audio/generation",
    "runtime_path": "muse.modalities.audio_generation.runtimes.stable_audio:StableAudioRuntime",
    "pip_extras": (
        "torch>=2.1.0",
        "diffusers>=0.27.0",
        "transformers>=4.36.0",
        "accelerate",
        "soundfile",
    ),
    "system_packages": ("ffmpeg",),
    "priority": 105,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
```

`_resolve` synthesizes a manifest with capabilities for the Stable Audio Open lineage.

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/modalities/audio_generation/test_hf_plugin.py -v
pytest tests/ -q -m "not slow"
```

**Commit:** `feat(audio-generation): HF plugin sniffs Stable Audio Open repos at priority 105 (#96, F/J)`

---

## Task G: Curated entry

One curated entry: `stable-audio-open-1.0` (bundled).

**Files:**
- Modify: `src/muse/curated.yaml`
- Modify: `tests/core/test_curated.py`

- [ ] **Step 1: Add failing assertion in tests**

Locate existing test for curated entries; add assertion that `stable-audio-open-1.0` appears in the curated list with `bundled: true`.

- [ ] **Step 2: Add curated entry**

In `src/muse/curated.yaml`, append at the end (after image/animation block):

```yaml
# ---------- audio/generation (music + sfx) ----------

- id: stable-audio-open-1.0
  bundled: true
```

- [ ] **Step 3: Verify tests pass**

```bash
pytest tests/core/test_curated.py -v
pytest tests/ -q -m "not slow"
```

**Commit:** `feat(audio-generation): curated entry stable-audio-open-1.0 (#96, G/J)`

---

## Task H: Slow e2e + integration tests

Slow e2e via in-process FastAPI (no subprocess). Integration test against MUSE_REMOTE_SERVER.

**Files:**
- Create: `tests/modalities/audio_generation/test_e2e.py`
- Create: `tests/integration/test_remote_audio_generation.py`
- Modify: `tests/integration/conftest.py` (add fixture)

- [ ] **Step 1: Slow e2e test**

In-process FastAPI app with FakeAudioGenerationModel registered. Fire requests at both `/v1/audio/music` and `/v1/audio/sfx`. Assert WAV header bytes look right (RIFF... fmt chunk).

- [ ] **Step 2: Integration test**

Fixture: skip if MUSE_REMOTE_SERVER unset OR audio-generation model not loaded. Test:
- `test_protocol_audio_generation_music_returns_audio_bytes`: hits `/v1/audio/music`, asserts response is audio bytes.
- `test_protocol_audio_generation_sfx_returns_audio_bytes`: same on `/v1/audio/sfx`.
- `test_protocol_capability_gate_returns_400`: post a prompt with model that lacks the capability flag (use a synthetic registered FakeModel via env-set, or skip if none available).

`MUSE_AUDIO_GENERATION_MODEL_ID` env override (default `stable-audio-open-1.0`).

- [ ] **Step 3: Verify**

```bash
pytest tests/modalities/audio_generation/test_e2e.py -v
pytest tests/ -q -m "not slow"
# Optional, opt-in:
MUSE_REMOTE_SERVER=http://192.168.0.225:8000 pytest tests/integration/test_remote_audio_generation.py
```

**Commit:** `test(audio-generation): slow e2e + integration tests (#96, H/J)`

---

## Task I: Documentation

CLAUDE.md + README.md updates.

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md` (if exists)
- Modify: `src/muse/__init__.py` (docstring)

- [ ] **Step 1: Update CLAUDE.md**

Add `audio/generation` to the modality list. Document:
- New: muse's first modality with TWO routes on one MIME tag.
- Capability gates: `supports_music`, `supports_sfx`.
- Bundled: stable-audio-open-1.0 (Apache 2.0).

- [ ] **Step 2: Update README.md**

Add modality + routes to any modality table; curl examples for both
`/v1/audio/music` and `/v1/audio/sfx`.

- [ ] **Step 3: Update src/muse/__init__.py docstring**

Bump "As of v0.19.0" -> "As of v0.20.0". Add `audio/generation: /v1/audio/music, /v1/audio/sfx (Stable Audio Open 1.0)` to the bundled modalities list.

- [ ] **Step 4: Verify**

```bash
pytest tests/ -q -m "not slow"
```

**Commit:** `docs(audio-generation): document audio/generation modality + routes (#96, I/J)`

---

## Task J: v0.20.0 release

Bump version, update changelog notes, tag, push, create GitHub release.

**Files:**
- Modify: `pyproject.toml`
- Verify: all prior tasks committed

- [ ] **Step 1: Bump pyproject.toml**

`version = "0.19.0"` -> `version = "0.20.0"`.

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -q -m "not slow"
pytest tests/ -q  # includes the slow lane
```

- [ ] **Step 3: Commit + tag**

```bash
git add pyproject.toml
git commit -m "chore(release): v0.20.0"
git tag -a v0.20.0 -m "v0.20.0: audio/generation modality (music + sfx)"
git push origin main
git push origin v0.20.0
```

- [ ] **Step 4: GitHub release**

```bash
gh release create v0.20.0 --title "v0.20.0: audio/generation (music + sfx)" --notes "..."
```

Release notes call out:
- 9th modality: `audio/generation`.
- Two routes on one modality: `/v1/audio/music` and `/v1/audio/sfx`.
- First bundled model: Stable Audio Open 1.0 (Apache 2.0, ~3.4GB).
- Generic StableAudioRuntime over `diffusers.StableAudioPipeline`.
- HF plugin sniffs Stable Audio Open-shaped repos (priority 105).
- Capability-gated routing (`supports_music`, `supports_sfx`) so future
  music-only / sfx-only models stay isolatable.

**Commit:** `chore(release): v0.20.0`

---

## Self-review checklist

After all tasks complete:

- [ ] 9 modalities total: audio/speech, audio/transcription, **audio/generation NEW**, chat/completion, embedding/text, image/animation, image/generation, text/classification, text/rerank.
- [ ] 8 of 9 with HF plugin coverage (audio/speech is bundled-only).
- [ ] Both `/v1/audio/music` and `/v1/audio/sfx` mount and respond.
- [ ] Capability gate works on each route (returns 400 if unsupported).
- [ ] Codec: wav (always), flac (with soundfile), mp3/opus (with pydub+ffmpeg, otherwise 400).
- [ ] Stable Audio Open 1.0 bundled and curated.
- [ ] Plugin discovery returns 8 plugins.
- [ ] `/v1/models` lists `stable-audio-open-1.0` with capabilities.
- [ ] `muse models probe stable-audio-open-1.0` works.
- [ ] All tests pass.
- [ ] v0.20.0 tagged and pushed; GitHub release published.
- [ ] No em-dashes anywhere.
