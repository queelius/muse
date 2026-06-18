# Supertonic-3 TTS engine - design (v0.47.0)

**Status:** approved (brainstorm), pending spec review -> writing-plans.

**Goal:** Add `Supertone/supertonic-3` as a new bundled CPU text-to-speech
engine in muse's `audio/speech` modality: a lightweight (~99M), ONNX-based,
on-device, 31-language TTS. Re-ordered ahead of the GPU-gated TRELLIS.2 /
Wan2.2 sub-projects because it is fully CPU-shippable (build + verify + release
from a CPU host).

## Why a bundled script (not a resolver entry)

muse's `audio/speech` engines are per-engine bundled scripts under
`src/muse/models/` (kokoro_82m.py, soprano_80m.py, bark_small.py): each
declares a top-level `MANIFEST` dict plus a `Model` class satisfying the
`audio/speech` protocol. There is no generic TTS runtime the HF resolver can
synthesize a manifest for (each engine has a bespoke SDK), so a new TTS engine
is always a new bundled script. Supertonic is self-contained (its own
`supertonic` ONNX SDK), so it needs no `backends/base.py` helper; it mirrors
the standalone `kokoro_82m.py` shape.

## Supertonic SDK API (from the model-card README; confirmed at B1)

```python
from supertonic import TTS
tts = TTS(auto_download=True)              # downloads ONNX assets from HF on first run
style = tts.get_voice_style(voice_name="M1")
wav, duration = tts.synthesize(text, voice_style=style, lang="en")
tts.save_audio(wav, "output.wav")
```

- ONNX Runtime, on-device, ~99M params across public ONNX assets.
- 31 languages (codes: en, ko, ja, ar, bg, cs, da, de, el, es, et, fi, fr, hi,
  hr, hu, id, it, lt, lv, nl, pl, pt, ro, ru, sk, sl, sv, tr, uk, vi).
- Fixed preset voice styles selected by `voice_name` (e.g. "M1"); full list +
  default + sample rate confirmed at B1.
- License: OpenRAIL (open-weight; responsible-use terms; noted in MANIFEST).

## Protocol contract (existing)

`muse.modalities.audio_speech.protocol`:
- `AudioResult(audio: np.ndarray float32 [-1,1], sample_rate: int, metadata: dict)`
- `AudioChunk(audio: np.ndarray float32 [-1,1], sample_rate: int)`
- `TTSModel`: `model_id` (property), `sample_rate` (property),
  `synthesize(text, **kwargs) -> AudioResult`,
  `synthesize_stream(text, **kwargs) -> Iterator[AudioChunk]`. Unknown kwargs
  must be silently ignored. Audio is float32 in [-1,1]; the codec converts to
  int16 PCM downstream.

## Architecture

### 1. Bundled script `src/muse/models/supertonic_3.py`

Mirror `kokoro_82m.py`:

- Top-level `MANIFEST`:
  ```python
  MANIFEST = {
      "model_id": "supertonic-3",
      "modality": "audio/speech",
      "hf_repo": "Supertone/supertonic-3",
      "description": "Supertonic 3: lightweight on-device TTS, 31 languages, ONNX (CPU)",
      "license": "OpenRAIL",
      "pip_extras": ("supertonic", "onnxruntime", "numpy", "soundfile"),
      "system_packages": (),                 # ONNX: no espeak-ng needed
      "capabilities": {
          "sample_rate": <B1>,               # confirmed at B1
          "voices": [<B1 preset list>],      # confirmed at B1
          "languages": ["en","ko","ja","ar","bg","cs","da","de","el","es",
                        "et","fi","fr","hi","hr","hu","id","it","lt","lv",
                        "nl","pl","pt","ro","ru","sk","sl","sv","tr","uk","vi"],
          "device": "cpu",
          "memory_gb": 0.5,                  # ~99M ONNX; CPU working set
      },
  }
  ```
- `Model` class (named `Model` per discovery convention):
  - Deferred import: `from supertonic import TTS` inside `__init__`.
  - `__init__(*, hf_repo="Supertone/supertonic-3", local_dir=None, device="auto", **_)`:
    construct `self._tts = TTS(auto_download=True)`. B1 confirms whether
    Supertonic resolves weights from the HF cache that `muse pull` populates
    (like Kokoro's repo_id-from-cache behavior); if it needs an explicit
    assets dir, pass `local_dir`. CPU-only engine; `device` accepted, ONNX
    runs on CPU.
  - `model_id` property -> "supertonic-3"; `sample_rate` property -> the B1 value.
  - `voices` property -> the preset list (so routes/registry surface them).
  - `synthesize(text, **kwargs) -> AudioResult`:
    ```python
    voice = kwargs.get("voice", "<default-from-B1>")
    lang = kwargs.get("lang", "en")
    style = self._tts.get_voice_style(voice_name=voice)
    wav, _duration = self._tts.synthesize(text, voice_style=style, lang=lang)
    audio = np.asarray(wav, dtype=np.float32)   # coerce to float32 [-1,1]
    return AudioResult(audio=audio, sample_rate=self.sample_rate,
                       metadata={"voice": voice, "lang": lang})
    ```
  - `synthesize_stream(text, **kwargs) -> Iterator[AudioChunk]`: Supertonic has
    no native streaming; yield the full synthesized result as a single
    `AudioChunk` (the documented fallback for one-shot engines).

### 2. B1 verification (CPU, runs in this environment)

`pip install supertonic` in a scratch venv, synthesize one line, and confirm:
1. Actual output `sample_rate` (fills `capabilities.sample_rate` + the property).
2. Preset voice names and a sensible default (fills `capabilities.voices` +
   the `synthesize` default).
3. `synthesize` return: `(wav, duration)`; wav dtype/range (confirm float32 in
   [-1,1], or document the coercion/normalization needed).
4. `TTS(auto_download=True)` cache behavior: does it resolve from the HF cache
   `muse pull` populates (offline-safe load), or always phone home / use its
   own dir. If the latter, document the local-assets path handling.
5. `get_voice_style` / `synthesize` exact signatures and `lang` validation.

Record findings in this spec's "B1 findings" note and the script docstring.

### 3. Curated alias + CI smoke matrix

- Add a `supertonic-3` curated entry (`bundled: true`) in `curated.yaml`,
  mirroring `kokoro-82m` / `soprano-80m`.
- Add `supertonic-3` to `.github/workflows/fresh-venv-smoke.yml` (lightweight
  CPU ONNX; ideal free-tier matrix candidate).

### 4. Wire - unchanged

`POST /v1/audio/speech` already forwards request kwargs (e.g. `voice`, `speed`)
to the backend's `synthesize`. Supertonic adds `lang` as a forwarded kwarg; no
route/protocol/codec change.

## Testing

- **Unit** (`tests/models/test_supertonic_3.py`): mock the `supertonic.TTS`
  SDK (deferred-import pattern; pre-populate / patch). Mirror
  `tests/models/test_kokoro_82m.py`: `synthesize` returns an `AudioResult` with
  float32 audio + the right sample_rate + voice/lang metadata; `get_voice_style`
  + `synthesize` called with the right args; default voice/lang; unknown kwargs
  ignored; `synthesize_stream` yields one `AudioChunk`; MANIFEST shape.
- **Real CPU e2e (B1-backed):** since CPU, run one real `pip install` +
  synth in a scratch venv during implementation to confirm the mocks match
  reality (the discipline that caught the v0.43 Shap-E mock-vs-reality break).
- **Smoke matrix:** the new fresh-venv-smoke.yml entry exercises a real install
  + load-probe in CI.

## Release

v0.47.0 (TRELLIS.2 parked): bump, full fast lane, build + **wheel
smoke-install** (verify version + the bundled script is discoverable + curated
loads, per the v0.46.0 packaging lesson), `twine upload`, tag, push, GitHub
release. Because Supertonic is CPU-runnable, a real end-to-end synth is part of
the pre-release verification.

## Out of scope

- Zero-shot custom voice cloning / Voice Builder embeddings (preset voices only).
- Native streaming (Supertonic is one-shot; single-chunk fallback).
- The other runtime upgrades (TRELLIS.2, ACE-Step, Wan2.2) - separate sub-projects.
- A shared ONNX-TTS base class (YAGNI; Supertonic is the only ONNX TTS).

## Open items resolved at B1 (CPU)

- `sample_rate`, preset voice list + default, wav dtype/range coercion,
  HF-cache vs auto-download behavior, exact `get_voice_style`/`synthesize`
  signatures.
