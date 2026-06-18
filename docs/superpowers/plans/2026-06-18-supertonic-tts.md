# Supertonic-3 TTS Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Supertone/supertonic-3` as a bundled CPU text-to-speech engine in muse's `audio/speech` modality, shipped as v0.47.0.

**Architecture:** A standalone bundled model script `src/muse/models/supertonic_3.py` (MANIFEST + Model) mirroring `kokoro_82m.py`. Wraps the `supertonic` ONNX SDK (`TTS.get_voice_style` + `TTS.synthesize`). Returns float32 `[-1,1]` audio per the audio/speech protocol. Fully CPU; the whole plan (including a real synth) runs in the current environment.

**Tech Stack:** Python, the `supertonic` package (ONNX Runtime), numpy, pytest with the SDK mocked.

## Global Constraints

- **ASCII only in committed files.** A pre-commit soul-voice hook rejects em-dashes and other non-ASCII. Use `-`, `:`, `,`, `()`, `->`.
- **Audio is float32 in `[-1, 1]`** at the protocol boundary (the codec converts to int16 PCM downstream).
- **Deferred heavy imports.** `from supertonic import TTS` happens inside `Model.__init__`, never at module top (so `muse --help` / discovery work without the SDK installed). Tests mock the SDK.
- **Bundled-script discovery.** Dropping `supertonic_3.py` into `src/muse/models/` is auto-discovered; no edits to catalog.py/registry.py/server.py.
- **CPU-only engine.** ONNX Runtime; `device: cpu`. No `espeak-ng` system package (unlike Kokoro).
- **License: OpenRAIL** (open-weight; noted in MANIFEST `license`).
- **Wheel smoke-install before publishing** (the v0.46.0 packaging lesson): a fresh-venv `pip install` of the built wheel must show the new script discoverable.

---

### Task 1: B1 verification of the real supertonic SDK (CPU, runs in this environment)

**Files:**
- Reference only: record findings in `docs/superpowers/specs/2026-06-18-supertonic-tts-design.md` ("B1 findings" note).

This task is fully runnable here (CPU). Its deliverable is the concrete values that Task 2 hardcodes into the script: `sample_rate`, the preset voice names + default, the wav dtype/range, and the cache behavior.

- [ ] **Step 1: Install + synth in a scratch venv**

```bash
python -m venv /tmp/supertonic-b1 && . /tmp/supertonic-b1/bin/activate
pip install supertonic
python - <<'PY'
from supertonic import TTS
import numpy as np, inspect
tts = TTS(auto_download=True)
print("get_voice_style sig:", inspect.signature(tts.get_voice_style))
print("synthesize sig:", inspect.signature(tts.synthesize))
# enumerate preset voices if the SDK exposes a list/method:
for attr in ("voices", "list_voices", "available_voices", "voice_names"):
    if hasattr(tts, attr):
        print("voices via", attr, ":", getattr(tts, attr)() if callable(getattr(tts, attr)) else getattr(tts, attr))
style = tts.get_voice_style(voice_name="M1")
wav, duration = tts.synthesize("A gentle breeze moved through the open window.", voice_style=style, lang="en")
wav = np.asarray(wav)
print("wav dtype:", wav.dtype, "shape:", wav.shape, "min/max:", float(wav.min()), float(wav.max()))
print("duration:", duration)
# infer sample_rate from len(wav)/duration (round to a standard rate)
print("implied sample_rate:", round(len(wav) / float(duration)))
PY
deactivate
```

- [ ] **Step 2: Record the answers**

From the run, capture and record:
1. `sample_rate` (the implied rate, rounded to the standard value, e.g. 24000 / 44100).
2. The preset voice names + a sensible default (e.g. "M1"). If the SDK exposes no enumeration method, list the documented presets and note the source.
3. wav dtype + range (confirm `np.asarray(wav, dtype=np.float32)` yields `[-1,1]`; if the SDK returns int16 or a different range, record the normalization needed).
4. Whether `TTS(auto_download=True)` re-downloads each run or caches (and whether it honors the HF cache `muse pull` populates). Record any local-assets-dir option.
5. Exact `get_voice_style(voice_name=...)` and `synthesize(text, voice_style=, lang=)` signatures.

Append a dated "B1 findings" note to the spec and commit it:

```bash
git add docs/superpowers/specs/2026-06-18-supertonic-tts-design.md
git commit -m "docs(spec): record Supertonic-3 B1 SDK-verification findings"
```

---

### Task 2: Bundled script + unit tests

**Files:**
- Create: `src/muse/models/supertonic_3.py`
- Test: `tests/models/test_supertonic_3.py`

**Interfaces:**
- Consumes: `muse.modalities.audio_speech.AudioResult`, `AudioChunk` (re-exported from the package `__init__`), `TTSModel` protocol.
- Produces: module constants `SUPERTONIC_SAMPLE_RATE: int`, `SUPERTONIC_VOICES: list[str]`, `SUPERTONIC_LANGUAGES: list[str]`, `DEFAULT_VOICE: str`; `MANIFEST: dict`; `Model` with `model_id`/`sample_rate`/`voices` properties, `synthesize(text, **kwargs) -> AudioResult`, `synthesize_stream(text, **kwargs) -> Iterator[AudioChunk]`.

Use the B1 values (Task 1 Step 2) for `SUPERTONIC_SAMPLE_RATE`, `SUPERTONIC_VOICES`, and `DEFAULT_VOICE` below.

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_supertonic_3.py`:

```python
"""Tests for muse.models.supertonic_3: Supertonic ONNX TTS adapter (SDK mocked)."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from muse.modalities.audio_speech.protocol import AudioChunk, AudioResult, TTSModel


def _make_adapter():
    """Build a Model bypassing __init__, with a mocked supertonic TTS."""
    from muse.models.supertonic_3 import Model, SUPERTONIC_SAMPLE_RATE

    tts = MagicMock()
    tts.get_voice_style = MagicMock(return_value="STYLE_OBJ")
    # synthesize returns (wav, duration); wav float32 in [-1, 1].
    wav = (np.random.rand(SUPERTONIC_SAMPLE_RATE).astype(np.float32) * 2) - 1
    tts.synthesize = MagicMock(return_value=(wav, 1.0))

    adapter = object.__new__(Model)
    adapter._tts = tts
    adapter._device = "cpu"
    return adapter


def test_protocol_conformance():
    assert isinstance(_make_adapter(), TTSModel)


def test_model_id():
    assert _make_adapter().model_id == "supertonic-3"


def test_sample_rate_matches_constant():
    from muse.models.supertonic_3 import SUPERTONIC_SAMPLE_RATE
    assert _make_adapter().sample_rate == SUPERTONIC_SAMPLE_RATE


def test_synthesize_returns_float32_audio_result():
    from muse.models.supertonic_3 import SUPERTONIC_SAMPLE_RATE
    r = _make_adapter().synthesize("Hello world")
    assert isinstance(r, AudioResult)
    assert r.sample_rate == SUPERTONIC_SAMPLE_RATE
    assert r.audio.dtype == np.float32
    assert float(r.audio.max()) <= 1.0 and float(r.audio.min()) >= -1.0


def test_synthesize_uses_voice_style_and_lang():
    from muse.models.supertonic_3 import DEFAULT_VOICE
    a = _make_adapter()
    a.synthesize("Hi", voice="M2", lang="ko")
    a._tts.get_voice_style.assert_called_once_with(voice_name="M2")
    # synthesize called with the resolved style + lang
    _, kwargs = a._tts.synthesize.call_args
    assert kwargs.get("voice_style") == "STYLE_OBJ"
    assert kwargs.get("lang") == "ko"


def test_synthesize_defaults_voice_and_lang():
    from muse.models.supertonic_3 import DEFAULT_VOICE
    a = _make_adapter()
    a.synthesize("Hi")
    a._tts.get_voice_style.assert_called_once_with(voice_name=DEFAULT_VOICE)
    assert a._tts.synthesize.call_args.kwargs.get("lang") == "en"


def test_synthesize_ignores_unknown_kwargs():
    # protocol: unknown kwargs silently ignored
    _make_adapter().synthesize("Hi", temperature=0.9, nonsense=True)


def test_stream_yields_one_chunk():
    chunks = list(_make_adapter().synthesize_stream("Hello"))
    assert len(chunks) == 1
    assert isinstance(chunks[0], AudioChunk)


def test_voices_property_returns_list():
    from muse.models.supertonic_3 import Model, SUPERTONIC_VOICES
    a = object.__new__(Model)
    assert a.voices is Model.VOICES
    assert a.voices == SUPERTONIC_VOICES
    assert len(a.voices) > 0


def test_manifest_required_fields():
    from muse.models.supertonic_3 import MANIFEST
    assert MANIFEST["model_id"] == "supertonic-3"
    assert MANIFEST["modality"] == "audio/speech"
    assert MANIFEST["hf_repo"] == "Supertone/supertonic-3"
    assert "supertonic" in MANIFEST["pip_extras"]
    assert MANIFEST["capabilities"]["device"] == "cpu"
    assert len(MANIFEST["capabilities"]["languages"]) == 31
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/models/test_supertonic_3.py -v`
Expected: FAIL with `ModuleNotFoundError: muse.models.supertonic_3`.

- [ ] **Step 3: Implement the script**

Create `src/muse/models/supertonic_3.py` (replace the three `<B1>`-sourced literals with the Task 1 values; example defaults shown):

```python
"""Supertonic 3 TTS: lightweight on-device multilingual TTS, ONNX (CPU).

Wraps Supertone/supertonic-3 via the `supertonic` package to implement the
audio/speech TTSModel protocol. ~99M params, ONNX Runtime, 31 languages,
fixed preset voice styles. License: OpenRAIL.

SDK API (TTS.get_voice_style + TTS.synthesize), verified at B1:
  from supertonic import TTS
  tts = TTS(auto_download=True)
  style = tts.get_voice_style(voice_name="M1")
  wav, duration = tts.synthesize(text, voice_style=style, lang="en")
"""
from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np

from muse.modalities.audio_speech import AudioChunk, AudioResult

logger = logging.getLogger(__name__)

# Filled from B1 (Task 1 Step 2):
SUPERTONIC_SAMPLE_RATE = 44100          # <- replace with the B1 value
SUPERTONIC_VOICES = ["M1", "F1"]        # <- replace with the B1 preset list
DEFAULT_VOICE = "M1"                    # <- replace with the B1 default

SUPERTONIC_LANGUAGES = [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi",
    "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro",
    "ru", "sk", "sl", "sv", "tr", "uk", "vi",
]

MANIFEST = {
    "model_id": "supertonic-3",
    "modality": "audio/speech",
    "hf_repo": "Supertone/supertonic-3",
    "description": "Supertonic 3: lightweight on-device TTS, 31 languages, ONNX (CPU)",
    "license": "OpenRAIL",
    "pip_extras": ("supertonic", "onnxruntime", "numpy", "soundfile"),
    "system_packages": (),
    "capabilities": {
        "sample_rate": SUPERTONIC_SAMPLE_RATE,
        "voices": SUPERTONIC_VOICES,
        "languages": SUPERTONIC_LANGUAGES,
        "device": "cpu",
        "memory_gb": 0.5,
    },
}


class Model:
    """Supertonic 3 TTS backend. Named `Model` per discovery convention."""

    MODEL_ID = "supertonic-3"
    VOICES = SUPERTONIC_VOICES

    @property
    def voices(self) -> list[str]:
        return self.VOICES

    def __init__(
        self,
        *,
        hf_repo: str = "Supertone/supertonic-3",
        local_dir: str | None = None,
        device: str = "auto",
        **_: Any,
    ) -> None:
        from supertonic import TTS

        # ONNX CPU engine. auto_download resolves assets from the HF cache
        # that `muse pull` populates (B1-confirmed); local_dir is accepted
        # for catalog-loader compatibility.
        logger.info("Loading Supertonic 3 (device=cpu)")
        self._tts = TTS(auto_download=True)
        self._device = "cpu"

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    @property
    def sample_rate(self) -> int:
        return SUPERTONIC_SAMPLE_RATE

    def synthesize(self, text: str, **kwargs: Any) -> AudioResult:
        """Synthesize speech. kwargs: voice (preset name), lang (ISO code)."""
        voice = kwargs.get("voice", DEFAULT_VOICE)
        lang = kwargs.get("lang", "en")
        style = self._tts.get_voice_style(voice_name=voice)
        wav, _duration = self._tts.synthesize(text, voice_style=style, lang=lang)
        audio = np.asarray(wav, dtype=np.float32)
        return AudioResult(
            audio=audio,
            sample_rate=SUPERTONIC_SAMPLE_RATE,
            metadata={"voice": voice, "lang": lang},
        )

    def synthesize_stream(self, text: str, **kwargs: Any) -> Iterator[AudioChunk]:
        """Supertonic has no native streaming: yield the full result as one chunk."""
        result = self.synthesize(text, **kwargs)
        yield AudioChunk(audio=result.audio, sample_rate=SUPERTONIC_SAMPLE_RATE)
```

- [ ] **Step 4: Run to verify all tests pass**

Run: `pytest tests/models/test_supertonic_3.py -v`
Expected: PASS (all).

- [ ] **Step 5: Real CPU synth (mocks-match-reality check)**

In the scratch venv from Task 1 (which has `supertonic` installed), confirm the script's real path end-to-end:
```bash
. /tmp/supertonic-b1/bin/activate
python - <<'PY'
import sys; sys.path.insert(0, "src")
from muse.models.supertonic_3 import Model
m = Model()
r = m.synthesize("Muse now speaks with Supertonic.", voice=Model.VOICES[0], lang="en")
print("sample_rate:", r.sample_rate, "samples:", len(r.audio), "dtype:", r.audio.dtype)
assert r.audio.dtype.name == "float32"
assert -1.0 <= float(r.audio.min()) and float(r.audio.max()) <= 1.0
print("OK real synth")
PY
deactivate
```
Expected: prints sample_rate + sample count + `OK real synth`. If anything mismatches the mocks, fix the script (and the B1 note) before committing.

- [ ] **Step 6: Commit**

```bash
git add src/muse/models/supertonic_3.py tests/models/test_supertonic_3.py
git commit -m "feat(audio/speech): Supertonic-3 ONNX CPU TTS engine"
```

---

### Task 3: Curated alias + discovery test

**Files:**
- Modify: `src/muse/curated.yaml` (add `supertonic-3` bundled alias in the audio/speech TTS section, after `soprano-80m`)
- Test: `tests/core/test_curated.py`

**Interfaces:**
- Consumes: `muse.core.curated.load_curated()`, discovery of the bundled script from Task 2.

- [ ] **Step 1: Write the failing test**

Add to `tests/core/test_curated.py`:

```python
def test_load_curated_includes_supertonic_3():
    """v0.47.0: Supertonic-3 bundled TTS alias."""
    by_id = {e.id: e for e in load_curated()}
    assert "supertonic-3" in by_id
    assert by_id["supertonic-3"].bundled is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/core/test_curated.py::test_load_curated_includes_supertonic_3 -v`
Expected: FAIL (`supertonic-3` not in curated).

- [ ] **Step 3: Add the curated alias**

In `src/muse/curated.yaml`, in the `audio/speech (TTS)` section after the `soprano-80m` entry, add:

```yaml
- id: supertonic-3
  bundled: true
```

- [ ] **Step 4: Run the curated tests**

Run: `pytest tests/core/test_curated.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/muse/curated.yaml tests/core/test_curated.py
git commit -m "feat(audio/speech): curated supertonic-3 bundled alias"
```

---

### Task 4: CI fresh-venv smoke matrix entry

**Files:**
- Modify: `.github/workflows/fresh-venv-smoke.yml` (add `supertonic-3` to `matrix.model`)

- [ ] **Step 1: Add the matrix entry**

In `.github/workflows/fresh-venv-smoke.yml`, add `- supertonic-3` to the `matrix.model` list (after `- mert-v1-95m`). Supertonic needs no system package, so no `if: matrix.model == ...` apt step is required.

```yaml
        model:
          - kokoro-82m
          - dinov2-small
          - bart-large-cnn
          - bge-reranker-v2-m3
          - mert-v1-95m
          - supertonic-3
```

- [ ] **Step 2: Local repro of the smoke check (CPU)**

Run: `python scripts/smoke_fresh_venv.py --model_id supertonic-3 --json`
Expected: a JSON result indicating the fresh venv installed `muse[server]` + the script's pip_extras and the load-probe succeeded (e.g. `supertonic-3: OK`). If `scripts/smoke_fresh_venv.py` exists and supports this, capture its output; this is the same check the CI matrix runs.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/fresh-venv-smoke.yml
git commit -m "ci: add supertonic-3 to fresh-venv smoke matrix"
```

---

### Task 5: Docs + v0.47.0 release

**Files:**
- Modify: `CLAUDE.md` (audio/speech line), `README.md` (TTS models, if it lists them), `pyproject.toml` (version)

- [ ] **Step 1: Update docs**

In `CLAUDE.md`, the `audio/speech` bullet: add Supertonic-3 as a bundled engine (ONNX, on-device, 31 languages, CPU) alongside Soprano/Kokoro/Bark. If `README.md` enumerates TTS models, add it there too. ASCII only.

- [ ] **Step 2: Bump version**

```bash
sed -i 's/^version = "0.46.1"/version = "0.47.0"/' pyproject.toml
grep '^version' pyproject.toml   # expect: version = "0.47.0"
```

- [ ] **Step 3: Full fast lane**

Run: `pytest tests/ -m "not slow" -q`
Expected: PASS (all green).

- [ ] **Step 4: Build + twine check + wheel smoke-install (script discoverable)**

```bash
rm -rf dist/ build/ src/museq.egg-info src/muse.egg-info
python -m build
twine check dist/*
python -m venv /tmp/museq-v0470 && /tmp/museq-v0470/bin/pip install -q dist/museq-0.47.0-py3-none-any.whl
/tmp/museq-v0470/bin/python -c "
from muse import __version__
from muse.core import curated, catalog
print('version:', __version__)
print('supertonic-3 curated:', 'supertonic-3' in [e.id for e in curated.load_curated()])
print('supertonic-3 discovered:', 'supertonic-3' in catalog.known_models())
"
```
Expected: version `0.47.0`; `supertonic-3 curated: True`; `supertonic-3 discovered: True` (the bundled script ships in the wheel and is discoverable).

- [ ] **Step 5: Commit, tag, push, publish, GitHub release**

```bash
git add CLAUDE.md README.md pyproject.toml
git commit -m "chore(release): v0.47.0 (Supertonic-3 CPU TTS engine)"
twine upload dist/*
git tag -a v0.47.0 -m "v0.47.0: Supertonic-3 on-device multilingual TTS engine"
git push origin main && git push origin v0.47.0
gh release create v0.47.0 --title "v0.47.0: Supertonic-3 TTS" --notes "<summary: new bundled audio/speech engine; ONNX on-device, 31 languages, CPU; OpenRAIL; first of the runtime-upgrade builds (the GPU-gated TRELLIS.2/Wan2.2 follow on GPU hardware)>"
```

- [ ] **Step 6: Verify the release**

```bash
curl -s https://pypi.org/pypi/museq/0.47.0/json | python -c "import json,sys;print('PyPI:', json.load(sys.stdin)['info']['version'])"
gh release view v0.47.0 --json tagName,isDraft -q '"GH: \(.tagName) draft=\(.isDraft)"'
```
Expected: `PyPI: 0.47.0`, `GH: v0.47.0 draft=false`.

---

## Self-Review

**Spec coverage:** bundled script (Task 2), MANIFEST shape incl. OpenRAIL + 31 langs + pip_extras (Task 2 + test_manifest), Model synthesize/stream float32 (Task 2), B1 CPU verification of sample_rate/voices/dtype/cache (Task 1 + Task 2 Step 5 real synth), curated bundled alias (Task 3), CI smoke matrix (Task 4), wire-unchanged (no task; route already forwards kwargs), docs + release + wheel smoke (Task 5). All covered.

**Placeholder scan:** The three `<B1>`-sourced literals (`SUPERTONIC_SAMPLE_RATE`, `SUPERTONIC_VOICES`, `DEFAULT_VOICE`) are produced by Task 1's exact command and plugged in Task 2 Step 3; example defaults are shown so the code is runnable, with an explicit "replace with the B1 value" marker. Not lazy TBDs (the resolving command is concrete and CPU-runnable here). Everything else is complete code.

**Type consistency:** `Model`, `model_id`/`sample_rate`/`voices` properties, `synthesize(text, **kwargs) -> AudioResult`, `synthesize_stream -> Iterator[AudioChunk]`, constants `SUPERTONIC_SAMPLE_RATE`/`SUPERTONIC_VOICES`/`DEFAULT_VOICE`/`SUPERTONIC_LANGUAGES`, id `supertonic-3` - consistent across Tasks 2-5 and the tests. `voices` property returns `Model.VOICES` (matches the kokoro `voices`-lowercase convention the registry/routes expect).
