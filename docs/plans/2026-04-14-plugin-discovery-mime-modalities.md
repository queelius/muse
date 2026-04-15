# Plugin Discovery + MIME-Style Modalities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure muse around two discovery-based plugin surfaces: (1) a flat `src/muse/models/` directory of drop-in model scripts (each a single file with `MANIFEST` + `Model` class), and (2) a `src/muse/modalities/` directory of self-declaring modality packages (each exports a `MODALITY` MIME-style tag + `build_router`). Eliminate the hardcoded `KNOWN_MODELS` dict, the hardcoded router imports in `worker.py`, and the `registry._extra()` allowlist. Rename all modality tags to MIME form (`audio.speech` → `audio/speech`, `embeddings` → `embedding/text`, `images.generations` → `image/generation`). Users can drop new model scripts into `~/.muse/models/` (or `$MUSE_MODELS_DIR`) to add backends without touching muse core; power users can add new modalities via `$MUSE_MODALITIES_DIR`.

**Architecture:** Discovery module (`muse.core.discovery`) scans three kinds of locations and produces two kinds of entries. Models: `src/muse/models/*.py` (bundled) → `~/.muse/models/*.py` (user) → `$MUSE_MODELS_DIR/*.py` (env). Each script defines `MANIFEST: dict` (metadata — model_id, modality, hf_repo, pip_extras, capabilities) and `Model: type` (class with the modality's expected methods). Modalities: `src/muse/modalities/*/` (bundled) → `$MUSE_MODALITIES_DIR/*/` (escape hatch). Each modality package exports `MODALITY: str` (MIME-style tag like `"audio/speech"`) and `build_router: Callable[[ModalityRegistry], APIRouter]`. On startup the supervisor calls `discover_modalities()` and `discover_models()`, catalog uses discovery in place of the old `KNOWN_MODELS` dict, worker mounts routers from discovery, registry stores manifests and passes them through to `/v1/models` without a hardcoded field allowlist.

**Tech Stack:** Python 3.10+, stdlib `importlib` + `pkgutil` for discovery (no new deps), existing FastAPI + uvicorn + httpx. MANIFEST is a plain dict, no schema framework. First-found-wins on collisions, log + skip on script import errors.

---

## File Structure (final)

```
src/muse/
├── core/
│   ├── discovery.py                 NEW  discover_models + discover_modalities
│   ├── registry.py                  MODIFIED  manifest-driven, no _extra allowlist
│   ├── catalog.py                   MODIFIED  KNOWN_MODELS becomes known_models()
│   ├── server.py                    unchanged
│   ├── venv.py                      unchanged
│   ├── install.py                   unchanged
│   └── errors.py                    unchanged
│
├── modalities/                      NEW  contains what was audio/speech, embeddings, images/generations
│   ├── __init__.py                  empty marker
│   ├── audio_speech/
│   │   ├── __init__.py              MODALITY = "audio/speech"; exports build_router + types
│   │   ├── protocol.py              AudioResult, AudioChunk, TTSModel
│   │   ├── routes.py                /v1/audio/speech router
│   │   ├── codec.py                 wav/opus encoding
│   │   └── client.py                SpeechClient
│   ├── embedding_text/
│   │   ├── __init__.py              MODALITY = "embedding/text"; exports build_router + types
│   │   ├── protocol.py              EmbeddingResult, EmbeddingsModel
│   │   ├── routes.py                /v1/embeddings router
│   │   ├── codec.py                 base64 float32 encoding
│   │   └── client.py                EmbeddingsClient
│   └── image_generation/
│       ├── __init__.py              MODALITY = "image/generation"; exports build_router + types
│       ├── protocol.py              ImageResult, ImageModel
│       ├── routes.py                /v1/images/generations router
│       ├── codec.py                 PNG/JPEG/base64 encoding
│       └── client.py                GenerationsClient
│
├── models/                          NEW  flat dir; each file = one model
│   ├── __init__.py                  empty marker
│   ├── soprano_80m.py               MANIFEST + Model (was audio/speech/backends/soprano.py + tts.py)
│   ├── kokoro_82m.py                MANIFEST + Model (was audio/speech/backends/kokoro.py)
│   ├── bark_small.py                MANIFEST + Model (was audio/speech/backends/bark.py)
│   ├── sd_turbo.py                  MANIFEST + Model (was images/generations/backends/sd_turbo.py)
│   ├── all_minilm_l6_v2.py          MANIFEST + Model (was embeddings/backends/minilm.py)
│   ├── qwen3_embedding_0_6b.py      MANIFEST + Model (was embeddings/backends/qwen3_embedding.py)
│   └── nv_embed_v2.py               MANIFEST + Model (was embeddings/backends/nv_embed_v2.py)
│
├── audio/                           DELETED
├── embeddings/                      DELETED
├── images/                          DELETED
│
└── cli_impl/
    ├── worker.py                    MODIFIED  mounts discovered routers, no hardcoded imports
    ├── serve.py                     unchanged
    ├── supervisor.py                unchanged
    └── gateway.py                   unchanged

tests/
├── core/
│   ├── test_discovery.py            NEW
│   ├── test_registry.py             MODIFIED  manifest-driven tests
│   └── test_catalog.py              MODIFIED  uses discovery
├── modalities/                      NEW  was tests/audio/speech, tests/embeddings, tests/images/generations
│   ├── audio_speech/
│   ├── embedding_text/
│   └── image_generation/
├── models/                          NEW  one test file per built-in model
│   ├── test_soprano_80m.py
│   ├── test_kokoro_82m.py
│   ├── test_bark_small.py
│   ├── test_sd_turbo.py
│   ├── test_all_minilm_l6_v2.py
│   ├── test_qwen3_embedding_0_6b.py
│   └── test_nv_embed_v2.py
└── cli_impl/
    └── test_worker.py               MODIFIED  assertions match new mount logic

docs/
└── MODEL_SCRIPTS.md                 NEW  how to write a muse model script
```

---

## Key design decisions (locked in)

1. **Model class is ALWAYS named `Model`.** Convention over configuration; discovery looks up `Model` by name. Users who want readable imports elsewhere can alias: `from muse.models.kokoro_82m import Model as KokoroModel`.

2. **MANIFEST is a plain dict, not a dataclass or Pydantic model.** Required keys: `model_id`, `modality`, `hf_repo`. Optional-recommended: `description`, `license`, `pip_extras`, `system_packages`, `capabilities`. Unknown keys pass through harmlessly. No central schema; validation happens on use (`KeyError` at first access).

3. **`capabilities` is a free-form sub-dict.** Conventions per modality emerge (embedders document `dimensions` / `matryoshka` / `context_length`; TTS documents `sample_rate` / `voices` / `languages`; image gen documents `default_size`). Documentation per-modality lists recommended keys as non-binding guidance.

4. **Discovery errors are warn-and-skip, not fatal.** One broken script shouldn't refuse server startup. Error messages include the source path so users can debug. An optional `MUSE_STRICT_DISCOVERY=1` env var could promote warnings to errors later.

5. **First-found-wins on collision.** Scan order: bundled → user dir → env override. A user script with a model_id matching a bundled one wins (and gets a warning). Two user scripts with the same model_id: first encountered wins (and gets a warning).

6. **Modality tags are MIME-style, Python package names are underscored MIME.** `"audio/speech"` as tag, `muse.modalities.audio_speech` as package. The slash can't go in a Python identifier; underscored is the natural mapping. URLs (`/v1/audio/speech`) stay where OpenAI put them.

7. **Modalities are discoverable but primary extension surface is models.** User-dir scan is models-only by default. Modality scan of `$MUSE_MODALITIES_DIR` exists as an escape hatch but isn't advertised as the path of least resistance. Rationale: writing a modality means writing a wire contract; that's architectural, not plug-in.

8. **Each commit leaves the tree green.** Migration is per-modality (Phase A) and per-modality-family (Phase C). No commit introduces an intermediate broken state.

---

## Task graph

```
A1 → A2 → A3 → A4        move + MIME rename per modality
B1 → B2                  discovery module
C1 → C2 → C3 → C4        model scripts per modality family + cleanup
D1 → D2                  catalog + registry use discovery
E1                       worker auto-mounts
F1 → F2                  user-dir + env-var scanning
G1 → G2 → G3             docs + verify + merge
```

18 tasks total. Phases A-C can proceed in parallel across modalities (each an independent subagent if desired); D-G are sequential.

---

## Part A — Layout refactor + MIME rename

### Task A1: Move audio.speech → modalities/audio_speech, rename tag to "audio/speech"

**Files:**
- Create: `src/muse/modalities/__init__.py` (empty marker)
- Move: all of `src/muse/audio/speech/*` → `src/muse/modalities/audio_speech/*`
- Move: `tests/audio/speech/*` → `tests/modalities/audio_speech/*`
- Modify: `src/muse/modalities/audio_speech/__init__.py` — add `MODALITY = "audio/speech"` constant, re-export types
- Modify: every import referencing the old path
- Modify: all `"audio.speech"` modality-tag strings → `"audio/speech"`

This is a substantial mechanical change. Do it with care; rely on sed + grep verification.

- [ ] **Step 1: Create worktree**

```bash
cd /home/spinoza/github/repos/muse
git worktree add ../muse-plugin-discovery -b feat/plugin-discovery-mime
cd ../muse-plugin-discovery
```

- [ ] **Step 2: Create new modality package structure**

```bash
mkdir -p src/muse/modalities/audio_speech
touch src/muse/modalities/__init__.py
```

- [ ] **Step 3: Move Python package files**

```bash
git mv src/muse/audio/speech/__init__.py       src/muse/modalities/audio_speech/__init__.py
git mv src/muse/audio/speech/protocol.py       src/muse/modalities/audio_speech/protocol.py
git mv src/muse/audio/speech/routes.py         src/muse/modalities/audio_speech/routes.py
git mv src/muse/audio/speech/codec.py          src/muse/modalities/audio_speech/codec.py
git mv src/muse/audio/speech/client.py         src/muse/modalities/audio_speech/client.py
git mv src/muse/audio/speech/backends          src/muse/modalities/audio_speech/backends
```

If there are other files under `src/muse/audio/speech/` that I haven't listed (e.g., `tts.py`, `alignment.py`, `encoded.py`, `decode_only.py`, `vocos/`, `utils/`), also move them:

```bash
git mv src/muse/audio/speech/tts.py           src/muse/modalities/audio_speech/tts.py
git mv src/muse/audio/speech/alignment.py     src/muse/modalities/audio_speech/alignment.py
git mv src/muse/audio/speech/encoded.py       src/muse/modalities/audio_speech/encoded.py
git mv src/muse/audio/speech/decode_only.py   src/muse/modalities/audio_speech/decode_only.py
git mv src/muse/audio/speech/vocos            src/muse/modalities/audio_speech/vocos
git mv src/muse/audio/speech/utils            src/muse/modalities/audio_speech/utils
```

Verify the old dir is empty or doesn't exist:

```bash
ls -la src/muse/audio/speech/ 2>&1 || echo "gone"
```

- [ ] **Step 4: Move test files**

```bash
mkdir -p tests/modalities/audio_speech
touch tests/modalities/__init__.py
touch tests/modalities/audio_speech/__init__.py

# Move every test file under the old path
for f in tests/audio/speech/*.py; do
  [[ "$(basename $f)" == "__init__.py" ]] && continue
  git mv "$f" "tests/modalities/audio_speech/$(basename $f)"
done

# Delete old empty test subdir
rmdir tests/audio/speech 2>&1 || echo "still has files"
rmdir tests/audio 2>&1 || echo "still has files"
```

- [ ] **Step 5: Rewrite imports in all moved source files**

```bash
find src/muse/modalities/audio_speech -name "*.py" -exec sed -i \
  -e 's|from muse\.audio\.speech|from muse.modalities.audio_speech|g' \
  -e 's|import muse\.audio\.speech|import muse.modalities.audio_speech|g' \
  {} +
```

- [ ] **Step 6: Rewrite imports in all test files for this modality**

```bash
find tests/modalities/audio_speech -name "*.py" -exec sed -i \
  -e 's|from muse\.audio\.speech|from muse.modalities.audio_speech|g' \
  -e 's|import muse\.audio\.speech|import muse.modalities.audio_speech|g' \
  {} +
```

- [ ] **Step 7: Rewrite imports anywhere ELSE in the codebase that references the old path**

```bash
grep -rn "muse\.audio\.speech" src/ tests/ --include="*.py"
```

Likely hits:
- `src/muse/cli_impl/worker.py` (hardcoded modality import)
- `src/muse/core/catalog.py` (backend_path strings)
- Possibly `src/muse/cli.py` or tests that reference backends by path

For each file found, sed-replace `muse.audio.speech` → `muse.modalities.audio_speech`. Verify afterwards:

```bash
! grep -rn "muse\.audio\.speech" src/ tests/ --include="*.py"
```

(The `!` inverts — we want NO hits.)

- [ ] **Step 8: Update modality-tag strings from "audio.speech" to "audio/speech"**

```bash
grep -rn '"audio\.speech"' src/ tests/ --include="*.py"
```

Hits will include:
- Catalog entries (`modality="audio.speech"`)
- Registry usages (`registry.register("audio.speech", model)`)
- Router tests and worker tests (`routers["audio.speech"] = ...`)
- Test fixtures (`reg.register("audio.speech", FakeTTS())`)

Rewrite each hit:

```bash
find src/ tests/ -name "*.py" -exec sed -i \
  's|"audio\.speech"|"audio/speech"|g' \
  {} +
```

Also check single-quoted variants:

```bash
find src/ tests/ -name "*.py" -exec sed -i \
  "s|'audio\.speech'|'audio/speech'|g" \
  {} +
```

Verify:

```bash
! grep -rn '"audio\.speech"\|'"'"'audio\.speech'"'"'' src/ tests/ --include="*.py"
```

- [ ] **Step 9: Update `src/muse/modalities/audio_speech/__init__.py` to export MODALITY constant**

File content (replace whatever's there currently with this):

```python
"""Audio speech modality: text-to-speech.

Wire contract: POST /v1/audio/speech with {input, model, voice?, speed?,
response_format? ('wav' | 'opus'), stream?} returns WAV/Opus audio bytes
(or SSE-streamed base64 PCM chunks when stream=True).

Models declaring `modality = "audio/speech"` in their MANIFEST and
satisfying the TTSModel protocol plug into this modality.
"""
from muse.modalities.audio_speech.client import SpeechClient
from muse.modalities.audio_speech.protocol import (
    AudioChunk,
    AudioResult,
    TTSModel,
)
from muse.modalities.audio_speech.routes import build_router

MODALITY = "audio/speech"

__all__ = [
    "MODALITY",
    "build_router",
    "SpeechClient",
    "AudioChunk",
    "AudioResult",
    "TTSModel",
]
```

- [ ] **Step 10: Install and run tests**

```bash
pip install -e ".[dev,server]"
pytest tests/ -q -m "not slow" 2>&1 | tail -5
```

Expected: all tests that previously passed still pass. The count should be unchanged from pre-refactor (no tests added or removed in this task, only moved + import-rewritten).

If tests fail because of residual old imports or modality strings, fix them. Only proceed to commit once green.

- [ ] **Step 11: Cleanup empty old parent dirs**

```bash
rmdir src/muse/audio/speech 2>&1 || echo "not empty"
rmdir src/muse/audio 2>&1 || echo "not empty"
```

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "refactor(audio.speech): move to muse/modalities/audio_speech + MIME tag

Python package: muse.audio.speech → muse.modalities.audio_speech
Modality tag: 'audio.speech' → 'audio/speech'

Package __init__ now exports MODALITY constant so discovery can pick
it up in a later task (still mounted via hardcoded import in
worker.py; auto-mount lands in Task E1).

Mechanical sed: all imports and tag strings rewritten. No behavior
change. Old muse/audio/ tree deleted."
```

**Acceptance:**
- `grep -r "muse\.audio\.speech" src/ tests/` returns no hits
- `grep -r '"audio\.speech"' src/ tests/` returns no hits
- `src/muse/modalities/audio_speech/__init__.py` exports `MODALITY = "audio/speech"`
- All tests pass
- Old `src/muse/audio/` directory deleted
- One commit

---

### Task A2: Move embeddings → modalities/embedding_text, rename tag to "embedding/text"

**Files:**
- Move: `src/muse/embeddings/*` → `src/muse/modalities/embedding_text/*`
- Move: `tests/embeddings/*` → `tests/modalities/embedding_text/*`
- Modify: `src/muse/modalities/embedding_text/__init__.py` — add MODALITY constant
- Modify: all imports and modality-tag strings

Same pattern as A1. Full steps:

- [ ] **Step 1: Move Python package files**

```bash
cd /home/spinoza/github/repos/muse-plugin-discovery

mkdir -p src/muse/modalities/embedding_text

git mv src/muse/embeddings/__init__.py   src/muse/modalities/embedding_text/__init__.py
git mv src/muse/embeddings/protocol.py   src/muse/modalities/embedding_text/protocol.py
git mv src/muse/embeddings/routes.py     src/muse/modalities/embedding_text/routes.py
git mv src/muse/embeddings/codec.py      src/muse/modalities/embedding_text/codec.py
git mv src/muse/embeddings/client.py     src/muse/modalities/embedding_text/client.py
git mv src/muse/embeddings/backends      src/muse/modalities/embedding_text/backends
```

- [ ] **Step 2: Move test files**

```bash
mkdir -p tests/modalities/embedding_text
touch tests/modalities/embedding_text/__init__.py

for f in tests/embeddings/*.py; do
  [[ "$(basename $f)" == "__init__.py" ]] && continue
  git mv "$f" "tests/modalities/embedding_text/$(basename $f)"
done

rmdir tests/embeddings 2>&1 || true
```

- [ ] **Step 3: Rewrite imports across the moved tree**

```bash
find src/muse/modalities/embedding_text tests/modalities/embedding_text -name "*.py" -exec sed -i \
  -e 's|from muse\.embeddings|from muse.modalities.embedding_text|g' \
  -e 's|import muse\.embeddings|import muse.modalities.embedding_text|g' \
  {} +
```

- [ ] **Step 4: Rewrite imports in the rest of the codebase**

```bash
grep -rn "muse\.embeddings" src/ tests/ --include="*.py"
```

For each file shown (likely `worker.py`, `catalog.py`, maybe others):

```bash
find src/ tests/ -name "*.py" -exec sed -i \
  -e 's|from muse\.embeddings|from muse.modalities.embedding_text|g' \
  -e 's|import muse\.embeddings|import muse.modalities.embedding_text|g' \
  {} +
```

Verify:

```bash
! grep -rn "muse\.embeddings" src/ tests/ --include="*.py"
```

- [ ] **Step 5: Rename modality-tag strings "embeddings" → "embedding/text"**

This one needs care: the bare word "embeddings" appears in comments, docstrings, etc. — we only want to replace it when it's used as a modality key string.

Find exact hits for modality-tag usage:

```bash
grep -rn '"embeddings"' src/ tests/ --include="*.py"
```

Review each hit. They'll be in:
- Catalog entries (`modality="embeddings"`)
- Registry register calls (`registry.register("embeddings", ...)`)
- Worker router mount (`routers["embeddings"] = ...`)
- Router prefix or MODALITY constant in routes.py
- Test fixtures (`reg.register("embeddings", ...)`)
- Gateway test route_paths assertions

For each file, sed-replace `"embeddings"` → `"embedding/text"` — but ONLY in contexts where it's a modality tag. Safe blanket replace works in this codebase because we don't use `"embeddings"` as a string literal for any other purpose. Run:

```bash
find src/ tests/ -name "*.py" -exec sed -i \
  -e 's|"embeddings"|"embedding/text"|g' \
  {} +
find src/ tests/ -name "*.py" -exec sed -i \
  -e "s|'embeddings'|'embedding/text'|g" \
  {} +
```

Verify by inspection (grep should show ONLY MIME form now):

```bash
grep -rn '"embedding/text"\|"embeddings"' src/ tests/ --include="*.py" | head
```

Then check the worker-route-tree test assertion — it checks `/v1/embeddings` (URL, keep as-is) vs `"embedding/text"` (modality, rename). Verify this distinction is preserved.

- [ ] **Step 6: Update `src/muse/modalities/embedding_text/__init__.py`**

```python
"""Embedding text modality: text-to-vector.

Wire contract: POST /v1/embeddings with {input (str | list[str]), model,
encoding_format? ('float' | 'base64'), dimensions?} returns list of
embedding vectors in OpenAI-compatible shape.

Models declaring `modality = "embedding/text"` in their MANIFEST and
satisfying the EmbeddingsModel protocol plug into this modality.
"""
from muse.modalities.embedding_text.client import EmbeddingsClient
from muse.modalities.embedding_text.protocol import (
    EmbeddingResult,
    EmbeddingsModel,
)
from muse.modalities.embedding_text.routes import build_router

MODALITY = "embedding/text"

__all__ = [
    "MODALITY",
    "build_router",
    "EmbeddingsClient",
    "EmbeddingResult",
    "EmbeddingsModel",
]
```

- [ ] **Step 7: Run tests**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -5
```

Expected: unchanged count, all pass.

- [ ] **Step 8: Cleanup empty old parent dir**

```bash
rmdir src/muse/embeddings 2>&1 || echo "not empty; investigate"
```

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor(embeddings): move to muse/modalities/embedding_text + MIME tag

Python package: muse.embeddings → muse.modalities.embedding_text
Modality tag: 'embeddings' → 'embedding/text'

The tag goes from plural-noun to singular-type-space form matching
the other modalities (audio/speech, image/generation). 'embedding/text'
makes room for embedding/image (CLIP-style) and embedding/multimodal
as future siblings.

Package __init__ now exports MODALITY constant."
```

---

### Task A3: Move images.generations → modalities/image_generation, rename tag to "image/generation"

Same pattern as A1/A2.

- [ ] **Step 1: Move Python package files**

```bash
cd /home/spinoza/github/repos/muse-plugin-discovery
mkdir -p src/muse/modalities/image_generation

git mv src/muse/images/generations/__init__.py  src/muse/modalities/image_generation/__init__.py
git mv src/muse/images/generations/protocol.py  src/muse/modalities/image_generation/protocol.py
git mv src/muse/images/generations/routes.py    src/muse/modalities/image_generation/routes.py
git mv src/muse/images/generations/codec.py     src/muse/modalities/image_generation/codec.py
git mv src/muse/images/generations/client.py    src/muse/modalities/image_generation/client.py
git mv src/muse/images/generations/backends     src/muse/modalities/image_generation/backends
```

- [ ] **Step 2: Move test files**

```bash
mkdir -p tests/modalities/image_generation
touch tests/modalities/image_generation/__init__.py

for f in tests/images/generations/*.py; do
  [[ "$(basename $f)" == "__init__.py" ]] && continue
  git mv "$f" "tests/modalities/image_generation/$(basename $f)"
done

rmdir tests/images/generations 2>&1 || true
rmdir tests/images 2>&1 || true
```

- [ ] **Step 3: Rewrite imports**

```bash
find src/muse/modalities/image_generation tests/modalities/image_generation -name "*.py" -exec sed -i \
  -e 's|from muse\.images\.generations|from muse.modalities.image_generation|g' \
  -e 's|import muse\.images\.generations|import muse.modalities.image_generation|g' \
  {} +

find src/ tests/ -name "*.py" -exec sed -i \
  -e 's|from muse\.images\.generations|from muse.modalities.image_generation|g' \
  -e 's|import muse\.images\.generations|import muse.modalities.image_generation|g' \
  {} +

! grep -rn "muse\.images\.generations" src/ tests/ --include="*.py"
```

- [ ] **Step 4: Rename modality-tag strings "images.generations" → "image/generation"**

```bash
find src/ tests/ -name "*.py" -exec sed -i \
  -e 's|"images\.generations"|"image/generation"|g' \
  {} +
find src/ tests/ -name "*.py" -exec sed -i \
  -e "s|'images\.generations'|'image/generation'|g" \
  {} +

! grep -rn '"images\.generations"\|'"'"'images\.generations'"'"'' src/ tests/ --include="*.py"
```

- [ ] **Step 5: Update `src/muse/modalities/image_generation/__init__.py`**

```python
"""Image generation modality: text-to-image.

Wire contract: POST /v1/images/generations with {prompt, model, n?, size?,
response_format? ('b64_json' | 'url'), negative_prompt?, steps?,
guidance?, seed?} returns list of generated images in OpenAI-compatible
shape (b64_json bytes or data URL).

Models declaring `modality = "image/generation"` in their MANIFEST and
satisfying the ImageModel protocol plug into this modality.
"""
from muse.modalities.image_generation.client import GenerationsClient
from muse.modalities.image_generation.protocol import ImageModel, ImageResult
from muse.modalities.image_generation.routes import build_router

MODALITY = "image/generation"

__all__ = [
    "MODALITY",
    "build_router",
    "GenerationsClient",
    "ImageResult",
    "ImageModel",
]
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -5
```

- [ ] **Step 7: Cleanup**

```bash
rmdir src/muse/images/generations 2>&1 || true
rmdir src/muse/images 2>&1 || true
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(images.generations): move to muse/modalities/image_generation + MIME tag

Python package: muse.images.generations → muse.modalities.image_generation
Modality tag: 'images.generations' → 'image/generation'

Pluralization normalized to singular (image, not images; generation,
not generations) to match MIME convention. Package __init__ exports
MODALITY constant."
```

---

### Task A4: Verify Phase A cleanup

**Files:** no changes, just verification.

- [ ] **Step 1: Confirm old directories are gone**

```bash
cd /home/spinoza/github/repos/muse-plugin-discovery

ls src/muse/audio 2>&1 && echo "STILL THERE" || echo "gone (good)"
ls src/muse/embeddings 2>&1 && echo "STILL THERE" || echo "gone (good)"
ls src/muse/images 2>&1 && echo "STILL THERE" || echo "gone (good)"

ls tests/audio 2>&1 && echo "STILL THERE" || echo "gone (good)"
ls tests/embeddings 2>&1 && echo "STILL THERE" || echo "gone (good)"
ls tests/images 2>&1 && echo "STILL THERE" || echo "gone (good)"
```

All should print `gone (good)`.

- [ ] **Step 2: Grep the codebase for stragglers**

```bash
grep -rn 'muse\.audio\.\|muse\.embeddings\|muse\.images\.\|"audio\.speech"\|"embeddings"\|"images\.generations"' src/ tests/ --include="*.py"
```

Should produce no output.

- [ ] **Step 3: Full test suite**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -5
```

Expected: all pass. Count should be unchanged from pre-refactor baseline (+/- 0).

- [ ] **Step 4: No commit (verification only)**

If step 2 or 3 surfaced issues, back up and fix in the appropriate earlier task (A1/A2/A3). The point of A4 is to catch any drift before Phase B builds on top.

---

## Part B — Discovery module

### Task B1: `muse.core.discovery` with `discover_models` and `discover_modalities`

**Files:**
- Create: `src/muse/core/discovery.py`
- Create: `tests/core/test_discovery.py`

TDD required.

- [ ] **Step 1: Write failing tests**

File: `tests/core/test_discovery.py`

```python
"""Tests for muse.core.discovery.

Discovery scans directories of .py files (models) or subpackages
(modalities) and extracts MANIFEST + Model class (models) or
MODALITY tag + build_router (modalities). Errors during discovery
are logged and skipped; discovery never raises.
"""
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from muse.core.discovery import (
    DiscoveredModel,
    discover_models,
    discover_modalities,
)


def _write_model_script(tmp_path: Path, filename: str, content: str) -> Path:
    """Helper: write a .py file with given content to tmp_path."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content).lstrip())
    return p


def _write_modality_package(tmp_path: Path, name: str, content: str) -> Path:
    """Helper: write a subpackage (__init__.py only) under tmp_path/name/."""
    pkg = tmp_path / name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(textwrap.dedent(content).lstrip())
    return pkg


# ---------- Model discovery ----------

class TestDiscoverModels:
    def test_empty_directory_yields_no_models(self, tmp_path):
        result = discover_models([tmp_path])
        assert result == {}

    def test_script_with_manifest_and_model_class_is_discovered(self, tmp_path):
        _write_model_script(tmp_path, "fake_model.py", """
            MANIFEST = {
                "model_id": "fake-model",
                "modality": "audio/speech",
                "hf_repo": "fake/repo",
            }
            class Model:
                model_id = "fake-model"
        """)
        result = discover_models([tmp_path])
        assert "fake-model" in result
        entry = result["fake-model"]
        assert isinstance(entry, DiscoveredModel)
        assert entry.manifest["model_id"] == "fake-model"
        assert entry.manifest["modality"] == "audio/speech"
        assert entry.model_class.__name__ == "Model"
        assert entry.source_path == tmp_path / "fake_model.py"

    def test_script_without_manifest_is_skipped(self, tmp_path, caplog):
        _write_model_script(tmp_path, "noisy.py", """
            class Model: ...
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([tmp_path])
        assert result == {}
        assert "MANIFEST" in caplog.text or "noisy" in caplog.text

    def test_script_without_model_class_is_skipped(self, tmp_path, caplog):
        _write_model_script(tmp_path, "manifest_only.py", """
            MANIFEST = {
                "model_id": "half-model",
                "modality": "audio/speech",
                "hf_repo": "x/y",
            }
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([tmp_path])
        assert result == {}
        assert "Model" in caplog.text or "half-model" in caplog.text or "manifest_only" in caplog.text

    def test_script_with_import_error_is_skipped(self, tmp_path, caplog):
        _write_model_script(tmp_path, "broken.py", """
            import definitely_not_a_real_module_xyz
            MANIFEST = {"model_id": "x", "modality": "y", "hf_repo": "z"}
            class Model: ...
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([tmp_path])
        assert result == {}
        # Discovery must not raise; just log
        assert "broken" in caplog.text or "ImportError" in caplog.text or "definitely_not" in caplog.text

    def test_files_starting_with_underscore_are_ignored(self, tmp_path):
        _write_model_script(tmp_path, "_private.py", """
            MANIFEST = {"model_id": "p", "modality": "m", "hf_repo": "r"}
            class Model: ...
        """)
        _write_model_script(tmp_path, "__init__.py", "")
        result = discover_models([tmp_path])
        assert result == {}

    def test_manifest_missing_required_fields_is_skipped(self, tmp_path, caplog):
        _write_model_script(tmp_path, "bad_manifest.py", """
            # Missing hf_repo
            MANIFEST = {"model_id": "x", "modality": "audio/speech"}
            class Model: ...
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([tmp_path])
        assert result == {}
        assert "hf_repo" in caplog.text or "required" in caplog.text.lower()

    def test_multiple_directories_scanned_in_order(self, tmp_path):
        d1 = tmp_path / "d1"
        d2 = tmp_path / "d2"
        d1.mkdir()
        d2.mkdir()
        _write_model_script(d1, "model_a.py", """
            MANIFEST = {"model_id": "a", "modality": "m", "hf_repo": "r"}
            class Model: ...
        """)
        _write_model_script(d2, "model_b.py", """
            MANIFEST = {"model_id": "b", "modality": "m", "hf_repo": "r"}
            class Model: ...
        """)
        result = discover_models([d1, d2])
        assert {"a", "b"} == set(result.keys())

    def test_first_found_wins_on_model_id_collision(self, tmp_path, caplog):
        d1 = tmp_path / "bundled"
        d2 = tmp_path / "user"
        d1.mkdir()
        d2.mkdir()
        _write_model_script(d1, "m.py", """
            MANIFEST = {"model_id": "collide", "modality": "m", "hf_repo": "bundled-repo"}
            class Model: ...
        """)
        _write_model_script(d2, "m.py", """
            MANIFEST = {"model_id": "collide", "modality": "m", "hf_repo": "user-repo"}
            class Model: ...
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([d1, d2])
        assert len(result) == 1
        # First dir (bundled) wins
        assert result["collide"].manifest["hf_repo"] == "bundled-repo"
        # Collision is logged
        assert "collide" in caplog.text

    def test_nonexistent_directory_is_silently_skipped(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        result = discover_models([missing])
        assert result == {}


# ---------- Modality discovery ----------

class TestDiscoverModalities:
    def test_empty_directory_yields_no_modalities(self, tmp_path):
        result = discover_modalities([tmp_path])
        assert result == {}

    def test_subpackage_with_MODALITY_and_build_router_is_discovered(self, tmp_path):
        _write_modality_package(tmp_path, "fake_modality", """
            MODALITY = "fake/type"
            def build_router(registry):
                from fastapi import APIRouter
                return APIRouter()
        """)
        result = discover_modalities([tmp_path])
        assert "fake/type" in result
        build_fn = result["fake/type"]
        assert callable(build_fn)

    def test_subpackage_without_MODALITY_is_skipped(self, tmp_path, caplog):
        _write_modality_package(tmp_path, "no_tag", """
            def build_router(registry):
                return None
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_modalities([tmp_path])
        assert result == {}
        assert "MODALITY" in caplog.text or "no_tag" in caplog.text

    def test_subpackage_without_build_router_is_skipped(self, tmp_path, caplog):
        _write_modality_package(tmp_path, "no_router", """
            MODALITY = "x/y"
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_modalities([tmp_path])
        assert result == {}
        assert "build_router" in caplog.text or "no_router" in caplog.text

    def test_plain_py_files_are_not_treated_as_modalities(self, tmp_path):
        (tmp_path / "not_a_package.py").write_text(
            'MODALITY = "wrong/form"\ndef build_router(r): pass\n'
        )
        result = discover_modalities([tmp_path])
        assert result == {}

    def test_first_found_wins_on_modality_tag_collision(self, tmp_path, caplog):
        d1 = tmp_path / "bundled"
        d2 = tmp_path / "escape"
        d1.mkdir()
        d2.mkdir()
        _write_modality_package(d1, "my_mod", """
            MODALITY = "collide/tag"
            def build_router(r): return ("bundled",)
        """)
        _write_modality_package(d2, "my_mod", """
            MODALITY = "collide/tag"
            def build_router(r): return ("escape",)
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_modalities([d1, d2])
        assert len(result) == 1
        assert result["collide/tag"](None) == ("bundled",)
        assert "collide/tag" in caplog.text
```

- [ ] **Step 2: Run — verify all fail**

```bash
cd /home/spinoza/github/repos/muse-plugin-discovery
pytest tests/core/test_discovery.py -v
```

Expected: `ModuleNotFoundError: No module named 'muse.core.discovery'`.

- [ ] **Step 3: Implement `src/muse/core/discovery.py`**

```python
"""Plugin discovery for models and modalities.

Models: each `.py` file in a scanned directory that defines a top-level
MANIFEST dict (with keys model_id, modality, hf_repo) and a top-level
Model class. Files starting with `_` (including `__init__.py`) are
skipped. Scripts failing to import (missing deps, syntax errors, etc.)
are logged and skipped — discovery never raises.

Modalities: each subdirectory under a scanned root that defines a
package (`__init__.py`) exporting a module-level MODALITY string
(MIME-style tag) and a build_router callable. Same error handling:
bad modality packages get logged and skipped.

Scan order is caller-defined (a list of directories). First-found-wins
on model_id or MODALITY tag collisions; subsequent duplicates produce
a warning log.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger(__name__)


REQUIRED_MANIFEST_KEYS = ("model_id", "modality", "hf_repo")


@dataclass
class DiscoveredModel:
    """A model that discovery found in one of the scanned dirs.

    - manifest: the MANIFEST dict the script defined
    - model_class: the class named `Model` exported by the script
    - source_path: filesystem path to the script (for error messages)
    """
    manifest: dict
    model_class: type
    source_path: Path


def discover_models(dirs: list[Path]) -> dict[str, DiscoveredModel]:
    """Scan dirs in order; return {model_id: DiscoveredModel}.

    First-found-wins on model_id collision; warns on duplicates.
    Nonexistent dirs are silently skipped. Script errors are logged.
    """
    found: dict[str, DiscoveredModel] = {}
    for d in dirs:
        if not d or not d.is_dir():
            continue
        for py_file in sorted(d.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                module = _load_script(py_file)
            except Exception as e:
                logger.warning(
                    "skipping model script %s: import failed (%s)",
                    py_file, e,
                )
                continue

            manifest = getattr(module, "MANIFEST", None)
            if not isinstance(manifest, dict):
                logger.warning(
                    "skipping model script %s: no top-level MANIFEST dict",
                    py_file,
                )
                continue

            missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
            if missing:
                logger.warning(
                    "skipping model script %s: MANIFEST missing required keys %s",
                    py_file, missing,
                )
                continue

            model_class = getattr(module, "Model", None)
            if not isinstance(model_class, type):
                logger.warning(
                    "skipping model script %s: no top-level Model class",
                    py_file,
                )
                continue

            model_id = manifest["model_id"]
            if model_id in found:
                existing = found[model_id].source_path
                logger.warning(
                    "model_id %r already discovered at %s; keeping that one, "
                    "skipping duplicate at %s",
                    model_id, existing, py_file,
                )
                continue

            found[model_id] = DiscoveredModel(
                manifest=manifest,
                model_class=model_class,
                source_path=py_file,
            )
    return found


def discover_modalities(dirs: list[Path]) -> dict[str, Callable]:
    """Scan dirs in order for modality subpackages.

    A modality is a directory with an __init__.py that exports
    module-level MODALITY (str, MIME-style tag) and build_router (callable).
    First-found-wins on MODALITY tag collision. Returns {tag: build_router}.
    """
    found: dict[str, Callable] = {}
    tag_sources: dict[str, Path] = {}

    for d in dirs:
        if not d or not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            if not sub.is_dir() or sub.name.startswith("_"):
                continue
            init_py = sub / "__init__.py"
            if not init_py.exists():
                continue
            try:
                module = _load_package(sub)
            except Exception as e:
                logger.warning(
                    "skipping modality package %s: import failed (%s)",
                    sub, e,
                )
                continue

            tag = getattr(module, "MODALITY", None)
            if not isinstance(tag, str) or not tag:
                logger.warning(
                    "skipping modality package %s: no MODALITY string",
                    sub,
                )
                continue

            build_router = getattr(module, "build_router", None)
            if not callable(build_router):
                logger.warning(
                    "skipping modality package %s: no callable build_router",
                    sub,
                )
                continue

            if tag in found:
                existing = tag_sources[tag]
                logger.warning(
                    "MODALITY tag %r already discovered at %s; keeping that one, "
                    "skipping duplicate at %s",
                    tag, existing, sub,
                )
                continue

            found[tag] = build_router
            tag_sources[tag] = sub
    return found


def _load_script(path: Path) -> Any:
    """Import a single-file .py module given its filesystem path."""
    # Use a mangled module name so multiple scripts with the same name
    # in different dirs don't collide in sys.modules
    mod_name = f"_muse_discover_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _load_package(pkg_dir: Path) -> Any:
    """Import a package (directory with __init__.py) from its path."""
    mod_name = f"_muse_discover_mod_{pkg_dir.parent.name}_{pkg_dir.name}"
    init_py = pkg_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        init_py,
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {pkg_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
```

- [ ] **Step 4: Run — pass**

```bash
pytest tests/core/test_discovery.py -v
```

Expected: all tests pass (17 in total across TestDiscoverModels + TestDiscoverModalities).

- [ ] **Step 5: Full regression**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -3
```

Expected: all pre-existing tests plus 17 new = baseline + 17.

- [ ] **Step 6: Commit**

```bash
git add src/muse/core/discovery.py tests/core/test_discovery.py
git commit -m "feat(core): add discovery module for models + modalities

discover_models(dirs) scans .py files, extracts MANIFEST + Model class.
discover_modalities(dirs) scans subpackages, extracts MODALITY tag +
build_router.

First-found-wins on collisions (warn). Script import errors logged
and skipped — discovery never raises. Nonexistent dirs silently
skipped. Files starting with underscore ignored (covers __init__.py
and user convention).

Used by catalog + worker in subsequent tasks (D1, E1)."
```

---

### Task B2: Harden discovery error handling

Look again at edge cases not yet covered and add tests + fixes.

**Files:**
- Modify: `src/muse/core/discovery.py` (minor hardening)
- Modify: `tests/core/test_discovery.py` (new tests)

- [ ] **Step 1: Append edge-case tests to `tests/core/test_discovery.py`**

```python
class TestDiscoveryEdgeCases:
    def test_non_dict_MANIFEST_is_skipped(self, tmp_path, caplog):
        _write_model_script(tmp_path, "bad.py", """
            MANIFEST = "not a dict"
            class Model: ...
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([tmp_path])
        assert result == {}
        assert "MANIFEST" in caplog.text

    def test_Model_as_instance_not_class_is_skipped(self, tmp_path, caplog):
        _write_model_script(tmp_path, "bad.py", """
            MANIFEST = {"model_id": "x", "modality": "m", "hf_repo": "r"}
            Model = object()  # instance, not a class
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_models([tmp_path])
        assert result == {}

    def test_non_string_MODALITY_is_skipped(self, tmp_path, caplog):
        _write_modality_package(tmp_path, "weird", """
            MODALITY = 42  # not a string
            def build_router(r): return None
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_modalities([tmp_path])
        assert result == {}

    def test_non_callable_build_router_is_skipped(self, tmp_path, caplog):
        _write_modality_package(tmp_path, "weird2", """
            MODALITY = "x/y"
            build_router = "not callable"
        """)
        import logging
        caplog.set_level(logging.WARNING)
        result = discover_modalities([tmp_path])
        assert result == {}

    def test_discovery_is_isolated_per_scan(self, tmp_path):
        """Each call to discover_models should work independently even if
        sys.modules has leftover entries from a previous call."""
        _write_model_script(tmp_path, "m.py", """
            MANIFEST = {"model_id": "m", "modality": "x", "hf_repo": "r"}
            class Model: ...
        """)
        first = discover_models([tmp_path])
        second = discover_models([tmp_path])
        assert "m" in first
        assert "m" in second
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/core/test_discovery.py -v
```

All should pass — the implementation from B1 already handles these because:
- `isinstance(manifest, dict)` check excludes string MANIFEST
- `isinstance(model_class, type)` check excludes instances
- `isinstance(tag, str)` check excludes non-string MODALITY
- `callable(build_router)` check excludes non-callables
- Mangled module names in `_load_script` prevent sys.modules pollution across scans

If any test fails, fix the discovery module to match. If all pass, commit the tests as a regression guard.

- [ ] **Step 3: Commit**

```bash
git add tests/core/test_discovery.py
git commit -m "test(discovery): add edge-case regression tests

Five additional tests covering non-dict MANIFEST, non-class Model,
non-string MODALITY, non-callable build_router, and repeated-scan
sys.modules independence. These are regression guards against future
refactors that might loosen the type checks."
```

---

## Part C — Model script migration

Each task converts a family of backends (per-modality) into flat model scripts under `muse/models/`. The goal is one file per model with MANIFEST + Model class + any module-local helpers.

Approach per model:
1. Open the existing backend file (e.g. `muse/modalities/audio_speech/backends/kokoro.py`)
2. Copy its content into `muse/models/kokoro_82m.py`, renaming the class to `Model`
3. Add a top-level `MANIFEST` dict matching the current `KNOWN_MODELS` entry for that model
4. Update imports of modality-internal types to use the new re-exports (`from muse.modalities.audio_speech import AudioResult`)
5. Move/rewrite the corresponding test file to `tests/models/<id>.py`
6. Update any helper classes (BaseModel, TransformersModel) — keep them in the modality package as shared utilities, or inline them into the model script if used by only one

### Task C1: Convert audio.speech backends → model scripts

**Files:**
- Create: `src/muse/models/__init__.py` (empty)
- Create: `src/muse/models/soprano_80m.py`, `kokoro_82m.py`, `bark_small.py`
- Move: helpers that remain shared → stay in `muse.modalities.audio_speech.backends.base` etc.
- Create: `tests/models/__init__.py`
- Move/Create: `tests/models/test_soprano_80m.py`, `test_kokoro_82m.py`, `test_bark_small.py`
- Modify: `src/muse/core/catalog.py` — update `backend_path` entries for these three models
- Remove (later task C4): `src/muse/modalities/audio_speech/backends/{soprano.py, kokoro.py, bark.py}`

- [ ] **Step 1: Create the `muse/models/` + `tests/models/` skeleton**

```bash
cd /home/spinoza/github/repos/muse-plugin-discovery
mkdir -p src/muse/models tests/models
touch src/muse/models/__init__.py tests/models/__init__.py
```

- [ ] **Step 2: Create `src/muse/models/kokoro_82m.py`**

Read the current file for reference:

```bash
cat src/muse/modalities/audio_speech/backends/kokoro.py
```

Then write the new script. Template:

```python
"""Kokoro 82M TTS: lightweight multi-voice text-to-speech, 24kHz.

Model by hexgrad (Apache 2.0). 82M parameters, 54 voices across 8+
languages. Runs on CPU in real-time; GPU gives ~5-10x real-time.
Requires espeak-ng at runtime (install via apt/brew).
"""
from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np

from muse.modalities.audio_speech import AudioChunk, AudioResult

logger = logging.getLogger(__name__)


_VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_liam", "am_michael", "am_onyx",
    "am_puck", "am_santa",
    "bf_emma", "bf_isabella", "bf_alice", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora",
    "em_alex", "em_santa",
    "ff_siwis",
    "hf_alpha", "hf_beta",
    "hm_omega", "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
]


MANIFEST = {
    "model_id": "kokoro-82m",
    "modality": "audio/speech",
    "hf_repo": "hexgrad/Kokoro-82M",
    "description": "Lightweight TTS, 54 voices, 24kHz",
    "license": "Apache 2.0",
    "pip_extras": ("kokoro", "soundfile", "misaki[en]"),
    "system_packages": ("espeak-ng",),
    "capabilities": {
        "sample_rate": 24000,
        "voices": _VOICES,
        "languages": ["en", "es", "fr", "it", "pt", "hi", "ja", "zh"],
    },
}


class Model:
    """Kokoro 82M TTS backend. Always named `Model` per muse discovery convention."""

    model_id = MANIFEST["model_id"]
    sample_rate = 24000

    @property
    def voices(self) -> list[str]:
        return _VOICES

    def __init__(
        self,
        *,
        hf_repo: str = "hexgrad/Kokoro-82M",
        local_dir: str | None = None,
        device: str = "auto",
        lang_code: str = "a",
        **_: Any,
    ) -> None:
        import torch
        from kokoro import KPipeline

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # local_dir is accepted for catalog-loader compatibility but NOT
        # forwarded to KPipeline — Kokoro validates repo_id as "namespace/name"
        # and rejects filesystem paths. The HF cache from snapshot_download
        # is still used transparently.
        logger.info("Loading Kokoro (lang=%s, device=%s)", lang_code, device)
        self._pipeline = KPipeline(
            lang_code=lang_code,
            repo_id=hf_repo,
            device=device,
        )
        self._device = device

    def synthesize(self, text: str, **kwargs) -> AudioResult:
        voice = kwargs.get("voice", "af_heart")
        speed = kwargs.get("speed", 1.0)
        chunks = []
        for result in self._pipeline(text, voice=voice, speed=speed):
            if result.audio is not None:
                chunks.append(result.audio.numpy())
        if chunks:
            audio = np.concatenate(chunks).astype(np.float32)
        else:
            audio = np.zeros(0, dtype=np.float32)
        return AudioResult(
            audio=audio,
            sample_rate=24000,
            metadata={"voice": voice} if voice else {},
        )

    def synthesize_stream(self, text: str, **kwargs) -> Iterator[AudioChunk]:
        voice = kwargs.get("voice", "af_heart")
        speed = kwargs.get("speed", 1.0)
        for result in self._pipeline(text, voice=voice, speed=speed):
            if result.audio is not None:
                audio = result.audio.numpy().astype(np.float32)
                yield AudioChunk(audio=audio, sample_rate=24000)
```

- [ ] **Step 3: Create `src/muse/models/soprano_80m.py`**

Read `src/muse/modalities/audio_speech/backends/soprano.py` for reference. The Soprano model is more complex because it wraps a `Narro` inner class (from `muse.modalities.audio_speech.tts`). The model script should import from the modality's internal modules as needed. Keep the Narro + Vocos helpers where they are (in the modality package) and import them from the script.

```python
"""Soprano 80M TTS: Qwen3 LLM backbone + Vocos decoder, 32kHz, 80M params."""
from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np

from muse.modalities.audio_speech import AudioChunk, AudioResult

logger = logging.getLogger(__name__)


MANIFEST = {
    "model_id": "soprano-80m",
    "modality": "audio/speech",
    "hf_repo": "ekwek/Soprano-1.1-80M",
    "description": "Qwen3 LLM backbone + Vocos decoder, 32kHz, 80M params",
    "license": "Apache 2.0",
    "pip_extras": ("transformers>=4.36.0", "scipy", "inflect", "unidecode"),
    "system_packages": (),
    "capabilities": {
        "sample_rate": 32000,
        "supports_alignment": True,
    },
}


class Model:
    """Soprano TTS backend.

    Delegates to the Narro engine (in muse.modalities.audio_speech.tts) for
    the actual inference; this wrapper adapts the narro API to the
    TTSModel protocol.
    """
    model_id = MANIFEST["model_id"]
    sample_rate = 32000

    def __init__(
        self,
        *,
        hf_repo: str = "ekwek/Soprano-1.1-80M",
        local_dir: str | None = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        from muse.modalities.audio_speech.tts import Narro

        src = local_dir or hf_repo
        logger.info("loading Soprano from %s (device=%s)", src, device)
        self._engine = Narro(model_path=src, device=device, **{
            k: v for k, v in kwargs.items()
            if k in ("compile", "quantize")
        })

    def synthesize(self, text: str, **kwargs) -> AudioResult:
        result = self._engine.synthesize(text, **kwargs)
        audio = result.audio if hasattr(result, "audio") else result
        metadata = getattr(result, "metadata", {}) or {}
        return AudioResult(
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=self.sample_rate,
            metadata=metadata,
        )

    def synthesize_stream(self, text: str, **kwargs) -> Iterator[AudioChunk]:
        for chunk in self._engine.synthesize_stream(text, **kwargs):
            audio = chunk.audio if hasattr(chunk, "audio") else chunk
            yield AudioChunk(
                audio=np.asarray(audio, dtype=np.float32),
                sample_rate=self.sample_rate,
            )
```

NOTE: if the current Soprano backend does something specific that the template above doesn't cover, adapt faithfully from the original file. The point is: the model file IS Soprano's public API as a muse model; anything it needs that's modality-internal (Narro engine, Vocos decoder) stays in `muse.modalities.audio_speech.*` and gets imported.

- [ ] **Step 4: Create `src/muse/models/bark_small.py`**

Same pattern. Read the current backend file and translate. Ship it.

```python
"""Bark Small TTS: multilingual + voice cloning, 24kHz."""
from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np

from muse.modalities.audio_speech import AudioChunk, AudioResult

logger = logging.getLogger(__name__)


_BARK_VOICES = [
    # Copy the voice list from the current bark.py backend
    # (likely includes en_speaker_0, en_speaker_1, ..., de_speaker_0, ...)
]


MANIFEST = {
    "model_id": "bark-small",
    "modality": "audio/speech",
    "hf_repo": "suno/bark-small",
    "description": "Multilingual TTS with voice cloning, 24kHz",
    "license": "MIT",
    "pip_extras": ("transformers>=4.36.0", "scipy"),
    "system_packages": (),
    "capabilities": {
        "sample_rate": 24000,
        "voices": _BARK_VOICES,
        "voice_cloning": True,
    },
}


class Model:
    model_id = MANIFEST["model_id"]
    sample_rate = 24000

    @property
    def voices(self) -> list[str]:
        return _BARK_VOICES

    def __init__(
        self,
        *,
        hf_repo: str = "suno/bark-small",
        local_dir: str | None = None,
        device: str = "auto",
        **_: Any,
    ) -> None:
        # Copy the __init__ body from the current bark.py backend.
        # Heavy imports deferred inside.
        from transformers import AutoProcessor, BarkModel as BarkModel_HF
        src = local_dir or hf_repo
        logger.info("loading Bark from %s (device=%s)", src, device)
        self._processor = AutoProcessor.from_pretrained(src)
        self._model = BarkModel_HF.from_pretrained(src)
        if device != "cpu":
            self._model = self._model.to(device)
        self._device = device

    def synthesize(self, text: str, **kwargs) -> AudioResult:
        # Copy from current bark.py synthesize
        # ...
        pass

    def synthesize_stream(self, text: str, **kwargs) -> Iterator[AudioChunk]:
        # Copy from current bark.py synthesize_stream
        # ...
        pass
```

Fill in the synthesize/synthesize_stream bodies from the current bark.py. This task MUST preserve behavior; only the file path and class name change.

- [ ] **Step 5: Move and rewrite tests**

```bash
git mv tests/modalities/audio_speech/test_kokoro.py tests/models/test_kokoro_82m.py
git mv tests/modalities/audio_speech/test_soprano.py tests/models/test_soprano_80m.py
git mv tests/modalities/audio_speech/test_bark.py tests/models/test_bark_small.py
```

Update imports in the moved test files:

```bash
find tests/models -name "test_kokoro_82m.py" -o -name "test_soprano_80m.py" -o -name "test_bark_small.py" | \
  xargs sed -i \
  -e 's|from muse\.modalities\.audio_speech\.backends\.kokoro import KokoroModel|from muse.models.kokoro_82m import Model as KokoroModel|g' \
  -e 's|from muse\.modalities\.audio_speech\.backends\.soprano import SopranoModel|from muse.models.soprano_80m import Model as SopranoModel|g' \
  -e 's|from muse\.modalities\.audio_speech\.backends\.bark import BarkModel|from muse.models.bark_small import Model as BarkModel|g' \
  -e 's|muse\.modalities\.audio_speech\.backends\.kokoro|muse.models.kokoro_82m|g' \
  -e 's|muse\.modalities\.audio_speech\.backends\.soprano|muse.models.soprano_80m|g' \
  -e 's|muse\.modalities\.audio_speech\.backends\.bark|muse.models.bark_small|g'
```

Also add a test per model that exercises the MANIFEST:

Add to each test file:

```python
def test_manifest_has_required_fields():
    from muse.models.kokoro_82m import MANIFEST
    assert MANIFEST["model_id"] == "kokoro-82m"
    assert MANIFEST["modality"] == "audio/speech"
    assert "hf_repo" in MANIFEST
    assert "pip_extras" in MANIFEST
```

(Adapt for each model.)

- [ ] **Step 6: Update `src/muse/core/catalog.py` to point at new backend paths**

Find the `KNOWN_MODELS` entries for soprano-80m, kokoro-82m, bark-small. Update each `backend_path` from e.g. `"muse.modalities.audio_speech.backends.kokoro:KokoroModel"` to `"muse.models.kokoro_82m:Model"`.

```python
"soprano-80m": CatalogEntry(
    ...
    backend_path="muse.models.soprano_80m:Model",
    ...
),
"kokoro-82m": CatalogEntry(
    ...
    backend_path="muse.models.kokoro_82m:Model",
    ...
),
"bark-small": CatalogEntry(
    ...
    backend_path="muse.models.bark_small:Model",
    ...
),
```

- [ ] **Step 7: Run tests**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -5
```

Expected: all audio.speech model tests pass (kokoro, soprano, bark). If any fail, diagnose — most likely cause is an import path or subtle behavior difference introduced during copy.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(audio.speech): convert backends to flat muse/models/ scripts

Each audio.speech model (kokoro, soprano, bark) becomes a single file
at src/muse/models/<id>.py containing both MANIFEST (metadata) and
Model class (implementation). Tests move to tests/models/. Catalog
backend_path strings updated.

Old files still in src/muse/modalities/audio_speech/backends/ — they
get deleted in Task C4 once all three modality families are migrated."
```

---

### Task C2: Convert images.generation backend → model script

Same pattern, single model.

- [ ] **Step 1: Create `src/muse/models/sd_turbo.py`**

Read `src/muse/modalities/image_generation/backends/sd_turbo.py` and translate:

```python
"""SD-Turbo: 1-step distilled Stable Diffusion, 512x512, 2GB."""
from __future__ import annotations

import logging
from typing import Any

from muse.modalities.image_generation import ImageResult

logger = logging.getLogger(__name__)


try:
    import torch
    from diffusers import AutoPipelineForText2Image
except ImportError:  # pragma: no cover
    torch = None
    AutoPipelineForText2Image = None


MANIFEST = {
    "model_id": "sd-turbo",
    "modality": "image/generation",
    "hf_repo": "stabilityai/sd-turbo",
    "description": "Stable Diffusion Turbo: 1-step distilled, 512x512",
    "license": "Stability AI Non-Commercial Research Community License",
    "pip_extras": ("diffusers>=0.27.0", "accelerate", "Pillow", "safetensors"),
    "system_packages": (),
    "capabilities": {
        "default_size": (512, 512),
        "recommended_steps": 1,
        "supports_cfg": False,
    },
}


class Model:
    model_id = MANIFEST["model_id"]
    default_size = (512, 512)

    def __init__(
        self,
        *,
        hf_repo: str = "stabilityai/sd-turbo",
        local_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        **_: Any,
    ) -> None:
        if AutoPipelineForText2Image is None:
            raise RuntimeError("diffusers not installed; run `muse pull sd-turbo`")
        # Copy the rest of __init__ from current sd_turbo.py
        ...

    def generate(self, prompt: str, **kwargs) -> ImageResult:
        # Copy from current sd_turbo.py
        ...
```

Fill in the bodies preserving current behavior.

- [ ] **Step 2: Move and rewrite tests**

```bash
git mv tests/modalities/image_generation/test_sd_turbo.py tests/models/test_sd_turbo.py

sed -i \
  -e 's|from muse\.modalities\.image_generation\.backends\.sd_turbo import SDTurboModel|from muse.models.sd_turbo import Model as SDTurboModel|g' \
  -e 's|muse\.modalities\.image_generation\.backends\.sd_turbo|muse.models.sd_turbo|g' \
  tests/models/test_sd_turbo.py
```

Add a manifest test:

```python
def test_manifest_has_required_fields():
    from muse.models.sd_turbo import MANIFEST
    assert MANIFEST["model_id"] == "sd-turbo"
    assert MANIFEST["modality"] == "image/generation"
    assert "hf_repo" in MANIFEST
```

- [ ] **Step 3: Update catalog entry**

Change `backend_path` for sd-turbo to `"muse.models.sd_turbo:Model"`.

- [ ] **Step 4: Run tests**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(image.generation): convert sd-turbo backend to muse/models/ script"
```

---

### Task C3: Convert embedding.text backends → model scripts

Three models: all-minilm-l6-v2, qwen3-embedding-0.6b, nv-embed-v2.

Steps mirror C1 but for embeddings. For each model:
- Create `src/muse/models/<id>.py` with MANIFEST + Model class
- Move test file to `tests/models/test_<id>.py`, rewrite imports
- Update catalog `backend_path`

Example for all-minilm-l6-v2:

```python
"""all-MiniLM-L6-v2: 384-dim sentence embeddings, CPU-friendly."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from muse.modalities.embedding_text import EmbeddingResult

logger = logging.getLogger(__name__)

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    torch = None
    SentenceTransformer = None


MANIFEST = {
    "model_id": "all-minilm-l6-v2",
    "modality": "embedding/text",
    "hf_repo": "sentence-transformers/all-MiniLM-L6-v2",
    "description": "MiniLM sentence embeddings: 384 dims, 22MB, CPU-friendly",
    "license": "Apache 2.0",
    "pip_extras": ("torch>=2.1.0", "sentence-transformers>=2.2.0"),
    "system_packages": (),
    "capabilities": {
        "dimensions": 384,
        "context_length": 256,
        "matryoshka": False,
    },
}


class Model:
    model_id = MANIFEST["model_id"]
    dimensions = 384

    def __init__(self, *, hf_repo, local_dir=None, device="auto", **_: Any):
        # Copy body from current minilm.py
        ...

    def embed(self, input, *, dimensions=None, **_: Any) -> EmbeddingResult:
        # Copy body from current minilm.py
        ...
```

Do the same for qwen3_embedding_0_6b.py and nv_embed_v2.py. Test files move to `tests/models/`.

- [ ] **Step 1-3: Create each of the three model scripts**

One per model, following the pattern above. Read each current backend file and translate.

- [ ] **Step 4: Move tests**

```bash
git mv tests/modalities/embedding_text/test_minilm.py         tests/models/test_all_minilm_l6_v2.py
git mv tests/modalities/embedding_text/test_qwen3_embedding.py tests/models/test_qwen3_embedding_0_6b.py
git mv tests/modalities/embedding_text/test_nv_embed_v2.py    tests/models/test_nv_embed_v2.py

# Rewrite imports
sed -i \
  -e 's|from muse\.modalities\.embedding_text\.backends\.minilm import MiniLMBackend|from muse.models.all_minilm_l6_v2 import Model as MiniLMBackend|g' \
  -e 's|muse\.modalities\.embedding_text\.backends\.minilm|muse.models.all_minilm_l6_v2|g' \
  tests/models/test_all_minilm_l6_v2.py

sed -i \
  -e 's|from muse\.modalities\.embedding_text\.backends\.qwen3_embedding import Qwen3Embedding06BBackend|from muse.models.qwen3_embedding_0_6b import Model as Qwen3Embedding06BBackend|g' \
  -e 's|muse\.modalities\.embedding_text\.backends\.qwen3_embedding|muse.models.qwen3_embedding_0_6b|g' \
  tests/models/test_qwen3_embedding_0_6b.py

sed -i \
  -e 's|from muse\.modalities\.embedding_text\.backends\.nv_embed_v2 import NVEmbedV2Backend|from muse.models.nv_embed_v2 import Model as NVEmbedV2Backend|g' \
  -e 's|muse\.modalities\.embedding_text\.backends\.nv_embed_v2|muse.models.nv_embed_v2|g' \
  tests/models/test_nv_embed_v2.py
```

- [ ] **Step 5: Update catalog entries**

Update `backend_path` for each of the three embedding models.

- [ ] **Step 6: Run tests**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -5
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(embedding.text): convert minilm/qwen3/nv-embed backends to flat scripts"
```

---

### Task C4: Delete old `backends/` directories in modalities

Now that all models live in `muse/models/*.py`, the per-modality `backends/` directories should be empty except for true helper code (base classes, shared utilities).

- [ ] **Step 1: List the contents of each remaining backends/ dir**

```bash
cd /home/spinoza/github/repos/muse-plugin-discovery
ls src/muse/modalities/audio_speech/backends/
ls src/muse/modalities/embedding_text/backends/
ls src/muse/modalities/image_generation/backends/
```

- [ ] **Step 2: Decide per file — helper or orphan**

For each file remaining, either:
- Keep it as a helper (e.g., `base.py` with shared methods used across multiple model scripts), OR
- Delete it (no longer imported anywhere)

Check what imports each file:

```bash
for f in src/muse/modalities/audio_speech/backends/*.py; do
  [[ "$(basename $f)" == "__init__.py" ]] && continue
  name=$(basename $f .py)
  echo "=== backends/$name.py ==="
  grep -rn "from muse.modalities.audio_speech.backends.$name\|import muse.modalities.audio_speech.backends.$name" src/ tests/ --include="*.py" || echo "(unused)"
done
```

For audio.speech, check if anything still imports from `backends.base` or `backends.transformers`. If not, delete them. If Soprano's script uses `BaseModel`, either:
- Move `BaseModel` into `soprano_80m.py` as a helper
- Or promote `base.py` out of backends and into `muse/modalities/audio_speech/utils/` or similar

- [ ] **Step 3: Act on the decisions**

Delete orphaned files:

```bash
git rm src/muse/modalities/audio_speech/backends/kokoro.py
git rm src/muse/modalities/audio_speech/backends/soprano.py
git rm src/muse/modalities/audio_speech/backends/bark.py
git rm src/muse/modalities/image_generation/backends/sd_turbo.py
git rm src/muse/modalities/embedding_text/backends/minilm.py
git rm src/muse/modalities/embedding_text/backends/qwen3_embedding.py
git rm src/muse/modalities/embedding_text/backends/nv_embed_v2.py
```

For helpers, either leave them in place or relocate (decide per-file; aim for minimal churn).

If an entire backends/ dir ends up empty after deleting orphans:

```bash
rmdir src/muse/modalities/audio_speech/backends 2>&1 || echo "still has files"
rmdir src/muse/modalities/embedding_text/backends 2>&1 || echo "still has files"
rmdir src/muse/modalities/image_generation/backends 2>&1 || echo "still has files"
```

- [ ] **Step 4: Test**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -3
```

All pass. If a test fails due to a missing helper that was deleted, restore the helper or inline it into the script that needed it.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: delete orphaned backends/ files after model migration

All old kokoro/soprano/bark/sd_turbo/minilm/qwen3/nv-embed backend
files are now duplicated by the flat scripts under muse/models/.
Deleting the originals. Any shared helpers (BaseModel etc.) stay in
the modality package's utility modules."
```

---

## Part D — Catalog and registry use discovery

### Task D1: Catalog reads from discovery, not hardcoded dict

**Files:**
- Modify: `src/muse/core/catalog.py`
- Modify: `tests/core/test_catalog.py`

The `KNOWN_MODELS` dict goes away; discovery drives everything.

- [ ] **Step 1: Write failing tests**

Add to `tests/core/test_catalog.py`:

```python
def test_known_models_now_includes_discovery_results():
    """Discovery populates KNOWN_MODELS from src/muse/models/ at import time."""
    from muse.core.catalog import known_models
    entries = known_models()
    # Bundled models from the Phase C migration
    assert "kokoro-82m" in entries
    assert "soprano-80m" in entries
    assert "bark-small" in entries
    assert "sd-turbo" in entries
    assert "all-minilm-l6-v2" in entries
    assert "qwen3-embedding-0.6b" in entries
    assert "nv-embed-v2" in entries


def test_catalog_entries_reflect_discovered_manifests():
    from muse.core.catalog import known_models
    entries = known_models()
    kokoro = entries["kokoro-82m"]
    assert kokoro.modality == "audio/speech"
    assert kokoro.hf_repo == "hexgrad/Kokoro-82M"
    # capabilities (was extra) passed through
    assert "sample_rate" in kokoro.extra
    assert kokoro.extra["sample_rate"] == 24000


def test_list_known_still_works_for_modality_filter():
    from muse.core.catalog import list_known
    audio = list_known("audio/speech")
    ids = {e.model_id for e in audio}
    assert {"kokoro-82m", "soprano-80m", "bark-small"}.issubset(ids)
    for e in audio:
        assert e.modality == "audio/speech"
```

- [ ] **Step 2: Run — these should pass immediately if catalog was already changed**

They won't pass yet because catalog still has the hardcoded dict. Expected:

```bash
pytest tests/core/test_catalog.py::test_known_models_now_includes_discovery_results -v
```

Likely fails with some sort of `AttributeError: module 'muse.core.catalog' has no attribute 'known_models'`.

- [ ] **Step 3: Refactor `src/muse/core/catalog.py`**

Replace the `KNOWN_MODELS` dict definition with a function that runs discovery. Keep `CatalogEntry` dataclass (maybe renamed but interface compat). Key changes:

```python
# Replace the huge KNOWN_MODELS = {...} dict with this:

from pathlib import Path as _Path
from muse.core.discovery import discover_models, DiscoveredModel


def _bundled_models_dir() -> _Path:
    """Return src/muse/models/ from the installed location."""
    # catalog.py is src/muse/core/catalog.py; models/ is src/muse/models/
    return _Path(__file__).resolve().parents[1] / "models"


def _model_dirs() -> list[_Path]:
    """Scan order for model discovery (bundled → user → env).

    User and env scanning land in Task F1; for now, bundled only.
    """
    return [_bundled_models_dir()]


def _manifest_to_catalog_entry(discovered: DiscoveredModel) -> CatalogEntry:
    """Convert a DiscoveredModel into the legacy CatalogEntry shape."""
    m = discovered.manifest
    return CatalogEntry(
        model_id=m["model_id"],
        modality=m["modality"],
        backend_path=f"{discovered.model_class.__module__}:{discovered.model_class.__name__}",
        hf_repo=m["hf_repo"],
        description=m.get("description", ""),
        pip_extras=tuple(m.get("pip_extras", ())),
        system_packages=tuple(m.get("system_packages", ())),
        extra=dict(m.get("capabilities", {})),
    )


# Cache so we don't re-import every scripts on every call
_known_models_cache: dict[str, CatalogEntry] | None = None


def known_models() -> dict[str, CatalogEntry]:
    """Discover and return all known models (bundled + user + env).

    Cached at first call; restart the process to pick up new scripts.
    (Hot-reload is a separate plan.)
    """
    global _known_models_cache
    if _known_models_cache is not None:
        return _known_models_cache
    discovered = discover_models(_model_dirs())
    _known_models_cache = {
        model_id: _manifest_to_catalog_entry(d)
        for model_id, d in discovered.items()
    }
    return _known_models_cache


# Legacy alias: KNOWN_MODELS used to be a dict. Keep as a property-like
# accessor so existing call sites don't break.
def _known_models_dict() -> dict[str, CatalogEntry]:
    return known_models()


# If any code does `KNOWN_MODELS` attribute access, this namespace hack gives them dict-like:
# But cleaner: have every caller switch to known_models() function call.
# We'll do both as a transition.
```

Then update all places that use `KNOWN_MODELS` (this module's `pull`, `load_backend`, `list_known`, `is_pulled` don't need changes if they go through `known_models()`). Find usages:

```bash
grep -rn "KNOWN_MODELS" src/ tests/ --include="*.py"
```

For each hit in `src/muse/core/catalog.py`, replace `KNOWN_MODELS` → `known_models()` so the function is called.

For `tests/core/test_catalog.py`, same replacement.

- [ ] **Step 4: Run tests**

```bash
pytest tests/core/test_catalog.py -v 2>&1 | tail -20
```

Fix any remaining issues. Likely some tests will need adjustment because they assumed the dict form vs function form.

- [ ] **Step 5: Full regression**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -3
```

- [ ] **Step 6: Commit**

```bash
git add src/muse/core/catalog.py tests/core/test_catalog.py
git commit -m "refactor(catalog): known_models() reads from discovery, drop KNOWN_MODELS dict

The old hardcoded KNOWN_MODELS = {...} dict (which had to be edited
every time a model was added) is gone. In its place, known_models()
runs discover_models([bundled_models_dir]) on first call and caches.

Script's MANIFEST keys map to the existing CatalogEntry fields:
  manifest[model_id] → CatalogEntry.model_id
  manifest[modality] → CatalogEntry.modality
  f'{module}:{Model}' → CatalogEntry.backend_path
  manifest[hf_repo] → CatalogEntry.hf_repo
  manifest[pip_extras] → CatalogEntry.pip_extras
  manifest[system_packages] → CatalogEntry.system_packages
  manifest[capabilities] → CatalogEntry.extra

User-dir and $MUSE_MODELS_DIR scanning added in Task F1."
```

---

### Task D2: Registry refactor — drop `_extra()` allowlist

**Files:**
- Modify: `src/muse/core/registry.py`
- Modify: `tests/core/test_registry.py`
- Modify: `src/muse/cli_impl/worker.py` (register with manifest, not just model)
- Possibly modify: `src/muse/core/server.py` (how `/v1/models` builds its output)

The old `_extra()` helper hardcoded attribute names. Now the registry stores a **manifest** per registered model and passes it through to `/v1/models`.

- [ ] **Step 1: Update `ModelInfo` dataclass**

```python
@dataclass
class ModelInfo:
    """Registry metadata for a loaded model.

    `manifest` holds the full MANIFEST dict from the model's script
    (plus runtime-specific keys added by the registry). Replaces the
    old `extra` field which was populated by a hardcoded attribute
    allowlist.
    """
    modality: str
    model_id: str
    manifest: dict
```

- [ ] **Step 2: Update `ModalityRegistry.register`**

```python
def register(self, modality: str, model: Any, manifest: dict | None = None) -> None:
    """Register a model under a modality.

    If `manifest` is provided, it's stored and surfaced via list_models
    / list_all. This is the metadata source for /v1/models responses.
    """
    models = self._models.setdefault(modality, {})
    model_id = model.model_id
    models[model_id] = model
    self._manifests.setdefault(modality, {})[model_id] = manifest or {
        "model_id": model_id,
        "modality": modality,
    }
    self._defaults.setdefault(modality, model_id)
```

- [ ] **Step 3: Drop `_extra` function, replace with manifest passthrough**

```python
def list_models(self, modality: str) -> list[ModelInfo]:
    return [
        ModelInfo(
            modality=modality,
            model_id=mid,
            manifest=self._manifests.get(modality, {}).get(mid, {}),
        )
        for mid in self._models.get(modality, {})
    ]
```

- [ ] **Step 4: Update `muse.core.server.create_app` so `/v1/models` uses manifest**

```python
@app.get("/v1/models")
def list_models():
    data = []
    for info in registry.list_all():
        # Start with manifest (capabilities, description, license, etc.)
        entry = {
            **info.manifest.get("capabilities", {}),
            **{k: info.manifest[k] for k in (
                "description", "license", "hf_repo",
            ) if k in info.manifest},
            "id": info.model_id,
            "modality": info.modality,
            "object": "model",
        }
        data.append(entry)
    return {"object": "list", "data": data}
```

- [ ] **Step 5: Update `muse.cli_impl.worker.run_worker` to pass manifest when registering**

```python
# In the loop that registers each loaded model:
from muse.core.catalog import known_models
catalog = known_models()
for model_id in to_load:
    ...
    entry = catalog[model_id]
    # Look up the manifest via the discovered model class module
    import importlib
    module_path, _ = entry.backend_path.split(":", 1)
    module = importlib.import_module(module_path)
    manifest = getattr(module, "MANIFEST", {})
    ...
    registry.register(entry.modality, backend, manifest=manifest)
```

- [ ] **Step 6: Update tests**

Existing registry tests use `ModelInfo(modality, model_id, extra={...})`. Replace with `ModelInfo(modality, model_id, manifest={...})`. Existing server tests (for /v1/models) assert the presence of specific fields — those should still pass because manifest forwarding preserves the fields that were previously in `extra`.

- [ ] **Step 7: Run tests**

```bash
pytest tests/ -q -m "not slow" 2>&1 | tail -5
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(registry): drop _extra allowlist, store manifest per model

ModelInfo.extra → ModelInfo.manifest. Full MANIFEST dict (plus any
runtime-added keys) is stored at registration time and surfaced via
/v1/models. No more hardcoded attribute allowlist; arbitrary manifest
capabilities pass through to clients.

Worker now reads each model's MANIFEST from its Python module at
registration time and forwards it to registry.register()."
```

---

## Part E — Worker auto-mount

### Task E1: Worker uses `discover_modalities` instead of hardcoded imports

**Files:**
- Modify: `src/muse/cli_impl/worker.py`
- Modify: `tests/cli_impl/test_worker.py`

- [ ] **Step 1: Update worker.py**

Replace the hardcoded block:

```python
from muse.modalities.audio_speech.routes import build_router as build_audio
from muse.modalities.embedding_text.routes import build_router as build_embeddings
from muse.modalities.image_generation.routes import build_router as build_images

routers["audio/speech"] = build_audio(registry)
routers["embedding/text"] = build_embeddings(registry)
routers["image/generation"] = build_images(registry)
```

With discovery:

```python
from pathlib import Path
from muse.core.discovery import discover_modalities


def _bundled_modalities_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "modalities"


def _modality_dirs() -> list[Path]:
    """Scan order for modality discovery (bundled, env-var in F2)."""
    return [_bundled_modalities_dir()]


def run_worker(*, host: str, port: int, models: list[str], device: str) -> int:
    ...
    discovered_modalities = discover_modalities(_modality_dirs())
    for tag, build_router in discovered_modalities.items():
        logger.info("mounting modality router for %s", tag)
        routers[tag] = build_router(registry)
    ...
```

- [ ] **Step 2: Update/add worker tests**

Existing test `test_worker_mounts_all_three_modality_routers` should continue to pass because all three bundled modalities export `build_router`. Add a new test:

```python
def test_worker_discovers_modalities_from_bundled_dir(monkeypatch):
    """run_worker should pick up routers via discovery without hardcoded imports."""
    # Patch discover_modalities to return a sentinel-mapped router; verify worker uses it
    ...
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/cli_impl/test_worker.py tests/cli_impl/test_e2e_supervisor.py -v --timeout=60
```

E2E test must still pass — verifies real subprocess + real discovery.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(worker): auto-mount modality routers via discovery

worker.py no longer has hardcoded imports of muse.modalities.*.routes.
Instead, run_worker calls discover_modalities([bundled_dir]) and mounts
each returned router. Adding a new modality requires zero changes to
worker.py — just drop a new subpackage under src/muse/modalities/.

$MUSE_MODALITIES_DIR escape hatch lands in Task F2."
```

---

## Part F — User-dir + env-var support

### Task F1: Discovery scans `~/.muse/models/` and `$MUSE_MODELS_DIR`

**Files:**
- Modify: `src/muse/core/catalog.py` — expand `_model_dirs()`
- Modify: `tests/core/test_catalog.py` — add user-dir tests

- [ ] **Step 1: Add failing tests**

```python
def test_known_models_includes_user_dir(tmp_path, monkeypatch):
    """Models in ~/.muse/models/ override bundled ones on collision."""
    user_dir = tmp_path / ".muse" / "models"
    user_dir.mkdir(parents=True)
    (user_dir / "my_custom.py").write_text('''
MANIFEST = {
    "model_id": "my-custom-model",
    "modality": "audio/speech",
    "hf_repo": "fake/repo",
}
class Model:
    model_id = "my-custom-model"
''')
    monkeypatch.setattr("muse.core.catalog._HOME_DIR", tmp_path)
    # Clear the cache
    import muse.core.catalog as cat
    cat._known_models_cache = None

    entries = cat.known_models()
    assert "my-custom-model" in entries


def test_muse_models_dir_env_var_scanned(tmp_path, monkeypatch):
    env_dir = tmp_path / "my-muse-models"
    env_dir.mkdir()
    (env_dir / "custom.py").write_text('''
MANIFEST = {"model_id": "env-model", "modality": "audio/speech", "hf_repo": "r"}
class Model: ...
''')
    monkeypatch.setenv("MUSE_MODELS_DIR", str(env_dir))
    # Clear cache
    import muse.core.catalog as cat
    cat._known_models_cache = None

    entries = cat.known_models()
    assert "env-model" in entries
```

- [ ] **Step 2: Expand `_model_dirs()` to include user dir + env**

```python
from pathlib import Path
import os

_HOME_DIR = Path.home()  # Can be monkeypatched in tests


def _user_models_dir() -> Path:
    return _HOME_DIR / ".muse" / "models"


def _env_models_dir() -> Path | None:
    env = os.environ.get("MUSE_MODELS_DIR")
    return Path(env) if env else None


def _model_dirs() -> list[Path]:
    """Scan order: bundled → user dir → env var override."""
    dirs = [_bundled_models_dir(), _user_models_dir()]
    env = _env_models_dir()
    if env is not None:
        dirs.append(env)
    return dirs
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/core/test_catalog.py -v 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(discovery): scan ~/.muse/models/ and \$MUSE_MODELS_DIR

discover_models is now called with [bundled, user, env] directories.
User drop-ins override bundled on model_id collision (warned); env
override runs last.

Users adding models needs zero muse source changes:
  cp my_model.py ~/.muse/models/
  muse models list  # now shows it"
```

---

### Task F2: Discovery scans `$MUSE_MODALITIES_DIR` (escape hatch)

**Files:**
- Modify: `src/muse/cli_impl/worker.py`
- Modify: `tests/cli_impl/test_worker.py`

Symmetric change: env-var for modalities. User-home isn't scanned (keeping modalities as advanced/escape-hatch territory).

- [ ] **Step 1: Expand worker's `_modality_dirs()`**

```python
import os

def _env_modalities_dir() -> Path | None:
    env = os.environ.get("MUSE_MODALITIES_DIR")
    return Path(env) if env else None


def _modality_dirs() -> list[Path]:
    """Scan order: bundled → env override. No user-home scan by design."""
    dirs = [_bundled_modalities_dir()]
    env = _env_modalities_dir()
    if env is not None:
        dirs.append(env)
    return dirs
```

- [ ] **Step 2: Add test**

```python
def test_muse_modalities_dir_env_scanned(tmp_path, monkeypatch):
    env_dir = tmp_path / "my-modalities"
    env_dir.mkdir()
    my_mod = env_dir / "my_new_thing"
    my_mod.mkdir()
    (my_mod / "__init__.py").write_text('''
MODALITY = "test/custom"
def build_router(registry):
    from fastapi import APIRouter
    return APIRouter()
''')
    monkeypatch.setenv("MUSE_MODALITIES_DIR", str(env_dir))

    # Run worker with discovery, verify custom modality was picked up
    from unittest.mock import patch
    with patch("muse.cli_impl.worker.uvicorn"):
        from muse.cli_impl.worker import run_worker, _modality_dirs
        dirs = _modality_dirs()
        # env dir is in the list
        assert env_dir in dirs
```

- [ ] **Step 3: Run tests + Commit**

```bash
pytest tests/cli_impl/test_worker.py -v
git add -A
git commit -m "feat(worker): scan \$MUSE_MODALITIES_DIR as escape hatch

Advanced users can define new modalities outside the bundled set by
dropping a subpackage in a dir pointed at by MUSE_MODALITIES_DIR.
Not advertised as a primary extension surface — writing a modality
means writing a wire contract, which is architectural work."
```

---

## Part G — Docs + verify + merge

### Task G1: Write `docs/MODEL_SCRIPTS.md` + update CLAUDE.md + README.md

**Files:**
- Create: `docs/MODEL_SCRIPTS.md`
- Modify: `CLAUDE.md`
- Modify: `README.md`

No em-dashes (soul voice hook bans U+2014). Use colons, commas, periods, parens.

- [ ] **Step 1: Write `docs/MODEL_SCRIPTS.md`** (~120 lines)

```markdown
# Writing a muse model script

A muse model script is a single Python file that declares one model.
Drop it in `~/.muse/models/` (or wherever `$MUSE_MODELS_DIR` points)
and muse picks it up at next startup.

## Minimum viable script

```python
# ~/.muse/models/my_embedder.py
"""My custom embedder: 512-dim sentence embeddings."""
from muse.modalities.embedding_text import EmbeddingResult


MANIFEST = {
    "model_id": "my-embedder-v1",
    "modality": "embedding/text",
    "hf_repo": "my-org/my-embedder",
    "description": "My custom sentence embedder, 512 dims",
    "license": "Apache 2.0",
    "pip_extras": ("torch>=2.1.0", "sentence-transformers>=2.2.0"),
    "capabilities": {
        "dimensions": 512,
        "context_length": 512,
    },
}


class Model:
    """Class name must be `Model` per discovery convention."""
    model_id = MANIFEST["model_id"]
    dimensions = 512

    def __init__(self, *, hf_repo, local_dir=None, device="auto", **_):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(local_dir or hf_repo, device=device)

    def embed(self, input, *, dimensions=None, **_) -> EmbeddingResult:
        texts = [input] if isinstance(input, str) else list(input)
        vectors = self._m.encode(texts, convert_to_numpy=True).tolist()
        return EmbeddingResult(
            embeddings=vectors,
            dimensions=len(vectors[0]),
            model_id=self.model_id,
            prompt_tokens=sum(len(t.split()) for t in texts),
        )
```

## MANIFEST schema

Required:
- `model_id: str` — unique, kebab-case by convention
- `modality: str` — MIME-style tag (e.g., `"audio/speech"`, `"embedding/text"`, `"image/generation"`); must match a discovered modality
- `hf_repo: str` — HuggingFace repo in `"namespace/name"` form

Optional (recommended):
- `description: str`
- `license: str` (SPDX-ish)
- `pip_extras: tuple[str, ...]` — packages installed in the model's venv on `muse pull`
- `system_packages: tuple[str, ...]` — warned about if missing from PATH
- `capabilities: dict[str, Any]` — free-form; surfaced in `/v1/models`

Recommended capability keys by modality:

| Modality | Recommended capabilities keys |
|---|---|
| `audio/speech` | `sample_rate`, `voices`, `languages` |
| `embedding/text` | `dimensions`, `context_length`, `matryoshka` |
| `image/generation` | `default_size`, `recommended_steps`, `supports_cfg` |

## The `Model` class

Must be named `Model`. Must satisfy the modality's runtime-checkable Protocol:

- `audio/speech`: `TTSModel` in `muse.modalities.audio_speech.protocol`
- `embedding/text`: `EmbeddingsModel` in `muse.modalities.embedding_text.protocol`
- `image/generation`: `ImageModel` in `muse.modalities.image_generation.protocol`

`__init__` signature: `(*, hf_repo, local_dir=None, device="auto", **_)`. The catalog loader calls with these kwargs.

## Scan order

Muse discovers models in this order, first-wins on model_id:
1. `src/muse/models/*.py` — bundled
2. `~/.muse/models/*.py` — user drop-in
3. `$MUSE_MODELS_DIR/*.py` — explicit override

A user script with a model_id matching a bundled one wins (and logs a warning).

## Error handling

Scripts that fail to import (missing deps, syntax error, etc.) are logged and skipped. Muse never refuses to start because one script broke. If a script is broken, `muse models list` won't show it; check logs.

## Writing a new modality

Rare. Modalities define wire contracts (Pydantic request shape, FastAPI router, codec). See `src/muse/modalities/audio_speech/` as a reference implementation. A modality subpackage must export `MODALITY: str` and `build_router: Callable[[ModalityRegistry], APIRouter]`. Drop your subpackage into `$MUSE_MODALITIES_DIR` to register without forking muse.
```

- [ ] **Step 2: Update CLAUDE.md**

In the "Architecture" section, add mentions of `muse.core.discovery`, `muse.modalities/`, `muse.models/`. Clarify that KNOWN_MODELS is now a function, not a dict.

- [ ] **Step 3: Update README.md**

In the "Architecture" section, update the package list to reflect the new layout (modalities, models).

- [ ] **Step 4: Commit**

```bash
git add docs/MODEL_SCRIPTS.md CLAUDE.md README.md
git commit -m "docs: MODEL_SCRIPTS.md guide + CLAUDE/README updates

MODEL_SCRIPTS.md walks through writing a model script with a full
example, MANIFEST schema, Model class requirements, scan order, and
error handling.

CLAUDE.md + README.md updated to reference the new muse/modalities/
and muse/models/ layout + discovery module."
```

---

### Task G2: Final verification sweep

Comprehensive run before merging.

- [ ] **Step 1: Fresh install**

```bash
cd /home/spinoza/github/repos/muse-plugin-discovery
pip uninstall -y muse 2>/dev/null || true
pip install -e ".[dev,server]"
```

- [ ] **Step 2: Full suite (not slow) with coverage**

```bash
pytest tests/ -q -m "not slow" --cov=muse --cov-report=term-missing 2>&1 | tail -40
```

- [ ] **Step 3: E2E slow test**

```bash
pytest tests/cli_impl/test_e2e_supervisor.py -v --timeout=60
```

- [ ] **Step 4: Import smokes**

```bash
python -c "from muse.core.discovery import discover_models, discover_modalities; print('ok')"
python -c "from muse.core.catalog import known_models; print(sorted(known_models()))"
python -c "from muse.modalities.audio_speech import MODALITY, build_router; print(MODALITY)"
python -c "from muse.modalities.embedding_text import MODALITY, build_router; print(MODALITY)"
python -c "from muse.modalities.image_generation import MODALITY, build_router; print(MODALITY)"
python -c "from muse.models.kokoro_82m import MANIFEST, Model; print(MANIFEST['model_id'])"
python -c "from muse.models.qwen3_embedding_0_6b import MANIFEST; print(MANIFEST['modality'])"
python -c "from muse.models.sd_turbo import MANIFEST; print(MANIFEST['modality'])"
```

- [ ] **Step 5: CLI smokes**

```bash
muse --help | head -15
muse models list
muse models info kokoro-82m
muse models list --modality "audio/speech"
```

Verify all work. `muse models list` should show all 7 bundled models with correct modality tags (`audio/speech`, `embedding/text`, `image/generation`).

- [ ] **Step 6: Gateway smoke**

```bash
muse serve --host 127.0.0.1 --port 8765 &
SERVER_PID=$!
sleep 4

curl -sf http://127.0.0.1:8765/health | python -m json.tool
curl -sf http://127.0.0.1:8765/v1/models | python -m json.tool

# Unknown-model 404 should produce OpenAI envelope
curl -s -X POST http://127.0.0.1:8765/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"hi","model":"no-such-model"}' \
  -w "\n---status: %{http_code}\n"

kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true
```

- [ ] **Step 7: Fix anything that fails**

Commit any fixes discovered.

- [ ] **Step 8: Report**

Verify `git log --oneline main..HEAD` shows ~18 commits, all focused. Report ready-for-merge status.

---

### Task G3: Merge back to main

```bash
cd /home/spinoza/github/repos/muse
git merge --no-ff feat/plugin-discovery-mime -m "feat: plugin-based model discovery + MIME-style modality tags

Reshapes muse around two discovery surfaces:

  muse/modalities/<name>/    bundled wire contracts (protocol+routes+codec+client)
  muse/models/<name>.py      flat drop-in scripts (MANIFEST + Model class)

Each modality's __init__.py exports MODALITY (MIME-style tag) and
build_router. Discovery scans both bundled dirs and user-provided
locations (~/.muse/models/, \$MUSE_MODELS_DIR, \$MUSE_MODALITIES_DIR)
and wires everything into the registry + catalog + worker.

Modality tags now use MIME form:
  'audio.speech'       → 'audio/speech'
  'embeddings'         → 'embedding/text'
  'images.generations' → 'image/generation'

New user extension surface: drop my_model.py into ~/.muse/models/
with MANIFEST + Model class; muse picks it up at next start with
zero source changes. docs/MODEL_SCRIPTS.md is the writing guide.

Code deletions:
  - KNOWN_MODELS hardcoded dict (replaced by known_models() function)
  - registry._extra() allowlist (replaced by manifest passthrough)
  - worker.py hardcoded modality imports (replaced by discover_modalities)
  - per-modality backends/ directories (models live in muse/models/)

See docs/plans/2026-04-14-plugin-discovery-mime-modalities.md."

pytest tests/ -q -m "not slow" 2>&1 | tail -3
git worktree remove ../muse-plugin-discovery
git branch -d feat/plugin-discovery-mime
git log --oneline -5
```

---

## Scope notes (deferred)

Intentionally NOT in this plan:

- **Hot reload**: muse re-scans discovery dirs and reconfigures running workers when catalog.json or model scripts change. Reuses this plan's discovery primitives. Separate plan.
- **`muse install <model-script-url>` / `muse install --from-hf <repo>`**: generate a model script from a URL or auto-populate one from HF metadata. Nice UX, separate plan.
- **`generic/json` fallback modality**: a bundled modality with a generic POST-JSON-in, POST-JSON-out shape for quick prototyping where users don't want to write a full modality subpackage. YAGNI for v1.
- **MANIFEST schema validation framework** (Pydantic etc.): plain dict access with clear error messages on missing required keys is sufficient for v1. Formalize the schema when conventions solidify.
- **`MUSE_STRICT_DISCOVERY=1`** env var to promote discovery warnings to errors: useful in CI; add when someone actually asks.
- **Class names other than `Model`**: we considered `MANIFEST["class"]` as an override field; deferred until someone wants it.

---

## Self-review

**Spec coverage:**
- Move modality trees + MIME rename: A1, A2, A3, A4 ✅
- Discovery primitives: B1, B2 ✅
- Flat model scripts with MANIFEST: C1, C2, C3, C4 ✅
- Catalog + registry use discovery: D1, D2 ✅
- Worker auto-mount: E1 ✅
- User-dir + env-var support: F1, F2 ✅
- Documentation: G1 ✅
- Verification + merge: G2, G3 ✅

**Placeholder scan:** Step bodies sometimes say "Copy from current file" rather than inlining every line — this is deliberate for files we're preserving behavior of (e.g., Soprano's synthesize). The engineer reads the current file and reproduces it verbatim in the new location. Not a placeholder, just efficient delegation; the test suite catches any behavior drift.

**Type consistency:**
- `MANIFEST` is always a top-level dict in model scripts. Required keys uniform across tasks.
- `MODALITY` is always a top-level str in modality `__init__.py`. Tag values consistent with Phase A renames.
- `discover_models` returns `dict[str, DiscoveredModel]`, `discover_modalities` returns `dict[str, Callable]` — consistent across B1 implementation and all downstream consumers (D1, E1, F1, F2).
- `ModelInfo.manifest` field name consistent across D2 implementation, D2 server-update, and registry tests.

Plan complete. 18 tasks across 7 phases.
