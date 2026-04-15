# Writing a muse model script

A muse model script is a single Python file that declares one model.
Drop it into `~/.muse/models/` (or any directory referenced by
`$MUSE_MODELS_DIR`) and muse picks it up on the next start.

There is no plugin API to learn, no registration call to make, and
no muse source file to edit. Discovery is pure file scanning.

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
    """Class name must be `Model` per the discovery convention."""
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

- `model_id: str` - unique, kebab-case by convention
- `modality: str` - MIME-style tag (e.g. `"audio/speech"`, `"embedding/text"`, `"image/generation"`); must match a discovered modality
- `hf_repo: str` - HuggingFace repo in `"namespace/name"` form

Optional (recommended):

- `description: str` - one-line summary; shown by `muse models list`
- `license: str` - SPDX-ish string
- `pip_extras: tuple[str, ...]` - packages installed into the model's venv on `muse pull`
- `system_packages: tuple[str, ...]` - warned about at pull time if missing from `PATH`
- `capabilities: dict[str, Any]` - free-form metadata; splatted into `/v1/models` entries

Recommended capability keys by modality:

| Modality | Recommended `capabilities` keys |
| --- | --- |
| `audio/speech` | `sample_rate`, `voices`, `languages` |
| `embedding/text` | `dimensions`, `context_length`, `matryoshka` |
| `image/generation` | `default_size`, `recommended_steps`, `supports_cfg` |

Unknown keys pass through harmlessly. Nothing validates against a
schema; consumers look up the keys they care about.

## The `Model` class

- Class name must be exactly `Model` (alias at the import site if you
  want a friendlier local name: `from muse.models.kokoro_82m import Model as KokoroModel`).
- Must satisfy the modality's runtime-checkable Protocol:
  - `audio/speech` - `TTSModel` in `muse.modalities.audio_speech.protocol`
  - `embedding/text` - `EmbeddingsModel` in `muse.modalities.embedding_text.protocol`
  - `image/generation` - `ImageModel` in `muse.modalities.image_generation.protocol`
- Constructor signature: `__init__(self, *, hf_repo, local_dir=None, device="auto", **_)`. The catalog loader calls with these kwargs; `**_` absorbs future additions without breaking existing scripts.
- Prefer `local_dir` over `hf_repo` when loading weights so HF cache reuse Just Works.
- Heavy imports (`torch`, `transformers`, `diffusers`, etc.) stay inside `__init__` or function bodies, not at module top-level. This keeps `muse --help` instant and lets `muse pull` work on a machine that hasn't installed the model's deps yet.

## Scan order

Muse discovers models in this order, first-wins on `model_id`:

1. `src/muse/models/*.py` - bundled with muse
2. `~/.muse/models/*.py` - per-user drop-in
3. `$MUSE_MODELS_DIR/*.py` - explicit override (optional)

Bundled models shadow user scripts that declare the same `model_id`;
collisions are logged. To replace a bundled model, rename it or remove
the bundled script. (This is a one-time operation; future bundled
updates won't silently reintroduce your id.)

Files starting with `_` (including `__init__.py`) are ignored. Any
script that fails to import (missing deps, syntax error, unresolved
reference) is logged and skipped - muse never refuses to start because
one script broke. If a script is broken, `muse models list` won't show
it; check logs for `skipping model script ...: import failed`.

## Writing a new modality

Rare. Modalities define wire contracts (Pydantic request shape, FastAPI
router, codec) and are shared across many models. See
`src/muse/modalities/audio_speech/` as a reference implementation.

A modality subpackage must:

- live in `src/muse/modalities/<mime_name>/` (bundled) or
  `$MUSE_MODALITIES_DIR/<mime_name>/` (escape hatch)
- export a module-level `MODALITY: str` (the MIME-style tag)
- export a module-level `build_router: Callable[[ModalityRegistry], APIRouter]`

Drop the subpackage into `$MUSE_MODALITIES_DIR` to register without
forking muse. The env-var approach is intentionally undocumented in
user-facing help: most users should extend via model scripts instead.

## Verifying your script

```bash
# Confirm discovery finds it without importing the heavy deps
python -c "from pathlib import Path; from muse.core.discovery import discover_models; \
  print(sorted(discover_models([Path.home() / '.muse' / 'models'])))"

# Register + install deps + download weights
muse pull my-embedder-v1

# Run it in the supervisor
muse serve
```

If `muse models list` doesn't include your model, run with
`PYTHONWARNINGS=default` and check the logs for the `skipping model script`
warning.
