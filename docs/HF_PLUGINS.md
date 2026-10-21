# Authoring HF resolver plugins

A plugin teaches muse's HF resolver how to recognize, resolve, and
search a particular model shape on HuggingFace. Each modality
contributes one plugin file; the resolver discovers them at startup.

## File location

`src/muse/modalities/<name>/hf.py` (bundled) or `<MUSE_MODALITIES_DIR>/<name>/hf.py` (user-contributed).

## The plugin contract

Top-level `HF_PLUGIN: dict` with these keys:

| Key | Type | Purpose |
|---|---|---|
| `modality` | `str` | MIME tag (must match the modality's `MODALITY` constant) |
| `runtime_path` | `str` | `"muse.modalities.X.runtimes.Y:Cls"` |
| `pip_extras` | `tuple[str, ...]` | per-model venv install args |
| `system_packages` | `tuple[str, ...]` | apt/brew packages required (may be empty) |
| `priority` | `int` | lower checked first; 100 for specific shapes, 200+ for tag-only catch-alls |
| `sniff` | `Callable[[info], bool]` | True iff this plugin claims `info` |
| `resolve` | `Callable[[repo_id, variant, info], ResolvedModel]` | build manifest + download closure |
| `search` | `Callable[[api, query, *, sort, limit], Iterable[SearchResult]]` | yield rows for `muse search` |

## Authoring rules

`hf.py` is loaded as a single-file module via `spec_from_file_location`,
bypassing the modality package's `__init__.py`. This keeps `muse pull`
working on a bare install (no fastapi). The cost: relative imports
(`from .protocol import ...`) and absolute sibling imports
(`from muse.modalities.X.codec import ...`) both fail because the
parent package is not initialized.

What you may import:
- stdlib
- `huggingface_hub` (base dep)
- `muse.core.*` (lightweight: resolvers, chat_formats, errors, and
  `hf_plugin_helpers` for the repeated id/license/search/allow-list code)

What you may NOT import:
- relative siblings (`.protocol`, `.codec`, `.routes`)
- absolute siblings (`muse.modalities.X.protocol`)
- heavy deps (torch, transformers, fastapi, llama_cpp)

## Priority conventions

| Range | When to use |
|---|---|
| 100 | File-pattern + tag (very specific). Examples: GGUF (`.gguf` siblings), CT2 (model.bin + config.json + ASR tag) |
| 110 | Tag OR config-file (medium-specific). Example: sentence-transformers |
| 200 | Tag-only (broad catch-all). Example: text-classification |

If two plugins ever sniff True on the same repo, lower priority wins.
Same priority resolves alphabetically by modality tag. The intent: the
narrower shape always wins.

## Example skeleton

```python
"""HF resolver plugin for <my modality>."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, snapshot_download

from muse.core.hf_plugin_helpers import (
    hf_model_id, hf_repo_license, hf_tag_search, transformers_allow_patterns,
)
from muse.core.resolvers import ResolvedModel, SearchResult


_RUNTIME_PATH = "muse.modalities.my_modality.runtimes.my_runtime:MyModel"
_PIP_EXTRAS = ("my-dep>=1.0",)


def _sniff(info) -> bool:
    tags = getattr(info, "tags", None) or []
    return "my-task-tag" in tags


def _resolve(repo_id: str, variant: str | None, info) -> ResolvedModel:
    manifest = {
        "model_id": hf_model_id(repo_id),
        "modality": "my/modality",
        "hf_repo": repo_id,
        "description": f"My modality: {repo_id}",
        "license": hf_repo_license(info),
        "pip_extras": list(_PIP_EXTRAS),
        "system_packages": [],
        "capabilities": {},
    }
    # fp16 safetensors when present, else safetensors; *.bin only when the
    # repo has no safetensors at all.
    allow_patterns = transformers_allow_patterns(info)

    def _download(cache_root: Path) -> Path:
        return Path(snapshot_download(
            repo_id=repo_id, allow_patterns=allow_patterns,
            cache_dir=str(cache_root) if cache_root else None,
        ))
    return ResolvedModel(
        manifest=manifest, backend_path=_RUNTIME_PATH, download=_download,
        # Lets the pull seed only these files from the shared HF cache.
        allow_patterns=tuple(allow_patterns),
    )


def _search(api: HfApi, query: str, *, sort: str, limit: int) -> Iterable[SearchResult]:
    return hf_tag_search(
        api, query, sort=sort, limit=limit, tag="my-task-tag", modality="my/modality",
    )


HF_PLUGIN = {
    "modality": "my/modality",
    "runtime_path": _RUNTIME_PATH,
    "pip_extras": _PIP_EXTRAS,
    "system_packages": (),
    "priority": 100,
    "sniff": _sniff,
    "resolve": _resolve,
    "search": _search,
}
```

## Testing

Each plugin should ship a `tests/modalities/<name>/test_hf_plugin.py`
covering:
- `HF_PLUGIN` has all required keys (`REQUIRED_HF_PLUGIN_KEYS` in
  `muse.core.discovery`)
- metadata correctness (modality, runtime_path, priority)
- `sniff` returns True on a positive synthetic info; False on a negative
- `resolve` returns a `ResolvedModel` with the right manifest shape
- `search` yields `SearchResult` instances with the right modality tag

Use `unittest.mock.MagicMock` for `info` and `api`; no real network
calls in unit tests.
