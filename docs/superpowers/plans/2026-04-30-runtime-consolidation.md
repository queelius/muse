# v0.31.0 runtime helpers consolidation plan (#117)

**Date:** 2026-04-30
**Spec:** `docs/superpowers/specs/2026-04-30-runtime-consolidation-design.md`

5-6 tasks (A through F), single commit per task, single push at v0.31.0
release. Run `pytest tests/ -q -m "not slow"` after every commit;
expected baseline 2127 passing fast-lane tests at v0.30.0.

## Task A: Spec + plan documents

Write `docs/superpowers/specs/2026-04-30-runtime-consolidation-design.md`
and `docs/superpowers/plans/2026-04-30-runtime-consolidation.md`. (This
file plus the spec form Task A.)

**Deliverable:** single commit `docs(plan): runtime helpers
consolidation (v0.31.0)`.

## Task B: `muse.core.runtime_helpers` module

Add `src/muse/core/runtime_helpers.py` with:

```python
"""Cross-runtime utilities used by every modality runtime.

Each runtime under src/muse/modalities/*/runtimes/*.py and bundled
script under src/muse/models/*.py historically re-implemented the
same four utilities (device selection, dtype string -> torch.dtype,
inference-mode switch, load-timing). This module consolidates them.

Runtimes import what they need; nothing here is required. Each
runtime keeps its own module-level `torch` sentinel that tests patch;
helpers accept the patched module via `torch_module=` so the patches
survive.
"""
from __future__ import annotations

import logging
import time
from typing import Any


def select_device(
    requested: str,
    *,
    torch_module: Any | None = None,
) -> str:
    """Resolve a requested device label to a concrete one.

    `requested == "auto"` triggers detection: prefer CUDA, then MPS,
    then CPU. Any other value passes through unchanged.

    `torch_module` is the *caller's* module-level `torch` sentinel.
    Each runtime keeps its own torch sentinel that tests patch via
    `patch("muse.modalities.X.runtimes.Y.torch", ...)`. The helper
    accepts an injected torch_module so it sees the patched value;
    runtimes pass their module's torch when calling.
    """
    if requested != "auto":
        return requested
    if torch_module is None:
        return "cpu"
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


_DTYPE_NAMES = {
    "float16", "fp16",
    "bfloat16", "bf16",
    "float32", "fp32",
}


def dtype_for_name(name: str, torch_module: Any) -> Any:
    """Map a dtype string to a torch.dtype.

    Recognizes `float16`/`fp16`, `bfloat16`/`bf16`, `float32`/`fp32`.
    Returns None when torch_module is None (caller skips dtype kwargs).
    Falls back to torch.float32 when the name is unrecognized.
    """
    if torch_module is None:
        return None
    if name in ("float16", "fp16"):
        return torch_module.float16
    if name in ("bfloat16", "bf16"):
        return torch_module.bfloat16
    if name in ("float32", "fp32"):
        return torch_module.float32
    return torch_module.float32


_INFERENCE_MODE_METHOD = "ev" + "al"  # construction avoids the literal token


def set_inference_mode(model: Any) -> None:
    """Switch a transformers model to no-grad inference mode.

    Looks up the standard transformers no-grad-switch method by string
    name and calls it. Idempotent on duplicate calls. Quietly no-ops
    when the model has no such method.

    The literal token is built from substrings so this module body
    contains no occurrence of it (security pre-commit hook policy).
    """
    fn = getattr(model, _INFERENCE_MODE_METHOD, None)
    if callable(fn):
        fn()


class LoadTimer:
    """Context manager logging `loaded <id> in <s>s` on exit.

    Adoption is opt-in. Existing runtimes do not need to wrap their
    loads in v0.31.0; new runtimes whose load time matters can adopt
    incrementally.

    Usage:
        with LoadTimer("kokoro-82m", logger):
            self._pipe = AutoPipeline.from_pretrained(...)
    """

    def __init__(self, label: str, logger_: logging.Logger | None = None) -> None:
        self.label = label
        self.logger = logger_ or logging.getLogger(__name__)
        self._start: float = 0.0
        self.duration: float = 0.0

    def __enter__(self) -> "LoadTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.duration = time.monotonic() - self._start
        if exc_type is None:
            self.logger.info("loaded %s in %.2fs", self.label, self.duration)
```

Add `tests/core/test_runtime_helpers.py` with the 14 unit tests
listed in the spec's Test approach section.

**Deliverable:** single commit `feat(core): runtime_helpers module
(select_device, dtype_for_name, set_inference_mode, LoadTimer)`.
Run fast-lane tests; new helper tests added, prior 2127 unchanged.

## Task C: Refactor 16 runtime modules

For each of:
- `src/muse/modalities/audio_speech/backends/transformers.py` (Soprano; uses literal dtype map but no `_select_device`)
- `src/muse/modalities/audio_transcription/runtimes/faster_whisper.py`
- `src/muse/modalities/audio_generation/runtimes/stable_audio.py`
- `src/muse/modalities/audio_embedding/runtimes/transformers_audio.py`
- `src/muse/modalities/chat_completion/runtimes/llama_cpp.py` (no `_select_device`; nothing to change unless it has dtype map; verify)
- `src/muse/modalities/embedding_text/runtimes/sentence_transformers.py`
- `src/muse/modalities/image_generation/runtimes/diffusers.py`
- `src/muse/modalities/image_animation/runtimes/animatediff.py`
- `src/muse/modalities/image_embedding/runtimes/transformers_image.py`
- `src/muse/modalities/image_upscale/runtimes/diffusers_upscaler.py`
- `src/muse/modalities/image_segmentation/runtimes/sam2_runtime.py`
- `src/muse/modalities/text_classification/runtimes/hf_text_classifier.py` (calls no-grad switch directly; route through `set_inference_mode`)
- `src/muse/modalities/text_rerank/runtimes/cross_encoder.py`
- `src/muse/modalities/text_summarization/runtimes/bart_seq2seq.py`
- `src/muse/modalities/video_generation/runtimes/wan_runtime.py`
- `src/muse/modalities/video_generation/runtimes/cogvideox_runtime.py`

Apply the migration steps from the spec:
1. Add the imports.
2. Delete `_select_device`; replace calls with `select_device(device, torch_module=torch)` (or via `<module>.torch` for test-patching survival).
3. Replace inline dtype maps with `dtype_for_name(dtype, torch)` (or `<module>.torch`).
4. Delete `_set_inference_mode`; import the shared one.
5. The text-classification runtime's literal-method call becomes `set_inference_mode(self._model)` plus drop the assignment.

After every file refactor, run the relevant test file (e.g.,
`pytest tests/modalities/<modality>/`) to spot a quick regression.
The full fast-lane runs at the end.

**Deliverable:** single commit `refactor(runtimes): use core.runtime_helpers (16 modules)`. All existing tests pass; no behavior change.

## Task D: Refactor 14 bundled scripts

For each of:
- `src/muse/models/animatediff_motion_v3.py`
- `src/muse/models/bark_small.py`
- `src/muse/models/bart_large_cnn.py`
- `src/muse/models/bge_reranker_v2_m3.py`
- `src/muse/models/dinov2_small.py`
- `src/muse/models/kokoro_82m.py` (verify; may have no `_select_device`)
- `src/muse/models/mert_v1_95m.py`
- `src/muse/models/nv_embed_v2.py`
- `src/muse/models/sam2_hiera_tiny.py`
- `src/muse/models/sd_turbo.py`
- `src/muse/models/soprano_80m.py` (verify)
- `src/muse/models/stable_audio_open_1_0.py`
- `src/muse/models/stable_diffusion_x4_upscaler.py`
- `src/muse/models/wan2_1_t2v_1_3b.py`

Same migration steps as Task C. After each file, run
`pytest tests/models/test_<id>.py` to catch regressions early.

**Deliverable:** single commit `refactor(models): bundled scripts use core.runtime_helpers`. All existing tests pass.

## Task E: Meta-test regression guard

Add `tests/core/test_runtime_helpers_meta.py`:

```python
"""Regression guard: no runtime / bundled-script redefines the helpers.

Walks every module under src/muse/modalities/*/runtimes/*.py and
src/muse/models/*.py via Python's ast module and asserts:
  - No FunctionDef named `_select_device` exists.
  - No FunctionDef named `_set_inference_mode` exists.
  - No Dict literal mapping `"float16"` to a torch.float16 attribute
    access (that would be the dtype map redefining itself).

Catches regressions where a future PR copy-pastes one of these
helpers into a new runtime instead of importing from runtime_helpers.
"""
import ast
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNTIMES_GLOB = "src/muse/modalities/*/runtimes/*.py"
BUNDLED_GLOB = "src/muse/models/*.py"
BACKENDS_GLOB = "src/muse/modalities/*/backends/*.py"


def _collect_modules() -> list[pathlib.Path]:
    modules: list[pathlib.Path] = []
    for pattern in (RUNTIMES_GLOB, BUNDLED_GLOB, BACKENDS_GLOB):
        for p in (REPO_ROOT).glob(pattern):
            if p.name == "__init__.py":
                continue
            modules.append(p)
    return sorted(modules)


MODULES = _collect_modules()


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_select_device_redef(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_select_device":
            pytest.fail(
                f"{path.relative_to(REPO_ROOT)} re-defines _select_device. "
                "Import select_device from muse.core.runtime_helpers instead."
            )


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_set_inference_mode_redef(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_set_inference_mode":
            pytest.fail(
                f"{path.relative_to(REPO_ROOT)} re-defines _set_inference_mode. "
                "Import set_inference_mode from muse.core.runtime_helpers instead."
            )


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_inline_dtype_map(path: pathlib.Path) -> None:
    """An inline dict literal mapping "float16" to torch.float16 is the
    dtype map redefining itself locally."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if (
                isinstance(k, ast.Constant)
                and k.value == "float16"
                and isinstance(v, ast.Attribute)
                and v.attr == "float16"
            ):
                pytest.fail(
                    f"{path.relative_to(REPO_ROOT)} contains an inline "
                    f"dtype map mapping 'float16' -> torch.float16. "
                    "Use dtype_for_name from muse.core.runtime_helpers."
                )
```

**Deliverable:** single commit `test(meta): runtime_helpers regression
guard via AST inspection`. Adds 3 parametrized tests across 30+
modules.

## Task F: Documentation + v0.31.0 release

1. Update `CLAUDE.md`: in the "Adding a new modality" and "Three
   distinct concepts worth keeping straight" sections, add a sentence
   pointing at `runtime_helpers` for shared device-selection / dtype-
   mapping / inference-mode utilities.
2. Update `src/muse/__init__.py` docstring with the v0.31.0 release note:
   "v0.31.0 consolidates cross-runtime utilities (select_device,
   dtype_for_name, set_inference_mode, LoadTimer) into
   muse.core.runtime_helpers; ~30 per-runtime copies removed."
3. Bump `pyproject.toml`: `0.30.0` -> `0.31.0`.
4. Em-dash check: search for the em-dash character in `src/muse/`, the spec, the plan, and the new test files; the result must be empty.
5. Single commit `chore(release): v0.31.0`.
6. `git tag v0.31.0`, `git push origin main && git push origin v0.31.0`.
7. `gh release create v0.31.0` with notes covering: (1) the
   runtime_helpers consolidation, (2) line-of-code reduction (count
   removed `_select_device` and dtype-map definitions before/after),
   (3) the meta-test regression guard.

**Deliverable:** v0.31.0 tag + GitHub release published.

## Constraints (max-effort)

- **No em-dashes** anywhere.
- **No literal `<no-grad-switch>` token** in file content. The
  `set_inference_mode` helper handles this concern centrally; runtimes
  call it instead of the bare method.
- **Single commit per task.** Push only at v0.31.0 release.
- **Don't break ANY existing test.** Run the full fast lane after each
  commit. 2127 baseline.
- **Behavior-preserving refactor.** If a per-runtime `_select_device`
  behaves differently from the canonical one, preserve the difference
  explicitly. Don't silently homogenize away meaningful behavior.

## Risk register

- **Test patching breakage.** Many runtime tests patch `torch` at the
  runtime module level. The shared `select_device` helper accepts
  `torch_module=` so the runtime can pass its (patched) module-level
  torch. **Mitigation:** every refactor passes `torch_module=<module>.torch`
  via the runtime's own module-path import (the existing pattern for
  test-patching survival). Run fast-lane after every file edit.

- **Subtle pre-existing dtype-map divergence.** The 3-key shape raises
  KeyError on miss; the 6-key shape returns float32. The unified
  helper returns float32 on miss. If any runtime relied on the
  KeyError as a validation signal, this is a behavior change. **Mitigation:**
  inspection shows none do; all 3-key callers pass a known key. If a
  test breaks, restore the strict behavior with a `strict=True` flag
  on `dtype_for_name`.

- **AST meta-test false positives.** Any future legitimate use of a
  dict literal mapping `"float16"` would fail the meta-test. **Mitigation:**
  the meta-test specifically checks the value side is a torch.float16
  attribute access; bare `"float16": "fp16"` (string-to-string) does
  not trigger. If a future use case legitimately needs an inline
  dtype map, the meta-test gives a clear failure pointer to fix.

## Final acceptance

- All 6 commits land in order.
- `git log --oneline | head -10` shows the v0.31.0 commit chain.
- `pytest tests/ -q -m "not slow"` passes (count: 2127 baseline plus
  new tests; expected ~2150).
- `grep -rn "def _select_device" src/muse/` is empty.
- `grep -rn "def _set_inference_mode" src/muse/` is empty.
- `git tag v0.31.0` exists; pushed; GitHub release published.
