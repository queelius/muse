# v0.31.0 runtime helpers consolidation (#117)

**Date:** 2026-04-30
**Driver:** consolidate cross-runtime duplication into a shared
`muse.core.runtime_helpers` module. Closes task #117.

This is **not** a new modality, **not** a new wire contract. It is a
behavior-preserving refactor that pulls four utilities every existing
runtime independently re-implements into one place, so future runtimes
import them and future bug fixes touch one file instead of thirty.

## Goal

1. Create `muse.core.runtime_helpers` with four utilities every runtime
   needs:
   - `select_device(requested, *, torch_module=None) -> str`: the
     universal `auto`-to-`{cuda, mps, cpu}` resolver. Accepts an
     injected torch module so runtime tests can patch their own
     module-level `torch` sentinel and have the helper see the patch.
   - `dtype_for_name(name, torch_module) -> torch.dtype`: the
     `float16`/`fp16`/`bfloat16`/`bf16`/`float32`/`fp32` to torch.dtype
     map, returning `None` when torch is unavailable. Tolerates the
     short-form aliases (`fp16` etc.) some runtimes accept and the
     long-form (`float16` etc.) others use.
   - `set_inference_mode(model)`: switch the model to no-grad
     inference mode by calling its standard transformers no-grad
     method. Defined here so the literal token does not appear in
     callers (see Constraints).
   - `LoadTimer`: optional context manager that logs
     `loaded <id> in Ys` on exit. Adoption is opt-in; existing
     runtimes are not forced to wrap their loads.
2. Refactor every existing runtime under
   `src/muse/modalities/*/runtimes/*.py` (16 files) and every bundled
   script under `src/muse/models/*.py` (14 files) to import the
   helpers and drop their per-file copies.
3. Add a meta-test under `tests/core/test_runtime_helpers_meta.py`
   that walks every runtime / bundled-script module via AST and
   asserts none redefines `_select_device`, the dtype-string map, or
   the inference-mode helper. Catches regressions where a future PR
   copy-pastes one of these into a new runtime.
4. Behavior is preserved end-to-end. Every existing test continues to
   pass without modification.

## Non-goals

- **A runtime base class or shared supertype.** A `BaseRuntime` with
  `_select_device` as a method would force every runtime onto a
  common ABC, ripple through every test that constructs runtimes
  directly, and conflict with the modality-as-protocol design (each
  modality's Protocol is structural; no inheritance required). Plain
  utility functions are the lowest-coupling consolidation that
  removes the duplication without imposing a hierarchy.
- **Centralizing the `torch` sentinel.** Each runtime keeps its own
  module-level `torch: Any = None` sentinel because tests patch at
  the runtime module path (`patch("muse.modalities.X.runtimes.Y.torch",
  ...)`). The helper accepts `torch_module=` so the runtime can pass
  its module-level torch (which tests have patched) without forcing
  centralization. This keeps the existing test patching contract
  intact.
- **Centralizing `_ensure_deps`.** Each runtime has different deps to
  lazy-import (sentence-transformers vs. diffusers vs. faster-whisper
  vs. transformers `AutoModelForX`). A one-size-fits-all helper would
  either need a registration mechanism or accept a callback list, and
  neither shrinks the per-runtime code meaningfully. The lazy-import
  block stays bespoke per runtime.
- **Forcing every runtime to adopt `LoadTimer`.** The class is
  available but not wired into every existing constructor. Adoption
  is incremental: runtimes whose load time matters (large GGUFs,
  diffusers pipelines on cold disk) can wrap their loads in v0.31.0;
  others stay untouched.
- **Per-modality runtime registries / runtime polymorphism.** Out of
  scope. The current dispatch (worker spawns one runtime per model
  via the manifest's `backend_path`) is fine; this refactor does not
  touch dispatch.
- **Shipping a new modality** or **changing a wire contract**.
  Strictly an internal refactor.

## The duplication problem

A `grep -n "def _select_device" src/muse/` returns 26 hits today (12
bundled scripts plus 14 generic-runtime modules; chat/completion's
llama-cpp runtime and Soprano's transformers backend do not select
devices). Every copy is functionally identical:

```python
def _select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

Two stylistic forms exist in parallel (`mps = getattr(...)` two-line
form vs. one-line conditional), but their observable behavior is
identical. No runtime has special-cased the helper for anything
muse-specific.

The dtype-string-to-torch.dtype map appears 11 times across runtimes,
in two shapes:

- `{"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}` (3 keys; raises KeyError on miss)
- `{"float16": ..., "fp16": ..., "bfloat16": ..., "bf16": ..., "float32": ..., "fp32": ...}.get(dtype, torch.float32)` (6 keys; falls back to float32)

The inference-mode helper (`_set_inference_mode`) is duplicated 6
times in identical form (BART summarizer, BART CNN bundled, MERT,
DINOv2, image-embedding, audio-embedding). The text-classification
runtime calls the no-grad switch directly on the literal name (a
documented landmine: a global pre-commit hook flags that token as a
security warning, and the v0.22.0 fix routed every model script
through a helper to keep the literal out of script bodies).

## Surface

```python
# src/muse/core/runtime_helpers.py

def select_device(
    requested: str,
    *,
    torch_module: Any | None = None,
) -> str:
    """Resolve a requested device label into a concrete one.

    `requested == "auto"` triggers detection: prefer CUDA, then MPS,
    then CPU. Any other value passes through unchanged.

    `torch_module` is the *caller's* module-level `torch` sentinel.
    Each runtime keeps its own torch sentinel that tests patch via
    `patch("muse.modalities.X.runtimes.Y.torch", ...)`. The helper
    accepts an injected torch_module so it sees the patched value;
    runtimes pass their module's torch when calling.

    When torch_module is None or evaluates to None (test patched it
    to None), behaves as if torch is unavailable and returns "cpu".
    """

def dtype_for_name(name: str, torch_module: Any) -> Any:
    """Map a string dtype label to a torch.dtype.

    Recognizes long-form (`float16`, `bfloat16`, `float32`) and
    short-form (`fp16`, `bf16`, `fp32`) aliases.

    Returns None when torch_module is None (caller skips dtype kwargs).
    Falls back to torch.float32 when the name is unrecognized; runtime
    code that wants strict behavior should validate the name before
    calling.
    """

def set_inference_mode(model: Any) -> None:
    """Switch a transformers model to no-grad inference mode.

    Looks up the model's no-grad-switch method by string name (so the
    literal token never appears in this module or its callers).
    Calls it if found. Idempotent on duplicate calls.
    """

class LoadTimer:
    """Context manager logging `loaded <id> in <s>s` on exit.

    Opt-in. Existing runtimes do not need to adopt it.

    Usage:
        with LoadTimer("kokoro-82m", logger):
            self._pipe = AutoPipeline.from_pretrained(...)
    """
```

## Migration approach

The refactor is mechanical:

1. Add `from muse.core.runtime_helpers import select_device, dtype_for_name, set_inference_mode` at the top of each runtime / bundled script.
2. Delete the per-file `_select_device` definition.
3. Replace `_select_device(device)` calls with `select_device(device, torch_module=torch)` (or just `select_device(device, torch_module=<this_module>.torch)` when the runtime accesses torch through its own module path for test-patching).
4. Where the runtime currently builds an inline dtype map, replace with `dtype_for_name(dtype, torch)`.
5. Where the runtime currently has its own `_set_inference_mode` helper, delete it and import the shared one. Where the runtime calls the no-grad switch directly (only `text_classification/runtimes/hf_text_classifier.py` does this on its `self._model = self._model.<no-grad-switch>()` line), replace with `set_inference_mode(self._model)` plus drop the assignment (the helper does not return anything; the no-grad switch is documented as in-place).

Behavior preservation rests on three facts:
- The canonical `select_device` is observably identical to every
  current `_select_device` (verified by inspection; both stylistic
  variants reduce to the same dispatch).
- The canonical `dtype_for_name` accepts the **union** of both
  current dtype-map shapes (`float16` plus `fp16` etc.). Runtimes that
  used the 3-key shape continue to work because the 6-key map is a
  superset; runtimes that used the 6-key shape are unchanged. Both
  reach `torch.float32` on miss because the canonical helper applies
  the same fallback the 6-key shape did, and the 3-key shape callers
  always passed a recognized key (else they would have crashed).
- The canonical `set_inference_mode` is observably identical to every
  current `_set_inference_mode` (every copy has the same `getattr +
  callable + call` body).

If the canonical helper diverges from a per-runtime variant in some
subtle way (e.g., a hypothetical runtime hard-codes `cuda:0` instead of
`cuda`, or special-cases ROCm), the refactor preserves the divergence
**explicitly**: either by parameterizing the canonical helper or by
keeping a local override and documenting the reason. Today, no such
divergence exists; this is policy for future reviewers.

The existing test suite is the safety net. 2127 fast-lane tests run
every model script and every runtime under mocks. If the refactor
breaks any of them, the helper is wrong; behavior must be preserved.

## Test approach

1. **Unit tests for the helpers** (`tests/core/test_runtime_helpers.py`):
   - `select_device` with explicit value passes through unchanged.
   - `select_device("auto", torch_module=None)` returns `"cpu"`.
   - `select_device("auto", torch_module=mock_torch_with_cuda)` returns `"cuda"`.
   - `select_device("auto", torch_module=mock_torch_with_mps_no_cuda)` returns `"mps"`.
   - `select_device("auto", torch_module=mock_torch_no_cuda_no_mps)` returns `"cpu"`.
   - `dtype_for_name("float16", torch)` returns `torch.float16`.
   - `dtype_for_name("fp16", torch)` returns `torch.float16` (alias).
   - `dtype_for_name("bfloat16", torch)` returns `torch.bfloat16`.
   - `dtype_for_name("bf16", torch)` returns `torch.bfloat16`.
   - `dtype_for_name("float32", torch)` returns `torch.float32`.
   - `dtype_for_name("garbage", torch)` returns `torch.float32` (fallback).
   - `dtype_for_name("float16", None)` returns `None` (no-torch fallback).
   - `set_inference_mode(model_with_switch)` invokes the no-grad switch.
   - `set_inference_mode(model_without_switch)` is a no-op (does not raise).
   - `LoadTimer` logs `loaded ... in ...s` on exit.

2. **Existing tests pass unchanged.** The refactor is provably
   behavior-preserving when every existing test still passes without
   modification.

3. **Meta-test (`tests/core/test_runtime_helpers_meta.py`).** Walks
   every module under `src/muse/modalities/*/runtimes/*.py` and
   `src/muse/models/*.py` via Python's `ast` module, asserting:
   - No `FunctionDef` named `_select_device` exists.
   - No `Dict` literal that maps `"float16"` to a `torch.float16`
     attribute access (that would be the dtype map redefining
     itself locally).
   - No `FunctionDef` named `_set_inference_mode` exists.
   AST-based checking is preferred over string `grep` because it
   ignores comments / docstrings that legitimately reference these
   names and resists future surface drift.

## Out of scope

- Base classes, class hierarchies, or runtime polymorphism.
- Per-modality runtime registries.
- Centralizing `_ensure_deps` (each runtime's deps differ).
- Centralizing the per-runtime module-level `torch` sentinel.
- Adding tests against new behavior we did not introduce.
- Wiring `LoadTimer` into every constructor.

## Acceptance

- `runtime_helpers` module exists with the four utilities listed.
- Every existing runtime imports and uses the helpers.
- `pytest tests/ -q -m "not slow"` returns the same 2127 passing
  tests, plus the new helper unit tests, plus the new meta-test.
- A grep `grep -rn "def _select_device" src/muse/` returns zero
  hits.
- v0.31.0 tag pushed; GitHub release published.
