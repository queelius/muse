# CI test lane + preflight guard - design

**Date:** 2026-06-24
**Status:** approved (design); pending implementation plan
**Author:** Alex Towell (with Claude Opus 4.8)

## Problem

muse's full unit/integration-contract test suite (`pytest -m "not slow"`,
~1000 tests) is **never run by CI**. The repository's only GitHub Actions
workflow is `fresh-venv-smoke.yml`, which loads a matrix of lightweight
bundled models in fresh per-model venvs - it validates the install/load
path, not the test suite. As a result the fast lane is gated **solely by a
manual local run** during the release ritual.

This caused a concrete failure: the v0.46.1 resolver test
`test_repo_info_repository_not_found_surfaces_without_retry` constructed
`RepositoryNotFoundError("...")` positionally. When `huggingface_hub`
floated up to 1.20.1 (its `HfHubHTTPError.__init__` gained a required
keyword-only `response`), the test began raising `TypeError` at setup - it
was **silently red across two releases** (v0.46.1, v0.47.0) and was caught
only by a later multi-agent code review (fixed in v0.47.1).

Two compounding factors:

1. **No automated suite run.** Nothing re-runs the suite when code or
   floated dependencies change.
2. **Local venv drift masks failures.** During the v0.47.1 work the dev
   venv had drifted to bare (no `torch`/`uvicorn`/`pytest-asyncio`), so the
   full lane could not even be run locally; verification fell back to the
   subset of tests that import without the ML stack. A drifted venv lets a
   release be "verified" while most of the suite never executes.

## Goal

Close both gaps:

- Run `pytest -m "not slow"` automatically on every push to `main` and every
  PR, against the current (floated) dependency versions.
- Make the local pre-release verification refuse to pass in a venv that
  cannot actually run the lane, so a release cannot be falsely "verified" in
  a bare venv.

## Decisions (resolved during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| CI scope | **Single job, Python 3.11** | Lean; one source of truth. No cross-version matrix. |
| Dependency versions | **Float on latest** (no lockfile) | Surfaces upstream drift (the hf_hub class of break) on the next push - the signal that was missing. |
| Local hygiene | **Preflight guard script** | Directly prevents the "verified in a bare venv" false-green. |

## Components

### A. CI test workflow - `.github/workflows/tests.yml`

A new workflow, sibling to `fresh-venv-smoke.yml`, reusing its proven shape
(CPU torch index, pip cache, same triggers).

- **Triggers:** `push: branches: [main]`, `pull_request: branches: [main]`,
  `workflow_dispatch`.
- **Runner:** `ubuntu-latest`, single job, `timeout-minutes: 20`.
- **Python:** 3.11 (via `actions/setup-python@v5`).
- **Install:**
  `pip install -e ".[dev,server,audio,images,embeddings]"` with
  `PIP_EXTRA_INDEX_URL: https://download.pytorch.org/whl/cpu` so torch
  resolves to the CPU wheel. These extras provide every heavy lib the
  non-mocked test paths import (torch, transformers, diffusers,
  sentence-transformers, scipy, Pillow, accelerate, fastapi, uvicorn,
  httpx, psutil, pytest-asyncio, etc.). Model tests are mocked, so **no
  model weights are downloaded** in this lane.
- **Cache:** `actions/cache@v4` on `~/.cache/pip` keyed by
  `hashFiles('pyproject.toml')` with a restore-key prefix. (No HF-weight
  cache - the fast lane downloads no weights.)
- **Run:** `pytest -m "not slow"`. This excludes the one slow e2e
  supervisor subprocess test and the opt-in `tests/integration/` suite
  (which auto-skips without `MUSE_REMOTE_SERVER`).
- **Float:** no constraints file. CI installs current PyPI versions each
  run, so an upstream breaking release fails the next push/PR. (Accepted
  cost: a dependency release can turn `main` red on a commit that did not
  change code - that is the intended drift signal.)

**Validation:** the workflow is validated by its own first CI run after the
green baseline (Component below) is established. If a test needs a dep not
covered by those extras, or is currently red on latest deps, it is fixed as
part of landing this change (fix-forward).

### B. Preflight guard - `scripts/preflight.py`

A small standalone script (no heavy imports at module top) that gates the
local pre-release verification.

Behavior:

1. **Dependency check.** An explicit list of `(import_name, extra, package)`
   sentinels covering the fast-lane requirements:
   `torch` (audio), `transformers` (audio), `scipy` (audio),
   `diffusers` (images), `PIL`/Pillow (images),
   `sentence_transformers` (embeddings),
   `fastapi` / `uvicorn` / `httpx` / `psutil` (server),
   `pytest_asyncio` (dev), plus core `numpy` / `yaml`. This list is
   representative; the exact set (including tricky-named server deps such as
   `multipart` from python-multipart and `pynvml` from nvidia-ml-py) is
   finalized against the green-baseline run, which reveals precisely which
   imports the lane requires.
   Each sentinel is attempted via `importlib.import_module`. Any failure is
   collected (not fail-fast) so the report lists every missing dep at once,
   then the script exits non-zero printing the exact remediation command:
   `pip install -e ".[dev,server,audio,images,embeddings]" --extra-index-url https://download.pytorch.org/whl/cpu`.
2. **Run the lane.** If all sentinels import, run
   `pytest -m "not slow"` (via `subprocess.run([sys.executable, "-m", "pytest", ...])`,
   forwarding any extra CLI args) and exit with pytest's return code.

Flags:

- `--check-only`: perform step 1 only (verify deps), skip the test run.
  Exit 0 if the venv is fast-lane-ready, non-zero otherwise.
- Trailing args after `--` are forwarded to pytest (e.g.
  `python scripts/preflight.py -- -k resolver`).

The sentinel list is an honest, maintained mapping rather than a derivation
from pyproject (package name != import name: `Pillow`->`PIL`,
`sentence-transformers`->`sentence_transformers`,
`python-multipart`->`multipart`, `nvidia-ml-py`->`pynvml`). Each entry names
its extra so the error message is actionable.

**Release-ritual integration:** the documented release flow replaces its
bare `pytest -m "not slow"` step with `python scripts/preflight.py`. A
release performed in a drifted venv now stops at preflight with a clear
remediation, instead of silently running a partial suite.

### C. Documentation

- **CLAUDE.md:**
  - In the "Fresh-venv smoke test (CI)" section (or a renamed "CI" section),
    note that `tests.yml` now runs the `pytest -m "not slow"` suite on
    push/PR, distinct from the per-model smoke matrix.
  - In "Development commands", document the canonical full-extras install
    (`pip install -e ".[dev,server,audio,images,embeddings]" --extra-index-url https://download.pytorch.org/whl/cpu`)
    and `python scripts/preflight.py`.
  - Update the release ritual to call `scripts/preflight.py` for the test
    step.
- **README.md (optional):** add a CI status badge for `tests.yml`.

## Establishing the green baseline (first implementation step)

Before `tests.yml` can land green, install the full CPU extras locally and
run `pytest -m "not slow"` to surface ALL current rot on latest deps - not
just the already-fixed resolver test. Fix whatever is red (additional test
fixes, or a missing dep added to the install set / a genuinely
environment-bound test marked appropriately). Only then add the workflow, so
its first run on `main` is green.

## Testing strategy

- **`scripts/preflight.py`** gets `tests/test_preflight.py`:
  - missing-dep path: monkeypatch the import to fail for one sentinel ->
    assert non-zero exit and that the remediation command string is emitted.
  - `--check-only` happy path: all sentinels present -> exit 0, pytest not
    invoked (patch `subprocess.run` and assert not called).
  - arg forwarding: trailing args reach the pytest invocation.
- **`tests.yml`** is validated by its first CI run (and by the local green
  baseline established above).

## Success criteria

1. A push to `main` (or a PR) triggers `tests.yml`, which installs the CPU
   stack and runs `pytest -m "not slow"` to a green result.
2. `python scripts/preflight.py` exits 0 and runs the lane in a fully-set-up
   venv; in a bare venv it exits non-zero naming the missing extras and the
   exact install command, without running a partial suite.
3. The release ritual in CLAUDE.md routes its test step through preflight.
4. A future dependency drift that breaks a test is caught by the next CI run
   rather than shipping silently.

## Non-goals (YAGNI)

- No Python version matrix (3.11 only).
- No dependency pinning / lockfile / constraints file.
- No tiered `requires_ml` core/full split or per-test ML markers.
- The slow e2e supervisor test and opt-in integration suite are NOT run in
  CI (subprocess- and live-server-bound respectively).
- No GPU CI (everything runs CPU-mocked).
- No branch-protection / required-status-check changes (a repo setting, not
  a workflow file; the direct-to-main flow has no PRs to block, so the value
  is a visible red run on `main`).
