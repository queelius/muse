# CI test lane + preflight guard - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run muse's `pytest -m "not slow"` suite automatically on every push/PR (it is currently gated only by a manual local run), and add a local preflight guard so a release cannot be "verified" in a venv that cannot actually run the lane.

**Architecture:** Add a new GitHub Actions workflow `tests.yml` (single Python-3.11 job, full CPU extras, float on latest) that runs the fast lane; add `scripts/preflight.py` that asserts the fast-lane deps import before invoking pytest; update docs. Establish a green local baseline first so the workflow's first run on `main` is green.

**Tech Stack:** GitHub Actions, pytest, Python 3.11, CPU PyTorch wheels (`download.pytorch.org/whl/cpu`).

## Global Constraints

- **Commits are ASCII-only** - a soul pre-commit hook rejects non-ASCII (no em-dashes, no Unicode arrows; use `->` and `-`). Applies to commit messages AND committed file contents.
- **Direct-to-main** - no feature branches; commit to `main`. (A throwaway branch + PR is acceptable ONLY to get a pre-merge CI run of `tests.yml`; see Task 5.)
- **CPU torch** - always install with `PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu` so `torch` resolves to the CPU wheel.
- **Float on latest** - no constraints file / lockfile; CI installs current PyPI versions each run.
- **CI Python is 3.11**; the workflow is a single job (no matrix).
- **No PyPI release** - this work changes no packaged code (`scripts/` and `.github/` are not in the wheel), so the `museq` artifact is unchanged. Do NOT bump the version or publish.
- **Commit trailers** - end every commit message with:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01E5TNYLRLwZCH4K1Db4xTbe
  ```

## File Structure

| File | Responsibility |
|---|---|
| `scripts/preflight.py` (create) | Assert fast-lane deps import; then run `pytest -m "not slow"`. Stdlib-only at module top so it loads in a bare venv. |
| `tests/test_preflight.py` (create) | Unit tests for preflight (dep-check, `--check-only`, arg forwarding). Stdlib-only, runs even in a bare venv. |
| `.github/workflows/tests.yml` (create) | CI: install CPU extras, run the fast lane on push/PR. |
| `CLAUDE.md` (modify) | Document the new CI lane, the canonical full-extras install, and `scripts/preflight.py`. |
| `README.md` (modify) | Add a `tests.yml` CI status badge. |

---

## Task 1: Establish the green baseline

**Goal:** Make `pytest -m "not slow"` pass locally with the full CPU stack installed, fixing any rot, so the CI workflow (Task 3) lands green. This task has no fixed deliverable code; its deliverable is a green lane plus commits for any fixes it required.

**Files:**
- Modify: whichever test/source files (if any) are red on current floated deps.

**Interfaces:**
- Consumes: nothing.
- Produces: a known-green `pytest -m "not slow"` baseline + the exact install command other tasks reuse.

- [ ] **Step 1: Install the full CPU dev stack into the active venv**

```bash
pip install -e ".[dev,server,audio,images,embeddings]" \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

Expected: completes; `python -c "import torch, transformers, diffusers, sentence_transformers, fastapi, uvicorn, pytest_asyncio; print('ok')"` prints `ok`.

- [ ] **Step 2: Run the full fast lane and capture the result**

```bash
pytest -m "not slow" -q 2>&1 | tail -40
```

Expected: a pass/fail summary. Record the count and any failing node IDs.

- [ ] **Step 3: Triage each failure with this rule**

For every failing test, classify and fix:
- **Test couples to a floated dependency's API** (e.g. constructing a library exception positionally, as the already-fixed `RepositoryNotFoundError` case did): fix the TEST to use the current API (e.g. pass required kwargs, or build via `MagicMock`). Do not pin the dependency.
- **A real product bug surfaced by a newer dep**: fix the SOURCE.
- **A genuinely missing dependency** the lane needs but no extra installs: add it to the install set in Step 1 AND to `tests.yml` (Task 3) AND to `scripts/preflight.py` sentinels (Task 2). Re-run.
- **A test that cannot run in this environment for a non-product reason** (e.g. needs a GPU): mark it `@pytest.mark.slow` (so the fast lane excludes it) OR add a precise `pytest.importorskip(...)` / `pytest.mark.skipif(...)` with a reason. Prefer fixing over skipping; skip only with a written justification in the marker reason.

- [ ] **Step 4: Re-run until green**

```bash
pytest -m "not slow" -q 2>&1 | tail -5
```

Expected: `N passed` (and possibly some `skipped`), `0 failed`.

- [ ] **Step 5: Commit any fixes (skip if the lane was already green)**

```bash
git add -A
git commit -m "test: green the fast lane on current floated deps

<one line per fix: what was red and why>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01E5TNYLRLwZCH4K1Db4xTbe"
```

Note: as of v0.47.1 the only known rot (the `RepositoryNotFoundError` resolver test) is already fixed; this task may find the lane already green, in which case make no commit and proceed.

---

## Task 2: Preflight guard script + tests

**Goal:** A stdlib-only script that verifies the venv can run the fast lane, then runs it; fully unit-tested.

**Files:**
- Create: `scripts/preflight.py`
- Test: `tests/test_preflight.py`

**Interfaces:**
- Consumes: the install command from Task 1 (embedded as `INSTALL_CMD`).
- Produces: `scripts/preflight.py` exposing `REQUIRED: list[tuple[str,str,str]]`, `missing_deps() -> list[tuple[str,str,str]]`, `report_missing(missing) -> None`, `main(argv: list[str] | None = None) -> int`. CLI: `python scripts/preflight.py [--check-only] [-- <pytest args>]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_preflight.py`:

```python
"""Tests for scripts/preflight.py (loaded as a module; stdlib-only)."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"


def _load():
    spec = importlib.util.spec_from_file_location("preflight", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_only_exits_zero_and_does_not_run_pytest(monkeypatch):
    pf = _load()
    monkeypatch.setattr(pf, "missing_deps", lambda: [])
    called = {}
    monkeypatch.setattr(pf.subprocess, "run",
                        lambda *a, **k: called.setdefault("ran", True))
    assert pf.main(["--check-only"]) == 0
    assert "ran" not in called


def test_missing_dep_exits_nonzero_and_prints_install_cmd(monkeypatch, capsys):
    pf = _load()
    monkeypatch.setattr(pf, "missing_deps",
                        lambda: [("torch", "audio", "torch")])
    rc = pf.main(["--check-only"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "torch" in err
    assert "download.pytorch.org/whl/cpu" in err


def test_runs_fast_lane_when_all_present(monkeypatch):
    pf = _load()
    monkeypatch.setattr(pf, "missing_deps", lambda: [])
    captured = {}

    class FakeProc:
        returncode = 0

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(pf.subprocess, "run", fake_run)
    assert pf.main([]) == 0
    assert "-m" in captured["cmd"]
    assert "not slow" in captured["cmd"]


def test_forwards_trailing_args_to_pytest(monkeypatch):
    pf = _load()
    monkeypatch.setattr(pf, "missing_deps", lambda: [])
    captured = {}

    class FakeProc:
        returncode = 0

    monkeypatch.setattr(pf.subprocess, "run",
                        lambda cmd, *a, **k: captured.__setitem__("cmd", cmd) or FakeProc())
    assert pf.main(["--", "-k", "resolver"]) == 0
    assert captured["cmd"][-2:] == ["-k", "resolver"]


def test_propagates_pytest_returncode(monkeypatch):
    pf = _load()
    monkeypatch.setattr(pf, "missing_deps", lambda: [])

    class FakeProc:
        returncode = 5

    monkeypatch.setattr(pf.subprocess, "run", lambda *a, **k: FakeProc())
    assert pf.main([]) == 5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_preflight.py -q`
Expected: FAIL / ERROR - `scripts/preflight.py` does not exist yet (`spec.loader.exec_module` raises `FileNotFoundError`).

- [ ] **Step 3: Write `scripts/preflight.py`**

```python
#!/usr/bin/env python
"""Preflight guard: verify the dev venv can run the fast test lane, then run it.

The muse fast lane (`pytest -m "not slow"`) imports the heavy ML stack
(torch, transformers, diffusers, sentence-transformers) plus server deps in
many test modules; several import them at collection time (not behind a
mock), so a venv missing those deps does not merely skip - it errors at
collection or silently runs a partial suite. This script asserts the
required deps import BEFORE running pytest, so a release cannot be
"verified" in a drifted venv.

Usage:
    python scripts/preflight.py                  # check deps, then run the lane
    python scripts/preflight.py --check-only     # check deps only, no tests
    python scripts/preflight.py -- -k resolver   # forward args after -- to pytest
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys

# (import_name, extra, pip_package). Import name != package name for several,
# hence the explicit mapping. Keep in sync with INSTALL_CMD and pyproject
# optional-dependencies.
REQUIRED: list[tuple[str, str, str]] = [
    ("numpy", "core", "numpy"),
    ("yaml", "core", "pyyaml"),
    ("torch", "audio", "torch"),
    ("transformers", "audio", "transformers"),
    ("scipy", "audio", "scipy"),
    ("diffusers", "images", "diffusers"),
    ("PIL", "images", "Pillow"),
    ("sentence_transformers", "embeddings", "sentence-transformers"),
    ("fastapi", "server", "fastapi"),
    ("uvicorn", "server", "uvicorn"),
    ("httpx", "server", "httpx"),
    ("psutil", "server", "psutil"),
    ("multipart", "server", "python-multipart"),
    ("pynvml", "server", "nvidia-ml-py"),
    ("pytest_asyncio", "dev", "pytest-asyncio"),
]

INSTALL_CMD = (
    'pip install -e ".[dev,server,audio,images,embeddings]" '
    "--extra-index-url https://download.pytorch.org/whl/cpu"
)


def missing_deps() -> list[tuple[str, str, str]]:
    """Return the sentinels that fail to import."""
    missing: list[tuple[str, str, str]] = []
    for import_name, extra, package in REQUIRED:
        try:
            importlib.import_module(import_name)
        except Exception:  # noqa: BLE001 - any import failure means "missing"
            missing.append((import_name, extra, package))
    return missing


def report_missing(missing: list[tuple[str, str, str]]) -> None:
    """Print an actionable error naming each missing dep and the fix."""
    print("preflight: venv is not fast-lane ready; missing imports:",
          file=sys.stderr)
    for import_name, extra, package in missing:
        print(f"  - {import_name}  (extra: {extra}, package: {package})",
              file=sys.stderr)
    print(f"\nInstall the full dev stack:\n  {INSTALL_CMD}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="muse fast-lane preflight guard")
    parser.add_argument("--check-only", action="store_true",
                        help="verify deps only; do not run pytest")
    parser.add_argument("pytest_args", nargs="*",
                        help="args forwarded to pytest (use -- to separate)")
    args = parser.parse_args(argv)

    missing = missing_deps()
    if missing:
        report_missing(missing)
        return 1
    print(f"preflight: all {len(REQUIRED)} fast-lane deps present.")
    if args.check_only:
        return 0

    cmd = [sys.executable, "-m", "pytest", "-m", "not slow", *args.pytest_args]
    print(f"preflight: running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_preflight.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Smoke the script's real dep-check (deps installed from Task 1)**

Run: `python scripts/preflight.py --check-only`
Expected: `preflight: all 15 fast-lane deps present.` and exit 0.
(If it reports a missing dep that the lane genuinely needs, add the matching extra to the Task-1 install set and the Task-3 workflow, then re-run. If it reports a dep the lane does NOT actually need, remove that sentinel from `REQUIRED`.)

- [ ] **Step 6: Commit**

```bash
git add scripts/preflight.py tests/test_preflight.py
git commit -m "feat(scripts): add preflight guard for the fast test lane

Asserts the fast-lane deps import before running pytest -m 'not slow', so a
release cannot be verified in a drifted venv. Stdlib-only at module top so it
loads in a bare venv; --check-only verifies deps without running tests.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01E5TNYLRLwZCH4K1Db4xTbe"
```

---

## Task 3: CI test workflow

**Goal:** A GitHub Actions workflow that installs the CPU stack and runs the fast lane on push/PR.

**Files:**
- Create: `.github/workflows/tests.yml`

**Interfaces:**
- Consumes: the install command and the green baseline from Task 1.
- Produces: a `tests` workflow named `tests.yml`, job id `fast-lane`, validated in Task 5.

- [ ] **Step 1: Write `.github/workflows/tests.yml`**

```yaml
name: tests

# Run the fast unit/contract lane (pytest -m "not slow") on every push to
# main and every PR. Distinct from fresh-venv-smoke.yml, which validates the
# per-model fresh-venv install/load path. Deps float on latest (no lockfile)
# so upstream drift surfaces here on the next push. CPU torch only; model
# tests are mocked, so no model weights are downloaded.

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:
  fast-lane:
    name: pytest -m "not slow" (py3.11, cpu)
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - name: checkout
        uses: actions/checkout@v4

      - name: set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: cache pip downloads
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-tests-${{ hashFiles('pyproject.toml') }}
          restore-keys: |
            ${{ runner.os }}-pip-tests-
            ${{ runner.os }}-pip-

      - name: install muse with full CPU extras
        env:
          PIP_EXTRA_INDEX_URL: https://download.pytorch.org/whl/cpu
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,server,audio,images,embeddings]"

      - name: run fast lane
        run: pytest -m "not slow" -q
```

- [ ] **Step 2: Lint the workflow if actionlint is available (optional)**

```bash
command -v actionlint >/dev/null && actionlint .github/workflows/tests.yml || echo "actionlint not installed; first CI run validates"
```

Expected: no errors, or the skip message. (Do not install actionlint just for this.)

- [ ] **Step 3: Verify the run command matches the local green lane**

Run: `pytest -m "not slow" -q 2>&1 | tail -3`
Expected: matches Task 1's green result (`0 failed`). This is the exact command CI runs.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/tests.yml
git commit -m "ci: run the fast test lane on push/PR (tests.yml)

Single Python-3.11 job, full CPU extras (download.pytorch.org/whl/cpu),
float on latest. Closes the gap where pytest -m 'not slow' was gated only
by a manual local run; upstream dependency drift now surfaces on the next
push instead of shipping silently (cf. the hf_hub 1.20.1 resolver-test rot).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01E5TNYLRLwZCH4K1Db4xTbe"
```

---

## Task 4: Documentation

**Goal:** Document the new CI lane, the canonical install, and the preflight command.

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: names from Tasks 2-3 (`scripts/preflight.py`, `tests.yml`).
- Produces: docs only.

- [ ] **Step 1: Add a CI note to CLAUDE.md**

In `CLAUDE.md`, find the line that begins the smoke-test section:

```
## Fresh-venv smoke test (CI)
```

Insert a new section immediately ABOVE it:

```markdown
## Continuous integration

Two GitHub Actions workflows run on every push to `main` and every PR:

- `.github/workflows/tests.yml` runs the fast lane (`pytest -m "not slow"`)
  in a single Python-3.11 job with the full CPU stack
  (`pip install -e ".[dev,server,audio,images,embeddings]"` against the
  `download.pytorch.org/whl/cpu` index). Dependencies float on latest (no
  lockfile), so an upstream breaking release surfaces here on the next push
  rather than shipping silently. Model tests are mocked, so no weights are
  downloaded.
- `.github/workflows/fresh-venv-smoke.yml` (below) validates the per-model
  fresh-venv install/load path.

Locally, run the fast lane through the preflight guard so a drifted venv
fails loudly instead of running a partial suite:

    python scripts/preflight.py            # check deps, then run the lane
    python scripts/preflight.py --check-only

```

- [ ] **Step 2: Add the canonical install + preflight to the Development commands block**

In `CLAUDE.md`, in the `## Development commands` fenced block, just under the
existing `# Install (dev)` line `pip install -e ".[dev,server,audio,images]"`,
add:

```bash
# Full dev stack (CPU torch) - what CI installs and what preflight expects
pip install -e ".[dev,server,audio,images,embeddings]" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Gate the fast lane on a non-drifted venv (used in the release ritual)
python scripts/preflight.py
```

- [ ] **Step 3: Add a CI badge to README.md**

At the top of `README.md`, immediately under the H1 title line, add:

```markdown
[![tests](https://github.com/queelius/muse/actions/workflows/tests.yml/badge.svg)](https://github.com/queelius/muse/actions/workflows/tests.yml)
```

- [ ] **Step 4: Verify ASCII cleanliness**

```bash
grep -nP '[^\x00-\x7F]' CLAUDE.md README.md && echo "NON-ASCII FOUND - fix before commit" || echo "clean"
```

Expected: `clean`.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document tests.yml CI lane + scripts/preflight.py

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01E5TNYLRLwZCH4K1Db4xTbe"
```

---

## Task 5: Land and verify CI

**Goal:** Get all commits onto `main` and confirm both CI workflows go green. No PyPI release (the package artifact is unchanged).

**Files:** none (push + verify only).

**Interfaces:**
- Consumes: commits from Tasks 1-4.
- Produces: a green `tests.yml` run on `main`.

- [ ] **Step 1: Final local gate via preflight**

Run: `python scripts/preflight.py`
Expected: `all 15 fast-lane deps present.` then `0 failed`. (This dogfoods preflight end-to-end.)

- [ ] **Step 2: Push to main**

```bash
git push origin main
```

- [ ] **Step 3: Watch the new workflow to completion**

```bash
sleep 10
gh run list --workflow tests.yml --limit 1
RUN_ID=$(gh run list --workflow tests.yml --limit 1 --json databaseId -q '.[0].databaseId')
gh run watch "$RUN_ID" --exit-status
```

Expected: `tests.yml` completes with `success`.

- [ ] **Step 4: Confirm fresh-venv-smoke still green and triage on failure**

```bash
gh run list --limit 5
```

Expected: the latest `tests` run is `success`. If `tests.yml` failed for a
reason the local lane did not catch (CI-only env difference, or a floated dep
that resolved differently on the runner), fix forward: reproduce by matching
the CI install command exactly, fix the test/source or add the missing dep to
both `tests.yml` and `scripts/preflight.py`'s `REQUIRED`, commit, and push
again until green.

- [ ] **Step 5: Route the release ritual through preflight (memory)**

Update the auto-memory file
`~/.claude/projects/-home-spinoza-github-repos-muse/memory/feedback_release_workflow.md`:
in the release-ritual description, change the pre-build test step from
`pytest -m "not slow"` to `python scripts/preflight.py`, and add a one-line
note that CI now runs the suite via `tests.yml`. (This is operator memory,
not a repo file; no commit.)

---

## Notes for the executor

- **This session's venv may be bare.** Task 1 Step 1 installs ~1-2 GB of CPU
  wheels (torch, diffusers, etc.); allow several minutes. If the executing
  environment genuinely cannot install the ML stack, the fallback is to land
  Tasks 2-4 and use the FIRST `tests.yml` run (Task 5) as the baseline,
  fixing forward - but the preferred path is a local green baseline first.
- **No version bump, no `twine`, no GitHub release** - nothing here ships in
  the wheel.
- **`espeak-ng` and other system libs** are intentionally NOT in `tests.yml`:
  the fast lane mocks model runtimes, so no phonemizer/system lib is needed.
  Only add an `apt-get` step if Task 1 proves a specific test needs it.
