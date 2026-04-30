# v0.32.0 fresh-venv smoke-test plan (#124)

**Date:** 2026-04-30
**Spec:** `docs/superpowers/specs/2026-04-30-fresh-venv-smoke-test-design.md`

5 tasks (A through E), single commit per task. Push only at the v0.32.0
release. Run `pytest tests/ -q -m "not slow"` after every commit;
expected baseline 2284 passing fast-lane tests at v0.31.0.

## Task A: Spec + plan documents

Write `docs/superpowers/specs/2026-04-30-fresh-venv-smoke-test-design.md`
and `docs/superpowers/plans/2026-04-30-fresh-venv-smoke-test.md`. (This
file plus the spec form Task A.)

**Deliverable:** single commit `docs(plan): fresh-venv smoke-test for CI (v0.32.0)`.

## Task B: smoke-test runner script + tests

Add `scripts/smoke_fresh_venv.py`. Approximately 150 LOC. Surface:

```
python scripts/smoke_fresh_venv.py --model_id <id> [--venv_root <path>] [--json]
```

Internal helpers (each named so unit tests can patch them):

```python
def _create_venv(target: Path) -> None: ...
def _install_muse(venv_python: Path, repo_root: Path) -> None: ...
def _install_pip_extras(venv_python: Path, packages: tuple[str, ...]) -> None: ...
def _run_load_only(venv_python: Path, model_id: str) -> tuple[int, str]: ...
def smoke_one(model_id: str, venv_root: Path) -> SmokeResult: ...
```

`smoke_one` returns a small dataclass:

```python
@dataclass
class SmokeResult:
    model_id: str
    ok: bool
    error: str | None
    duration_s: float
    label: str   # "kokoro-82m: OK (12.3s)" or "kokoro-82m: FAIL (missing dep: librosa)"
```

The script uses `muse.core.catalog.known_models()` and `MANIFEST` from
the bundled script to discover `pip_extras`. It then shells out to:

1. `<venv>/bin/python -m pip install -e <repo>[server]`
2. `<venv>/bin/python -m pip install <pip_extras>`
3. `<venv>/bin/python -m muse.cli _probe_worker --model <id> --device cpu --no-inference`

The probe worker already exists (`src/muse/cli_impl/probe_worker.py`)
and exits 0 on successful load + JSON record on stdout. Errors land
on stderr; the smoke runner extracts the most informative line for the
failure label.

Add `tests/scripts/test_smoke_fresh_venv.py` covering:

1. `test_smoke_one_success_path`: mock subprocess.run + venv creation; assert
   the four expected commands are issued in order; assert SmokeResult.ok is True.
2. `test_smoke_one_unknown_model`: assert exit code 2 and a clear error.
3. `test_smoke_one_pip_install_fails`: subprocess.run returns non-zero on
   the muse[server] install; assert SmokeResult.ok is False and the failure
   label mentions pip.
4. `test_smoke_one_load_fails`: subprocess.run returns non-zero on the
   load step; assert SmokeResult.ok is False and the failure label mentions
   the captured stderr.
5. `test_smoke_one_extracts_missing_dep_label`: stderr contains
   `ModuleNotFoundError: No module named 'librosa'`; assert the failure
   label mentions `librosa`.
6. `test_main_json_output`: invoke `main(["--model_id", "kokoro-82m", "--json"])`;
   assert stdout is parseable JSON with the expected keys.
7. `test_main_human_output`: invoke without `--json`; assert stdout is
   the human-readable label.

All tests mock subprocess.run + `Path.exists` / venv creation; no real
venvs created in unit tests.

**Deliverable:** single commit `feat(ci): smoke-test runner for fresh
per-model venvs`. Run fast-lane tests; expected delta: +7 tests.

## Task C: GitHub Actions workflow

Add `.github/workflows/fresh-venv-smoke.yml`:

```yaml
name: fresh-venv-smoke
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  smoke:
    name: smoke-${{ matrix.model }}
    runs-on: ubuntu-latest
    timeout-minutes: 30
    strategy:
      fail-fast: false
      matrix:
        model:
          - kokoro-82m
          - dinov2-small
          - bart-large-cnn
          - bge-reranker-v2-m3
          - mert-v1-95m

    steps:
      - uses: actions/checkout@v4

      - name: set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: cache pip downloads
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: cache HF model weights
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: ${{ runner.os }}-hf-${{ hashFiles('pyproject.toml') }}-${{ matrix.model }}
          restore-keys: |
            ${{ runner.os }}-hf-${{ hashFiles('pyproject.toml') }}-
            ${{ runner.os }}-hf-

      - name: install host muse
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]" \
            --extra-index-url https://download.pytorch.org/whl/cpu

      - name: system deps for kokoro
        if: matrix.model == 'kokoro-82m'
        run: sudo apt-get update && sudo apt-get install -y espeak-ng

      - name: smoke-test fresh venv
        env:
          PIP_EXTRA_INDEX_URL: https://download.pytorch.org/whl/cpu
        run: |
          python scripts/smoke_fresh_venv.py \
            --model_id ${{ matrix.model }} \
            --json | tee smoke-result.json

      - name: upload logs on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: smoke-${{ matrix.model }}-logs
          path: |
            smoke-result.json
            ${{ runner.temp }}/muse-smoke-${{ matrix.model }}/.venv/pip-log.txt
```

Note that espeak-ng is a system package required by kokoro; the conditional
step installs it only on the kokoro job to keep the other jobs lean.

Verify the YAML is well-formed locally with `python -c "import yaml; yaml.safe_load(open('.github/workflows/fresh-venv-smoke.yml'))"`.

**Deliverable:** single commit `ci: matrix-tested fresh-venv smoke for
five lightweight bundled models`.

## Task D: Documentation

Two doc updates in one commit:

1. CLAUDE.md: add a "Fresh-venv smoke test" subsection under
   "Project-specific conventions". Body covers:
   - The bug class: bundled-script venvs have to install ONLY what
     `pip_extras` declares; transitive imports that work in the
     muse[dev,server,audio,images] dev environment may not be present
     in the per-model venv.
   - Why the v0.30.0 audit (#110) caught direct-import gaps via AST
     scan but cannot catch transitive `from_pretrained` imports.
   - The CI workflow (`fresh-venv-smoke.yml`) closes that gap by
     actually creating the venv and running the loader. Five models
     cover the common patterns; expand the matrix when a new pattern
     lands.
   - Local repro: `python scripts/smoke_fresh_venv.py --model_id <id>`.

2. README.md: add a CI status badge near the top.
   ```markdown
   ![fresh-venv-smoke](https://github.com/queelius/muse/actions/workflows/fresh-venv-smoke.yml/badge.svg)
   ```

**Deliverable:** single commit `docs: fresh-venv smoke-test (CLAUDE.md +
README.md badge)`.

## Task E: v0.32.0 release

1. Bump `pyproject.toml`: `0.31.0` -> `0.32.0`.
2. Update `src/muse/__init__.py` docstring with a v0.32.0 release note:
   "v0.32.0 adds CI smoke-tests of fresh per-model venvs (#124). The
   workflow `.github/workflows/fresh-venv-smoke.yml` matrix-tests five
   lightweight bundled models on every push and PR; each job creates a
   fresh venv, installs only what `muse pull` would install, and
   verifies the model loads. Catches the production failure mode where
   a bundled script's pip_extras misses a transitive dep that the dev
   environment happens to have. Local repro:
   `python scripts/smoke_fresh_venv.py --model_id <id>`."
3. Em-dash check: `grep -rn '\xe2\x80\x94' .github/ scripts/ src/muse/__init__.py docs/superpowers/specs/2026-04-30-fresh-venv-smoke-test-design.md docs/superpowers/plans/2026-04-30-fresh-venv-smoke-test.md` returns empty.
4. Single commit `chore(release): v0.32.0`.
5. `git tag v0.32.0`.
6. `git push origin main && git push origin v0.32.0`.
7. `gh release create v0.32.0` with notes covering: the smoke-test
   workflow, the five-model matrix, the local-repro script, the
   pip-extras audit follow-on context.

**Deliverable:** v0.32.0 tag + GitHub release published.

## Constraints (max-effort)

- **No em-dashes** anywhere.
- **No literal banned-token** (the Python builtin starting with `e`, three-letter, that runs a string as code) in any new file content.
- **Single commit per task.**
- **Push only at v0.32.0 release.**
- **Don't break existing tests.** All 2284 fast-lane tests must keep passing.
- **Smoke-runner unit tests mock subprocess + venv ops.** No real venvs created in unit tests.

## Risk register

- **Workflow YAML syntax errors.** GitHub Actions silently skips
  malformed workflows in some cases. **Mitigation:** parse locally with
  `python -c "import yaml; yaml.safe_load(open(path))"` before commit;
  optionally run `actionlint` if the binary is available.

- **Disk space on free-tier runners.** GitHub-hosted ubuntu-latest has
  ~14 GB free. Five models at a few hundred MB each plus torch wheels
  (~1.5 GB CPU-only) plus muse + transformers + sentence-transformers
  fits easily. **Mitigation:** keep matrix entries lightweight; document
  the size limit in spec.

- **Cache pollution between matrix jobs.** Each matrix job is a fresh
  runner, so caches don't cross-contaminate. The HF cache key includes
  the model id, so models don't collide on weight cache writes.

- **PyTorch CUDA wheels accidentally pulled.** `pip install torch` would
  default to the CUDA wheel on a Linux runner with NVIDIA drivers
  (none on free-tier, but defensive). **Mitigation:** force
  `--extra-index-url https://download.pytorch.org/whl/cpu` and
  `PIP_EXTRA_INDEX_URL` env var.

- **espeak-ng missing for kokoro-82m.** The kokoro runtime imports
  espeak-ng's libraries via misaki at synthesis time, but load-only
  smoke tests may or may not trigger that path depending on where
  misaki initializes. **Mitigation:** install espeak-ng via apt on the
  kokoro job conditionally.

- **HF rate limiting on cold runs.** First run downloads ~1 GB of
  weights across five models. Cached after first success.
  **Mitigation:** the cache key fallback (`restore-keys`) lets a
  cache hit on the muse-version-prefix even if the model-specific
  key misses, accelerating subsequent runs even after a muse bump.

- **Workflow won't run until pushed.** Cannot test the actual workflow
  until v0.32.0 lands on main. **Mitigation:** verify YAML well-
  formedness locally; verify the smoke-runner script works locally for
  at least one model id; rely on the first push to surface remaining
  workflow issues, fix in v0.32.1 if needed.

## Final acceptance

- 5 commits land in order (A, B, C, D, E).
- `git log --oneline | head -10` shows the v0.32.0 commit chain.
- `pytest tests/ -q -m "not slow"` passes; expected ~2291 (baseline +7
  smoke-runner tests).
- `scripts/smoke_fresh_venv.py` exists and is invocable.
- `.github/workflows/fresh-venv-smoke.yml` exists and parses as valid YAML.
- README.md has the CI badge.
- CLAUDE.md has the "Fresh-venv smoke test" section.
- `git tag v0.32.0` exists; pushed; GitHub release published.
