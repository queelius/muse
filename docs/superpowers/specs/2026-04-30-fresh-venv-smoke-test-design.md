# v0.32.0 fresh-venv smoke-test for CI (#124)

**Date:** 2026-04-30
**Driver:** add a CI workflow that creates a fresh per-model venv, installs ONLY the deps `muse pull <id>` would install, and verifies the model loads. Closes task #124.

This is **not** a new modality, **not** a new wire contract, **not** a runtime behavior change. It is pure CI infrastructure that catches the "muse pull works in dev, fails in production" regression class.

## The bug pattern

The user has been bitten by this multiple times in production. The shape:

1. A bundled script declares `pip_extras` with what its `_ensure_deps()` directly imports (e.g., `transformers`, `torch`).
2. CI passes because the test environment preinstalls broad muse extras (`pip install -e ".[dev,server,audio,images]"`), so transitive deps that the runtime ALSO imports at load time (e.g., `accelerate`, `safetensors`, `Pillow`, `numpy`, `librosa`) happen to be present.
3. The user runs `muse pull <id>` on a clean machine. `pull` creates a fresh venv at `~/.muse/venvs/<id>/` and installs ONLY:
   - `muse[server]` (in editable mode)
   - The model's declared `pip_extras`
4. The worker tries to load the model. `from_pretrained` (or sentence-transformers, or diffusers) imports a transitive dep that's not in `pip_extras`. ImportError. Worker exits.

The v0.30.0 audit (#110) caught seven such gaps and added a regression-guard test (`tests/models/test_pip_extras_audit.py`) that scans each script for direct imports and asserts they're declared. That test catches the "the script imports `accelerate` but doesn't declare it" class.

The remaining gap is: **transitive deps that no script source-imports but `from_pretrained` pulls in at load time**. The v0.30.0 audit cannot see these because they don't appear in the source AST. The user's production fix has been: install the model in a clean venv, watch the ImportError, add the missing dep, repeat until the model loads.

The fix #124: a CI workflow that does exactly that, automatically, on every push and PR.

## Goal

1. A standalone Python script `scripts/smoke_fresh_venv.py` that:
   - Takes `--model_id <id>` (and optional `--venv_root <path>`, `--no-pull` to skip download for unit-test pathing).
   - Creates a fresh venv (mimicking what `muse pull` does).
   - Invokes `muse pull <id>` against that venv.
   - Spawns a worker subprocess in that venv that does `load_backend(model_id)` (load only, no inference).
   - Reports success or failure as a JSON record on stdout, exits non-zero on failure.
2. A GitHub Actions workflow `.github/workflows/fresh-venv-smoke.yml` that:
   - Runs on every `push` to `main` and on every `pull_request`.
   - Matrix-tests a curated set of lightweight bundled models.
   - Caches HF weights and pip downloads between runs (keyed on muse version + model id).
   - Surfaces failures with a label like `kokoro-82m: missing dep 'librosa'`.
3. A README.md mention of the CI badge so contributors see green/red at a glance.
4. A CLAUDE.md "Fresh-venv smoke test" section explaining the failure mode and how the CI guards it.

The work is pure CI; runtime behavior does not change. v0.32.0 is informational.

## Smoke-test runner design

`scripts/smoke_fresh_venv.py` is a self-contained Python script that:

- Is invocable both locally (`python scripts/smoke_fresh_venv.py --model_id kokoro-82m`) and in CI.
- Uses only stdlib + the host muse install (no extra deps).
- Mocks-cleanly: the heavy operations (subprocess.run, venv creation, HF download) are routed through small named helpers so unit tests can patch them.

### CLI surface

```
python scripts/smoke_fresh_venv.py --model_id <id> [--venv_root <path>] [--json]
```

- `--model_id` (required): the bundled model to smoke-test.
- `--venv_root` (optional, default `<tmp>/muse-smoke-<id>`): where to put the fresh venv. CI uses a workspace-relative path so cache directives can target it.
- `--json` (optional): emit a JSON record on stdout (default human-readable).

### Behavior

```
1. Validate <id> is in known_models() (else exit 2 with "unknown model" error).
2. Create venv at <venv_root>/.venv (using sys.executable -m venv, same as muse pull).
3. Install muse[server] (editable, repo root) into the venv.
4. Install the model's pip_extras (from MANIFEST) into the venv.
5. Download HF weights to a cache (or reuse if present).
6. Spawn a worker subprocess in the new venv:
       <venv>/bin/python -m muse.cli _probe_worker --model <id> --device cpu --no-inference
   The probe worker loads via `load_backend(model_id, device=cpu)` and exits 0 on success.
7. Capture exit code + stderr.
8. Report:
       JSON: {"model_id": "...", "ok": true|false, "error": null|"...", "duration_s": N}
       Human: "kokoro-82m: OK (12.3s)" or "kokoro-82m: FAIL (missing dep: librosa)"
9. Exit 0 on success; non-zero on failure.
```

The script reuses `muse._probe_worker` because that already loads the model in the venv via `load_backend`. The smoke test passes `--no-inference` to skip the actual inference (faster, doesn't need GPU).

### Why a separate script and not a `muse smoke` subcommand

The smoke test is a CI utility, not a user-facing command. Users never want to "smoke-test their kokoro install"; they want it to either work or to have failed in CI before they pulled muse. Making it a script keeps the CLI surface clean.

### Local invocation

```bash
python scripts/smoke_fresh_venv.py --model_id kokoro-82m
# kokoro-82m: OK (12.3s)

python scripts/smoke_fresh_venv.py --model_id kokoro-82m --json
# {"model_id":"kokoro-82m","ok":true,"error":null,"duration_s":12.3}
```

## CI matrix

`.github/workflows/fresh-venv-smoke.yml` runs five lightweight models in parallel:

- `kokoro-82m` (audio/speech, ~160 MB, kokoro pulls torch transitively, has surprised us)
- `dinov2-small` (image/embedding, ~88 MB, transformers + Pillow)
- `bart-large-cnn` (text/summarization, ~400 MB, transformers seq2seq)
- `bge-reranker-v2-m3` (text/rerank, ~568 MB, sentence-transformers cross-encoder)
- `mert-v1-95m` (audio/embedding, ~370 MB, transformers + librosa, has surprised us before)

Heavy / GPU-required models (`sd-turbo`, `animatediff-motion-v3`, `stable-audio-open-1-0`, `wan2-1-t2v-1-3b`) are deferred. They need GPU runners (not free-tier) or would bust the 14 GB GitHub-hosted runner disk limit. `sam2-hiera-tiny` and others may be added later if budget allows.

`fail-fast: false` so one failing model does not abort the rest. Each job has a 30-minute timeout.

## Caching strategy

Two caches:

1. **HF weight cache** at `~/.cache/huggingface`, keyed on `${{ runner.os }}-hf-${{ hashFiles('pyproject.toml') }}-${{ matrix.model }}`. The pyproject hash invalidates the cache on every muse version bump; the model id invalidates when the curated allow-patterns change. Weights are typically a few hundred MB to a few GB; caching saves bandwidth and time.

2. **Pip download cache** at `~/.cache/pip`, keyed on `${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}`. Saves re-downloading torch/transformers wheels on every CI run.

CPU-only torch wheels are forced via `pip install --extra-index-url https://download.pytorch.org/whl/cpu`. CUDA wheels would bust the runner disk space and aren't usable on GitHub-hosted free-tier runners anyway.

## Failure surfacing

When a model's smoke test fails:

1. The job exits non-zero with a one-line label visible in the GitHub Actions UI:
   ```
   kokoro-82m: FAIL (missing dep: librosa)
   ```
2. The full pip log + worker stderr is uploaded as an artifact (`smoke-<model>-logs`) for diagnosis.
3. If the workflow has been wired with annotations, GitHub annotates the failing line in the bundled script (e.g., `src/muse/models/kokoro_82m.py` line N) so the diagnosis lands at the source.

The CI status badge in README.md surfaces overall green/red.

## Out of scope

- **Real inference** (slow, GPU-dependent). We verify the model can be loaded, not that it produces correct output. End-to-end inference is what the integration test suite covers (opt-in via `MUSE_REMOTE_SERVER`).
- **Heavy / GPU-only models** (sd-turbo, animatediff, stable-audio, wan, large LLMs). These need paid runners or different infrastructure. Deferred until budget allows.
- **Multi-platform** (Mac, Windows). v0.32.0 ships ubuntu-latest only. Mac and Windows can come in a follow-up if the regression rate justifies them.
- **Resolver-pulled models** (GGUFs via `hf://` URIs). The CI smoke-test currently targets bundled scripts because they're the muse-shipped surface; resolver pulls inherit the same dep-install path through `_pull_via_resolver`, which is exercised by unit tests. If a resolver-pulled model regresses on dep handling, that surfaces in integration tests.
- **`muse smoke <id>` CLI subcommand.** Not user-facing; a script is sufficient.
- **Smoke-testing every bundled model.** Five lightweight models cover the common dep patterns (transformers, sentence-transformers, kokoro, librosa). Adding more is cheap (one matrix entry); judgment call on cost vs. coverage.

## Acceptance

- `scripts/smoke_fresh_venv.py` exists and is invocable locally with `--model_id`.
- Unit tests in `tests/scripts/test_smoke_fresh_venv.py` cover the success path, missing-model path, pip-install-failure path, load-failure path, JSON output mode. Heavy operations (venv creation, subprocess runs) are mocked.
- `.github/workflows/fresh-venv-smoke.yml` exists with the five-model matrix.
- The workflow YAML is well-formed (verified locally; cannot run until pushed).
- Caching is configured for HF weights and pip downloads.
- README.md mentions the CI badge.
- CLAUDE.md has a "Fresh-venv smoke test" section.
- v0.32.0 tag pushed; GitHub release published.
- All 2284 fast-lane tests still pass; new smoke-runner tests added.
- No em-dashes anywhere in new content.
