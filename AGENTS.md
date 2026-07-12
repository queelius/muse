# Repository Guidelines

## Project Structure & Module Organization

Muse is a Python 3.10+ package using a `src/` layout. Core discovery, catalogs, configuration, and runtime helpers live in `src/muse/core/`. Each API capability is a self-contained package under `src/muse/modalities/`; modality names use underscores in Python and MIME-style tags at runtime (for example, `audio_speech` declares `audio/speech`). Bundled model manifests live in `src/muse/models/`, while CLI implementation, admin APIs, MCP tools, and observability code have sibling packages under `src/muse/`. Tests mirror this organization in `tests/`. Use `examples/` for runnable clients, `scripts/` for maintenance tooling, and `docs/` for architecture, specifications, and plans. Runtime YAML assets such as `src/muse/curated.yaml` are packaged with the wheel.

## Build, Test, and Development Commands

```bash
pip install -e ".[dev,server,audio,images,embeddings]" \
  --extra-index-url https://download.pytorch.org/whl/cpu
python scripts/preflight.py
pytest -m "not slow" -q
pytest tests/modalities/audio_speech/ -q
pytest tests/ -q
muse serve --host 127.0.0.1 --port 8000
```

The editable install matches CI’s CPU stack. Preflight checks dependency drift and runs the release gate. The fast pytest lane excludes subprocess-heavy tests; run the full suite before broad runtime or supervisor changes. Integration tests require `MUSE_REMOTE_SERVER` and a reachable Muse instance.

## Coding Style & Naming Conventions

Follow existing Python style: four-space indentation, type annotations on public interfaces, short docstrings for non-obvious contracts, `snake_case` for modules/functions, `PascalCase` for classes, and uppercase constants. Keep heavyweight ML imports deferred inside runtime loaders. Put CLI argument wiring in `src/muse/cli.py` and behavior in `src/muse/cli_impl/`. No mandatory formatter is configured; keep changes focused and consistent with neighboring code.

## Testing Guidelines

Pytest is the test framework. Name files `test_*.py` and tests `test_<behavior>`. Add tests beside the affected layer: modality protocol/routes/runtime tests under `tests/modalities/<name>/`, model tests under `tests/models/`, and discovery/catalog tests under `tests/core/`. Mark real subprocess end-to-end tests with `@pytest.mark.slow`. No fixed coverage percentage is enforced, but every bug fix should include a regression test.

## Commit & Pull Request Guidelines

History follows Conventional Commit-style subjects such as `feat(translation): ...` and `fix(core): ...`. Use an imperative, scoped subject and keep each commit cohesive. Pull requests should explain behavior and compatibility changes, link relevant issues, list exact test commands, and include request/response examples for API changes. Add screenshots only for dashboard/UI changes; never commit model weights, secrets, local virtual environments, or generated caches.
