"""`muse _worker` implementation: runs ONE worker (optionally hosting
multiple models from the same venv) and starts uvicorn.

Invoked by the supervisor (`muse serve`) via subprocess:
    <venv>/bin/python -m muse.cli _worker --port 9001 --model soprano-80m

Can also be run standalone for debugging. Not advertised in top-level help.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn

from muse.core.catalog import get_manifest, is_pulled, known_models, load_backend
from muse.core.discovery import discover_modalities
from muse.core.registry import ModalityRegistry
from muse.core.server import create_app

log = logging.getLogger(__name__)


def _bundled_modalities_dir() -> Path:
    """Directory containing muse's built-in modality packages."""
    # worker.py sits at src/muse/cli_impl/worker.py; parents[1] is src/muse/.
    return Path(__file__).resolve().parents[1] / "modalities"


def _env_modalities_dir() -> Path | None:
    """Optional extra modalities dir from `$MUSE_MODALITIES_DIR` env var.

    Intended as an escape hatch for power users experimenting with new
    modality contracts, not a normal extension surface. Most users
    should extend via model scripts instead (see $MUSE_MODELS_DIR).
    """
    env = os.environ.get("MUSE_MODALITIES_DIR")
    return Path(env) if env else None


def _modality_dirs() -> list[Path]:
    """Scan order for modality discovery: bundled first, then env override.

    First-found-wins on MODALITY tag collision, so bundled modalities
    shadow env-dir entries that declare the same MIME tag.
    """
    dirs = [_bundled_modalities_dir()]
    env = _env_modalities_dir()
    if env is not None:
        dirs.append(env)
    return dirs


def run_worker(*, host: str, port: int, models: list[str], device: str) -> int:
    """Load the specified models into a registry and run uvicorn.

    `models` is the exact set of model-ids to load into this process.
    The supervisor decides which models share a worker; the worker just
    loads what it's told.

    Fail-fast contract: if any assigned model fails to load for any
    reason (unknown id, not pulled, backend import error), the worker
    returns exit code 2 BEFORE starting uvicorn. A partial worker
    masquerading as healthy is worse than a crashed one: the
    supervisor's restart-then-mark-dead machinery only engages when
    the worker process actually exits, and /health only reports
    'degraded' when the supervisor sees a worker unreachable or dead.

    `models == []` is a valid test configuration (empty-registry
    router mounting smoke test); it does not trigger the fail-fast.
    """
    registry = ModalityRegistry()
    routers: dict = {}
    failures: list[str] = []

    catalog = known_models()
    to_load = [m for m in models if m in catalog]
    unknown = [m for m in models if m not in catalog]
    if unknown:
        log.warning("ignoring unknown models: %s", unknown)
        failures.extend(unknown)

    for model_id in to_load:
        if not is_pulled(model_id):
            log.error("model %s not pulled; worker cannot host it", model_id)
            failures.append(model_id)
            continue
        entry = catalog[model_id]
        log.info("loading %s (%s)", model_id, entry.modality)
        try:
            backend = load_backend(model_id, device=device)
        except Exception as e:
            log.error("failed to load %s: %s", model_id, e)
            failures.append(model_id)
            continue
        manifest = get_manifest(model_id)
        registry.register(entry.modality, backend, manifest=manifest)

    if failures:
        log.error(
            "worker exiting (exit 2): %d/%d assigned models failed to load: %s",
            len(failures), len(models), failures,
        )
        return 2

    # Always mount every discovered modality router so empty-registry
    # requests get the OpenAI envelope rather than FastAPI's default
    # {"detail": "Not Found"}. Adding a new modality requires zero
    # changes here: drop a subpackage under src/muse/modalities/ that
    # exports MODALITY + build_router, and discovery picks it up.
    for tag, build_router in discover_modalities(_modality_dirs()).items():
        log.info("mounting modality router for %s", tag)
        routers[tag] = build_router(registry)

    app = create_app(registry=registry, routers=routers)
    uvicorn.run(app, host=host, port=port, log_config=None)
    return 0
