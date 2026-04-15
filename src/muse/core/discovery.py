"""Plugin discovery for models and modalities.

Models: each `.py` file in a scanned directory that defines a top-level
MANIFEST dict (with keys model_id, modality, hf_repo) and a top-level
Model class. Files starting with `_` (including `__init__.py`) are
skipped. Scripts failing to import (missing deps, syntax errors, etc.)
are logged and skipped. Discovery never raises.

Modalities: each subdirectory under a scanned root that defines a
package (`__init__.py`) exporting a module-level MODALITY string
(MIME-style tag) and a build_router callable. Same error handling:
bad modality packages get logged and skipped.

Scan order is caller-defined (a list of directories). First-found-wins
on model_id or MODALITY tag collisions; subsequent duplicates produce
a warning log.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger(__name__)


REQUIRED_MANIFEST_KEYS = ("model_id", "modality", "hf_repo")


@dataclass
class DiscoveredModel:
    """A model that discovery found in one of the scanned dirs.

    - manifest: the MANIFEST dict the script defined
    - model_class: the class named `Model` exported by the script
    - source_path: filesystem path to the script (for error messages)
    """
    manifest: dict
    model_class: type
    source_path: Path


def discover_models(dirs: list[Path]) -> dict[str, DiscoveredModel]:
    """Scan dirs in order; return {model_id: DiscoveredModel}.

    First-found-wins on model_id collision; warns on duplicates.
    Nonexistent dirs are silently skipped. Script errors are logged.
    """
    found: dict[str, DiscoveredModel] = {}
    for d in dirs:
        if not d or not d.is_dir():
            continue
        for py_file in sorted(d.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            try:
                module = _load_script(py_file)
            except Exception as e:
                logger.warning(
                    "skipping model script %s: import failed (%s)",
                    py_file, e,
                )
                continue

            manifest = getattr(module, "MANIFEST", None)
            if not isinstance(manifest, dict):
                logger.warning(
                    "skipping model script %s: no top-level MANIFEST dict",
                    py_file,
                )
                continue

            missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in manifest]
            if missing:
                logger.warning(
                    "skipping model script %s: MANIFEST missing required keys %s",
                    py_file, missing,
                )
                continue

            model_class = getattr(module, "Model", None)
            if not isinstance(model_class, type):
                logger.warning(
                    "skipping model script %s: no top-level Model class",
                    py_file,
                )
                continue

            model_id = manifest["model_id"]
            if model_id in found:
                existing = found[model_id].source_path
                logger.warning(
                    "model_id %r already discovered at %s; keeping that one, "
                    "skipping duplicate at %s",
                    model_id, existing, py_file,
                )
                continue

            found[model_id] = DiscoveredModel(
                manifest=manifest,
                model_class=model_class,
                source_path=py_file,
            )
    return found


def discover_modalities(dirs: list[Path]) -> dict[str, Callable]:
    """Scan dirs in order for modality subpackages.

    A modality is a directory with an __init__.py that exports
    module-level MODALITY (str, MIME-style tag) and build_router (callable).
    First-found-wins on MODALITY tag collision. Returns {tag: build_router}.
    """
    found: dict[str, Callable] = {}
    tag_sources: dict[str, Path] = {}

    for d in dirs:
        if not d or not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            if not sub.is_dir() or sub.name.startswith("_"):
                continue
            init_py = sub / "__init__.py"
            if not init_py.exists():
                continue
            try:
                module = _load_package(sub)
            except Exception as e:
                logger.warning(
                    "skipping modality package %s: import failed (%s)",
                    sub, e,
                )
                continue

            tag = getattr(module, "MODALITY", None)
            if not isinstance(tag, str) or not tag:
                logger.warning(
                    "skipping modality package %s: no MODALITY string",
                    sub,
                )
                continue

            build_router = getattr(module, "build_router", None)
            if not callable(build_router):
                logger.warning(
                    "skipping modality package %s: no callable build_router",
                    sub,
                )
                continue

            if tag in found:
                existing = tag_sources[tag]
                logger.warning(
                    "MODALITY tag %r already discovered at %s; keeping that one, "
                    "skipping duplicate at %s",
                    tag, existing, sub,
                )
                continue

            found[tag] = build_router
            tag_sources[tag] = sub
    return found


def _load_script(path: Path) -> Any:
    """Import a single-file .py module given its filesystem path."""
    mod_name = f"_muse_discover_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _load_package(pkg_dir: Path) -> Any:
    """Import a package (directory with __init__.py) from its path."""
    mod_name = f"_muse_discover_mod_{pkg_dir.parent.name}_{pkg_dir.name}"
    init_py = pkg_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        init_py,
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {pkg_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
