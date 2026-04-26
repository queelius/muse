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

import ast
import importlib.util
import logging
import os
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


def modality_tags() -> list[str]:
    """Single source of truth: which modality MIME tags does this muse know?

    Walks the bundled modalities tree plus `$MUSE_MODALITIES_DIR` (if set)
    and returns the sorted MODALITY tag list. Used by CLI argparse choices,
    test invariants, and any caller that needs "what modalities exist?"
    without paying the full router-build cost of `discover_modalities`.

    Reads each modality `__init__.py` via AST parsing rather than executing
    it, so callers like `muse --help` and `muse pull` don't transitively
    import fastapi (which the modality routes depend on but the bare
    `pip install muse` user may not have).

    First-found-wins on collision (matches `discover_modalities` semantics).
    """
    bundled = Path(__file__).resolve().parents[1] / "modalities"
    env = os.environ.get("MUSE_MODALITIES_DIR")
    dirs = [bundled] + ([Path(env)] if env else [])

    seen: set[str] = set()
    for d in dirs:
        if not d or not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            if not sub.is_dir() or sub.name.startswith("_"):
                continue
            init_py = sub / "__init__.py"
            if not init_py.exists():
                continue
            tag = _ast_extract_modality(init_py)
            if tag and tag not in seen:
                seen.add(tag)
    return sorted(seen)


def _ast_extract_modality(init_py: Path) -> str | None:
    """Pull the `MODALITY = "..."` literal from an __init__.py without exec.

    Only handles the simple case (top-level `MODALITY = "string-literal"`),
    which matches the convention every bundled modality follows. Anything
    fancier (computed values, conditional assignment) returns None.
    """
    try:
        tree = ast.parse(init_py.read_text())
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Name) and target.id == "MODALITY"):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    return None


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


REQUIRED_HF_PLUGIN_KEYS = (
    "modality", "runtime_path", "pip_extras", "system_packages",
    "priority", "sniff", "resolve", "search",
)


def discover_hf_plugins(dirs: list[Path]) -> list[dict]:
    """Scan dirs for `<dir>/<name>/hf.py` files exporting HF_PLUGIN.

    Each hf.py is loaded as a single-file module via spec_from_file_location
    (mangled name) so the modality package's __init__.py is not executed.
    This preserves the bare-install contract: `muse pull` works without
    fastapi installed because plugins don't transitively pull in routes.

    Validation: HF_PLUGIN must be a dict with all REQUIRED_HF_PLUGIN_KEYS.
    Missing keys, type mismatches, or import errors log a warning and skip
    the plugin. Discovery never raises.

    Returns plugins sorted by (priority asc, modality asc) so the dispatcher
    iterates specific shapes before catch-alls and the order is deterministic
    across machines.
    """
    found: list[dict] = []
    for d in dirs:
        if not d or not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            if not sub.is_dir() or sub.name.startswith("_"):
                continue
            hf_py = sub / "hf.py"
            if not hf_py.exists():
                continue
            try:
                module = _load_hf_plugin_script(hf_py)
            except Exception as e:
                logger.warning(
                    "skipping HF plugin %s: import failed (%s)", hf_py, e,
                )
                continue
            plugin = getattr(module, "HF_PLUGIN", None)
            if not isinstance(plugin, dict):
                logger.warning(
                    "skipping HF plugin %s: no top-level HF_PLUGIN dict", hf_py,
                )
                continue
            missing = [k for k in REQUIRED_HF_PLUGIN_KEYS if k not in plugin]
            if missing:
                logger.warning(
                    "skipping HF plugin %s: missing required keys %s",
                    hf_py, missing,
                )
                continue
            found.append(plugin)
    return sorted(found, key=lambda p: (p["priority"], p["modality"]))


def _load_hf_plugin_script(path: Path) -> Any:
    """Import hf.py as a single-file module. Bypasses package __init__.py.

    The mangled module name avoids sys.modules collisions when multiple
    modalities each ship a hf.py.
    """
    mod_name = f"_muse_hf_plugin_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _load_script(path: Path) -> Any:
    """Import a single-file .py module given its filesystem path.

    Two paths depending on where the script lives:

      1. Inside the installed muse package tree (canonical name exists,
         e.g. `muse.models.kokoro_82m`): use standard
         `importlib.import_module`. This ensures the parent-child module
         chain is fully set up in `sys.modules` AND that the parent
         package's __dict__ has the submodule as an attribute. Without
         that, downstream `importlib.import_module("muse.models.foo")`
         calls walk the hierarchy via the normal machinery and fail
         with "cannot import name 'models' from 'muse'" because
         `spec_from_file_location` registers the module under the
         canonical name but doesn't rebuild the parent chain.

      2. Outside the muse tree (user/env dirs, no canonical name):
         use `spec_from_file_location` with a mangled private name.
         Keeps user scripts from colliding with real modules in
         `sys.modules`.
    """
    canonical = _canonical_module_name(path)
    if canonical is not None:
        # Canonical path: let Python's normal import machinery handle
        # parent-package bookkeeping. Works because `muse` and
        # `muse.models` (or whichever intermediate packages) are real
        # on-disk packages reachable via sys.path.
        return importlib.import_module(canonical)

    # Non-canonical (external user/env dir): fall back to the
    # spec_from_file_location approach with a mangled name. These
    # scripts by design DON'T have a canonical parent chain; the
    # mangled name avoids sys.modules collisions with real modules.
    mod_name = f"_muse_discover_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_module_name(path: Path) -> str | None:
    """Return `'muse.models.kokoro_82m'` if path is inside the muse package.

    Otherwise None: script lives outside muse (user or env dir) and will
    get a mangled private module name instead.
    """
    try:
        import muse  # local import so discovery module has no hard dep
    except ImportError:
        return None
    muse_init = getattr(muse, "__file__", None)
    if not muse_init:
        return None
    muse_dir = Path(muse_init).resolve().parent
    try:
        rel = path.resolve().relative_to(muse_dir)
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if not parts or parts[0] == "__pycache__":
        return None
    return ".".join(["muse", *parts])


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


def _default_hf_plugin_dirs() -> list[Path]:
    """Default scan paths: bundled modalities + $MUSE_MODALITIES_DIR if set.

    Mirrors `modality_tags()` and `discover_modalities` ordering so all
    discovery surfaces walk the same roots in the same precedence.
    """
    bundled = Path(__file__).resolve().parents[1] / "modalities"
    env = os.environ.get("MUSE_MODALITIES_DIR")
    return [bundled] + ([Path(env)] if env else [])
