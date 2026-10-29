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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from muse.core import config

logger = logging.getLogger(__name__)


REQUIRED_MANIFEST_KEYS = ("model_id", "modality", "hf_repo")

# A script declaring `MANIFEST["weights"] = "none"` ships no weights (its
# code is the whole model), so it need not name an `hf_repo`.
WEIGHTLESS = "none"


def is_weightless_manifest(manifest: dict) -> bool:
    """True when a script MANIFEST opts into the weightless seam."""
    return manifest.get("weights") == WEIGHTLESS


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

            required = REQUIRED_MANIFEST_KEYS
            if is_weightless_manifest(manifest):
                required = tuple(k for k in required if k != "hf_repo")
            missing = [k for k in required if k not in manifest]
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


def modality_dirs() -> list[Path]:
    """Modality scan roots: the bundled tree, then `$MUSE_MODALITIES_DIR`.

    Every discovery surface (worker routers, HF plugins, the AST-read
    seams) walks these roots in this order, so first-found-wins on a
    MODALITY tag collision means bundled shadows the env dir.
    """
    bundled = Path(__file__).resolve().parents[1] / "modalities"
    env = config.get("paths.modalities_dir")
    return [bundled] + ([Path(env)] if env else [])


# Kept for existing importers (muse.core.resolvers_hf, tests).
_default_hf_plugin_dirs = modality_dirs


def _modality_inits(dirs: list[Path], label: str):
    """Yield (MODALITY tag, __init__.py) per modality package, AST-read and
    first-found-wins per tag. An unreadable scan root is logged and skipped,
    so a bad `$MUSE_MODALITIES_DIR` never raises on the request path."""
    seen_tags: set[str] = set()
    for d in dirs:
        if not d or not d.is_dir():
            continue
        try:
            entries = sorted(d.iterdir())
        except OSError as e:
            logger.warning("%s: cannot list directory %s: %s", label, d, e)
            continue
        for sub in entries:
            init_py = sub / "__init__.py"
            if not sub.is_dir() or sub.name.startswith("_") or not init_py.exists():
                continue
            tag = _ast_extract_modality(init_py)
            if not tag or tag in seen_tags:
                continue
            seen_tags.add(tag)
            yield tag, init_py


def modality_tags() -> list[str]:
    """Single source of truth: which modality MIME tags does this muse know?

    Walks `modality_dirs()` and returns the sorted MODALITY tag list. Used
    by CLI argparse choices, test invariants, and any caller that needs
    "what modalities exist?" without paying the full router-build cost of
    `discover_modalities`.

    Reads each modality `__init__.py` via AST parsing rather than executing
    it, so callers like `muse --help` and `muse pull` don't transitively
    import fastapi (which the modality routes depend on but the bare
    `pip install muse` user may not have).
    """
    return sorted(tag for tag, _ in _modality_inits(modality_dirs(), "modality_tags"))


def _ast_extract_modality(init_py: Path) -> str | None:
    """The `MODALITY = "..."` string literal from an __init__.py, without
    exec; None when absent, computed, or not a string."""
    tag = _ast_extract_literal(init_py, "MODALITY")
    return tag if isinstance(tag, str) else None


def _reset_model_optional_paths_cache() -> None:
    """Test seam: clear the memoized default (dirs=None) scans behind
    model_optional_paths() and the other AST-read declaration seams."""
    _DECLARATION_CACHE.clear()


def model_optional_paths(dirs: list[Path] | None = None) -> dict[str, str]:
    """{request_path: modality_tag} for every modality package that
    exports MODEL_OPTIONAL_PATHS.

    Consumed by the gateway's default-model seam (see
    `muse.cli_impl.gateway`): a request to one of these paths with no
    `model` field resolves to the modality's first enabled catalog model
    instead of 400ing `model_required`. AST-based (no imports) and, for
    the default scan, memoized for the process lifetime, so it stays cheap
    enough to call on every gateway request and keeps the bare-install
    contract (no fastapi import just to compute this map).

    First-found-wins per MODALITY tag, mirroring `discover_modalities`'s
    collision rule: a duplicate tag discovered in a later dir contributes
    no paths, even if its own MODEL_OPTIONAL_PATHS differs. Pass explicit
    `dirs` to force a fresh scan (tests do this).
    """
    out: dict[str, str] = {}
    for tag, value in _declarations("MODEL_OPTIONAL_PATHS", dirs):
        if not isinstance(value, (tuple, list)):
            logger.warning("ignoring malformed MODEL_OPTIONAL_PATHS for %s", tag)
            continue
        for p in value:
            if isinstance(p, str):
                out.setdefault(p, tag)
    return out


def _ast_extract_literal(init_py: Path, name: str) -> Any:
    """The literal value of a top-level `NAME = <literal>` in an __init__.py,
    without exec; None when absent, computed, or unparsable."""
    try:
        tree = ast.parse(init_py.read_text())
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return None
    return None


def _declarations(name: str, dirs: list[Path] | None) -> list[tuple[str, Any]]:
    """[(modality tag, literal NAME)] per modality package, first-found-wins
    per tag. The default scan (dirs=None, over `modality_dirs()`) is
    memoized for the process lifetime: modality packages cannot change at
    runtime, and the gateway consults these on the request path."""
    use_cache = dirs is None
    if use_cache and name in _DECLARATION_CACHE:
        return _DECLARATION_CACHE[name]
    out: list[tuple[str, Any]] = []
    for tag, init_py in _modality_inits(dirs if dirs is not None else modality_dirs(), name):
        value = _ast_extract_literal(init_py, name)
        if value is not None:
            out.append((tag, value))
    if use_cache:
        _DECLARATION_CACHE[name] = out
    return out


_DECLARATION_CACHE: dict[str, list[tuple[str, Any]]] = {}
_CEILING_KEYS = ("max_concurrency", "max_queue_depth")


def concurrency_ceilings(dirs: list[Path] | None = None) -> dict[str, dict[str, int]]:
    """{modality tag: {"max_concurrency": n, "max_queue_depth": n}} for every
    modality package that exports CONCURRENCY_CEILING.

    A ceiling is a hard upper bound the gateway applies to every model of
    that modality, whatever its manifest or catalog entry says (a catalog
    edit cannot raise it). Positive global limits may still lower it.
    Malformed declarations are logged and ignored.
    """
    out: dict[str, dict[str, int]] = {}
    for tag, value in _declarations("CONCURRENCY_CEILING", dirs):
        if (
            isinstance(value, dict)
            and set(value) <= set(_CEILING_KEYS)
            and all(type(v) is int and v > 0 for v in value.values())
        ):
            out[tag] = dict(value)
        else:
            logger.warning("ignoring malformed CONCURRENCY_CEILING for %s", tag)
    return out


_DISCOVERY_ROUTE_KEYS = {
    "method": str, "path": str, "response_content_type": str, "mcp_tool": str,
    "file_fields": list, "media_fields": list, "example": dict,
}


def _str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _valid_text_media(value: Any) -> bool:
    """Optional ``text_media_fields``: {JSON path: MIME type} of plain-text
    media (e.g. SVG markup) that clients write to files as UTF-8."""
    return isinstance(value, dict) and all(
        isinstance(k, str) and k and isinstance(v, str) and "/" in v
        for k, v in value.items()
    )


def _valid_mcp_defaults(value: Any) -> bool:
    """Optional ``mcp_defaults``: {request field: scalar} that the generated
    MCP tool sends when the caller omits the field (e.g. a zip
    ``response_format`` whose bundle keeps cross-file references intact,
    where per-file outputs would be saved under content-hash names)."""
    return isinstance(value, dict) and all(
        isinstance(k, str) and k and isinstance(v, (str, int, float, bool))
        for k, v in value.items()
    )


_OPTIONAL_ROUTE_KEYS = {"text_media_fields", "mcp_defaults"}


def _valid_discovery_route(route: Any) -> bool:
    if not isinstance(route, dict):
        return False
    required = set(_DISCOVERY_ROUTE_KEYS) | {"request_content_type", "compat"}
    if set(route) - _OPTIONAL_ROUTE_KEYS != required:
        return False
    if "text_media_fields" in route and not _valid_text_media(route["text_media_fields"]):
        return False
    if "mcp_defaults" in route and not _valid_mcp_defaults(route["mcp_defaults"]):
        return False
    if not all(isinstance(route[k], t) for k, t in _DISCOVERY_ROUTE_KEYS.items()):
        return False
    if route["method"] not in ("GET", "POST"):
        return False
    # A GET has no body, so its content type is None and `example` holds
    # query parameters; a POST declares what it accepts.
    if route["request_content_type"] is None:
        if route["method"] != "GET":
            return False
    elif route["request_content_type"] not in (
        "application/json", "multipart/form-data",
    ):
        return False
    if route["compat"] is not None and not isinstance(route["compat"], str):
        return False
    return _str_list(route["file_fields"]) and _str_list(route["media_fields"])


def _valid_discovery(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"summary", "use_for", "not_for", "routes"}
        and isinstance(value["summary"], str)
        and _str_list(value["use_for"])
        and _str_list(value["not_for"])
        and isinstance(value["routes"], list)
        and bool(value["routes"])
        and all(_valid_discovery_route(r) for r in value["routes"])
    )


def modality_discovery(dirs: list[Path] | None = None) -> dict[str, dict]:
    """{modality tag: DISCOVERY dict} for every modality package that exports
    a well-formed DISCOVERY literal (what the capabilities document, OpenAPI
    descriptions and generated MCP tools are built from). Malformed
    declarations are logged and skipped.
    """
    out: dict[str, dict] = {}
    for tag, value in _declarations("DISCOVERY", dirs):
        if _valid_discovery(value):
            out[tag] = value
        else:
            logger.warning("ignoring malformed DISCOVERY for %s", tag)
    return out


def strict_request_parsers(dirs: list[Path] | None = None) -> dict[str, str]:
    """{request path: "module:callable"} for every modality package that
    exports STRICT_REQUEST_PARSERS.

    The gateway calls the parser on the raw body of a request to that path
    BEFORE model selection, and routes by the parsed result's `.model`
    instead of its own lenient JSON read. A parser raises an exception
    carrying safe `.status`, `.code` and `.message` to reject the request.
    """
    out: dict[str, str] = {}
    for tag, value in _declarations("STRICT_REQUEST_PARSERS", dirs):
        if not isinstance(value, dict) or not all(
            isinstance(k, str) and k.startswith("/") and isinstance(v, str) and ":" in v
            for k, v in value.items()
        ):
            logger.warning("ignoring malformed STRICT_REQUEST_PARSERS for %s", tag)
            continue
        for path, target in value.items():
            out.setdefault(path, target)
    return out


def probe_timeouts(dirs: list[Path] | None = None) -> dict[str, float]:
    """{modality tag: seconds} for every modality exporting a literal
    PROBE_TIMEOUT_SECONDS, for modalities whose representative inference
    legitimately outlasts the default probe budget (video generation)."""
    out: dict[str, float] = {}
    for tag, value in _declarations("PROBE_TIMEOUT_SECONDS", dirs):
        if type(value) in (int, float) and value > 0:
            out[tag] = float(value)
        else:
            logger.warning("ignoring malformed PROBE_TIMEOUT_SECONDS for %s", tag)
    return out


def disconnect_cancellable_paths(dirs: list[Path] | None = None) -> frozenset[str]:
    """Request paths whose modality exports CANCEL_ON_DISCONNECT_PATHS.

    For these paths the gateway drops its worker connection when the client
    disconnects (releasing its slot and refcount at once) instead of waiting
    for the worker's response. A modality declares this only when its worker
    does its own admission and stops abandoned work, so freeing the gateway
    slot early can never put two runs on the backend at once.
    """
    out: set[str] = set()
    for tag, value in _declarations("CANCEL_ON_DISCONNECT_PATHS", dirs):
        if isinstance(value, tuple) and all(
            isinstance(v, str) and v.startswith("/") for v in value
        ):
            out.update(value)
        else:
            logger.warning("ignoring malformed CANCEL_ON_DISCONNECT_PATHS for %s", tag)
    return frozenset(out)


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
            if not hf_py.is_file():
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

