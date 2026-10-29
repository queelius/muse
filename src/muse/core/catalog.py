"""Known-models catalog: what can be pulled, what's been pulled.

The set of known models is not hardcoded. It is discovered at first
access by scanning `src/muse/models/*.py` for scripts that define a
top-level `MANIFEST` dict and a `Model` class (see `muse.core.discovery`).
Each MANIFEST's fields are projected onto the stable `CatalogEntry`
shape that the rest of muse (CLI, server, worker) consumes.

Structure:
    known_models() -> dict[model_id, CatalogEntry]  # cached, discovery-driven
    catalog.json (on disk): dict[model_id, {
        pulled_at,                     # ISO 8601 timestamp
        hf_repo,                       # original HF repo id
        revision,                      # immutable primary HF commit
        artifact_provenance,           # complete immutable download receipt
        local_dir,                     # HF snapshot_download cache path
        venv_path,                     # dedicated venv for this model
        python_path,                   # <venv_path>/bin/python for workers
    }]

A weightless script (`MANIFEST["weights"] = "none"`) has no hf_repo or
revision: its row carries `weights: "none"`, an empty owned `local_dir`
under `<catalog>/local/`, and a one-item `code` receipt over the script
source (`kind`, `model_id`, `muse_version`, `source_sha256`).

This module is the public catalog facade. It owns per-model operator state
(enable/disable, device and GPU-layer pins) and backend loading with its
provenance checks, and re-exports the names callers use from the modules
that implement the rest:

    catalog_store   catalog.json location, I/O, schema, transaction lock
    catalog_locks   per-model pull lock, resource lease, storage lock
    catalog_models  CatalogEntry, known_models(), get_manifest()
    catalog_pull    pull() and its bundled/resolver transactions
    catalog_remove  remove() and owned-path purging
"""
from __future__ import annotations

import importlib
import os
import stat
from pathlib import Path
from typing import Any

# Imported names this module does not use itself are re-exported: callers
# import them from muse.core.catalog.
from muse.core.catalog_locks import (
    _model_pull_lock,
    _model_resource_lease,
    _storage_cache_lock,
)
from muse.core.catalog_models import (
    _KNOWN_MODELS_LOCK,
    CatalogEntry,
    _bundled_models_dir,
    _manifest_to_catalog_entry,
    _reset_known_models_cache,
    get_manifest,
    is_weightless,
    known_models,
    list_known,
)
from muse.core.catalog_pull import (
    _bundled_download_spec,
    _installed_muse_requirement,
    _muse_repo_root,
    _muse_server_install_args,
    _pull_bundled,
    _pull_bundled_transaction,
    _pull_resolved_transaction,
    _pull_via_resolver,
    _validate_lora_capabilities,
    check_system_packages,
    pull,
)
from muse.core.catalog_remove import (
    _other_catalog_path_reference,
    _owned_weights_purge_path,
    _purge_owned_directory,
    remove,
)
from muse.core.catalog_store import (
    _CATALOG_WRITE_LOCK,
    CatalogError,
    ModelIdConflictError,
    ModelInUseError,
    StorageBusyError,
    _catalog_dir,
    _catalog_path,
    _ensure_owned_directory,
    _read_catalog,
    _reset_read_catalog_cache,
    _validate_model_id_for_fs,
    _write_catalog,
    is_pulled,
)


def is_enabled(model_id: str) -> bool:
    """Return True if model is pulled AND enabled in the catalog."""
    catalog = _read_catalog()
    if model_id not in catalog:
        return False
    return catalog[model_id].get("enabled", True)


def set_enabled(model_id: str, enabled: bool) -> None:
    """Toggle the `enabled` flag for a pulled model.

    Raises KeyError if the model is not in the catalog (not pulled).
    Other catalog fields are preserved.

    Holds _CATALOG_WRITE_LOCK for the full read->mutate->write sequence so
    concurrent mutations on different keys do not lose each other's updates.
    """
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        if model_id not in catalog:
            raise KeyError(f"model {model_id!r} is not pulled")
        catalog[model_id]["enabled"] = bool(enabled)
        _write_catalog(catalog)
    # `enabled` flows through known_models() into the CatalogEntry
    # consumers see; without the reset, `muse models list` would
    # display the stale state until the process restarts.
    _reset_known_models_cache()


VALID_DEVICE_OVERRIDES = ("auto", "cpu", "cuda", "mps")


def set_device_override(model_id: str, device: str | None) -> None:
    """Set or clear the per-model device override for a pulled model.

    `device` in {auto, cpu, cuda, mps} pins the model's load device,
    overriding both the manifest `capabilities.device` pin and the
    supervisor `--device` flag (see `load_backend`'s precedence). The
    special value "auto" un-pins a cpu-pinned model so the runtime's
    `select_device` picks cuda when a GPU is present. Passing ``None``
    removes the override entirely (revert to manifest pin / --device).

    Catalog state only: takes effect on the model's next cold load. To
    apply it to an already-resident worker, evict or restart that worker.

    Raises ValueError for an unrecognized device label and KeyError when
    the model is not pulled. Holds _CATALOG_WRITE_LOCK for the full
    read->mutate->write so concurrent mutations on different keys do not
    clobber each other (mirrors `set_enabled`).
    """
    if device is not None and device not in VALID_DEVICE_OVERRIDES:
        raise ValueError(
            f"invalid device {device!r}; expected one of "
            f"{', '.join(VALID_DEVICE_OVERRIDES)} or None to clear"
        )
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        if model_id not in catalog:
            raise KeyError(f"model {model_id!r} is not pulled")
        if device is None:
            catalog[model_id].pop("device_override", None)
        else:
            catalog[model_id]["device_override"] = device
        _write_catalog(catalog)
    # device_override is read live from the catalog in load_backend, but
    # `muse models info` surfaces it via known_models()-adjacent reads;
    # reset for display consistency (mirrors set_enabled).
    _reset_known_models_cache()


def set_gpu_layers_override(model_id: str, n: int | None) -> None:
    """Set or clear the per-model llama.cpp GPU-layer pin for a pulled model.

    `n` is the llama.cpp `n_gpu_layers` value: -1 = offload every layer the
    GPU fits, 0 = pure CPU, N > 0 = first N layers on GPU (rest on CPU).
    Stored as the TOP-LEVEL catalog field `gpu_layers_override` (operator
    state, mirroring `device_override` -- NOT part of the manifest). Passing
    ``None`` removes the pin (revert to capabilities.n_gpu_layers / the
    runtime default).

    Catalog state only: takes effect on the model's next cold load. To
    apply it to an already-resident worker, evict or restart that worker.

    Raises ValueError for a non-int or n < -1 and KeyError when the model
    is not pulled. Holds _CATALOG_WRITE_LOCK for the full
    read->mutate->write (mirrors `set_device_override`).
    """
    if n is not None:
        if isinstance(n, bool) or not isinstance(n, int) or n < -1:
            raise ValueError(
                f"invalid gpu layers {n!r}; expected an int >= -1 "
                "(-1 = all layers on GPU, 0 = pure CPU) or None to clear"
            )
    with _CATALOG_WRITE_LOCK:
        catalog = _read_catalog()
        if model_id not in catalog:
            raise KeyError(f"model {model_id!r} is not pulled")
        if n is None:
            catalog[model_id].pop("gpu_layers_override", None)
        else:
            catalog[model_id]["gpu_layers_override"] = n
        _write_catalog(catalog)
    _reset_known_models_cache()


def _import_backend_module(module_path: str):
    """Local indirection for `importlib.import_module`.

    Why this wrapper exists: tests need to stub out the catalog's backend
    import to install a fake module without fetching real ML deps. Patching
    `importlib.import_module` directly (or `muse.core.catalog.importlib.
    import_module`) mutates the shared global `importlib` module object, so
    the stub also intercepts calls from `muse.core.discovery` and anywhere
    else that imports at test time. Indirecting through a catalog-local
    function gives tests a patching target (`muse.core.catalog.
    _import_backend_module`) that only affects catalog's import path.
    """
    return importlib.import_module(module_path)


def _validate_remote_code_catalog_pins(
    model_id: str,
    manifest: dict,
    entry_data: dict,
) -> None:
    """Refuse a local remote-code snapshot that predates current review pins."""
    capabilities = manifest.get("capabilities") or {}
    if not capabilities.get("trust_remote_code"):
        return

    mismatches: list[str] = []
    expected_revision = manifest.get("revision")
    if expected_revision and entry_data.get("revision") != expected_revision:
        mismatches.append("model revision")
    expected_code_revision = capabilities.get("code_revision")
    if (
        expected_code_revision
        and entry_data.get("code_revision") != expected_code_revision
    ):
        mismatches.append("external code revision")
    if mismatches:
        raise RuntimeError(
            f"model {model_id!r} does not match the current reviewed "
            f"remote-code pin ({', '.join(mismatches)}); re-run "
            f"`muse pull {model_id}` before loading it"
        )


def _validate_bundled_catalog_provenance(
    model_id: str,
    entry: CatalogEntry,
    entry_data: dict,
) -> None:
    """Reject script snapshots that do not match their complete receipt."""
    # Resolver downloads persist their own manifest/source and use resolver
    # pin validation. A receipt is written only by the script-based pull path.
    if "manifest" in entry_data:
        return
    if is_weightless(entry):
        _validate_weightless_catalog_provenance(model_id, entry, entry_data)
        return
    if not entry.bundled and "artifact_provenance" not in entry_data:
        return

    spec = _bundled_download_spec(model_id, entry)
    expected = (
        list(spec.artifact_provenance)
        if spec.artifact_provenance is not None
        else None
    )
    mismatches: list[str] = []
    if entry_data.get("revision") != spec.revision:
        mismatches.append("primary revision")
    if entry_data.get("artifact_provenance") != expected:
        mismatches.append("artifact receipt")
    if mismatches:
        raise RuntimeError(
            f"model {model_id!r} does not match the current immutable "
            f"artifact provenance ({', '.join(mismatches)}); re-run "
            f"`muse pull {model_id}` before loading it"
        )


# The receipt fields that identify the pulled code. `muse_version` records
# which release pulled it; it is informational, so a `muse models refresh`
# (which upgrades the venv's muse without re-pulling) does not block loads.
_CODE_RECEIPT_IDENTITY = ("kind", "model_id", "source_sha256")


def _validate_weightless_catalog_provenance(
    model_id: str,
    entry: CatalogEntry,
    entry_data: dict,
) -> None:
    """Reject a weightless row whose code receipt no longer matches the script."""
    # Re-validates the manifest (no revision / hf_repo / downloads) and
    # hashes the script as it is on disk now.
    spec = _bundled_download_spec(model_id, entry)
    expected = (spec.artifact_provenance or ({},))[0]
    mismatches: list[str] = []
    if entry_data.get("weights") != "none" or "revision" in entry_data:
        mismatches.append("weightless catalog row")
    receipt = entry_data.get("artifact_provenance")
    stored = receipt[0] if isinstance(receipt, list) and len(receipt) == 1 else None
    if not isinstance(stored, dict):
        mismatches.append("code receipt")
    else:
        if any(stored.get(key) != expected[key] for key in _CODE_RECEIPT_IDENTITY):
            mismatches.append("code receipt")
    if mismatches:
        raise RuntimeError(
            f"model {model_id!r} does not match the current code "
            f"provenance ({', '.join(mismatches)}); re-run "
            f"`muse pull {model_id}` before loading it"
        )


def _validate_resolver_catalog_provenance(
    model_id: str,
    manifest: dict,
    entry_data: dict,
) -> None:
    """Reject resolver snapshots stale against current pins or their receipt."""
    if "manifest" not in entry_data or not entry_data.get("source"):
        return

    mismatches: list[str] = []
    expected_revision = manifest.get("revision")
    if expected_revision and entry_data.get("revision") != expected_revision:
        mismatches.append("primary revision")

    expected_receipt = manifest.get("artifact_provenance")
    if expected_receipt is not None:
        if entry_data.get("artifact_provenance") != expected_receipt:
            mismatches.append("artifact receipt")
        if expected_revision:
            primary_repo = manifest.get("hf_repo")
            primary_revisions = {
                item.get("revision")
                for item in expected_receipt
                if isinstance(item, dict) and item.get("repo_id") == primary_repo
            } if isinstance(expected_receipt, list) else set()
            if expected_revision not in primary_revisions:
                mismatches.append("manifest artifact receipt")
    if mismatches:
        raise RuntimeError(
            f"model {model_id!r} does not match the current immutable resolver "
            f"provenance ({', '.join(mismatches)}); re-run "
            f"`muse pull {model_id}` before loading it"
        )


def _validate_model_provenance(
    model_id: str, entry: CatalogEntry, entry_data: dict, manifest: dict,
) -> None:
    """One artifact policy for loader execution and serve preflight."""
    _validate_remote_code_catalog_pins(model_id, manifest, entry_data)
    _validate_resolver_catalog_provenance(model_id, manifest, entry_data)
    _validate_bundled_catalog_provenance(model_id, entry, entry_data)


def _stat_if_present(path: str) -> os.stat_result | None:
    """``stat`` a path (following symlinks); None only when it is absent.

    ``Path.is_dir()`` / ``is_file()`` cannot tell "absent" from
    "inaccessible": since Python 3.14 they are ``os.path.isdir`` /
    ``isfile``, which swallow every ``OSError`` (EACCES, EIO, ...) as
    False, whereas 3.10-3.13 re-raised errors other than ENOENT, ENOTDIR,
    EBADF and ELOOP. Only absence is mapped to None here; any other
    ``OSError`` propagates so the caller reports an inaccessible
    environment on every supported Python. That deliberately includes a
    symlink loop (ELOOP), which ``is_dir()`` used to report as missing: a
    loop is a broken environment to inspect, not an absent download.
    """
    try:
        return Path(path).stat()
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return None


def model_load_blocker(model_id: str, entry_data: dict) -> str | None:
    """Return a local cold-load prerequisite failure without loading a model.

    This checks reviewed provenance and the recorded worker environment,
    not dependency imports, inference, or memory capacity. It neither
    downloads nor rewrites artifacts. Legacy embedded ``load_backend``
    callers need not use a per-model interpreter; serving does.
    """
    try:
        manifest = get_manifest(model_id)
        entry = known_models()[model_id]
        _validate_model_provenance(model_id, entry, entry_data, manifest)
    except KeyError:
        return f"model {model_id!r} is not discoverable; re-run `muse pull {model_id}`"
    except (RuntimeError, ValueError, OSError) as exc:
        return str(exc)

    try:
        local_dir = entry_data.get("local_dir")
        local_stat = _stat_if_present(local_dir) if local_dir else None
        if local_stat is None or not stat.S_ISDIR(local_stat.st_mode):
            return f"model files are missing; re-run `muse pull {model_id}`"
        python_path = entry_data.get("python_path")
        python_stat = _stat_if_present(python_path) if python_path else None
        if (
            python_stat is None or not stat.S_ISREG(python_stat.st_mode)
            or not os.access(python_path, os.X_OK)
        ):
            return f"worker interpreter is missing or not executable; re-run `muse pull {model_id}`"
    except OSError:
        return (
            "model environment is inaccessible; check filesystem permissions "
            f"and mounts or re-run `muse pull {model_id}`"
        )
    return None


def load_backend(model_id: str, **kwargs) -> Any:
    """Import backend class and instantiate it.

    `backend_path` has the form "package.module:ClassName". The class
    is expected to accept (hf_repo, local_dir, **kwargs) in its constructor.

    For resolver-pulled models, manifest.capabilities are merged into the
    kwargs (caller's explicit kwargs win). This lets generic runtimes
    like LlamaCppModel pull `gguf_file`, `chat_template`, `context_length`
    out of the persisted manifest without the worker having to know
    those keys exist. `model_id` is also injected so generic runtimes
    (one class, many models) know which model they're loading.
    """
    catalog_known = known_models()
    if model_id not in catalog_known:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(catalog_known)}")
    if not is_pulled(model_id):
        raise RuntimeError(f"model {model_id!r} not pulled; run `muse pull {model_id}`")
    entry = catalog_known[model_id]
    catalog = _read_catalog()
    entry_data = catalog[model_id]
    manifest = get_manifest(model_id)
    _validate_model_provenance(model_id, entry, entry_data, manifest)
    module_path, class_name = entry.backend_path.split(":", 1)
    module = _import_backend_module(module_path)
    cls = getattr(module, class_name)
    local_dir = entry_data["local_dir"]
    # entry.extra holds capabilities from either the bundled MANIFEST
    # (read live from source by known_models() each call) or the persisted
    # manifest in catalog.json (resolver-pulled). Reading from entry here
    # means bundled scripts' capabilities (e.g. device: cpu on kokoro) are
    # honored at load time, not just resolver-pulled ones.
    capabilities = entry.extra
    merged: dict = {"model_id": model_id, **capabilities, **kwargs}
    # Device precedence, most authoritative first:
    #   1. catalog `device_override`  (operator, via `muse models set-device`)
    #   2. manifest `capabilities.device` pin (model-author affinity, e.g.
    #      kokoro's "cpu") -- overrides the supervisor --device flag
    #   3. caller kwargs device (the --device flag), already folded into kwargs
    #   4. "auto" (runtime select_device picks cuda if available, else cpu)
    # The override beats even the manifest pin so an operator can force a
    # cpu-pinned model onto cuda (or back to cpu to save VRAM) per deployment
    # without editing the bundled script. override="auto" un-pins a model to
    # auto-detect. Other capability keys still lose to kwargs (the documented
    # contract); device is the exception because it is a placement preference.
    override = entry_data.get("device_override")
    if override:
        merged["device"] = override
    elif "device" in capabilities and capabilities["device"] != "auto":
        merged["device"] = capabilities["device"]
    # GPU-layers precedence (spec 2026-07-08), most authoritative first:
    #   1. catalog `gpu_layers_override` (operator, via
    #      `muse models set-gpu-layers`)
    #   2. manifest `capabilities.n_gpu_layers` (already in `merged` via the
    #      capabilities splat above)
    #   3. runtime default (-1 in LlamaCppModel: everything the GPU fits)
    # Applied AFTER the kwargs merge, like device_override, so the operator
    # pin also beats caller kwargs: it is a placement preference.
    gpu_layers = entry_data.get("gpu_layers_override")
    if gpu_layers is not None:
        merged["n_gpu_layers"] = gpu_layers
    return cls(hf_repo=entry.hf_repo, local_dir=local_dir, **merged)


def _human_size(b: int) -> str:
    """Format bytes as 'N.N GB' / 'N MB' / 'N KB' for table display.

    Returns '-' for 0 (callers use 0 as the missing-size sentinel).
    GB uses one decimal; MB and KB are integer-rounded to keep the
    column narrow enough for table alignment.
    """
    if b == 0:
        return "-"
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.0f} MB"
    return f"{b / 1024:.0f} KB"


