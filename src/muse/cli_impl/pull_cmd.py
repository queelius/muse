"""`muse pull` and the catalog-only model verbs (remove, set-device,
set-gpu-layers).

`muse.cli` only binds arguments; each function here returns the command's
exit code and prints exactly what the command shows.
"""
from __future__ import annotations

import typer


def run_pull(
    identifier: str,
    *,
    no_probe: bool,
    verbose: bool,
    base: str | None,
    model_id: str | None,
) -> int:
    """Pull, then probe (unless `no_probe`); a failed probe disables it.

    Unrecognized pull failures re-raise so real bugs keep their traceback.
    """
    from muse.core.catalog import pull as _pull
    from muse.core.catalog_store import (
        ModelIdConflictError,
        ModelInUseError,
        _validate_model_id_for_fs,
    )
    from muse.core.venv import install_output_mode
    # Always register the HF resolver before dispatching. The arg may
    # be a URI directly, OR a curated alias that expands to a URI
    # inside pull(); the old conditional "only import when :// is in
    # the arg" missed the second case and crashed with "no resolver
    # for scheme 'hf'". Importing is near-free (heavy huggingface_hub
    # imports happen on actual resolve(), not on module import).
    import muse.core.resolvers_hf  # noqa: F401  (registers HFResolver on import)
    import muse.core.resolvers_file  # noqa: F401  (registers FileResolver on import)

    if model_id is not None:
        if "://" not in identifier:
            typer.echo(
                "error: --id only applies to resolver URIs (hf://..., file://...)",
                err=True,
            )
            return 2
        try:
            _validate_model_id_for_fs(model_id)
        except ValueError as e:
            typer.echo(f"error: {e}", err=True)
            return 2

    with install_output_mode(verbose=verbose):
        try:
            pulled_id = _pull(identifier, base_override=base, model_id=model_id)
        except KeyError as e:
            typer.echo(f"error: {e}", err=True)
            return 2
        except (ModelIdConflictError, ModelInUseError) as e:
            typer.echo(f"error: {e}", err=True)
            return 2
        except Exception as e:  # noqa: BLE001
            # A gated/private HF repo pulled without auth raises a deep
            # huggingface_hub error whose uncaught traceback is ~200 lines
            # of noise. Translate the recognized access failures into a
            # one-line actionable message; re-raise anything else so real
            # bugs still surface with a full traceback.
            from muse.cli_impl.pull_errors import friendly_pull_error
            msg = friendly_pull_error(identifier, e)
            if msg is None:
                raise
            typer.echo(msg, err=True)
            return 1
        typer.echo(f"pulled {identifier}")

        if no_probe:
            return 0

        # Probe-on-pull: a successful pull means the model loads and runs a
        # representative inference. A failed probe leaves the weights and
        # venv in place (a retry needs no re-download) but disables the model.
        try:
            from muse.cli_impl.probe import run_probe
            rc = run_probe(
                model_id=pulled_id, no_inference=False, device=None, as_json=False,
            )
        except Exception as e:  # noqa: BLE001
            typer.echo(f"error: probe-on-pull failed: {e}", err=True)
            rc = 5
        if rc != 0:
            fail_pull_probe(pulled_id)
            return 1
    return 0


def fail_pull_probe(model_id: str) -> None:
    """Disable a model whose post-pull probe failed and say how to recover."""
    from muse.core.catalog import _read_catalog, set_enabled

    try:
        installed = model_id in _read_catalog()
    except Exception:  # noqa: BLE001
        installed = False
    if not installed:
        typer.echo(
            "error: the pull finished but its post-pull probe could not run; "
            "run `muse models probe <id>` to verify the model before serving it",
            err=True,
        )
        return
    try:
        set_enabled(model_id, False)
        state = "has been disabled"
    except Exception as e:  # noqa: BLE001
        state = f"could not be disabled ({e})"
    typer.echo(
        f"error: {model_id} was installed but failed its post-pull probe, so it "
        f"{state}. Fix the cause above, then run `muse models probe {model_id}` "
        f"and `muse models enable {model_id}`. (Pulling for another device? "
        f"Use --no-probe.)",
        err=True,
    )


def run_remove(model_id: str, *, purge: bool) -> int:
    """`muse models remove`."""
    from muse.core.catalog import CatalogError, remove
    try:
        remove(model_id, purge=purge)
    except CatalogError as e:
        typer.echo(f"error: {e}", err=True)
        return 2
    suffix = " (purged unshared Muse-owned files)" if purge else ""
    typer.echo(f"removed {model_id} from catalog{suffix}")
    return 0


def run_set_device(model_id: str, device: str | None, *, clear: bool) -> int:
    """`muse models set-device` (`device` is the enum value or None)."""
    from muse.core.catalog import set_device_override

    if clear:
        target = None
    elif device is None:
        typer.echo(
            "error: provide a device (auto|cpu|cuda|mps) or pass --clear",
            err=True,
        )
        return 2
    else:
        target = device

    try:
        set_device_override(model_id, target)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        return 2

    if target is None:
        typer.echo(
            f"cleared device override for {model_id} "
            "(takes effect on next cold load)"
        )
    else:
        typer.echo(
            f"set {model_id} device override -> {target} "
            "(takes effect on next cold load)"
        )
    return 0


def run_set_gpu_layers(
    model_id: str, n: int | None, *, all_layers: bool, clear: bool,
) -> int:
    """`muse models set-gpu-layers`: exactly one of `n`, --all, --clear."""
    from muse.core.catalog import known_models, set_gpu_layers_override

    given = (n is not None, all_layers, clear)
    if sum(given) != 1:
        typer.echo(
            "error: provide exactly one of a layer count (int >= 0), "
            "--all, or --clear",
            err=True,
        )
        return 2

    if clear:
        target = None
    elif all_layers:
        target = -1
    else:
        target = n

    if target is not None:
        entry = known_models().get(model_id)
        # entry is None (unknown / never-pulled model): skip the GGUF
        # capability check here and fall through to set_gpu_layers_override
        # below, whose KeyError produces the uniform "not pulled" error
        # instead of the misleading "is not a GGUF model" refusal.
        if entry is not None:
            capabilities = entry.extra or {}
            if not capabilities.get("gguf_file"):
                typer.echo(
                    f"error: {model_id!r} is not a GGUF model (no "
                    "capabilities.gguf_file); n_gpu_layers only applies to "
                    "llama.cpp runtimes and would be silently ignored",
                    err=True,
                )
                return 2

    try:
        set_gpu_layers_override(model_id, target)
    except (KeyError, ValueError) as e:
        typer.echo(f"error: {e}", err=True)
        return 2

    if target is None:
        typer.echo(
            f"cleared gpu-layers pin for {model_id} "
            "(takes effect on next cold load)"
        )
    else:
        typer.echo(
            f"set {model_id} gpu layers -> {target} "
            "(takes effect on next cold load; run `muse models probe "
            f"{model_id}` to re-measure VRAM)"
        )
    return 0
