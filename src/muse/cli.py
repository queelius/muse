"""`muse` CLI -- admin commands only.

The CLI surface is deliberately minimal and modality-agnostic:

    muse serve                    start the HTTP server
    muse pull <model-id>          download weights + install deps
    muse models list              list known/pulled models (all modalities)
    muse models info <model-id>   show catalog entry
    muse models remove <model-id> unregister from catalog

Generation endpoints are reached via HTTP (the canonical interface):
    - Python: muse.modalities.audio_speech.SpeechClient,
              muse.modalities.image_generation.GenerationsClient
    - Shell:  curl -X POST http://host:8000/v1/audio/speech ...
    - LLMs:   muse mcp (MCP server bridging muse to LLM clients)

Built on typer (a thin click wrapper); argparse-era v0.x.x shipped its
own parser, the typer port landed in v0.39.0. Existing tests that
invoke `python -m muse.cli ...` keep working unchanged because typer
preserves argv handling.

Heavy imports (torch, diffusers) are kept out of this module so
`muse --help` stays instant. Command implementations live in
`muse.cli_impl.*` and import what they need when invoked.
"""
from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated


log = logging.getLogger("muse")


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Device(str, Enum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


class SearchSort(str, Enum):
    downloads = "downloads"
    lastModified = "lastModified"
    likes = "likes"


class McpFilter(str, Enum):
    all = "all"
    admin = "admin"
    inference = "inference"


class TelemetryFormat(str, Enum):
    json = "json"
    csv = "csv"


# Top-level Typer app. rich_markup_mode lets help strings use [bold]
# and similar inline rich tags; we don't lean on it heavily but it
# costs nothing.
_TYPER_SETTINGS = {
    "no_args_is_help": True,
    "rich_markup_mode": "rich",
    "context_settings": {"help_option_names": ["-h", "--help"]},
}
app = typer.Typer(
    name="muse",
    help="Multi-modality generation server + admin CLI",
    **_TYPER_SETTINGS,
)


def _subapp(name: str, help: str) -> typer.Typer:
    """Create a command group with the top-level app's shared settings."""
    sub = typer.Typer(name=name, help=help, **_TYPER_SETTINGS)
    app.add_typer(sub, name=name)
    return sub


models_app = _subapp("models", "manage the model catalog")
config_app = _subapp("config", "inspect and edit muse's config.yaml")
doctor_app = _subapp(
    "doctor", "diagnose local muse state and owned runtime resources",
)
storage_app = _subapp("storage", "inspect and reclaim Muse-owned disk space")
telemetry_app = _subapp(
    "telemetry", "view, export, and manage request traces and resource metrics",
)


def _version_callback(value: bool) -> bool:
    """Print the installed distribution version for eager version flags."""
    if value:
        from muse import __version__

        typer.echo(__version__)
        raise typer.Exit()
    return value


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="show the installed muse version and exit",
        ),
    ] = False,
    log_level: Annotated[
        LogLevel,
        typer.Option("--log-level", help="root logger verbosity"),
    ] = LogLevel.INFO,
) -> None:
    """Root callback: configure logging before any subcommand runs."""
    logging.basicConfig(
        level=getattr(logging, log_level.value), format="%(message)s",
    )


# Top-level subcommands ------------------------------------------------------


@doctor_app.command("resources")
def doctor_resources(
    repair: Annotated[
        bool,
        typer.Option(
            "--repair",
            help=(
                "remove stale records and terminate only identity-verified "
                "orphan Muse process leaders (supervisors, workers, and "
                "admin jobs)"
            ),
        ),
    ] = False,
    grace: Annotated[
        float,
        typer.Option(
            "--grace",
            min=0.0,
            help="seconds to wait before force-killing a verified orphan",
        ),
    ] = 5.0,
) -> None:
    """Inspect Muse's registry without scanning unrelated host processes."""
    from muse.cli_impl.resource_doctor import run_resource_doctor

    raise typer.Exit(run_resource_doctor(repair=repair, grace=grace))


@doctor_app.command("storage")
def doctor_storage(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="emit a machine-readable storage report"),
    ] = False,
) -> None:
    """Inspect owned and adjacent shared storage without changing it."""
    from muse.cli_impl.storage import run_storage_doctor

    raise typer.Exit(run_storage_doctor(json_output=json_output))


@storage_app.command("prune")
def storage_prune(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="show the cleanup plan without deleting"),
    ] = False,
    include_unreferenced: Annotated[
        bool,
        typer.Option(
            "--include-unreferenced",
            help=(
                "also delete old Muse-owned venvs/weights with no catalog "
                "reference; these may have been intentionally retained"
            ),
        ),
    ] = False,
    older_than_hours: Annotated[
        float,
        typer.Option(
            "--older-than-hours",
            min=0.0,
            help="only prune items untouched for at least this many hours",
        ),
    ] = 24.0,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="emit machine-readable results"),
    ] = False,
) -> None:
    """Delete old partial/staging data; unreferenced models require opt-in."""
    from muse.cli_impl.storage import run_storage_prune

    raise typer.Exit(run_storage_prune(
        dry_run=dry_run,
        include_unreferenced=include_unreferenced,
        older_than_hours=older_than_hours,
        json_output=json_output,
    ))


@storage_app.command("dedupe")
def storage_dedupe(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="report what would be linked without changing files"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="emit machine-readable results"),
    ] = False,
) -> None:
    """Hardlink identical package files (>= 1 MiB) across model venvs."""
    from muse.cli_impl.storage import run_storage_dedupe

    raise typer.Exit(run_storage_dedupe(dry_run=dry_run, json_output=json_output))


def _run_telemetry(fn, /, *, param_hint: str | None = None, **kwargs) -> None:
    try:
        code = fn(**kwargs)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint=param_hint) from exc
    raise typer.Exit(code)


@telemetry_app.command("status")
def telemetry_status(
    as_json: Annotated[bool, typer.Option("--json", help="machine-readable output")] = False,
) -> None:
    """Show the local telemetry database location, bounds, and event counts."""
    from muse.cli_impl.telemetry import run_status
    raise typer.Exit(run_status(as_json=as_json))


@telemetry_app.command("summary")
def telemetry_summary(
    since: Annotated[str, typer.Option(help="lookback duration, e.g. 6h or 7d")] = "24h",
    as_json: Annotated[bool, typer.Option("--json", help="machine-readable output")] = False,
) -> None:
    """Compare cold and hot latency, peak VRAM, and request-linked evictions."""
    from muse.cli_impl.telemetry import run_summary
    _run_telemetry(run_summary, param_hint="--since", since=since, as_json=as_json)


@telemetry_app.command("traces")
def telemetry_traces(
    since: Annotated[str, typer.Option(help="lookback duration, e.g. 1h or 2d")] = "24h",
    limit: Annotated[int, typer.Option(min=1, max=1000, help="maximum traces")] = 50,
    model_id: Annotated[Optional[str], typer.Option("--model", help="exact model id filter")] = None,
    modality: Annotated[Optional[str], typer.Option(help="exact modality filter")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="machine-readable output")] = False,
) -> None:
    """List individual end-to-end request traces."""
    from muse.cli_impl.telemetry import run_traces
    _run_telemetry(
        run_traces,
        param_hint="--since",
        since=since,
        limit=limit,
        model_id=model_id,
        modality=modality,
        as_json=as_json,
    )


@telemetry_app.command("vram")
def telemetry_vram(
    since: Annotated[str, typer.Option(help="lookback duration, e.g. 1h or 24h")] = "1h",
    width: Annotated[int, typer.Option(min=10, max=200, help="terminal graph width")] = 72,
    as_json: Annotated[bool, typer.Option("--json", help="return bucketed VRAM series as JSON")] = False,
) -> None:
    """Render a terminal graph of the device-wide GPU working set."""
    from muse.cli_impl.telemetry import run_vram
    _run_telemetry(
        run_vram, param_hint="--since", since=since, width=width, as_json=as_json,
    )


@telemetry_app.command("export")
def telemetry_export(
    output: Annotated[Path, typer.Argument(help="destination .json or .csv file")],
    since: Annotated[str, typer.Option(help="lookback duration")] = "7d",
    format: Annotated[TelemetryFormat, typer.Option(help="export encoding")] = TelemetryFormat.json,
    force: Annotated[bool, typer.Option("--force", help="overwrite an existing file")] = False,
) -> None:
    """Export raw telemetry events for notebooks or a blog artifact."""
    from muse.cli_impl.telemetry import run_export
    _run_telemetry(
        run_export, since=since, output=output, format=format.value, force=force,
    )


@telemetry_app.command("prune")
def telemetry_prune(
    older_than: Annotated[str, typer.Option(help="delete events older than this duration")] = "7d",
    dry_run: Annotated[bool, typer.Option("--dry-run", help="count without deleting")] = False,
) -> None:
    """Prune old telemetry events from the local SQLite store."""
    from muse.cli_impl.telemetry import run_prune
    _run_telemetry(
        run_prune, param_hint="--older-than", older_than=older_than, dry_run=dry_run,
    )


def _resolve_serve_device(cli_device: "Device | None") -> str:
    """Resolve the effective `muse serve` device.

    Precedence: explicit `--device` flag > `server.device` config
    (MUSE_DEVICE env / config.yaml) > registry default ("auto").
    `cli_device` is None when the flag was not passed, which is what
    lets the config/env value take over; an explicit flag always wins
    since it is passed as `override`.
    """
    from muse.core import config
    override = cli_device.value if cli_device is not None else None
    return config.get("server.device", override=override)


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option(help="bind address")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="bind port")] = 8000,
    device: Annotated[
        Optional[Device],
        typer.Option(
            help="default device for models (auto|cpu|cuda|mps); "
            "default: $MUSE_DEVICE or config server.device, else auto",
        ),
    ] = None,
) -> None:
    """Start the HTTP gateway (spawns one worker per venv)."""
    from muse.cli_impl.serve import run_serve
    raise typer.Exit(
        run_serve(host=host, port=port, device=_resolve_serve_device(device)) or 0
    )


@app.command("pull")
def pull(
    identifier: Annotated[
        str,
        typer.Argument(
            help=(
                "bundled model_id (e.g. `kokoro-82m`) OR resolver URI "
                "(e.g. `hf://Qwen/Qwen3-8B-GGUF@q4_k_m`)"
            ),
        ),
    ],
    no_probe: Annotated[
        bool,
        typer.Option(
            "--no-probe",
            help=(
                "skip the post-pull memory probe (use for cross-device pulls, "
                "e.g. pulling on a CPU box for later GPU deployment)"
            ),
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v",
            help=(
                "stream pip's full dep-resolution output and HF tqdm "
                "progress bars (default: quiet, only stage markers + "
                "errors)"
            ),
        ),
    ] = False,
    base: Annotated[
        str | None,
        typer.Option(
            "--base",
            help=(
                "(LoRA pulls) pair the adapter with this base model: a "
                "pulled muse id (e.g. sdxl-turbo) or an HF repo "
                "(org/name). Overrides the base the repo declares."
            ),
        ),
    ] = None,
    model_id: Annotated[
        str | None,
        typer.Option(
            "--id",
            help=(
                "(resolver URIs) catalog id to install under, e.g. when "
                "another publisher's same-named repo already took the "
                "synthesized id"
            ),
        ),
    ] = None,
) -> None:
    """Download weights + install deps for a model.

    By default the pull is quiet: pip runs with `-q`, HF download
    progress bars are suppressed, and the user sees only the stage
    markers (creating venv, installing museq[server], installing
    pip_extras, downloaded weights, probed). On non-zero pip exit
    the captured output is printed to stderr so dep failures are
    diagnosable. Use `-v` / `--verbose` to stream the full pip +
    HF firehose (the v0.40.2 default).

    The post-pull probe runs by default: it loads the model, runs a
    representative inference, and records `measurements.<device>`. If it
    fails, the pull exits 1 and the model is left installed but disabled.
    `--no-probe` opts out for cross-device scenarios.
    """
    from muse.cli_impl.pull_cmd import run_pull
    rc = run_pull(
        identifier, no_probe=no_probe, verbose=verbose, base=base, model_id=model_id,
    )
    if rc:
        raise typer.Exit(rc)


@app.command("search")
def search(
    query: Annotated[str, typer.Argument(help="search query")],
    modality: Annotated[
        Optional[str],
        typer.Option(help="filter by modality (e.g., audio/speech)"),
    ] = None,
    limit: Annotated[int, typer.Option(help="max rows to return")] = 20,
    sort: Annotated[SearchSort, typer.Option()] = SearchSort.downloads,
    max_size_gb: Annotated[
        Optional[float],
        typer.Option(
            "--max-size-gb",
            help=(
                "filter out rows whose size exceeds this "
                "(rows with unknown size pass through)"
            ),
        ),
    ] = None,
    backend: Annotated[
        Optional[str],
        typer.Option(
            help="resolver backend (default: hf; use --backend file for local GGUF dir)",
        ),
    ] = None,
) -> None:
    """Search resolvers (e.g. HuggingFace or local GGUF dir) for pullable models."""
    from muse.cli_impl.search import run_search
    from muse.core.discovery import modality_tags
    if modality is not None and modality not in modality_tags():
        valid = ", ".join(sorted(modality_tags()))
        raise typer.BadParameter(
            f"unknown modality {modality!r}; valid: {valid}",
            param_hint="--modality",
        )
    # Register resolver backends. Default to "hf" when --backend is omitted
    # so registering multiple backends (hf, file) stays ergonomic.
    import muse.core.resolvers_hf  # noqa: F401
    import muse.core.resolvers_file  # noqa: F401
    raise typer.Exit(run_search(
        query=query, modality=modality, limit=limit, sort=sort.value,
        max_size_gb=max_size_gb, backend=backend or "hf",
    ) or 0)


@app.command("mcp")
def mcp(
    http: Annotated[
        bool,
        typer.Option(
            "--http/--stdio",
            help="Streamable HTTP mode instead of stdio (default: stdio for desktop apps)",
        ),
    ] = False,
    port: Annotated[
        int, typer.Option(help="port for Streamable HTTP mode"),
    ] = 8088,
    server: Annotated[
        Optional[str],
        typer.Option(help="muse server URL (default: $MUSE_SERVER or http://localhost:8000)"),
    ] = None,
    admin_token: Annotated[
        Optional[str],
        typer.Option(
            "--admin-token",
            help="admin bearer token for /v1/admin/* tools (default: $MUSE_ADMIN_TOKEN)",
        ),
    ] = None,
    filter_kind: Annotated[
        McpFilter,
        typer.Option(
            "--filter",
            help="restrict tool surface (default: all)",
        ),
    ] = McpFilter.all,
) -> None:
    """Run an MCP server bridging muse to LLM clients (Claude Desktop, Cursor)."""
    from muse.cli_impl.mcp_server import run_mcp_server
    from muse.core import config
    server_url = server or config.get("client.server_url")
    admin_token = admin_token or config.get("admin.token")
    raise typer.Exit(run_mcp_server(
        http=http, port=port, server_url=server_url,
        admin_token=admin_token, filter_kind=filter_kind.value,
    ) or 0)


# Hidden internal commands invoked by the supervisor as subprocesses.
# They never show up in `muse --help` and aren't documented for users.

@app.command("_worker", hidden=True)
def _worker(
    port: Annotated[int, typer.Option()],
    model: Annotated[
        list[str],
        typer.Option(help="model to load (repeatable)"),
    ],
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    device: Annotated[Device, typer.Option()] = Device.auto,
) -> None:
    """internal: run a single worker (invoked by `muse serve`)."""
    from muse.cli_impl.worker import run_worker
    raise typer.Exit(run_worker(
        host=host, port=port, models=model, device=device.value,
    ) or 0)


@app.command("_probe_worker", hidden=True)
def _probe_worker(
    model: Annotated[str, typer.Option()],
    device: Annotated[str, typer.Option()] = "auto",
    no_inference: Annotated[
        bool,
        typer.Option("--no-inference", help="load only; skip inference"),
    ] = False,
) -> None:
    """internal: run a probe in this venv (invoked by `muse models probe`)."""
    from muse.cli_impl.probe_worker import run_probe_worker
    raise typer.Exit(run_probe_worker(
        model_id=model, device=device, run_inference=not no_inference,
    ) or 0)


# `muse models <verb>` ------------------------------------------------------


@models_app.command("list")
def models_list(
    modality: Annotated[
        Optional[str],
        typer.Option(help="filter by modality (e.g., audio/speech)"),
    ] = None,
    installed: Annotated[
        bool,
        typer.Option(
            "--installed",
            help="only models with a catalog.json entry (enabled or disabled)",
        ),
    ] = False,
    available: Annotated[
        bool,
        typer.Option(
            "--available",
            help="only models you could install (recommended or available bundled)",
        ),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="machine-readable JSON instead of the human table",
        ),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="disable color and table styling"),
    ] = False,
) -> None:
    """List known models (bundled scripts + curated recommendations + pulled)."""
    from muse.cli_impl.models_list import run_models_list
    raise typer.Exit(run_models_list(
        modality=modality, installed=installed, available=available,
        as_json=as_json, no_color=no_color,
    ) or 0)


@models_app.command("info")
def models_info(
    model_id: Annotated[str, typer.Argument()],
) -> None:
    """Show catalog entry for a model.

    Recognizes three sources, in priority order:
      1. Bundled scripts and resolver-pulled entries (`known_models()`)
      2. Curated recommendations not yet pulled (`curated.yaml`)
      3. Unknown -> 2 with a clear error.

    For curated-only rows the output shows description / hf_repo /
    license / suggested memory plus a `muse pull <id>` hint, so
    `muse models info <id>` matches the rows surfaced by
    `muse models list` symmetrically.
    """
    from muse.cli_impl.admin_actions import run_models_info
    rc = run_models_info(model_id)
    if rc:
        raise typer.Exit(rc)


@models_app.command("remove")
def models_remove(
    model_id: Annotated[str, typer.Argument()],
    purge: Annotated[
        bool,
        typer.Option(
            "--purge",
            help=(
                "also delete the per-model venv and unshared Muse-owned "
                "weights; legacy weights in a shared HF cache are preserved"
            ),
        ),
    ] = False,
) -> None:
    """Unregister a model from the catalog."""
    from muse.cli_impl.pull_cmd import run_remove
    rc = run_remove(model_id, purge=purge)
    if rc:
        raise typer.Exit(rc)


@models_app.command("enable")
def models_enable(
    model_id: Annotated[str, typer.Argument()],
) -> None:
    """Enable a pulled model for serving.

    When MUSE_ADMIN_TOKEN is set AND the supervisor is running, this
    routes through the admin API so the running gateway picks up the
    change live (worker spawn or restart-in-place). Otherwise it falls
    back to mutating catalog.json directly with a warning.
    """
    _set_model_enabled(model_id, True)


@models_app.command("disable")
def models_disable(
    model_id: Annotated[str, typer.Argument()],
) -> None:
    """Disable a pulled model.

    Stays in catalog, not loaded by `muse serve`. Same admin-API
    routing as `enable`.
    """
    _set_model_enabled(model_id, False)


def _set_model_enabled(model_id: str, enabled: bool) -> None:
    """Shared body of `models enable` / `models disable`."""
    from muse.cli_impl.admin_actions import set_model_enabled
    rc = set_model_enabled(model_id, enabled)
    if rc:
        raise typer.Exit(rc)


@models_app.command("set-device")
def models_set_device(
    model_id: Annotated[str, typer.Argument()],
    device: Annotated[
        Optional[Device],
        typer.Argument(
            help="device to pin the model to (auto|cpu|cuda|mps); omit with --clear",
        ),
    ] = None,
    clear: Annotated[
        bool,
        typer.Option("--clear", help="remove the override (revert to manifest / --device)"),
    ] = False,
) -> None:
    """Pin a model's load device, overriding its manifest device + --device flag.

    Writes a per-model `device_override` to the catalog. Precedence at
    load time: override > manifest capabilities.device pin > --device flag
    > auto. Use `cuda` to force a cpu-pinned model (e.g. kokoro-82m) onto
    the GPU, `cpu` to keep a model off scarce VRAM, or `auto` to un-pin and
    let muse pick cuda-if-available. `--clear` removes the override.

    Catalog-only state: takes effect on the model's NEXT cold load. To
    apply it to an already-resident worker, evict it (idle/memory pressure)
    or restart the supervisor.
    """
    from muse.cli_impl.pull_cmd import run_set_device
    rc = run_set_device(
        model_id, device.value if device is not None else None, clear=clear,
    )
    if rc:
        raise typer.Exit(rc)


@models_app.command("set-gpu-layers")
def models_set_gpu_layers(
    model_id: Annotated[str, typer.Argument()],
    n: Annotated[
        Optional[int],
        typer.Argument(
            help="llama.cpp n_gpu_layers: N >= 0 (0 = pure CPU, N > 0 = "
                 "first N layers on GPU); use --all for -1 (every layer on "
                 "GPU) or --clear to remove the pin",
        ),
    ] = None,
    all_layers: Annotated[
        bool,
        typer.Option(
            "--all",
            help="offload every layer to the GPU (writes -1); shells "
                 "reject a bare -1 positional (click has no negative-"
                 "number heuristic), so use this flag instead",
        ),
    ] = False,
    clear: Annotated[
        bool,
        typer.Option("--clear", help="remove the pin (revert to manifest / runtime default)"),
    ] = False,
) -> None:
    """Pin a GGUF model's llama.cpp GPU/CPU layer split (operator override).

    Writes a per-model `gpu_layers_override` to the catalog. Precedence at
    load time: pin > manifest capabilities.n_gpu_layers > runtime default
    (-1, everything the GPU fits). GGUF-only: refuses models without
    capabilities.gguf_file, since other runtimes silently ignore the kwarg.

    Exactly one of a layer count, --all, or --clear is required:

        muse models set-gpu-layers <id> 30      split: first 30 layers on GPU
        muse models set-gpu-layers <id> 0       pure CPU
        muse models set-gpu-layers <id> --all   everything on GPU (writes -1)
        muse models set-gpu-layers <id> --clear remove the pin

    click has no negative-number heuristic for positional arguments, so a
    bare `-1` is parsed as an unrecognized option and the command dies at
    the shell before this code ever runs -- that is why -1 is not
    documented as a positional value above; `--all` is the supported way
    to request it. Shell die-hards can still pass a literal -1 via the
    `--` end-of-options marker (`muse models set-gpu-layers <id> -- -1`),
    which forces click to treat everything after it as positional.

    Catalog-only state: takes effect on the model's NEXT cold load. To
    apply it to an already-resident worker, evict it or restart the
    supervisor. Run `muse models probe <id>` after pinning so admission
    sizing measures the split's real (smaller) VRAM peak.
    """
    from muse.cli_impl.pull_cmd import run_set_gpu_layers
    rc = run_set_gpu_layers(model_id, n, all_layers=all_layers, clear=clear)
    if rc:
        raise typer.Exit(rc)


@models_app.command("warmup")
def models_warmup(
    model_id: Annotated[str, typer.Argument()],
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            help=(
                "max seconds to wait for the cold load to complete "
                "(default: 300; bump for very large models)"
            ),
        ),
    ] = 300.0,
) -> None:
    """Pre-load a model via the running supervisor.

    Routes through the admin API's `POST /v1/admin/models/{id}/warmup`,
    which calls `LoadDirector.warmup` to bring the model into the loaded
    set without consuming a request slot. Subsequent inference requests
    skip the cold-load latency.

    Warmup is purely a runtime operation: there is no offline equivalent
    (catalog state alone cannot pre-load anything). When
    `MUSE_ADMIN_TOKEN` is unset OR the supervisor isn't reachable, this
    command exits non-zero with a clear error rather than silently
    succeeding.
    """
    from muse.cli_impl.admin_actions import run_warmup
    rc = run_warmup(model_id, timeout=timeout)
    if rc:
        raise typer.Exit(rc)


@models_app.command("probe")
def models_probe(
    model_id: Annotated[
        Optional[str],
        typer.Argument(help="model to probe (omit to probe all enabled)"),
    ] = None,
    no_inference: Annotated[
        bool,
        typer.Option(
            "--no-inference",
            help="load only; skip representative inference (faster but undersells peak)",
        ),
    ] = False,
    device: Annotated[
        Optional[Device],
        typer.Option(help="override the model's device preference for this probe"),
    ] = None,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="machine-readable output instead of human-readable summary",
        ),
    ] = False,
) -> None:
    """Measure VRAM/RAM by loading the model and running representative inference."""
    from muse.cli_impl.probe import run_probe, run_probe_all
    device_str = device.value if device is not None else None
    if model_id is None:
        rc = run_probe_all(
            no_inference=no_inference, device=device_str, as_json=as_json,
        )
    else:
        rc = run_probe(
            model_id=model_id, no_inference=no_inference,
            device=device_str, as_json=as_json,
        )
    raise typer.Exit(rc or 0)


@models_app.command("refresh")
def models_refresh(
    model_id: Annotated[
        Optional[str],
        typer.Argument(help="model to refresh (omit if using --all or --enabled)"),
    ] = None,
    all_: Annotated[
        bool,
        typer.Option("--all", help="refresh every pulled venv"),
    ] = False,
    enabled_only: Annotated[
        bool,
        typer.Option("--enabled", help="only refresh enabled venvs"),
    ] = False,
    no_extras: Annotated[
        bool,
        typer.Option(
            "--no-extras",
            help="only refresh museq[server]; skip the model's pip_extras",
        ),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="machine-readable output instead of human-readable summary",
        ),
    ] = False,
) -> None:
    """Re-install museq[server] + the model's pip_extras into per-model venvs."""
    from muse.cli_impl.refresh import run_refresh
    raise typer.Exit(run_refresh(
        model_id=model_id, all_=all_, enabled_only=enabled_only,
        no_extras=no_extras, as_json=as_json,
    ) or 0)


# `muse config <verb>` -------------------------------------------------------


@config_app.command("path")
def config_path() -> None:
    """Print the resolved config.yaml path."""
    from muse.cli_impl.config_cmd import run_path
    raise typer.Exit(run_path() or 0)


@config_app.command("generate")
def config_generate(
    force: Annotated[
        bool,
        typer.Option("--force", help="overwrite an existing config.yaml"),
    ] = False,
) -> None:
    """Write a fully-commented config.yaml template to the config path."""
    from muse.cli_impl.config_cmd import run_generate
    raise typer.Exit(run_generate(force) or 0)


@config_app.command("show")
def config_show(
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="machine-readable JSON instead of the human table",
        ),
    ] = False,
) -> None:
    """Show every setting's resolved value + source (env/file/default).

    `admin.token` is always redacted to `set`/`unset`.
    """
    from muse.cli_impl.config_cmd import run_show
    raise typer.Exit(run_show(as_json) or 0)


@config_app.command("get")
def config_get(
    key: Annotated[str, typer.Argument(help="dotted setting key, e.g. limits.rerank_max_documents")],
) -> None:
    """Print one setting's resolved value."""
    from muse.cli_impl.config_cmd import run_get
    raise typer.Exit(run_get(key) or 0)


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="dotted setting key, e.g. server.gpu_headroom_gb")],
    value: Annotated[str, typer.Argument(help="raw value; coerced per the setting's type")],
) -> None:
    """Write one setting into config.yaml (strict type validation)."""
    from muse.cli_impl.config_cmd import run_set
    raise typer.Exit(run_set(key, value) or 0)


@config_app.command("unset")
def config_unset(
    key: Annotated[str, typer.Argument(help="dotted setting key to remove from config.yaml")],
) -> None:
    """Remove one setting from config.yaml so it falls back to env/default."""
    from muse.cli_impl.config_cmd import run_unset
    raise typer.Exit(run_unset(key) or 0)


def main(argv: list[str] | None = None) -> int:
    """Programmatic entry point. Accepts argv override for tests.

    `standalone_mode=False` lets us translate `typer.Exit` into a
    return code instead of letting click call `sys.exit`. The price
    is that any other click exception (`NoArgsIsHelpError` when the
    user runs `muse` with no subcommand; `UsageError` for bad flags;
    `Abort` for SIGINT) also propagates here. We catch the click
    base class `ClickException` and use its `exit_code`.

    Critically, with `standalone_mode=False` neither click nor typer's
    own `_main` override ever calls `ClickException.show()` -- both
    just re-raise (see `click.core.BaseCommand.main` and
    `typer.core._main`'s `if not standalone_mode: raise` branches).
    Rendering the error is therefore entirely our job here; skipping
    `e.show()` (a past bug) meant every usage error (unknown
    command/option, missing argument, bad value) exited nonzero with
    completely empty stdout+stderr on the shipped `muse` binary.
    `NoArgsIsHelpError` is the one exception already visible before we
    get here: typer's rich help renderer prints as a side effect of
    building `ctx.get_help()` while the exception is constructed, so
    calling `.show()` on it too only adds a harmless blank line to
    stderr, not a duplicate help render.
    """
    import click
    _click_exceptions = [click.exceptions.ClickException]
    try:
        # typer >= 0.26 vendors click as `typer._click`; its exception
        # classes are DISTINCT from the installed click package's, so
        # catching only `click.exceptions.ClickException` lets every
        # usage error escape as an uncaught traceback on the shipped
        # `muse` binary. Catch both hierarchies.
        from typer import _click as _typer_click
        _click_exceptions.append(_typer_click.exceptions.ClickException)
    except ImportError:
        pass  # older typer: the installed click IS typer's click
    _click_exceptions = tuple(_click_exceptions)
    try:
        rv = app(args=argv, standalone_mode=False)
        # With standalone_mode=False, click catches `typer.Exit` itself and
        # *returns* its exit code as app()'s return value (it does not
        # re-raise). Honor that int so a command's `raise typer.Exit(2)`
        # actually exits 2 on the real `muse` binary; a normal command
        # returns None -> 0. (Subprocess tests use `python -m muse.cli` ->
        # app() in standalone mode, so they never exercised this path.)
        return rv if isinstance(rv, int) and not isinstance(rv, bool) else 0
    except typer.Exit as e:
        return e.exit_code
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    except _click_exceptions as e:
        # Render the error/usage message ourselves: standalone_mode=False
        # suppresses click's own show()+sys.exit() so nothing gets
        # printed otherwise (see docstring above).
        e.show()
        return e.exit_code


if __name__ == "__main__":
    app()
