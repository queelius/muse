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


# Top-level Typer app. rich_markup_mode lets help strings use [bold]
# and similar inline rich tags; we don't lean on it heavily but it
# costs nothing.
app = typer.Typer(
    name="muse",
    help="Multi-modality generation server + admin CLI",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)
models_app = typer.Typer(
    name="models",
    help="manage the model catalog",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)
app.add_typer(models_app, name="models")


@app.callback()
def _root(
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


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option(help="bind address")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="bind port")] = 8000,
    device: Annotated[Device, typer.Option()] = Device.auto,
) -> None:
    """Start the HTTP gateway (spawns one worker per venv)."""
    from muse.cli_impl.serve import run_serve
    raise typer.Exit(run_serve(host=host, port=port, device=device.value) or 0)


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
) -> None:
    """Download weights + install deps for a model."""
    from muse.core.catalog import pull as _pull
    # Always register the HF resolver before dispatching. The arg may
    # be a URI directly, OR a curated alias that expands to a URI
    # inside pull(); the old conditional "only import when :// is in
    # the arg" missed the second case and crashed with "no resolver
    # for scheme 'hf'". Importing is near-free (heavy huggingface_hub
    # imports happen on actual resolve(), not on module import).
    import muse.core.resolvers_hf  # noqa: F401  (registers HFResolver on import)
    try:
        _pull(identifier)
    except KeyError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    typer.echo(f"pulled {identifier}")


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
            help="resolver backend (default: only-registered, or pick when ambiguous)",
        ),
    ] = None,
) -> None:
    """Search resolvers (e.g. HuggingFace) for pullable models."""
    from muse.cli_impl.search import run_search
    from muse.core.discovery import modality_tags
    if modality is not None and modality not in modality_tags():
        valid = ", ".join(sorted(modality_tags()))
        raise typer.BadParameter(
            f"unknown modality {modality!r}; valid: {valid}",
            param_hint="--modality",
        )
    # Register resolver backends. Today only HF; future backends slot
    # in by importing their resolvers_<scheme> module here.
    import muse.core.resolvers_hf  # noqa: F401
    raise typer.Exit(run_search(
        query=query, modality=modality, limit=limit, sort=sort.value,
        max_size_gb=max_size_gb, backend=backend,
    ) or 0)


@app.command("mcp")
def mcp(
    http: Annotated[
        bool,
        typer.Option(
            "--http/--stdio",
            help="HTTP+SSE mode instead of stdio (default: stdio for desktop apps)",
        ),
    ] = False,
    port: Annotated[
        int, typer.Option(help="port for HTTP+SSE mode"),
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
            help="restrict tool surface (default: all 29 tools)",
        ),
    ] = McpFilter.all,
) -> None:
    """Run an MCP server bridging muse to LLM clients (Claude Desktop, Cursor)."""
    import os as _os
    from muse.cli_impl.mcp_server import run_mcp_server
    server_url = server or _os.environ.get("MUSE_SERVER", "http://localhost:8000")
    admin_token = admin_token or _os.environ.get("MUSE_ADMIN_TOKEN")
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
    """Show catalog entry for a model."""
    from muse.core.catalog import _read_catalog, known_models
    from muse.cli_impl.models_info_display import format_info

    catalog_known = known_models()
    if model_id not in catalog_known:
        typer.echo(f"error: unknown model {model_id!r}", err=True)
        raise typer.Exit(2)
    catalog_data = _read_catalog().get(model_id, {}) or {}
    online_status = _probe_online_worker_status(model_id)
    typer.echo(format_info(
        model_id,
        catalog_known=catalog_known,
        catalog_data=catalog_data,
        online_status=online_status,
    ))


@models_app.command("remove")
def models_remove(
    model_id: Annotated[str, typer.Argument()],
    purge: Annotated[
        bool,
        typer.Option(
            "--purge",
            help=(
                "also delete the per-model venv (HF weights cache is left alone; "
                "use huggingface-cli delete-cache to reclaim it)"
            ),
        ),
    ] = False,
) -> None:
    """Unregister a model from the catalog."""
    from muse.core.catalog import remove
    remove(model_id, purge=purge)
    suffix = " (purged venv)" if purge else ""
    typer.echo(f"removed {model_id} from catalog{suffix}")


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
    if _try_admin_action("enable", model_id):
        return

    from muse.core.catalog import set_enabled
    try:
        set_enabled(model_id, True)
    except KeyError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    typer.echo(
        f"enabled {model_id} (catalog only; supervisor will pick this up on restart)"
    )


@models_app.command("disable")
def models_disable(
    model_id: Annotated[str, typer.Argument()],
) -> None:
    """Disable a pulled model.

    Stays in catalog, not loaded by `muse serve`. Same admin-API
    routing as `enable`.
    """
    if _try_admin_action("disable", model_id):
        return

    from muse.core.catalog import set_enabled
    try:
        set_enabled(model_id, False)
    except KeyError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(2)
    typer.echo(
        f"disabled {model_id} (catalog only; supervisor will pick this up on restart)"
    )


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
            help="only refresh muse[server]; skip the model's pip_extras",
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
    """Re-install muse[server] + the model's pip_extras into per-model venvs."""
    from muse.cli_impl.refresh import run_refresh
    raise typer.Exit(run_refresh(
        model_id=model_id, all_=all_, enabled_only=enabled_only,
        no_extras=no_extras, as_json=as_json,
    ) or 0)


# Helpers (preserved verbatim from the argparse era) -----------------------


def _probe_online_worker_status(model_id: str) -> dict | None:
    """Best-effort lookup of live worker status via the admin API.

    Returns a dict shaped like the /v1/admin/models/{id}/status response
    on success; None when the supervisor isn't reachable, the admin
    endpoint isn't enabled (no MUSE_ADMIN_TOKEN), or the credentials are
    rejected. Caller treats None as "no live data; show offline view."
    """
    import os
    try:
        from muse.admin.client import AdminClient, AdminClientError
    except Exception:  # noqa: BLE001
        return None
    if not os.environ.get("MUSE_ADMIN_TOKEN"):
        return None
    client = AdminClient(timeout=2.0)
    try:
        return client.status(model_id)
    except AdminClientError:
        return None
    except Exception:  # noqa: BLE001
        return None


def _try_admin_action(action: str, model_id: str) -> bool:
    """Try the admin-API path for enable/disable; return True iff used."""
    import os
    if not os.environ.get("MUSE_ADMIN_TOKEN"):
        return False
    try:
        from muse.admin.client import AdminClient, AdminClientError
    except Exception:  # noqa: BLE001
        return False
    client = AdminClient(timeout=5.0)
    try:
        if action == "enable":
            out = client.enable(model_id)
            job_id = out.get("job_id")
            if job_id:
                typer.echo(f"enable job submitted: {job_id}")
                try:
                    final = client.wait(job_id, timeout=120.0, poll=0.5)
                    if final.get("state") == "done":
                        port = (final.get("result") or {}).get("worker_port")
                        msg = f"enabled {model_id}"
                        if port:
                            msg += f" (worker port {port})"
                        typer.echo(msg)
                        return True
                    typer.echo(
                        f"error: enable failed: {final.get('error')}", err=True,
                    )
                    return True
                except TimeoutError:
                    typer.echo(
                        f"warning: enable still running; poll job {job_id} for status",
                        err=True,
                    )
                    return True
        elif action == "disable":
            out = client.disable(model_id)
            typer.echo(f"disabled {model_id}")
            if out.get("worker_terminated"):
                typer.echo(f"  worker on port {out.get('worker_port')} stopped")
            elif out.get("worker_port"):
                remaining = out.get("remaining_models_in_worker") or []
                typer.echo(
                    f"  worker on port {out['worker_port']} restarted; "
                    f"remaining models: {', '.join(remaining)}"
                )
            return True
    except AdminClientError as e:
        if e.status == 503 or e.code == "admin_disabled":
            return False
        typer.echo(f"error: {e.message}", err=True)
        return True
    except Exception as e:  # noqa: BLE001
        typer.echo(
            f"warning: admin API unreachable ({e}); falling back to catalog-only update",
            err=True,
        )
        return False
    return False


def main(argv: list[str] | None = None) -> int:
    """Programmatic entry point. Accepts argv override for tests."""
    try:
        app(args=argv, standalone_mode=False)
        return 0
    except typer.Exit as e:
        return e.exit_code
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0


if __name__ == "__main__":
    app()
