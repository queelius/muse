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

Deliberate non-goals:
    - Per-modality CLI subcommands (e.g., `muse speak`, `muse audio ...`).
      They'd require hardcoded modality→verb mappings that grow every
      time a new modality lands. Keeping the CLI modality-agnostic means
      embeddings / transcriptions / video all work without CLI changes.

Heavy imports (torch, diffusers) are kept out of this module so
`muse --help` stays instant. Command implementations live in
`muse.cli_impl.*` and import what they need when invoked.
"""
from __future__ import annotations

import argparse
import logging
import sys

log = logging.getLogger("muse")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="muse",
        description="Multi-modality generation server + admin CLI",
    )
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="cmd", required=False)

    # serve
    sp_serve = sub.add_parser("serve", help="start the HTTP gateway (spawns one worker per venv)")
    sp_serve.add_argument("--host", default="0.0.0.0")
    sp_serve.add_argument("--port", type=int, default=8000)
    sp_serve.add_argument("--device", default="auto",
                          choices=["auto", "cpu", "cuda", "mps"])
    sp_serve.set_defaults(func=_cmd_serve)

    # pull (accepts bare model_id OR resolver URI)
    sp_pull = sub.add_parser(
        "pull",
        help=(
            "download weights + install deps for a model "
            "(bundled id like `kokoro-82m` OR resolver URI like "
            "`hf://Qwen/Qwen3-8B-GGUF@q4_k_m`)"
        ),
    )
    sp_pull.add_argument(
        "identifier",
        help="bundled model_id OR resolver URI (e.g. hf://org/repo@variant)",
    )
    sp_pull.set_defaults(func=_cmd_pull)

    # search (HuggingFace + future resolvers)
    sp_search = sub.add_parser(
        "search",
        help="search resolvers (e.g. HuggingFace) for pullable models",
    )
    sp_search.add_argument("query", help="search query")
    # Choices come from disk via AST scan: every bundled modality knows
    # its own MIME tag, so adding text/classification or audio/transcription
    # later doesn't require an edit here. modality_tags() avoids fastapi
    # imports, so `muse --help` works on a bare install.
    from muse.core.discovery import modality_tags
    sp_search.add_argument(
        "--modality",
        choices=modality_tags(),
        default=None,
        help="filter by modality (omit to search all supported)",
    )
    sp_search.add_argument("--limit", type=int, default=20)
    sp_search.add_argument(
        "--sort",
        choices=["downloads", "lastModified", "likes"],
        default="downloads",
    )
    sp_search.add_argument(
        "--max-size-gb",
        type=float,
        default=None,
        help="filter out rows whose size exceeds this (rows with unknown size pass through)",
    )
    sp_search.add_argument(
        "--backend",
        default=None,
        help="resolver backend to use (default: only-registered, or pick when ambiguous)",
    )
    sp_search.set_defaults(func=_cmd_search)

    # _worker (hidden; invoked by supervisor via subprocess)
    sp_worker = sub.add_parser("_worker", help="internal: run a single worker (invoked by muse serve)")
    sp_worker.add_argument("--host", default="127.0.0.1",
                           help="bind address (default: 127.0.0.1, workers are internal)")
    sp_worker.add_argument("--port", type=int, required=True)
    sp_worker.add_argument("--model", action="append", default=[], required=True,
                           help="model to load (repeatable; one worker can host multiple compatible models)")
    sp_worker.add_argument("--device", default="auto",
                           choices=["auto", "cpu", "cuda", "mps"])
    sp_worker.set_defaults(func=_cmd_worker)

    # models (catalog admin)
    sp_models = sub.add_parser("models", help="manage the model catalog")
    models_sub = sp_models.add_subparsers(dest="models_cmd", required=True)

    sp_list = models_sub.add_parser(
        "list",
        help="list known models (bundled scripts + curated recommendations + pulled)",
    )
    sp_list.add_argument("--modality", default=None,
                         help="filter by modality (e.g., audio/speech)")
    sp_list.add_argument(
        "--installed",
        action="store_true",
        help="only models with a catalog.json entry (enabled or disabled)",
    )
    sp_list.add_argument(
        "--available",
        action="store_true",
        help="only models you could install (recommended or available bundled)",
    )
    sp_list.set_defaults(func=_cmd_models_list)

    sp_info = models_sub.add_parser("info", help="show catalog entry for a model")
    sp_info.add_argument("model_id")
    sp_info.set_defaults(func=_cmd_models_info)

    sp_remove = models_sub.add_parser("remove", help="unregister a model from the catalog")
    sp_remove.add_argument("model_id")
    sp_remove.add_argument(
        "--purge", action="store_true",
        help="also delete the per-model venv (HF weights cache is left alone; "
             "use huggingface-cli delete-cache to reclaim it)",
    )
    sp_remove.set_defaults(func=_cmd_models_remove)

    sp_enable = models_sub.add_parser("enable", help="enable a pulled model for serving")
    sp_enable.add_argument("model_id")
    sp_enable.set_defaults(func=_cmd_models_enable)

    sp_disable = models_sub.add_parser("disable", help="disable a pulled model (stays in catalog, not loaded by muse serve)")
    sp_disable.add_argument("model_id")
    sp_disable.set_defaults(func=_cmd_models_disable)

    sp_probe = models_sub.add_parser(
        "probe",
        help=(
            "measure VRAM/RAM by loading the model and (default) running "
            "representative inference; omit model_id to probe every "
            "enabled+pulled model"
        ),
    )
    sp_probe.add_argument(
        "model_id",
        nargs="?",
        default=None,
        help="model to probe (omit to probe all enabled)",
    )
    sp_probe.add_argument(
        "--no-inference",
        action="store_true",
        help="load only; skip representative inference (faster but undersells peak)",
    )
    sp_probe.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda", "mps"],
        help="override the model's device preference for this probe",
    )
    sp_probe.add_argument(
        "--json",
        action="store_true",
        help="machine-readable output instead of human-readable summary",
    )
    sp_probe.set_defaults(func=_cmd_models_probe)

    # refresh: re-install muse[server,<extras>] into per-model venvs
    sp_refresh = models_sub.add_parser(
        "refresh",
        help=(
            "re-install muse[server,<modality-extras>] + the model's "
            "pip_extras into one or more per-model venvs (use after "
            "`pip install -U muse` to propagate new server-side deps)"
        ),
    )
    sp_refresh.add_argument(
        "model_id", nargs="?", default=None,
        help="model to refresh (omit if using --all or --enabled)",
    )
    sp_refresh.add_argument(
        "--all", action="store_true", dest="all_",
        help="refresh every pulled venv",
    )
    sp_refresh.add_argument(
        "--enabled", action="store_true", dest="enabled_only",
        help="only refresh enabled venvs",
    )
    sp_refresh.add_argument(
        "--no-extras", action="store_true", dest="no_extras",
        help="only refresh muse[server]; skip the model's pip_extras",
    )
    sp_refresh.add_argument(
        "--json", action="store_true", dest="as_json",
        help="machine-readable output instead of human-readable summary",
    )
    sp_refresh.set_defaults(func=_cmd_models_refresh)

    # mcp (Model Context Protocol server)
    sp_mcp = sub.add_parser(
        "mcp",
        help=(
            "run an MCP (Model Context Protocol) server bridging muse "
            "to LLM clients (Claude Desktop, Cursor, etc.)"
        ),
    )
    sp_mcp.add_argument(
        "--http", action="store_true",
        help="run in HTTP+SSE mode instead of stdio (default: stdio for desktop apps)",
    )
    sp_mcp.add_argument(
        "--port", type=int, default=8088,
        help="port for HTTP+SSE mode (default: 8088)",
    )
    sp_mcp.add_argument(
        "--server", default=None,
        help="muse server URL (default: $MUSE_SERVER or http://localhost:8000)",
    )
    sp_mcp.add_argument(
        "--admin-token", default=None,
        help="admin bearer token for /v1/admin/* tools (default: $MUSE_ADMIN_TOKEN)",
    )
    sp_mcp.add_argument(
        "--filter", default="all", dest="filter_kind",
        choices=["all", "admin", "inference"],
        help="restrict tool surface (default: all 29 tools)",
    )
    sp_mcp.set_defaults(func=_cmd_mcp)

    # _probe_worker (hidden; invoked by `muse models probe` via subprocess)
    sp_probe_worker = sub.add_parser(
        "_probe_worker",
        help="internal: run a probe in this venv (invoked by `muse models probe`)",
    )
    sp_probe_worker.add_argument("--model", required=True)
    sp_probe_worker.add_argument("--device", default="auto")
    sp_probe_worker.add_argument("--no-inference", action="store_true")
    sp_probe_worker.set_defaults(func=_cmd_probe_worker)

    return p


# Command implementations (deferred imports for fast startup)

def _cmd_serve(args):
    from muse.cli_impl.serve import run_serve
    return run_serve(host=args.host, port=args.port, device=args.device)


def _cmd_pull(args):
    from muse.core.catalog import pull
    # Always register the HF resolver before dispatching. The arg may be
    # a URI directly, OR a curated alias that expands to a URI inside
    # pull(); the old conditional "only import when :// is in the arg"
    # missed the second case and crashed with "no resolver for scheme 'hf'".
    # Importing is near-free (heavy huggingface_hub imports happen on
    # actual resolve(), not on module import).
    import muse.core.resolvers_hf  # noqa: F401  (registers HFResolver on import)
    try:
        pull(args.identifier)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"pulled {args.identifier}")
    return 0


def _cmd_search(args):
    from muse.cli_impl.search import run_search
    # Register resolver backends. Today only HF; future backends slot in
    # by importing their resolvers_<scheme> module here.
    import muse.core.resolvers_hf  # noqa: F401
    return run_search(
        query=args.query,
        modality=args.modality,
        limit=args.limit,
        sort=args.sort,
        max_size_gb=args.max_size_gb,
        backend=args.backend,
    )


def _cmd_worker(args):
    from muse.cli_impl.worker import run_worker
    return run_worker(
        host=args.host, port=args.port,
        models=args.model, device=args.device,
    )


def _cmd_models_list(args):
    """Print models from three sources: bundled scripts + curated + pulled.

    Status precedence:
      - pulled (in catalog.json) -> 'enabled' or 'disabled' (catalog wins)
      - curated and not pulled  -> 'recommended' (curated trumps bundled-available)
      - bundled and not pulled  -> 'available'

    Memory column (v0.18.2+):
      - measured peak from `muse models probe` (no prefix, most honest)
      - annotated capabilities.memory_gb (~ prefix, peak-inference estimate)
      - "-" otherwise
      Tagged GPU/CPU based on capabilities.device.

    Filters (mutually compatible):
      --modality M     : only entries whose modality == M
      --installed      : only entries with a catalog.json record
      --available      : only entries you could install (recommended/available)
    """
    from muse.core.catalog import (
        _read_catalog,
        is_enabled,
        is_pulled,
        list_known,
    )
    from muse.core.curated import load_curated

    bundled_entries = {e.model_id: e for e in list_known(None)}
    curated_entries = {c.id: c for c in load_curated()}
    catalog_data = _read_catalog()

    # Build the unified row set keyed by id. Each row is a dict carrying
    # whatever metadata we have, plus the computed status.
    rows: list[dict] = []
    seen: set[str] = set()

    # 1. Bundled scripts and resolver-pulled entries (everything in known_models).
    for model_id, e in bundled_entries.items():
        seen.add(model_id)
        pulled = is_pulled(model_id)
        if pulled:
            status = "enabled" if is_enabled(model_id) else "disabled"
        elif model_id in curated_entries:
            status = "recommended"
        else:
            status = "available"
        mem_str, mem_gb, mem_device = _model_memory_display(
            e.extra, catalog_data.get(model_id)
        )
        rows.append({
            "id": model_id,
            "modality": e.modality,
            "description": e.description,
            "status": status,
            "mem_str": mem_str,
            "mem_gb": mem_gb,
            "mem_device": mem_device,
        })

    # 2. Curated entries that aren't already covered by a bundled/pulled
    #    entry above (i.e. resolver-pulled curated aliases).
    for cid, c in curated_entries.items():
        if cid in seen:
            continue
        if is_pulled(cid):
            status = "enabled" if is_enabled(cid) else "disabled"
        else:
            status = "recommended"
        mem_str, mem_gb, mem_device = _model_memory_display(
            c.capabilities or {}, catalog_data.get(cid)
        )
        rows.append({
            "id": cid,
            "modality": c.modality or "?",
            "description": c.description or "",
            "status": status,
            "mem_str": mem_str,
            "mem_gb": mem_gb,
            "mem_device": mem_device,
        })

    # Filters
    if args.modality:
        rows = [r for r in rows if r["modality"] == args.modality]
    if args.installed:
        rows = [r for r in rows if r["status"] in ("enabled", "disabled")]
    if args.available:
        rows = [r for r in rows if r["status"] in ("recommended", "available")]

    if not rows:
        suffixes = []
        if args.modality:
            suffixes.append(f"modality {args.modality!r}")
        if args.installed:
            suffixes.append("--installed")
        if args.available:
            suffixes.append("--available")
        suffix = (" matching " + ", ".join(suffixes)) if suffixes else ""
        print(f"no models{suffix}")
        return 0

    rows.sort(key=lambda r: (r["status"], r["id"]))
    for r in rows:
        print(
            f"  {r['id']:20s} [{r['status']:11s}] "
            f"{r['modality']:22s} {r['mem_str']:>14s}  {r['description']}"
        )

    # Footer: aggregate memory by device, counting only ENABLED models so
    # users can see the budget they'd actually demand at runtime. GPU and
    # CPU are reported separately because they're different physical
    # resources.
    gpu_total = sum(
        r["mem_gb"] for r in rows
        if r["status"] == "enabled" and r["mem_gb"] is not None
        and r["mem_device"] == "GPU"
    )
    cpu_total = sum(
        r["mem_gb"] for r in rows
        if r["status"] == "enabled" and r["mem_gb"] is not None
        and r["mem_device"] == "CPU"
    )
    n_enabled = sum(1 for r in rows if r["status"] == "enabled")
    print()
    print(
        f"Enabled: {gpu_total:.1f} GB GPU + {cpu_total:.1f} GB CPU "
        f"({n_enabled} models)"
    )
    print("Measured values (from `muse models probe`) shown without prefix;")
    print("annotated estimates (peak inference) shown with ~ prefix.")
    return 0


def _model_memory_display(extra: dict, catalog_entry: dict | None):
    """Return (display_str, gb_for_aggregate, device_label).

    Resolution order, in decreasing fidelity:
      1. measured peak from `muse models probe` (per-device measurement)
      2. annotated `capabilities.memory_gb`
      3. None (display "-")

    The device label ("CPU" or "GPU") is derived from `capabilities.device`:
    cpu -> CPU; everything else (cuda/auto/mps/unset) -> GPU.
    """
    extra = extra or {}
    cap_device = (extra.get("device") or "auto").lower()
    if cap_device == "cpu":
        device_label = "CPU"
        measurement_keys = ("cpu",)
    else:
        device_label = "GPU"
        # Prefer cuda; fall back to a generic "auto" key if a future
        # probe records under that name.
        measurement_keys = ("cuda", "auto")

    measurements = (catalog_entry or {}).get("measurements") or {}
    for key in measurement_keys:
        m = measurements.get(key)
        if not m:
            continue
        peak = m.get("peak_bytes") or 0
        if peak > 0:
            gb = peak / (1024**3)
            return f"{gb:.1f} GB {device_label}", gb, device_label

    annotation = extra.get("memory_gb")
    if annotation is not None:
        try:
            gb = float(annotation)
        except (TypeError, ValueError):
            return "-", None, device_label
        return f"~{gb:.1f} GB {device_label}", gb, device_label

    return "-", None, device_label


def _cmd_models_info(args):
    from muse.core.catalog import _read_catalog, known_models
    from muse.cli_impl.models_info_display import format_info

    catalog_known = known_models()
    if args.model_id not in catalog_known:
        print(f"error: unknown model {args.model_id!r}", file=sys.stderr)
        return 2
    catalog_data = _read_catalog().get(args.model_id, {}) or {}
    online_status = _probe_online_worker_status(args.model_id)
    print(format_info(
        args.model_id,
        catalog_known=catalog_known,
        catalog_data=catalog_data,
        online_status=online_status,
    ))
    return 0


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
        # httpx ConnectError, etc. -> "supervisor not running" path
        return None


def _cmd_models_remove(args):
    from muse.core.catalog import remove
    remove(args.model_id, purge=args.purge)
    suffix = " (purged venv)" if args.purge else ""
    print(f"removed {args.model_id} from catalog{suffix}")
    return 0


def _cmd_models_enable(args):
    """Enable a model.

    When MUSE_ADMIN_TOKEN is set AND the supervisor is running, this
    routes through the admin API so the running gateway picks up the
    change live (worker spawn or restart-in-place). Otherwise it falls
    back to mutating catalog.json directly with a warning.
    """
    if _try_admin_action("enable", args.model_id):
        return 0

    from muse.core.catalog import set_enabled
    try:
        set_enabled(args.model_id, True)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"enabled {args.model_id} (catalog only; supervisor will pick this up on restart)")
    return 0


def _cmd_models_disable(args):
    """Disable a model.

    When MUSE_ADMIN_TOKEN is set AND the supervisor is running, this
    routes through the admin API to unload the worker live. Otherwise
    it falls back to mutating catalog.json directly with a warning.
    """
    if _try_admin_action("disable", args.model_id):
        return 0

    from muse.core.catalog import set_enabled
    try:
        set_enabled(args.model_id, False)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"disabled {args.model_id} (catalog only; supervisor will pick this up on restart)")
    return 0


def _try_admin_action(action: str, model_id: str) -> bool:
    """Try the admin-API path for enable/disable; return True iff used.

    Returns False to signal the caller should fall back to the legacy
    catalog-only mutation. The fallback runs when:
      - MUSE_ADMIN_TOKEN isn't set (admin disabled)
      - supervisor isn't reachable (httpx.ConnectError)
      - server returned 503 admin_disabled (token not configured server-side)

    Other errors (404 model_not_found, etc.) are treated as fatal and
    the caller does NOT fall through.
    """
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
                print(f"enable job submitted: {job_id}")
                try:
                    final = client.wait(job_id, timeout=120.0, poll=0.5)
                    if final.get("state") == "done":
                        port = (final.get("result") or {}).get("worker_port")
                        msg = f"enabled {model_id}"
                        if port:
                            msg += f" (worker port {port})"
                        print(msg)
                        return True
                    print(
                        f"error: enable failed: {final.get('error')}",
                        file=sys.stderr,
                    )
                    return True
                except TimeoutError:
                    print(
                        f"warning: enable still running; poll job {job_id} for status",
                        file=sys.stderr,
                    )
                    return True
        elif action == "disable":
            out = client.disable(model_id)
            print(f"disabled {model_id}")
            if out.get("worker_terminated"):
                print(f"  worker on port {out.get('worker_port')} stopped")
            elif out.get("worker_port"):
                remaining = out.get("remaining_models_in_worker") or []
                print(
                    f"  worker on port {out['worker_port']} restarted; "
                    f"remaining models: {', '.join(remaining)}"
                )
            return True
    except AdminClientError as e:
        if e.status == 503 or e.code == "admin_disabled":
            return False
        print(f"error: {e.message}", file=sys.stderr)
        # Treat as fatal so caller doesn't double-emit
        return True
    except Exception as e:  # noqa: BLE001
        # httpx.ConnectError, etc.: supervisor not reachable -> fall back
        print(
            f"warning: admin API unreachable ({e}); falling back to catalog-only update",
            file=sys.stderr,
        )
        return False
    return False


def _cmd_models_probe(args):
    from muse.cli_impl.probe import run_probe, run_probe_all
    if args.model_id is None:
        return run_probe_all(
            no_inference=args.no_inference,
            device=args.device,
            as_json=args.json,
        )
    return run_probe(
        model_id=args.model_id,
        no_inference=args.no_inference,
        device=args.device,
        as_json=args.json,
    )


def _cmd_models_refresh(args):
    from muse.cli_impl.refresh import run_refresh
    return run_refresh(
        model_id=args.model_id,
        all_=args.all_,
        enabled_only=args.enabled_only,
        no_extras=args.no_extras,
        as_json=args.as_json,
    )


def _cmd_mcp(args):
    import os as _os
    from muse.cli_impl.mcp_server import run_mcp_server
    server_url = args.server or _os.environ.get("MUSE_SERVER", "http://localhost:8000")
    admin_token = args.admin_token or _os.environ.get("MUSE_ADMIN_TOKEN")
    return run_mcp_server(
        http=args.http,
        port=args.port,
        server_url=server_url,
        admin_token=admin_token,
        filter_kind=args.filter_kind,
    )


def _cmd_probe_worker(args):
    from muse.cli_impl.probe_worker import run_probe_worker
    return run_probe_worker(
        model_id=args.model,
        device=args.device,
        run_inference=not args.no_inference,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
