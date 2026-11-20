"""Admin-API-first CLI verbs: `muse models enable/disable/warmup/info`.

`muse.cli` only binds arguments; the logic lives here. Every function that
backs a command returns an exit code (0 on success) and prints exactly what
the command shows.
"""
from __future__ import annotations

from enum import Enum

import typer


class AdminActionOutcome(str, Enum):
    """Three-way result of `try_admin_action`.

    - SUCCESS: the admin API handled the action; caller returns (exit 0).
    - FAILED: the admin API was consulted and the action definitively
      failed (job finished non-done, or an AdminClientError other than
      admin_disabled); caller exits 2.
    - TIMEOUT: a distinguished FAILED sub-case -- the async job's status
      poll timed out before reaching a terminal state, or the request
      connected but did not complete. This is genuinely ambiguous (the
      action may still complete server-side), so it is NOT folded into
      FAILED; callers exit 3 to give operators a distinct signal from a
      hard failure.
    - NOT_HANDLED: no admin token configured, the admin API cannot be
      connected to, or it reports admin_disabled -- caller falls
      through to the catalog-only path exactly as if no admin API
      existed. A request that connected but did not complete (e.g. a
      read timeout) is TIMEOUT, never NOT_HANDLED.
    """

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NOT_HANDLED = "not_handled"


def admin_client(timeout: float):
    """An AdminClient when an admin token is configured, else None.

    The import is deferred (and may raise ImportError on an install
    without the server extras' httpx) so `muse --help` stays light.
    """
    from muse.core import config
    if not config.get("admin.token"):
        return None
    from muse.admin.client import AdminClient
    return AdminClient(timeout=timeout)


def probe_online_worker_status(model_id: str) -> dict | None:
    """Best-effort lookup of live worker status.

    Prefers the admin API (rich worker pid / uptime / restart detail)
    when MUSE_ADMIN_TOKEN is set and the supervisor is reachable. Falls
    back to the PUBLIC /v1/models endpoint, which reports loaded state
    without a token, so `muse models info` shows loaded/not-loaded for
    anyone who can reach the server. Returns None only when nothing is
    reachable; the caller then shows the offline "unreachable" view.

    Shapes:
      - admin:  the /v1/admin/models/{id}/status dict (loaded + worker_*).
      - public: {"loaded": bool, "detail_source": "public"}.
      - None:   nothing reachable.
    """
    admin = probe_admin_worker_status(model_id)
    if admin is not None:
        return admin
    return public_loaded_status(model_id)


def probe_admin_worker_status(model_id: str) -> dict | None:
    """Admin-API worker status, or None without a token / on any failure."""
    try:
        client = admin_client(timeout=2.0)
        return client.status(model_id) if client is not None else None
    except Exception:  # noqa: BLE001
        return None


def public_loaded_status(model_id: str) -> dict | None:
    """Loaded state from the public /v1/models endpoint (no admin token).

    Returns {"loaded": bool, "detail_source": "public"} when the server
    is reachable, else None. A reachable server that does not list the
    model reports loaded=False (reachable, not resident).
    """
    from muse.cli_impl.runtime_state import fetch_public_models, loaded_ids
    data = fetch_public_models()
    if data is None:
        return None
    # Reuse loaded_ids so `muse models info` and `muse models list` answer
    # "is this loaded?" with the SAME predicate (strict `loaded is True`).
    return {"loaded": model_id in loaded_ids(data), "detail_source": "public"}


def try_admin_action(action: str, model_id: str) -> AdminActionOutcome:
    """Try the admin-API path for enable/disable.

    Returns an `AdminActionOutcome` (see its docstring for the meaning
    of each value). Callers must NOT collapse this back to a bool: a
    FAILED or TIMEOUT outcome means the admin API WAS consulted and the
    action did not cleanly succeed, which is different from NOT_HANDLED
    (no admin token / unreachable / admin_disabled), where falling
    through to the catalog-only path is correct.
    """
    try:
        # Disable is synchronous and includes the worker teardown (up to a
        # TERM grace plus KILL), so it gets far more than enable's 202.
        client = admin_client(timeout=5.0 if action == "enable" else 60.0)
    except Exception:  # noqa: BLE001
        return AdminActionOutcome.NOT_HANDLED
    if client is None:
        return AdminActionOutcome.NOT_HANDLED
    from muse.admin.client import AdminClientError
    try:
        if action == "enable":
            out = client.enable(model_id)
            job_id = out.get("job_id")
            if not job_id:
                # client.enable() didn't raise, so the admin API already
                # accepted the request; a missing/empty job_id is a
                # degenerate but still-successful response shape. Treat
                # it as handled so the caller does NOT fall through to
                # the catalog-only fallback (which would both re-do the
                # mutation and print a misleading "catalog only" message).
                typer.echo(f"enabled {model_id} (admin API accepted)")
                return AdminActionOutcome.SUCCESS
            typer.echo(f"enable job submitted: {job_id}")
            try:
                final = client.wait(job_id, timeout=120.0, poll=0.5)
                if final.get("state") == "done":
                    port = (final.get("result") or {}).get("worker_port")
                    msg = f"enabled {model_id}"
                    if port:
                        msg += f" (worker port {port})"
                    typer.echo(msg)
                    return AdminActionOutcome.SUCCESS
                typer.echo(
                    f"error: enable failed: {final.get('error')}", err=True,
                )
                return AdminActionOutcome.FAILED
            except TimeoutError:
                typer.echo(
                    f"warning: enable still running; poll job {job_id} for status",
                    err=True,
                )
                return AdminActionOutcome.TIMEOUT
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
            return AdminActionOutcome.SUCCESS
    except AdminClientError as e:
        # Only "no admin API configured" falls back. Every other refusal
        # (worker_shutdown_incomplete, server_shutting_down, ...) is the
        # live supervisor answering; editing the catalog behind it is wrong.
        if e.code == "admin_disabled":
            return AdminActionOutcome.NOT_HANDLED
        typer.echo(f"error: {e.message}", err=True)
        return AdminActionOutcome.FAILED
    except Exception as e:  # noqa: BLE001
        import httpx
        if isinstance(e, (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout)):
            typer.echo(
                f"warning: admin API unreachable ({e}); falling back to catalog-only update",
                err=True,
            )
            return AdminActionOutcome.NOT_HANDLED
        # The request reached the supervisor (or may have): the action may
        # still complete there, so neither report success nor fall back.
        typer.echo(
            f"warning: {action} request to the admin API did not complete "
            f"({e}); check `muse models info {model_id}`",
            err=True,
        )
        return AdminActionOutcome.TIMEOUT
    return AdminActionOutcome.NOT_HANDLED


def set_model_enabled(model_id: str, enabled: bool) -> int:
    """Shared body of `models enable` / `models disable`."""
    action = "enable" if enabled else "disable"
    outcome = try_admin_action(action, model_id)
    if outcome is AdminActionOutcome.SUCCESS:
        return 0
    if outcome is AdminActionOutcome.FAILED:
        return 2
    if outcome is AdminActionOutcome.TIMEOUT:
        return 3
    # NOT_HANDLED: no admin token / admin unreachable / admin disabled --
    # fall through to the catalog-only path.

    from muse.core.catalog import set_enabled
    try:
        set_enabled(model_id, enabled)
    except KeyError as e:
        typer.echo(f"error: {e}", err=True)
        return 2
    typer.echo(
        f"{action}d {model_id} (catalog only; a running supervisor applies it on the next request)"
    )
    return 0


def run_warmup(model_id: str, *, timeout: float) -> int:
    """`muse models warmup`: pre-load through the admin API, else fail."""
    try:
        client = admin_client(timeout=timeout)
    except Exception as e:  # noqa: BLE001
        typer.echo(f"error: admin client unavailable: {e}", err=True)
        return 2
    if client is None:
        typer.echo(
            "error: warmup requires a running `muse serve` with MUSE_ADMIN_TOKEN set",
            err=True,
        )
        return 2

    from muse.admin.client import AdminClientError
    try:
        # Pass the per-call timeout explicitly: cold loads take 10-60s
        # for typical models and longer (5+ min) for large diffusion or
        # video models. The constructor default (30s) would routinely
        # truncate; the CLI knob lets operators tune up.
        out = client.warmup(model_id, timeout=timeout)
    except AdminClientError as e:
        if e.status == 503 and e.code == "admin_disabled":
            typer.echo(
                "error: warmup requires a running `muse serve` with MUSE_ADMIN_TOKEN set",
                err=True,
            )
            return 2
        typer.echo(f"error: {e.message}", err=True)
        return 1
    except Exception as e:  # noqa: BLE001
        typer.echo(
            f"error: warmup requires a running `muse serve`; could not reach supervisor ({e})",
            err=True,
        )
        return 2

    port = out.get("worker_port") if isinstance(out, dict) else None
    if port:
        typer.echo(f"warmed up {model_id} (worker port {port})")
    else:
        typer.echo(f"warmed up {model_id}")
    return 0


def run_models_info(model_id: str) -> int:
    """`muse models info`: catalog card plus live worker status.

    Recognizes three sources, in priority order: bundled scripts and
    resolver-pulled entries (`known_models()`), curated recommendations not
    yet pulled (`curated.yaml`), else exit 2 with a clear error.
    """
    from muse.core.catalog import _read_catalog, known_models
    from muse.cli_impl.models_info_display import format_curated_info, format_info
    from muse.core.curated import load_curated

    catalog_known = known_models()
    if model_id in catalog_known:
        catalog_data = _read_catalog().get(model_id, {}) or {}
        online_status = probe_online_worker_status(model_id)
        typer.echo(format_info(
            model_id,
            catalog_known=catalog_known,
            catalog_data=catalog_data,
            online_status=online_status,
        ))
        return 0

    # Fall through: maybe it's a curated-only recommendation that
    # hasn't been pulled. Render a minimal info card from the YAML.
    curated_match = next(
        (c for c in load_curated() if c.id == model_id), None,
    )
    if curated_match is not None:
        typer.echo(format_curated_info(curated_match))
        return 0

    typer.echo(f"error: unknown model {model_id!r}", err=True)
    return 2
