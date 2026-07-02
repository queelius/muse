"""FastAPI gateway: proxy requests by model-id to the right worker.

The gateway is the user-facing process (port 8000 by default). Workers
live on internal ports (9001+). The gateway:
  1. (v0.40.0+) Routes requests via state.director.acquire(model_id, manifest=...).
     The director may load the model on demand if it isn't currently
     loaded (lazy load), evict an LRU model under memory pressure, or
     short-circuit 503 if the requested model exceeds device capacity.
  2. Extracts `model` from each request (body for POST, query for GET)
  3. Forwards the request to the hosting worker, streaming the response (D4)
  4. Aggregates /v1/models and /health across all workers (D3)
  5. Mounts /v1/admin/* with bearer-token auth (v0.28.0+)

The legacy static-routes path (build_gateway(routes=...)) survives for
tests that pre-date the lazy-load wiring. When `state.director` is set,
the proxy skips the static map entirely and calls director.acquire on
every request; release fires on completion in a finally clause that
spans both buffered and streaming responses.

Proxy routing is modality-agnostic: any request with a `model` field
routes to the worker hosting that model, regardless of URL path. This
means future modalities (/v1/embeddings, /v1/audio/transcriptions, ...)
work without gateway changes.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# Imported at module-top so tests can patch
# `muse.cli_impl.gateway.get_manifest` / `._read_catalog` to inject
# manifests or a catalog snapshot without touching state on disk.
# `_read_catalog` backs the /v1/models listing of enabled-but-unloaded
# models (v0.47.3); it is mtime-cached so per-request reads are cheap.
from muse.core.catalog import _read_catalog, get_manifest
from muse.core.errors import error_type_for_status
from muse.core.server import _format_loaded_at, build_model_entry

# Module-top (no import cycle: supervisor imports gateway only lazily).
# `revalidate_servability` re-checks a stale boot unservable stamp against
# the live catalog; `backfill_manifest_memory` sizes the load from the
# catalog when the manifest declares no memory_gb. Both are on the request
# hot path, so importing per-request would pay a needless cost.
from muse.cli_impl.supervisor import (
    backfill_manifest_memory,
    revalidate_servability,
)

# OperationError lives at module-top because `_route_via_director` must
# trap director.acquire() failures; importing inside a hot-path function
# pays a re-import cost on every request even though Python caches the
# module object.
from muse.admin.operations import OperationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerRoute:
    """One entry in the gateway's routing table.

    A worker may host multiple models; each gets its own WorkerRoute
    pointing at the same worker_url.
    """
    model_id: str
    worker_url: str


async def extract_model_from_request(request: Any) -> str | None:
    """Extract the `model` field from a request.

    - POST with JSON body: body["model"]
    - POST with multipart/form-data body: form["model"]
      (used by /v1/audio/transcriptions, /v1/audio/translations, and
      future multipart endpoints like /v1/images/edits.)
    - GET: query_params["model"]
    - Anything else: None

    Returns None (not raises) on missing/invalid. The caller decides
    what "no model specified" means (400, or fall back to default).
    """
    if request.method == "GET":
        return request.query_params.get("model")

    if request.method == "POST":
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body_bytes = await request.body()
                body = json.loads(body_bytes)
                if not isinstance(body, dict):
                    return None
                return body.get("model")
            except (json.JSONDecodeError, ValueError):
                return None
        if "multipart/form-data" in content_type:
            try:
                # Read body FIRST so Starlette caches it on _body.
                # request.form() consumes the receive stream; without
                # caching, _forward's later request.body() raises
                # "Stream consumed". With _body set, stream() yields
                # the cached bytes and form() parses from those.
                await request.body()
                form = await request.form()
                model = form.get("model")
                # form values are strings or UploadFile; only the
                # string value is a valid model id.
                return model if isinstance(model, str) else None
            except Exception:  # noqa: BLE001
                return None

    return None


def _openai_error(
    status: int,
    code: str,
    message: str,
    *,
    error_type: str = "invalid_request_error",
) -> JSONResponse:
    """OpenAI-compatible error envelope.

    `error_type` defaults to `invalid_request_error` (the OpenAI shape for
    400 / 404). 5xx paths (lazy-load 503 "model_unservable" and
    "model_too_large_for_device", or any director-raised OperationError
    with status >= 500) pass `server_error` so SDK clients that branch on
    type can distinguish "this request is malformed" from "this server is
    full." This matches the chat_completion router's `server_error` choice
    for backend-stream failures (`muse.modalities.chat_completion.routes`).
    """
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "type": error_type}},
    )


def build_gateway(
    routes: list[WorkerRoute] | None = None,
    timeout: float = 300.0,
    *,
    state: "object | None" = None,
) -> FastAPI:
    """Build the gateway FastAPI app.

    Two routing modes:

      - state-driven (preferred for `muse serve`): pass `state`, a
        SupervisorState. Routes are re-derived per request from
        state.workers, filtered to status=='running'. Late-promoting
        workers join the routing table without an app rebuild and
        pending workers don't surface in /v1/models. The
        SupervisorState's lock guards each snapshot.
      - static: pass `routes`, a fixed list of WorkerRoute. Used by
        tests that want a frozen routing table.

    The app exposes:

      - aggregated /v1/models, /health (registered first so the catch-all
        proxy below doesn't shadow them)
      - admin /v1/admin/* (registered before the proxy so admin paths win;
        each admin route requires Authorization: Bearer <MUSE_ADMIN_TOKEN>)
      - catch-all /{full_path:path} that forwards by `model` field to the
        worker hosting that model.
    """
    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI):
        # Lifecycle: when uvicorn shuts the gateway down, drain any
        # outstanding admin job threads (pull subprocesses are SIGTERM'd
        # via JobStore.shutdown's join loop).
        try:
            yield
        finally:
            try:
                from muse.admin.jobs import get_default_store
                get_default_store().shutdown(timeout=5.0)
            except Exception as e:  # noqa: BLE001
                logger.warning("admin job-store shutdown failed: %s", e)

    app = FastAPI(title="Muse Gateway", lifespan=_lifespan)
    app.state.routes_state = state
    app.state.static_routes = (
        {r.model_id: r for r in routes} if routes is not None else {}
    )
    app.state.timeout = timeout

    def _routes_now() -> dict[str, WorkerRoute]:
        """Return the current routing table.

        State-driven: filter state.workers to status=='running' and build
        the table fresh; this means a worker that promotes after gateway
        boot becomes routable on the next request.
        Static: return the frozen dict captured at build_gateway time.
        """
        if app.state.routes_state is not None:
            out: dict[str, WorkerRoute] = {}
            s = app.state.routes_state
            with s.lock:
                for spec in s.workers:
                    if getattr(spec, "status", None) != "running":
                        continue
                    url = f"http://127.0.0.1:{spec.port}"
                    for m in spec.models:
                        out[m] = WorkerRoute(model_id=m, worker_url=url)
            return out
        return dict(app.state.static_routes)

    app.state.routes_now = _routes_now

    @app.get("/_gateway-info")
    def info():
        cur = app.state.routes_now()
        return {
            "routes": [
                {"model_id": r.model_id, "worker_url": r.worker_url}
                for r in cur.values()
            ],
        }

    @app.get("/v1/models")
    async def list_models():
        cur = app.state.routes_now()
        worker_urls = {r.worker_url for r in cur.values()}
        aggregated: list[dict] = []
        async with httpx.AsyncClient(timeout=5.0) as client:
            async def _one(url: str) -> list[dict]:
                try:
                    r = await client.get(f"{url}/v1/models")
                    if r.status_code != 200:
                        return []
                    return r.json().get("data", [])
                except httpx.HTTPError as e:
                    logger.warning("worker %s unreachable: %s", url, e)
                    return []
            results = await asyncio.gather(*[_one(u) for u in worker_urls])
        for items in results:
            aggregated.extend(items)

        # v0.47.4: fill last_loaded_at on resident entries. Workers
        # self-report it as null (they run outside a supervisor and have
        # no director); the gateway owns the load timestamps via
        # state.director.loaded[id].loaded_at, so we join them in here
        # before appending the unloaded rows.
        state = app.state.routes_state
        if state is not None:
            _enrich_loaded_at(state, aggregated)
        # v0.47.3: also advertise enabled-but-unloaded catalog models so
        # /v1/models reflects the catalog, not just resident workers. The
        # loaded workers above are authoritative for loaded=True; here we
        # append every enabled catalog row not already present, with
        # loaded=False + its boot unservable_reason (if any). Skipped in
        # legacy static-routes mode (no SupervisorState to read the
        # unservable map from).
        if state is not None:
            aggregated.extend(_unloaded_catalog_entries(state, aggregated))
        return {"object": "list", "data": aggregated}

    @app.get("/health")
    async def health():
        cur = app.state.routes_now()
        worker_urls = {r.worker_url for r in cur.values()}
        modalities: set[str] = set()
        models: set[str] = set()
        any_down = False
        async with httpx.AsyncClient(timeout=5.0) as client:
            async def _one(url: str) -> dict | None:
                try:
                    r = await client.get(f"{url}/health")
                    if r.status_code != 200:
                        return None
                    return r.json()
                except httpx.HTTPError:
                    return None
            results = await asyncio.gather(*[_one(u) for u in worker_urls])
        for body in results:
            if body is None:
                any_down = True
                continue
            modalities.update(body.get("modalities", []))
            models.update(body.get("models", []))
        # v0.47.4: also report enabled-but-unloaded catalog models so the
        # serviceable surface matches /v1/models. A request naming one of
        # these triggers an on-demand load; reporting only resident
        # workers under-stated what the gateway can actually serve.
        state = app.state.routes_state
        if state is not None:
            already = [{"id": m} for m in models]
            for entry in _unloaded_catalog_entries(state, already):
                models.add(entry["id"])
                if entry.get("modality"):
                    modalities.add(entry["modality"])
        return {
            "status": "degraded" if any_down else "ok",
            "modalities": sorted(modalities),
            "models": sorted(models),
        }

    # Admin routes mounted BEFORE the catch-all so /v1/admin/* paths
    # win the dispatch even though the proxy below would also match
    # them. Auth is enforced on every admin route via the Depends in
    # build_admin_router, so an anonymous /v1/admin/* request returns
    # 503 (no token configured) or 401/403 (token mismatch) before
    # touching any worker. Importing inside build_gateway keeps the
    # admin module out of the import path of muse.cli_impl.worker
    # (which loads inside per-model venvs that may not have fastapi
    # extras installed).
    from muse.admin.errors import install_admin_error_handler
    from muse.admin.routes import build_admin_router
    app.include_router(build_admin_router())
    # Unwrap the admin auth dependency's OpenAI-shaped HTTPException
    # details so /v1/admin/* auth errors emit a bare {"error": {...}}
    # envelope, matching the route-level admin errors instead of the
    # default {"detail": {"error": {...}}} double-wrap.
    install_admin_error_handler(app)

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(request: Request, full_path: str):
        # NOTE: aggregated endpoints (/v1/models, /health) land in D3 as
        # explicit routes registered BEFORE this catch-all, so they win
        # via FastAPI's registration order.

        model_id = await extract_model_from_request(request)
        if model_id is None:
            return _openai_error(
                400, "model_required",
                "request is missing a `model` field (required for gateway routing)",
            )

        # Director-driven path (v0.40.0+ lazy load). When the SupervisorState
        # carries a LoadDirector, every request acquires the worker port
        # via director.acquire and releases on completion. The director
        # transparently handles cold load, LRU eviction, and singleton-
        # load coordination. Static-routes mode (build_gateway(routes=...)
        # with no state.director) still works for the legacy test path.
        s = app.state.routes_state
        if s is not None and getattr(s, "director", None) is not None:
            return await _route_via_director(
                request, full_path, model_id, s, app.state.timeout,
            )

        # Legacy static-routes path: look up the worker URL in a frozen
        # dict and forward. No acquire/release. Used by tests that
        # pre-date Task F.
        cur = app.state.routes_now()
        route = cur.get(model_id)
        if route is None:
            return _openai_error(
                404, "model_not_found",
                f"model {model_id!r} is not registered with any worker; "
                f"known: {sorted(cur)}",
            )

        target_url = f"{route.worker_url}/{full_path}"
        return await _forward(request, target_url, app.state.timeout)

    return app


def _enrich_loaded_at(state: Any, aggregated: list[dict]) -> None:
    """Fill last_loaded_at on resident /v1/models entries, in place.

    Per-model workers self-report last_loaded_at=None because they run
    outside a supervisor and own no LoadDirector. The gateway holds the
    load timestamps in state.director.loaded[id].loaded_at (monotonic
    seconds), so it joins them onto the worker-reported rows here.

    Best-effort and non-fatal: a missing director, a non-dict `loaded`
    (e.g. a bare MagicMock in tests), or any read failure leaves the
    entries untouched. Entries that already carry a non-null
    last_loaded_at are never overwritten.
    """
    director = getattr(state, "director", None)
    if director is None:
        return
    loaded = getattr(director, "loaded", None)
    if not isinstance(loaded, dict):
        return
    # Snapshot id -> loaded_at under the director lock (held briefly); the
    # monotonic->wall-clock render happens outside the lock.
    lock = getattr(director, "lock", None)
    try:
        if lock is not None:
            with lock:
                snap = {
                    mid: getattr(e, "loaded_at", None)
                    for mid, e in loaded.items()
                }
        else:
            snap = {
                mid: getattr(e, "loaded_at", None)
                for mid, e in loaded.items()
            }
    except Exception as e:  # noqa: BLE001
        logger.warning("/v1/models: loaded_at join failed: %s", e)
        return
    for entry in aggregated:
        if entry.get("last_loaded_at") is not None:
            continue
        mono = snap.get(entry.get("id"))
        if mono is None:
            continue
        entry["last_loaded_at"] = _format_loaded_at({"loaded_at": mono})


def _unloaded_catalog_entries(state: Any, loaded: list[dict]) -> list[dict]:
    """Build /v1/models entries for enabled-but-unloaded catalog models.

    `loaded` is the entries already aggregated from resident workers; their
    ids are skipped. Returns one entry per enabled catalog row with a
    `python_path` that isn't already loaded, marked loaded=False with its
    boot `unservable_reason` (if any). Best-effort: a catalog read failure
    yields no extra entries rather than failing the endpoint.
    """
    loaded_ids = {e.get("id") for e in loaded}
    try:
        catalog = _read_catalog()
    except Exception as e:  # noqa: BLE001
        logger.warning("/v1/models: catalog read failed: %s", e)
        return []
    with state.lock:
        reasons = dict(getattr(state, "unservable_reasons", {}) or {})
    out: list[dict] = []
    for model_id, entry in catalog.items():
        if model_id in loaded_ids:
            continue
        if not entry.get("enabled", True):
            continue
        if not entry.get("python_path"):
            continue  # pre-worker entry; cannot load
        try:
            manifest = get_manifest(model_id)
        except KeyError:
            manifest = entry.get("manifest", {}) or {}
        modality = manifest.get("modality") or entry.get("modality", "")
        out.append(build_model_entry(
            model_id, modality, manifest,
            loaded=False, last_loaded_at=None,
            unservable_reason=reasons.get(model_id),
        ))
    return out


async def _route_via_director(
    request: Request,
    full_path: str,
    model_id: str,
    state: Any,
    timeout: float,
) -> Response:
    """Director-driven request routing for v0.40.0 lazy load.

    Order of operations:

      1. Check `state.unservable_reasons[model_id]`. If set, return 503
         with the reason text. The director is NOT called for unservable
         models: the boot-validation step already decided they cannot be
         served, so attempting a load would only stall the request before
         failing.
      2. Resolve the manifest via `muse.core.catalog.get_manifest`. The
         director's load + eviction decisions read `capabilities.memory_gb`
         and `capabilities.device` from this dict. KeyError (model not in
         catalog) -> 404 model_not_found.
      3. Call `state.director.acquire(model_id, manifest=...)`. This may
         block on cold load + eviction. On `OperationError` (the director's
         user-facing failure type, e.g. model_too_large_for_device), map
         to the corresponding HTTP status. Other exceptions propagate to
         FastAPI's default 500 path.
      4. On a successful acquire, forward the request to the returned port.
         The release call MUST fire on completion regardless of outcome.
         For non-streaming responses, the finally clause covers it. For
         streaming responses, the relay generator's own finally clause
         calls release at end-of-iteration (NOT at first-chunk dispatch).
    """
    # 1. Unservable short-circuit (BEFORE acquire). The boot-time
    # validate_catalog_at_boot snapshot can go stale: a `muse models probe`
    # (or weights landing on disk) that arrives after boot makes a
    # previously-unsizable model sizable, and freed memory can make a model
    # that did not fit at boot fit now. When a stamp exists, re-derive the
    # full verdict for THIS one model against the LIVE catalog + live free
    # memory, so the probe (or memory change) takes effect without a
    # supervisor restart. revalidate_servability clears the stamp (returns
    # None) only when the model is sizable AND fits; a genuine "exceeds
    # device capacity" stamp is RETAINED, so an impossible-to-fit model 503s
    # here and never reaches the director's eviction loop (which would
    # otherwise evict the whole idle working set before 503'ing). Reading +
    # mutating the dict happens under state.lock inside revalidate_servability.
    with state.lock:
        unservable_reason = state.unservable_reasons.get(model_id)
    if unservable_reason is not None:
        unservable_reason = revalidate_servability(state, model_id)
    if unservable_reason is not None:
        return _openai_error(
            503,
            "model_unservable",
            f"model {model_id!r} cannot be served: {unservable_reason}",
            error_type="server_error",
        )

    # 2. Resolve manifest. Per-request lookup is cheap: `known_models()`
    # memoizes its merge against catalog.json's (path, mtime_ns) and
    # `_read_catalog()` is mtime-cached the same way, so unchanged
    # catalogs return immediately. Any catalog write -- including the
    # admin pull endpoint's `muse pull` subprocess or an operator's CLI
    # pull beside this supervisor -- bumps the mtime and both re-read, so
    # models pulled after boot route without a restart. KeyError -> 404
    # model_not_found.
    try:
        manifest = get_manifest(model_id)
    except KeyError:
        return _openai_error(
            404, "model_not_found",
            f"model {model_id!r} is not in the catalog",
        )

    # 2b. Size the load. The director sizes loads + drives LRU eviction
    # from capabilities.memory_gb; a probed-only or never-probed model
    # declares none, so backfill it from the catalog (probe measurement,
    # else on-disk weights) -- otherwise the director treats the model as
    # 0 GB and never evicts to make room for it.
    manifest = backfill_manifest_memory(manifest, model_id)

    # 3. Acquire, OFF the event loop. director.acquire may block for tens of
    # seconds on a cold load (worker spawn + health poll). Running it
    # synchronously here would freeze the single gateway event loop and stall
    # EVERY concurrent request, including ones for already-hot models
    # (measured: a 37s cold load stalled 6 hot requests for ~35s each).
    # asyncio.to_thread dispatches it to a worker thread; the director's
    # RLock + in-flight memory reservation make concurrent off-loop acquires
    # safe (they cannot over-admit the device). asyncio.shield keeps the
    # acquire running if THIS request is cancelled (client disconnect /
    # timeout) so a load that already bumped refcount is released rather than
    # leaked -- a leaked refcount would pin the model non-evictable forever.
    acquire_future = asyncio.ensure_future(
        asyncio.to_thread(state.director.acquire, model_id, manifest=manifest)
    )
    try:
        worker_port = await asyncio.shield(acquire_future)
    except OperationError as exc:
        return _openai_error(
            exc.status, exc.code, exc.message,
            error_type=error_type_for_status(exc.status),
        )
    except asyncio.CancelledError:
        # This request was cancelled while the acquire ran in its detached
        # thread. shield() let the acquire keep running; if it SUCCEEDS it
        # bumped refcount, so release once it settles (a threading load
        # cannot be interrupted). A failed acquire never bumped refcount, so
        # only release on a clean result.
        def _release_if_acquired(fut: "asyncio.Future") -> None:
            if not fut.cancelled() and fut.exception() is None:
                try:
                    state.director.release(model_id)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "release after cancelled acquire failed for %r",
                        model_id, exc_info=True,
                    )
        acquire_future.add_done_callback(_release_if_acquired)
        raise

    # 4. Forward. The release call is wired into the response shape:
    # for buffered responses, fire in a finally clause once the body is
    # read; for SSE streams, fire from inside the relay generator's
    # finally clause when the upstream iteration ends (or raises). Both
    # paths converge in `_forward_with_release`.
    target_url = f"http://127.0.0.1:{worker_port}/{full_path}"
    return await _forward_with_release(
        request, target_url, timeout,
        director=state.director, model_id=model_id,
    )


async def _forward_with_release(
    request: Request,
    target_url: str,
    timeout: float,
    *,
    director: Any,
    model_id: str,
) -> Response:
    """Forward + release variant: same shape as `_forward`, but wires the
    director.release call into the response lifecycle.

    Buffered (non-stream) response: release runs after `aread()` and
    before the Response is returned. The TestClient consumes the buffer
    before its `client.post` call returns, so the release fires inside
    the request lifecycle.

    Streaming (SSE) response: release runs inside the relay generator's
    `finally` clause. The clause executes only after the FastAPI runtime
    finishes iterating (full body sent) or the iteration raises (worker
    died, client disconnected). This matches the spec's "release on
    stream-close" requirement: a release on first-chunk dispatch would
    decrement refcount before the request actually finished, opening a
    window where the model could be evicted mid-response.

    Both paths also call director.release on the early-failure branches
    (request-body read, stream-open raise, body-aread raise), so refcount
    is never stranded.
    """
    # The body read happens AFTER the caller's director.acquire() bumped
    # the refcount. A ClientDisconnect (client vanished mid-body) here --
    # reachable for a body-bearing GET, whose body is unread during model
    # extraction -- would otherwise skip release and strand the refcount,
    # wedging eviction. Release the slot first, then re-raise (L16).
    try:
        body = await request.body()
    except BaseException:
        # BaseException, not just Exception: a CancelledError here (client
        # disconnect / request cancellation mid body-read) is a BaseException,
        # so `except Exception` would skip the release and strand the refcount
        # -- pinning the model non-evictable. We re-raise, so cancellation
        # still propagates.
        try:
            director.release(model_id)
        except Exception:  # noqa: BLE001
            logger.warning(
                "director.release(%r) raised during body-read cleanup",
                model_id, exc_info=True,
            )
        raise
    excluded = {"host", "content-length", "transfer-encoding", "connection"}
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in excluded}

    client = httpx.AsyncClient(timeout=timeout)
    stream_ctx = client.stream(
        method=request.method,
        url=target_url,
        headers=fwd_headers,
        content=body,
        params=dict(request.query_params),
    )
    try:
        response = await stream_ctx.__aenter__()
    except BaseException:
        # BaseException, not just Exception: a CancelledError here (the client
        # disconnected while we were opening the worker connection) must
        # release the refcount too, or the hot model is pinned non-evictable
        # forever. We re-raise below, so cancellation still propagates.
        # __aenter__ raised: release the refcount FIRST, then aclose
        # the AsyncClient. Order matters for cascading failure: if we
        # awaited aclose() first and IT raised, control would exit via
        # the new exception and director.release(model_id) would never
        # run. Refcount leaks, eviction wedges. By releasing the
        # director slot first we guarantee the refcount returns to
        # baseline regardless of what aclose() does. The aclose() call
        # below is wrapped so any failure there is logged but does not
        # mask the original stream-open exception we re-raise.
        # Regression watchdog: tests/cli_impl/test_gateway.py
        # ::TestAsyncClientLifecycle::test_stream_open_failure_aclose_client
        # (aclose runs); tests/cli_impl/test_gateway_lazy.py
        # ::TestAcquireRelease::test_release_runs_even_when_aclose_raises_during_stream_open_failure
        # (release runs even when aclose explodes).
        try:
            director.release(model_id)
        except Exception:  # noqa: BLE001
            logger.warning(
                "director.release(%r) raised during stream-open cleanup",
                model_id, exc_info=True,
            )
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            logger.warning(
                "AsyncClient.aclose() raised during stream-open cleanup",
                exc_info=True,
            )
        raise

    content_type = response.headers.get("content-type", "")
    is_stream = "text/event-stream" in content_type

    resp_headers = {
        k: v for k, v in response.headers.items()
        if k.lower() not in excluded
    }

    if is_stream:
        async def relay():
            # The relay generator owns the release call: it fires when
            # the upstream iteration completes (full body sent) OR when
            # it raises (worker died, client disconnected). Either way,
            # the model's refcount drops back to baseline at end-of-life.
            #
            # Order: release FIRST, then close the stream and client.
            # If aexit / aclose raised before release, the refcount would
            # leak. Putting release at the top of the finally chain
            # decouples it from cascading failures in the cleanup path.
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                try:
                    director.release(model_id)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "director.release(%r) raised at stream-close",
                        model_id, exc_info=True,
                    )
                try:
                    await stream_ctx.__aexit__(None, None, None)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "stream_ctx.__aexit__ raised at stream-close",
                        exc_info=True,
                    )
                try:
                    await client.aclose()
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "AsyncClient.aclose() raised at stream-close",
                        exc_info=True,
                    )

        return StreamingResponse(
            relay(),
            status_code=response.status_code,
            headers=resp_headers,
            media_type=content_type,
        )

    # Non-streaming: read the buffered body, then release the director
    # slot FIRST, then aexit the stream and aclose the client. The
    # release-first order matches the stream-open-failure branch and
    # the relay generator: no cleanup-cascading failure can strand the
    # refcount. A failure in aread propagates after release runs in the
    # finally chain.
    try:
        content = await response.aread()
    finally:
        try:
            director.release(model_id)
        except Exception:  # noqa: BLE001
            logger.warning(
                "director.release(%r) raised during buffered-response cleanup",
                model_id, exc_info=True,
            )
        try:
            await stream_ctx.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            logger.warning(
                "stream_ctx.__aexit__ raised during buffered-response cleanup",
                exc_info=True,
            )
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001
            logger.warning(
                "AsyncClient.aclose() raised during buffered-response cleanup",
                exc_info=True,
            )

    return Response(
        content=content,
        status_code=response.status_code,
        headers=resp_headers,
    )


async def _forward(request: Request, target_url: str, timeout: float) -> Response:
    """Forward a request to target_url.

    Detects streaming content-types (text/event-stream) and relays chunks
    via StreamingResponse. Non-streaming responses are read fully and
    returned in one go.

    The httpx client and stream context are held open for the duration of
    a streaming response so chunks dispatch as they arrive from the worker
    (not after full synthesis completes). Same producer-consumer shape as
    the audio/speech router's internal streaming.
    """
    body = await request.body()
    excluded = {"host", "content-length", "transfer-encoding", "connection"}
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in excluded}

    client = httpx.AsyncClient(timeout=timeout)
    stream_ctx = client.stream(
        method=request.method,
        url=target_url,
        headers=fwd_headers,
        content=body,
        params=dict(request.query_params),
    )
    try:
        response = await stream_ctx.__aenter__()
    except Exception:
        # __aenter__ raised: stream_ctx is not entered, so __aexit__
        # is not appropriate. Close the client to release the
        # underlying connection pool slot, then re-raise.
        await client.aclose()
        raise

    content_type = response.headers.get("content-type", "")
    is_stream = "text/event-stream" in content_type

    resp_headers = {
        k: v for k, v in response.headers.items()
        if k.lower() not in excluded
    }

    if is_stream:
        async def relay():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await stream_ctx.__aexit__(None, None, None)
                await client.aclose()

        return StreamingResponse(
            relay(),
            status_code=response.status_code,
            headers=resp_headers,
            media_type=content_type,
        )

    # Non-streaming: read once, close stream + client, return buffered.
    try:
        content = await response.aread()
    finally:
        await stream_ctx.__aexit__(None, None, None)
        await client.aclose()

    return Response(
        content=content,
        status_code=response.status_code,
        headers=resp_headers,
    )
