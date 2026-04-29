"""Admin tools wrapping muse.admin.client.AdminClient.

Eleven tools cover the admin surface (`/v1/admin/*`):

  - muse_list_models           catalog overview
  - muse_get_model_info        per-model status (live worker fields)
  - muse_search_models         resolver-search for pullable HF repos
  - muse_pull_model            download weights + install deps
  - muse_remove_model          drop from catalog
  - muse_enable_model          load into a worker
  - muse_disable_model         unload from a worker
  - muse_probe_model           measure VRAM / RAM
  - muse_get_memory_status     aggregate memory
  - muse_get_workers           per-worker status + uptime
  - muse_get_jobs              poll a single job or list recent jobs

Long-running ops (pull, probe, enable) return a job_id to the LLM and
hint at polling via muse_get_jobs.

Auth: each handler that calls AdminClient methods inherits the token
the MuseClient was constructed with (from MUSE_ADMIN_TOKEN by default).
``muse_list_models`` works without a token because /v1/models is
unauthenticated; it falls through to MuseClient.list_models().
"""
from __future__ import annotations

import json
from typing import Any

from mcp.types import Tool

from muse.admin.client import AdminClientError
from muse.mcp.client import MuseClient
from muse.mcp.tools import ADMIN_TOOLS, ToolEntry


def _json_block(payload: Any) -> dict:
    """Pack any JSON-serializable payload as an MCP TextContent block."""
    return {"type": "text", "text": json.dumps(payload, indent=2)}


def _admin_error_block(e: AdminClientError) -> dict:
    return _json_block({
        "error": {
            "code": e.code,
            "message": e.message,
            "status": e.status,
            "type": "admin_api_error",
        },
    })


# ---- handlers ----

def _handle_list_models(client: MuseClient, args: dict) -> list[dict]:
    out = client.list_models()
    rows = out.get("data", [])
    if (m := args.get("filter_modality")):
        rows = [r for r in rows if r.get("modality") == m]
    if (s := args.get("filter_status")):
        rows = [r for r in rows if r.get("status") == s]
    return [_json_block({"data": rows, "count": len(rows)})]


def _handle_get_model_info(client: MuseClient, args: dict) -> list[dict]:
    model_id = args["model_id"]
    try:
        out = client.admin.status(model_id)
    except AdminClientError as e:
        return [_admin_error_block(e)]
    return [_json_block(out)]


def _handle_search_models(client: MuseClient, args: dict) -> list[dict]:
    out = client.search_models(
        query=args["query"],
        modality=args.get("modality"),
        max_size_gb=args.get("max_size_gb"),
        limit=args.get("limit", 20),
    )
    return [_json_block({
        "results": out["results"],
        "count": len(out["results"]),
    })]


def _handle_pull_model(client: MuseClient, args: dict) -> list[dict]:
    try:
        out = client.admin.pull(args["identifier"])
    except AdminClientError as e:
        return [_admin_error_block(e)]
    return [_json_block({
        "job_id": out.get("job_id"),
        "status": out.get("status"),
        "poll_with": (
            f"muse_get_jobs(job_id='{out.get('job_id')}')"
        ),
        "estimated_time": (
            "30 seconds to 5 minutes depending on model size and "
            "download speed"
        ),
    })]


def _handle_remove_model(client: MuseClient, args: dict) -> list[dict]:
    try:
        out = client.admin.remove(
            args["model_id"],
            purge=bool(args.get("purge", False)),
        )
    except AdminClientError as e:
        return [_admin_error_block(e)]
    return [_json_block(out)]


def _handle_enable_model(client: MuseClient, args: dict) -> list[dict]:
    try:
        out = client.admin.enable(args["model_id"])
    except AdminClientError as e:
        return [_admin_error_block(e)]
    return [_json_block({
        "job_id": out.get("job_id"),
        "status": out.get("status"),
        "poll_with": f"muse_get_jobs(job_id='{out.get('job_id')}')",
        "estimated_time": (
            "5 to 60 seconds depending on model load time"
        ),
    })]


def _handle_disable_model(client: MuseClient, args: dict) -> list[dict]:
    try:
        out = client.admin.disable(args["model_id"])
    except AdminClientError as e:
        return [_admin_error_block(e)]
    return [_json_block(out)]


def _handle_probe_model(client: MuseClient, args: dict) -> list[dict]:
    try:
        out = client.admin.probe(
            args["model_id"],
            no_inference=bool(args.get("no_inference", False)),
            device=args.get("device"),
        )
    except AdminClientError as e:
        return [_admin_error_block(e)]
    return [_json_block({
        "job_id": out.get("job_id"),
        "status": out.get("status"),
        "poll_with": f"muse_get_jobs(job_id='{out.get('job_id')}')",
        "estimated_time": "10 seconds to 5 minutes per model",
    })]


def _handle_get_memory_status(client: MuseClient, args: dict) -> list[dict]:  # noqa: ARG001
    try:
        out = client.admin.memory()
    except AdminClientError as e:
        return [_admin_error_block(e)]
    return [_json_block(out)]


def _handle_get_workers(client: MuseClient, args: dict) -> list[dict]:  # noqa: ARG001
    try:
        out = client.admin.workers()
    except AdminClientError as e:
        return [_admin_error_block(e)]
    return [_json_block(out)]


def _handle_get_jobs(client: MuseClient, args: dict) -> list[dict]:
    try:
        if args.get("job_id"):
            out = client.admin.job(args["job_id"])
        else:
            out = client.admin.jobs()
    except AdminClientError as e:
        return [_admin_error_block(e)]
    return [_json_block(out)]


# ---- tool definitions ----

ADMIN_TOOLS.extend([
    ToolEntry(
        tool=Tool(
            name="muse_list_models",
            description=(
                "List all models in muse's catalog. Use this to discover "
                "what's available before pulling, or to check what's "
                "currently loaded. Returns a list with id, modality, "
                "and capabilities."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filter_modality": {
                        "type": "string",
                        "description": (
                            "Filter to one modality, e.g. 'chat/completion', "
                            "'audio/speech', 'image/generation'."
                        ),
                    },
                    "filter_status": {
                        "type": "string",
                        "description": "Filter by status, e.g. 'enabled' or 'disabled'.",
                    },
                },
            },
        ),
        handler=_handle_list_models,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_get_model_info",
            description=(
                "Get detailed live status for one model: catalog "
                "metadata, capabilities, measured memory, plus worker "
                "status (pid, port, uptime, restart count) when loaded."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Catalog id, e.g. 'kokoro-82m', 'sd-turbo'.",
                    },
                },
                "required": ["model_id"],
            },
        ),
        handler=_handle_get_model_info,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_search_models",
            description=(
                "Search HuggingFace for muse-pullable models matching a "
                "query. Use this to discover candidate models before "
                "calling muse_pull_model. Filter by modality and "
                "max size to narrow the result set."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords (e.g. 'qwen3', 'whisper').",
                    },
                    "modality": {
                        "type": "string",
                        "description": "Optional modality filter (e.g. 'chat/completion').",
                    },
                    "max_size_gb": {
                        "type": "number",
                        "description": "Skip results larger than this many GB.",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": ["query"],
            },
        ),
        handler=_handle_search_models,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_pull_model",
            description=(
                "Download model weights and install per-model "
                "dependencies. Long-running: returns a job_id; poll "
                "muse_get_jobs to track progress. Accepts a curated "
                "alias (e.g. 'kokoro-82m') or resolver URI (e.g. "
                "'hf://Qwen/Qwen3-9B-GGUF@q4_k_m'). Requires admin token."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "identifier": {
                        "type": "string",
                        "description": (
                            "Curated alias or resolver URI to pull."
                        ),
                    },
                },
                "required": ["identifier"],
            },
        ),
        handler=_handle_pull_model,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_remove_model",
            description=(
                "Remove a model from muse's catalog. Pass purge=true "
                "to also delete its per-model venv. Refuses to remove "
                "loaded models; disable first. Requires admin token."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Catalog id to remove.",
                    },
                    "purge": {
                        "type": "boolean",
                        "default": False,
                        "description": "Also delete the per-model venv.",
                    },
                },
                "required": ["model_id"],
            },
        ),
        handler=_handle_remove_model,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_enable_model",
            description=(
                "Enable a pulled model: marks it enabled in the catalog "
                "and loads it into a worker. Long-running: returns "
                "job_id; poll muse_get_jobs. Requires admin token."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                },
                "required": ["model_id"],
            },
        ),
        handler=_handle_enable_model,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_disable_model",
            description=(
                "Disable a model: marks it disabled in the catalog and "
                "unloads it from its worker. Synchronous (returns "
                "immediately). Requires admin token."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                },
                "required": ["model_id"],
            },
        ),
        handler=_handle_disable_model,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_probe_model",
            description=(
                "Measure a model's VRAM/RAM by loading + running "
                "representative inference. Long-running: returns "
                "job_id; poll muse_get_jobs. Requires admin token."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "no_inference": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "Skip representative inference (faster but "
                            "undersells peak memory)."
                        ),
                    },
                    "device": {
                        "type": "string",
                        "enum": ["auto", "cpu", "cuda", "mps"],
                        "description": "Override the model's device preference.",
                    },
                },
                "required": ["model_id"],
            },
        ),
        handler=_handle_probe_model,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_get_memory_status",
            description=(
                "Aggregate memory across all loaded models, by device "
                "(GPU/CPU). Includes per-model breakdowns. Useful "
                "before enabling a heavy model to check headroom. "
                "Requires admin token."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        handler=_handle_get_memory_status,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_get_workers",
            description=(
                "List all workers, the models each hosts, plus pid, "
                "port, uptime, and restart count. Useful for live "
                "ops visibility. Requires admin token."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        handler=_handle_get_workers,
    ),
    ToolEntry(
        tool=Tool(
            name="muse_get_jobs",
            description=(
                "Poll a long-running admin job by id, or list recent "
                "jobs when called without job_id. State machine: "
                "pending -> running -> done | failed. Use after "
                "muse_pull_model, muse_enable_model, or muse_probe_model "
                "to track progress. Requires admin token."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": (
                            "Specific job id to poll; omit to list recent."
                        ),
                    },
                },
            },
        ),
        handler=_handle_get_jobs,
    ),
])
