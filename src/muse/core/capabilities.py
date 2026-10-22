"""The capabilities document an agent reads to learn what a muse server hosts.

`GET /v1/capabilities` (gateway and worker) serves one JSON document:

    {"object": "muse.capabilities",
     "server": {version, node, links, mcp},
     "modalities": [{modality, summary, use_for, not_for, routes, models}],
     "models": [{id, modality, hf_repo, license, revision, components,
                 summary, strengths, limits, quality_tier, capabilities,
                 loaded, memory_gb, last_cold_load_seconds, queue_depth,
                 unservable_reason}]}

Facts about a model split in two: `ModelFacts` (identity, notes,
capabilities, size) change only when the catalog does, so callers cache them;
`ModelState` (resident, queue, last cold load, servability) is live and
recomputed per call. `build_document` and `render_llms_txt` are pure: callers
gather the facts (I/O) and pass them in.

Public: never put pids, tokens, job records or hidden capability keys here.
Stdlib only.
"""
from __future__ import annotations

import copy
import math
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from muse.core.capability_keys import public_capabilities
from muse.core.model_notes import ModelNotes, model_notes
from muse.core.provenance import ModelRevision

MCP_INSTALL = 'uvx --from "museq[mcp]" muse mcp'
# The env var the local MCP program reads to find this server (a value we
# hand the agent, not one muse reads here).
MCP_SERVER_ENV = "MUSE_SERVER"
LINKS = {
    "openapi": "/openapi.json",
    "models": "/v1/models",
    "capabilities": "/v1/capabilities",
    "llms_txt": "/llms.txt",
}


@dataclass(frozen=True)
class ModelFacts:
    """Static facts about one model; cacheable on the catalog's mtime."""

    id: str
    modality: str
    hf_repo: str | None
    license: str | None
    revision: ModelRevision
    notes: ModelNotes
    capabilities: dict  # raw manifest capabilities; filtered on output
    memory_gb: float | None


@dataclass(frozen=True)
class ModelState:
    """Live state of one model; recomputed on every call."""

    loaded: bool
    last_cold_load_seconds: float | None
    queue_depth: int | None
    unservable_reason: str | None


class UnknownFilter(LookupError):
    """A `?modality=` or `?model=` value names nothing in the document."""


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _gb(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        gb = float(value)
    except (TypeError, ValueError):
        return None
    return gb if math.isfinite(gb) and gb >= 0 else None


def model_facts(
    model_id: str,
    manifest: Mapping[str, Any],
    *,
    revision: ModelRevision,
    memory_gb: Any,
    modality: str | None = None,
) -> ModelFacts:
    """Project a manifest (plus resolved revision and size) onto ModelFacts.

    Lenient about types: a hand-edited catalog manifest degrades field by
    field rather than failing the document.
    """
    caps = manifest.get("capabilities")
    return ModelFacts(
        id=model_id,
        modality=_str(manifest.get("modality")) or modality or "",
        hf_repo=_str(manifest.get("hf_repo")),
        license=_str(manifest.get("license")),
        revision=revision,
        notes=model_notes(model_id, manifest),
        capabilities=dict(caps) if isinstance(caps, Mapping) else {},
        memory_gb=_gb(memory_gb),
    )


def node_name(state: Any) -> str:
    """The node identity the telemetry summary reports."""
    for attr in ("node_url", "node_id"):
        value = getattr(state, attr, None) if state is not None else None
        if value:
            return str(value)
    return socket.gethostname()


def server_block(*, version: str, node: str, base_url: str) -> dict:
    return {
        "version": version,
        "node": node,
        "links": dict(LINKS),
        "mcp": {"install": MCP_INSTALL, "env": {MCP_SERVER_ENV: base_url}},
    }


def _model_entry(facts: ModelFacts, state: ModelState | None) -> dict:
    notes = facts.notes
    rev = facts.revision
    return {
        "id": facts.id,
        "modality": facts.modality,
        "hf_repo": facts.hf_repo,
        "license": facts.license,
        "revision": rev.revision,
        "components": [
            {"role": c.role, "repo": c.repo, "revision": c.revision}
            for c in rev.components
        ],
        "summary": notes.summary,
        "strengths": list(notes.strengths),
        "limits": list(notes.limits),
        "quality_tier": notes.quality_tier,
        "capabilities": copy.deepcopy(public_capabilities(facts.capabilities)),
        "loaded": True if state is None else state.loaded,
        "memory_gb": facts.memory_gb,
        "last_cold_load_seconds": None if state is None else state.last_cold_load_seconds,
        "queue_depth": None if state is None else state.queue_depth,
        "unservable_reason": None if state is None else state.unservable_reason,
    }


def build_document(
    *,
    server: dict,
    discovery: Mapping[str, dict],
    models: list[tuple[ModelFacts, ModelState | None]],
    modality: str | None = None,
    model: str | None = None,
) -> dict:
    """Assemble the capabilities document. Pure: no I/O.

    `models` with `ModelState=None` (a worker's own registry) report
    `loaded: true` and null live-state fields. Modalities come from
    `discovery` plus any modality a model serves that declares no
    DISCOVERY. `modality` / `model` slice the document; an unknown value
    (or a model outside the requested modality) raises UnknownFilter.
    """
    entries = sorted(
        (_model_entry(f, s) for f, s in models), key=lambda e: e["id"],
    )
    tags = set(discovery) | {e["modality"] for e in entries if e["modality"]}
    modalities = []
    for tag in sorted(tags):
        d = discovery.get(tag) or {}
        modalities.append({
            "modality": tag,
            "summary": d.get("summary"),
            "use_for": copy.deepcopy(d.get("use_for", [])),
            "not_for": copy.deepcopy(d.get("not_for", [])),
            "routes": copy.deepcopy(d.get("routes", [])),
            "models": [e["id"] for e in entries if e["modality"] == tag],
        })
    doc = {
        "object": "muse.capabilities",
        "server": copy.deepcopy(server),
        "modalities": modalities,
        "models": entries,
    }
    return filter_document(doc, modality=modality, model=model)


def filter_document(
    doc: dict, *, modality: str | None = None, model: str | None = None,
) -> dict:
    """Slice a built (or federation-merged) capabilities document. Pure.

    Returns a new document; other top-level keys (server,
    unreachable_nodes, ...) are kept. `modality` keeps that modality and
    its models; `model` keeps that one model and its modality, with the
    modality's `models` narrowed to it. An unknown value, or a model
    outside the requested modality, raises UnknownFilter.
    """
    if modality is None and model is None:
        return copy.deepcopy(doc)
    entries = [e for e in doc.get("models") or [] if isinstance(e, dict)]
    mods = {
        m["modality"]: m for m in doc.get("modalities") or []
        if isinstance(m, dict) and isinstance(m.get("modality"), str)
    }
    tags = set(mods) | {e.get("modality") for e in entries if e.get("modality")}

    if modality is not None and modality not in tags:
        raise UnknownFilter(f"unknown modality {modality!r}")
    if model is not None:
        match = [e for e in entries if e.get("id") == model]
        if not match:
            raise UnknownFilter(f"unknown model {model!r}")
        if modality is not None and match[0].get("modality") != modality:
            raise UnknownFilter(f"model {model!r} does not serve modality {modality!r}")
        entries = match[:1]
        keep = match[0].get("modality")
    else:
        entries = [e for e in entries if e.get("modality") == modality]
        keep = modality

    out = {k: copy.deepcopy(v) for k, v in doc.items() if k not in ("modalities", "models")}
    base = mods.get(keep) or {
        "modality": keep, "summary": None, "use_for": [], "not_for": [],
        "routes": [], "models": [],
    }
    kept = copy.deepcopy(base)
    if model is not None:
        kept["models"] = [model]
    out["modalities"] = [kept] if keep else []
    out["models"] = copy.deepcopy(entries)
    # Preserve the canonical key order (object, server, modalities, models, ...).
    order = [k for k in doc if k in out] + [k for k in out if k not in doc]
    return {k: out[k] for k in order}


def render_llms_txt(doc: dict) -> str:
    """A short markdown page (llms.txt) generated from the document."""
    server = doc.get("server") or {}
    links = server.get("links") or LINKS
    mcp = server.get("mcp") or {}
    base = (mcp.get("env") or {}).get(MCP_SERVER_ENV) or "http://host:8000"
    install = mcp.get("install") or MCP_INSTALL
    lines = [
        "# muse",
        "",
        f"> A multi-modality generation server (muse {server.get('version', '?')},"
        f" node {server.get('node', '?')}). Read {links['capabilities']} first:"
        " it lists every modality, model, route and current cost as JSON.",
        "",
        "## Discovery",
        "",
        f"- [Capabilities]({links['capabilities']}): modalities, models, routes,"
        " revisions and live state; filter with ?modality= or ?model=",
        f"- [OpenAPI]({links['openapi']}): request and response schemas for every route",
        f"- [Models]({links['models']}): OpenAI-compatible model list",
        "",
        "## MCP",
        "",
        f"`claude mcp add muse -e MUSE_SERVER={base} -- {install}`",
        "",
        "## Modalities",
        "",
    ]
    for m in doc.get("modalities") or []:
        summary = m.get("summary") or "(no description)"
        lines.append(f"- **{m['modality']}**: {summary}")
        for r in m.get("routes") or []:
            compat = f" ({r['compat']})" if r.get("compat") else ""
            lines.append(f"  - `{r['method']} {r['path']}`{compat}")
        ids = m.get("models") or []
        lines.append(f"  - models: {', '.join(ids) if ids else 'none enabled'}")
    lines.append("")
    return "\n".join(lines)
