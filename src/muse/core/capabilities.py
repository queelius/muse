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
import re
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from muse.core.capability_keys import public_capabilities
from muse.core.model_notes import ModelNotes, model_notes
from muse.core.provenance import ModelRevision

MCP_INSTALL = 'uvx --from "museq[mcp]" muse mcp'
_RELEASE = re.compile(r"(\d+)\.(\d+)\.\d+")
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


def mcp_install(version: str) -> str:
    """The `muse mcp` install command for a server at ``version``.

    A release version pins the MCP client to at least its minor line
    (``museq[mcp]>=0.70``): uvx otherwise reuses whatever older museq it
    cached, which may predate DISCOVERY keys the server sends. A dev or
    local version may not be on PyPI, so it gets no pin.
    """
    m = _RELEASE.fullmatch(version or "")
    if not m:
        return MCP_INSTALL
    return f'uvx --from "museq[mcp]>={m.group(1)}.{m.group(2)}" muse mcp'


def server_block(*, version: str, node: str, base_url: str) -> dict:
    return {
        "version": version,
        "node": node,
        "links": dict(LINKS),
        "mcp": {"install": mcp_install(version), "env": {MCP_SERVER_ENV: base_url}},
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
    """Slice a built capabilities document. Pure.

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


LLMS_SUMMARY_CHARS = 120
LLMS_MODELS_PER_MODALITY = 8
_TIER_ORDER = {"best": 0, "good": 1, "draft": 2}


def _one_line(text: Any, limit: int) -> str:
    """``text`` folded onto one line, cut to ``limit`` chars with an ellipsis."""
    flat = " ".join(text.split()) if isinstance(text, str) else ""
    return flat if len(flat) <= limit else flat[:limit - 3].rstrip() + "..."


def _model_line(entry: dict) -> str:
    """``id (tier; state): summary`` for one enabled model."""
    if entry.get("unservable_reason"):
        # Its first clause ("model files are missing"); the full reason is
        # in the capabilities document.
        reason = str(entry["unservable_reason"]).split(";")[0]
        state = f"unservable: {_one_line(reason, 80)}"
    elif entry.get("loaded"):
        state = "loaded"
    else:
        seconds = entry.get("last_cold_load_seconds")
        state = "cold" + (f", last load {seconds:g} s"
                          if isinstance(seconds, (int, float)) else "")
    tier = entry.get("quality_tier") or "untiered"
    summary = _one_line(entry.get("summary"), LLMS_SUMMARY_CHARS)
    return f"{entry['id']} ({tier}; {state})" + (f": {summary}" if summary else "")


def _first_to_try(entry: dict) -> tuple:
    """Sort key: loaded first, then best, good, draft, untiered, then id."""
    return (not entry.get("loaded"), _TIER_ORDER.get(entry.get("quality_tier"), 3),
            str(entry.get("id")))


def render_llms_txt(doc: dict) -> str:
    """The llms.txt page (llmstxt.org shape) generated from the document.

    H1, a one-line blockquote, the workflow, then H2 link lists: discovery
    links, MCP, the modalities that have an enabled model (each linked to
    its filtered capabilities document, with routes and models), and an
    Optional section naming the modalities with no model enabled here.
    Links are relative: behind a TLS or Host-rewriting proxy a base derived
    from the request would be wrong. Only the MUSE_SERVER install line uses
    the derived base. Bounded for big catalogs: each model is one line with
    its summary cut to LLMS_SUMMARY_CHARS, and a modality lists at most
    LLMS_MODELS_PER_MODALITY models (loaded first, then by tier) plus a
    pointer to its ?modality= document for the rest.
    """
    server = doc.get("server") or {}
    links = server.get("links") or LINKS
    mcp = server.get("mcp") or {}
    base = (mcp.get("env") or {}).get(MCP_SERVER_ENV) or "http://host:8000"
    install = mcp.get("install") or MCP_INSTALL
    caps = links["capabilities"]
    by_id = {m.get("id"): m for m in doc.get("models") or [] if isinstance(m, dict)}
    mods = [m for m in doc.get("modalities") or [] if isinstance(m, dict)]
    served = [m for m in mods if m.get("models")]
    idle = [m["modality"] for m in mods if not m.get("models")]

    lines = [
        "# muse",
        "",
        f"> Self-hosted multi-modality generation server (muse"
        f" {server.get('version', '?')}, node {server.get('node', '?')}):"
        " speech, transcription, chat, images, pixel art, 3D, video, embeddings"
        " and more over one HTTP API, OpenAI-compatible where OpenAI has a route.",
        "",
        "How to use it: pick a modality below, read its entry with"
        f" `{links['capabilities']}?modality=<tag>` (what it is for and not for,"
        " request examples, and each model's summary, limits and quality_tier),"
        f" or one model with `?model=<id>`; then POST to the route with that"
        " `model`. Routes marked \"model optional\" use the modality's first"
        " enabled model when `model` is omitted. A model that is not `loaded`"
        " (cold) loads on its first request, which can take seconds to minutes;"
        " later requests are fast. Request and response schemas are in"
        f" `{links['openapi']}`. Every response names the model and revision in"
        " `X-Muse-Model` / `X-Muse-Model-Revision`.",
        "",
        "## Discovery",
        "",
        f"- [Capabilities]({caps}): JSON for every modality, model, route and"
        " live state; filter with ?modality= or ?model=",
        f"- [OpenAPI]({links['openapi']}): request and response schemas"
        " for every route",
        f"- [Models]({links['models']}): OpenAI-compatible model list with"
        " `loaded`",
        "",
        "## MCP",
        "",
        "For agents with MCP (Claude Code, Claude Desktop): tools for the inference routes"
        " plus `muse_describe`; media outputs are written to local files and"
        " returned as paths with sha256, never inline.",
        "",
        f"`claude mcp add muse -e MUSE_SERVER={base} -- {install}`",
        "",
        "## Modalities",
        "",
    ]
    if not served:
        lines.append("- No model is enabled on this server yet.")
    for m in served:
        tag = m["modality"]
        summary = m.get("summary") or "(no description)"
        lines.append(f"- [{tag}]({caps}?modality={tag}): {summary}")
        for r in m.get("routes") or []:
            notes = [r["compat"]] if r.get("compat") else []
            if r.get("model_optional"):
                notes.append("model optional")
            note = f" ({'; '.join(notes)})" if notes else ""
            lines.append(f"  - `{r['method']} {r['path']}`{note}")
        entries = sorted((by_id.get(mid) or {"id": mid} for mid in m["models"]),
                         key=_first_to_try)
        lines += [f"  - model {_model_line(e)}"
                  for e in entries[:LLMS_MODELS_PER_MODALITY]]
        if len(entries) > LLMS_MODELS_PER_MODALITY:
            lines.append(f"  - and {len(entries) - LLMS_MODELS_PER_MODALITY} more"
                         f" (see {caps}?modality={tag})")
    lines += ["", "## Optional", ""]
    if idle:
        lines.append(
            "- Supported by muse, no model enabled here (an operator can"
            " `muse pull` one): " + ", ".join(idle))
    lines.append(f"- [Full capabilities document]({caps}): includes strengths,"
                 " limits, capabilities and revisions for every model")
    lines.append("")
    return "\n".join(lines)
