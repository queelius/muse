"""Merge per-node `/v1/capabilities` documents into one cluster document.

Pure and stdlib-only. `None` for a node means it was unreachable (or sent
something unusable) for this call.
"""
from __future__ import annotations

import copy


def _int_or_none(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def merge_capabilities(
    node_docs: list[tuple[str, dict | None]],
    *,
    server: dict | None = None,
) -> dict:
    unreachable: list[str] = []
    models: dict[str, dict] = {}
    modalities: dict[str, dict] = {}
    for name, doc in node_docs:
        if not isinstance(doc, dict):
            unreachable.append(name)
            continue
        for m in doc.get("models") or []:
            if not isinstance(m, dict) or not isinstance(m.get("id"), str):
                continue
            entry = models.get(m["id"])
            if entry is None:
                entry = models[m["id"]] = {"base": copy.deepcopy(m), "nodes": []}
            entry["nodes"].append({
                "name": name,
                "loaded": bool(m.get("loaded")),
                "queue_depth": _int_or_none(m.get("queue_depth")),
                "revision": m.get("revision"),
            })
        for mod in doc.get("modalities") or []:
            if not isinstance(mod, dict) or not isinstance(mod.get("modality"), str):
                continue
            cur = modalities.get(mod["modality"])
            if cur is None:
                cur = modalities[mod["modality"]] = copy.deepcopy(mod)
                cur["models"] = []
            for mid in mod.get("models") or []:
                if mid not in cur["models"]:
                    cur["models"].append(mid)

    out_models = []
    for mid in sorted(models):
        entry = models[mid]
        m = entry["base"]
        revs = {n["revision"] for n in entry["nodes"]}
        m["revision"] = revs.pop() if len(revs) == 1 else None
        m["loaded"] = any(n["loaded"] for n in entry["nodes"])
        depths = [n["queue_depth"] for n in entry["nodes"] if n["queue_depth"] is not None]
        m["queue_depth"] = sum(depths) if depths else None
        m["nodes"] = entry["nodes"]
        out_models.append(m)
    for mod in modalities.values():
        mod["models"].sort()
    return {
        "object": "muse.capabilities",
        "server": copy.deepcopy(server) if server is not None else {},
        "modalities": [modalities[t] for t in sorted(modalities)],
        "models": out_models,
        "unreachable_nodes": unreachable,
    }
