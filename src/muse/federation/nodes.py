"""Node-membership model for the muse federation coordinator.

A "node" is a remote muse `serve` instance the coordinator forwards
requests to. This module is intentionally light: stdlib + `yaml` only
(no torch, no fastapi), so it can be imported early without pulling in
heavy ML deps.

Two node sources merge into one list:
  - CLI entries: plain URLs (`"http://host:8000"`) or named entries
    (`"name=http://host:8000"`).
  - A yaml file with shape `nodes: [{url, name?, token?}, ...]`.

Both sources are merged and deduped by normalized url; the first
occurrence wins (CLI entries take precedence over the yaml file).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yaml


def _normalize_url(url: str) -> str:
    """Strip a single trailing slash so 'http://h:8000/' == 'http://h:8000'."""
    return url[:-1] if url.endswith("/") else url


def _default_name(url: str) -> str:
    """Derive a default node name from the url's host, falling back to
    the url itself when it has no parseable hostname."""
    return urlparse(url).hostname or url


@dataclass(frozen=True)
class NodeSpec:
    url: str
    name: str
    token: str | None = None


def _node_from_cli_entry(entry: str) -> NodeSpec:
    """Parse one CLI node entry: 'http://h:8000' or 'name=http://h:8000'."""
    name, sep, rest = entry.partition("=")
    if sep and "://" in rest:
        url = _normalize_url(rest)
        return NodeSpec(url=url, name=name, token=None)
    url = _normalize_url(entry)
    return NodeSpec(url=url, name=_default_name(url), token=None)


def _nodes_from_yaml(config_path: str | Path) -> list[NodeSpec]:
    path = Path(config_path)
    try:
        text = path.read_text()
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return []
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        return []
    entries = data.get("nodes") or []
    nodes: list[NodeSpec] = []
    for entry in entries:
        if not isinstance(entry, dict) or "url" not in entry:
            continue
        url = _normalize_url(str(entry["url"]))
        name = entry.get("name") or _default_name(url)
        token = entry.get("token")
        nodes.append(NodeSpec(url=url, name=name, token=token))
    return nodes


def load_nodes(
    cli_nodes: list[str] | None = None,
    config_path: str | Path | None = None,
) -> list[NodeSpec]:
    """Merge CLI-provided node entries with a yaml node-list file.

    Dedup by normalized url; the first occurrence wins, so CLI entries
    take precedence over entries from the yaml file with the same url.
    """
    nodes: list[NodeSpec] = []
    for entry in cli_nodes or []:
        nodes.append(_node_from_cli_entry(entry))
    if config_path is not None:
        nodes.extend(_nodes_from_yaml(config_path))

    seen: set[str] = set()
    deduped: list[NodeSpec] = []
    for node in nodes:
        if node.url in seen:
            continue
        seen.add(node.url)
        deduped.append(node)
    return deduped
