"""MCP tools generated from a muse server's own discovery documents.

One tool per route in ``/v1/capabilities`` (each route's DISCOVERY entry
names its tool via ``mcp_tool``), with the input schema taken from the
route's request schema in ``/openapi.json``. New modalities and models
therefore appear with no MCP code changes, and MCP cannot disagree with
HTTP.

Input changes against the route's own schema:
  - each DISCOVERY ``file_fields`` entry becomes ``<name>_path`` (a local
    file, read and uploaded) and ``<name>_url`` (http(s) or data: URL); a
    list field (``input[]``) becomes ``<name>_paths`` / ``<name>_urls``.
    "Exactly one of" is checked at call time, not with a top-level
    ``oneOf``, because the Claude API rejects top-level oneOf/anyOf/allOf;
  - ``model`` is a string whose description lists the models serving the
    route's modality (not an ``enum``: clients validate against a cached
    schema, which would reject a model enabled mid-session);
  - ``stream`` is removed and always sent as false (MCP results are whole);
  - optional ``out_path``, and ``preview`` on image-producing routes.

Outputs never come back inline: binary bodies and every base64 (or data:
URL) string at a DISCOVERY ``media_fields`` path, or in a sibling ``url``,
are written to local files (``files.py``), as is each plain-text string at a
``text_media_fields`` path (e.g. SVG markup, written as UTF-8 with the
declared MIME type); other values, such as RLE mask dicts, are left alone. A result is a short text block, a ``resource_link``
per file, an optional PNG preview, and ``structuredContent`` (``OUTPUT_SCHEMA``):
``model`` / ``revision`` / ``artifacts`` from the ``X-Muse-Model*`` response
headers, ``files`` (path, sha256, mime, bytes, duration or size), and
``response``, the JSON body with media replaced by file paths (null for a
binary body; a summary plus a written ``.json`` file past 20,000 chars).

Content blocks are plain dicts; the pseudo-block ``{"type": "structured",
"data": {...}}`` carries the ``structuredContent`` the server lifts into the
result. Light by design: httpx, mcp.types and muse.core.config only.
"""
from __future__ import annotations

import base64
import binascii
import copy
import json
import os
import pathlib
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any

import mcp.types as types

from muse.mcp import files

DESCRIBE = "muse_describe"
LARGE_JSON_CHARS = 20_000
MAX_INPUT_BYTES = 256 * 1024 * 1024
_REF = "#/components/schemas/"
# Response headers that carry data (not provenance) -> key in ``response``.
DATA_HEADERS = {"x-alignment": "alignment"}
# Property names MCP clients accept (the Claude API enforces this pattern).
_PROP_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_MAX_DEPTH = 64  # JSON nesting; bounds a cyclic $ref

_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"}, "sha256": {"type": "string"},
        "mime": {"type": "string"}, "bytes": {"type": "integer"},
    },
    "required": ["path", "sha256", "mime", "bytes"],
}
# Repeated on every tool, so kept terse; the module docstring explains it.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "model": {"type": ["string", "null"]},
        "revision": {"type": ["string", "null"]},
        "artifacts": {"type": ["string", "null"]},
        "files": {"type": "array", "items": _FILE_SCHEMA},
        "response": {},
    },
    "required": ["model", "revision", "artifacts", "files", "response"],
}


@dataclass(frozen=True)
class GeneratedTool:
    name: str
    tool: types.Tool
    route: dict                      # the DISCOVERY route entry
    modality: str = ""
    required_files: tuple[str, ...] = ()
    fixed: dict = field(default_factory=dict)   # body values always sent
    image_output: bool = False
    aliases: dict = field(default_factory=dict)  # schema name -> wire name


# ---- schema generation ----

def _resolve(node: Any, schemas: dict, depth: int = 0) -> Any:
    """Inline ``#/components/schemas`` refs so the schema is self-contained."""
    if depth > _MAX_DEPTH:
        return {}
    if isinstance(node, list):
        return [_resolve(v, schemas, depth + 1) for v in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str):
        target = schemas.get(ref[len(_REF):], {}) if ref.startswith(_REF) else {}
        rest = {k: v for k, v in node.items() if k != "$ref"}
        return {**_resolve(target, schemas, depth + 1), **_resolve(rest, schemas, depth + 1)}
    return {k: _resolve(v, schemas, depth + 1) for k, v in node.items()}


def _example_schema(route: dict) -> dict:
    """The route's example keys as untyped properties (no published schema)."""
    example = route.get("example") or {}
    props = {
        k: {"description": "Example: " + json.dumps(v, ensure_ascii=False)[:300]}
        for k, v in example.items()
    }
    return {"properties": props, "required": [k for k in example if k != "model"],
            "additionalProperties": True}


def _request_schema(route: dict, openapi: dict) -> tuple[dict, bool]:
    """``(schema, from_openapi)`` for the route's request."""
    schemas = (openapi.get("components") or {}).get("schemas") or {}
    op = ((openapi.get("paths") or {}).get(route["path"]) or {}).get(
        route["method"].lower()) or {}
    if route["method"].upper() == "GET":
        params = [p for p in op.get("parameters") or [] if p.get("in") == "query"]
        if params:
            return {
                "properties": {p["name"]: _resolve(p.get("schema") or {}, schemas)
                               for p in params},
                "required": [p["name"] for p in params if p.get("required")],
            }, True
        return _example_schema(route), False
    content = (op.get("requestBody") or {}).get("content") or {}
    media = content.get(route.get("request_content_type") or "") or next(
        iter(content.values()), {})
    schema = _resolve(media.get("schema") or {}, schemas)
    if isinstance(schema, dict) and schema.get("properties"):
        return schema, True
    return _example_schema(route), False


def _file_props(name: str, many: bool, required: bool) -> dict:
    need = (f"Exactly one of {name}_path{'s' if many else ''} or "
            f"{name}_url{'s' if many else ''} is required." if required else
            f"At least one image is required via {name}_paths or {name}_urls."
            if many else "Optional.")
    if many:
        arr = {"type": "array", "items": {"type": "string"}}
        return {
            f"{name}_paths": {**arr, "description": f"Local files for `{name}`, read "
                              f"and uploaded. {need}"},
            f"{name}_urls": {**arr, "description": f"http(s) or data: URLs for `{name}`. {need}"},
        }
    return {
        f"{name}_path": {"type": "string", "description": f"Local file for `{name}`, "
                         f"read and uploaded. {need}"},
        f"{name}_url": {"type": "string", "description": f"http(s) or data: URL for `{name}`. {need}"},
    }


def _has_media(route: dict) -> bool:
    return bool(route.get("media_fields") or route.get("text_media_fields")) or not (
        route.get("response_content_type") or "").endswith("json")


def _is_image_output(modality: str, route: dict) -> bool:
    if (route.get("response_content_type") or "").startswith("image/"):
        return True
    return modality.startswith("image/") and bool(route.get("media_fields"))


def _description(mod: dict, route: dict, models: list[dict], from_openapi: bool) -> str:
    lines = [mod.get("summary") or f"muse {mod['modality']}", ""]
    compat = f" (follows {route['compat']})" if route.get("compat") else ""
    lines.append(f"Route: {route['method']} {route['path']}{compat}.")
    if models:
        lines.append("Models:")
        lines += [f"- {m['id']} — {m.get('quality_tier') or 'untiered'} — "
                  f"{m.get('summary') or ''}".rstrip(" —") for m in models]
    else:
        lines.append("No enabled model serves this modality right now.")
    if not from_openapi:
        lines.append("The server publishes no request schema for this route; the "
                     "fields come from its example and extra fields pass through.")
    if _has_media(route):
        lines.append("Media outputs are written to local files (out_path, else "
                     "MUSE_MCP_OUTPUT_DIR); the result links them.")
    return "\n".join(lines)


def _build(mod: dict, route: dict, openapi: dict, models: list[dict]) -> GeneratedTool:
    base, from_openapi = _request_schema(route, openapi)
    props = {k: {kk: vv for kk, vv in (v if isinstance(v, dict) else {}).items()
                 if kk != "title"} for k, v in (base.get("properties") or {}).items()}
    required = [r for r in base.get("required") or [] if r in props]
    aliases = {}
    for wire in [k for k in props if not _PROP_NAME.match(k)]:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", wire).strip("_")[:64] or "field"
        if name in props:
            continue
        props[name] = props.pop(wire)
        aliases[name] = wire
        required = [name if r == wire else r for r in required]
    required_files: list[str] = []
    for f in route.get("file_fields") or []:
        many = f.endswith("[]")
        name = f[:-2] if many else f
        was_required = name in required or f in required
        # The wire name (`image[]`) and the bare name are one logical input.
        props.pop(f, None)
        props.pop(name, None)
        required = [r for r in required if r not in (name, f)]
        props.update(_file_props(name, many, was_required))
        if was_required:
            required_files.append(name)
    fixed = {}
    if "stream" in props:
        props.pop("stream")
        required = [r for r in required if r != "stream"]
        fixed["stream"] = False
    ids = list(mod.get("models") or [])
    # Ids go in the description, never an `enum`: MCP SDKs validate calls
    # against the client's cached schema, so an enum would reject a model
    # enabled mid-session. The server is the authority on unknown ids.
    model_prop = {"type": "string", "description": (
        "Model id. Serving now: " + ", ".join(ids) + " (see the list above; "
        "a model enabled later works too)." if ids else
        "Model id; no enabled model serves this route right now.")}
    if ids:
        required.append("model")
    else:
        required = [r for r in required if r != "model"]
    props["model"] = model_prop
    has_media = _has_media(route)
    props["out_path"] = {"type": "string", "description": (
        "File for the media output (later outputs get -1, -2... suffixes)."
        if has_media else "File for the JSON result when it is too large to "
        "return inline (over 20,000 characters).")}
    image_output = _is_image_output(mod["modality"], route)
    if image_output:
        props["preview"] = {"type": "boolean", "default": False, "description":
                            "Also return one inline PNG preview (long side 512 px)."}
    schema = {"type": "object", "properties": props,
              "required": sorted(set(required)),
              "additionalProperties": bool(base.get("additionalProperties", not from_openapi))}
    tool = types.Tool(name=route["mcp_tool"],
                      description=_description(mod, route, models, from_openapi),
                      inputSchema=schema, outputSchema=OUTPUT_SCHEMA)
    return GeneratedTool(name=route["mcp_tool"], tool=tool, route=copy.deepcopy(route),
                         modality=mod["modality"], required_files=tuple(required_files),
                         fixed=fixed, image_output=image_output, aliases=aliases)


def generate_tools(capabilities: dict, openapi: dict) -> list[GeneratedTool]:
    """One tool per declared route; the first route wins a name clash."""
    by_id = {m.get("id"): m for m in capabilities.get("models") or []}
    out: list[GeneratedTool] = []
    seen = {DESCRIBE}
    for mod in capabilities.get("modalities") or []:
        models = [by_id[i] for i in mod.get("models") or [] if i in by_id]
        for route in mod.get("routes") or []:
            name = route.get("mcp_tool")
            if not name or name in seen or not route.get("path"):
                continue
            seen.add(name)
            out.append(_build(mod, route, openapi, models))
    return out


def describe_tool() -> types.Tool:
    return types.Tool(
        name=DESCRIBE,
        description=(
            "Call this before choosing a muse model or tool. Returns the muse "
            "server's capabilities document: every modality (what it is for and "
            "not for), its routes and tool names, and every model with summary, "
            "strengths, limits, quality_tier, revision and live state (loaded, "
            "memory_gb, queue_depth, last_cold_load_seconds). Filter with "
            "`modality` (e.g. audio/speech) or `model` (an id) to keep it small."
        ),
        inputSchema={"type": "object", "properties": {
            "modality": {"type": "string", "description": "Only this modality."},
            "model": {"type": "string", "description": "Only this model id."},
        }, "additionalProperties": False},
        outputSchema={"type": "object"},
    )


# ---- calls ----

def _decode_data_url(url: str) -> tuple[bytes, str]:
    head, sep, payload = url.partition(",")
    if not url.startswith("data:") or not sep:
        raise ValueError("not a data: URL")
    meta = head[5:].split(";")
    mime = meta[0] or "text/plain"
    if "base64" in meta[1:]:
        return base64.b64decode(payload, validate=True), mime
    return urllib.parse.unquote_to_bytes(payload), mime


def _fetch_input(url: str, field_name: str) -> tuple[str, bytes, str]:
    """``(filename, data, mime)`` for a multipart upload given by URL."""
    if url.startswith("data:"):
        try:
            data, mime = _decode_data_url(url)
        except (ValueError, binascii.Error) as e:
            raise ValueError(f"{field_name}_url is not a valid data: URL: {e}") from e
        if len(data) > MAX_INPUT_BYTES:
            raise ValueError(f"{field_name}_url exceeds {MAX_INPUT_BYTES} bytes")
        return field_name, data, mime
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"{field_name}_url must be an http(s) or data: URL")
    import mimetypes

    from muse.core.net_fetch import fetch_url_bytes

    data = fetch_url_bytes(url, max_bytes=MAX_INPUT_BYTES)
    name = os.path.basename(urllib.parse.urlparse(url).path) or field_name
    return name, data, mimetypes.guess_type(name)[0] or "application/octet-stream"


def _take_files(gen: GeneratedTool, args: dict, multipart: bool) -> list[tuple]:
    """Move file arguments out of ``args``; returns multipart parts.

    The parts are ``(field, (filename, bytes, mime))`` pairs, as httpx
    ``files=`` accepts. A ``[]`` field on a multipart route sends one part
    per file, each named with the declared ``X[]`` (the OpenAI SDK's name).
    """
    parts: list[tuple] = []
    for f in gen.route.get("file_fields") or []:
        many = f.endswith("[]")
        name = f[:-2] if many else f
        s = "s" if many else ""
        path, url = args.pop(f"{name}_path{s}", None), args.pop(f"{name}_url{s}", None)
        if path and url:
            raise ValueError(f"pass only one of {name}_path{s} or {name}_url{s}")
        if not path and not url:
            if name in gen.required_files:
                raise ValueError(f"{name}_path{s} or {name}_url{s} is required")
            continue
        if many and multipart:
            if path:
                parts.extend((f, _read_named(p, name)) for p in path)
            else:
                parts.extend((f, _fetch_input(u, name)) for u in url)
        elif many:
            args[name] = ([files.as_data_url(*_read(p, name)) for p in path]
                          if path else list(url))
        elif multipart:
            parts.append((name, _read_named(path, name) if path else _fetch_input(url, name)))
        else:
            args[name] = files.as_data_url(*_read(path, name)) if path else url
    return parts


def _read_named(path: str, name: str) -> tuple[str, bytes, str]:
    data, filename, mime = files.read_input(path, field=name, max_bytes=MAX_INPUT_BYTES)
    return filename, data, mime


def _read(path: str, name: str) -> tuple[bytes, str]:
    data, _, mime = files.read_input(path, field=name, max_bytes=MAX_INPUT_BYTES)
    return data, mime


def _form(value: Any) -> str | list[str]:
    """A multipart form value; a list becomes repeated fields."""
    if isinstance(value, list):
        return [v if isinstance(v, str) else json.dumps(v) for v in value]
    return value if isinstance(value, str) else json.dumps(value)


def call_generated(client: Any, gen: GeneratedTool, args: dict) -> list[dict]:
    """Run one generated tool; returns plain content-block dicts."""
    args = dict(args or {})
    out_path = args.pop("out_path", None)
    preview = bool(args.pop("preview", False)) and gen.image_output
    route = gen.route
    method = route["method"].upper()
    multipart = route.get("request_content_type") == "multipart/form-data"
    parts = _take_files(gen, args, multipart)
    body = {gen.aliases.get(k, k): v for k, v in args.items() if v is not None}
    if method == "GET":
        resp = client.call(route, params=body)
    elif multipart:
        resp = client.call(route, data={k: _form(v) for k, v in body.items()},
                           files=parts)
    else:
        resp = client.call(route, json={**body, **gen.fixed})
    return _render(resp, gen, out_path=out_path, preview=preview,
                   model=body.get("model"))


# ---- results ----

_MAGIC = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"),
          (b"GIF8", "image/gif"), (b"glTF", "model/gltf-binary"),
          (b"\x1a\x45\xdf\xa3", "video/webm"), (b"OggS", "audio/ogg"),
          (b"fLaC", "audio/flac"), (b"ID3", "audio/mpeg"))


def _sniff(data: bytes) -> str:
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF":
        return {b"WEBP": "image/webp", b"WAVE": "audio/wav"}.get(
            data[8:12], "application/octet-stream")
    if data[4:8] == b"ftyp":
        return "video/mp4"
    if data.lstrip()[:5] in (b"<svg ", b"<?xml"):
        return "image/svg+xml"
    return "application/octet-stream"


class _Writer:
    """Writes outputs, honouring ``out_path`` (suffixed after the first)."""

    def __init__(self, out_path: str | None):
        self.out_path = out_path
        self.written: list[tuple[files.WrittenFile, bytes]] = []

    def write(self, data: bytes, mime: str) -> files.WrittenFile:
        target = None
        if self.out_path:
            n = len([w for w in self.written if w[0].mime != "application/json"])
            root, ext = os.path.splitext(self.out_path)
            target = self.out_path if n == 0 else f"{root}-{n}{ext}"
        wf = files.write_output(data, mime, out_path=target)
        self.written.append((wf, data))
        return wf

    def media(self, value: str) -> str:
        """Write a base64 / data: URL value; return its path (or the value)."""
        try:
            if value.startswith("data:"):
                data, mime = _decode_data_url(value)
            else:
                data = base64.b64decode(value, validate=True)
                mime = _sniff(data)
        except (ValueError, binascii.Error):
            return value
        return self.write(data, mime).path if data else value


def _walk(node: Any, parts: list[str], convert, *, sibling_url: bool = True) -> None:
    """Replace string values at a ``a.b[].c`` path with ``convert(value)``;
    others untouched. ``sibling_url`` also converts a data: URL in the
    leaf's sibling ``url`` (base64 media envelopes)."""
    if not parts or not isinstance(node, dict):
        return
    head, rest = parts[0], parts[1:]
    many = head.endswith("[]")
    key = head[:-2] if many else head
    if not many and not rest:
        if isinstance(node.get(key), str) and node[key]:
            node[key] = convert(node[key])
        url = node.get("url")
        if sibling_url and key != "url" and isinstance(url, str) and url.startswith("data:"):
            node["url"] = convert(url)
        return
    if not many:
        return _walk(node.get(key), rest, convert, sibling_url=sibling_url)
    items = node.get(key)
    if not isinstance(items, list):
        return
    for i, item in enumerate(items):
        if rest:
            _walk(item, rest, convert, sibling_url=sibling_url)
        elif isinstance(item, str) and item:
            items[i] = convert(item)


def _outline(value: Any, depth: int = 0) -> Any:
    if isinstance(value, dict):
        if depth >= 2:
            return f"object({len(value)} keys)"
        return {k: _outline(v, depth + 1) for k, v in list(value.items())[:50]}
    if isinstance(value, list):
        return f"array({len(value)})"
    if isinstance(value, str) and len(value) > 200:
        return value[:200] + "..."
    return value


def _file_entry(wf: files.WrittenFile, data: bytes) -> dict:
    facts = files.media_facts(data, wf.mime) if wf.mime != "application/json" else {}
    return {"path": wf.path, "sha256": wf.sha256, "mime": wf.mime,
            "bytes": wf.bytes, **facts}


def _maybe_spill(body: Any, writer: _Writer) -> Any:
    """``body``, or a summary plus a written .json when it is too large."""
    text = json.dumps(body, ensure_ascii=False)
    if len(text) <= LARGE_JSON_CHARS:
        return body
    # out_path names the spill only when no media took it.
    target = writer.out_path if not writer.written else None
    wf = files.write_output(text.encode("utf-8"), "application/json", out_path=target)
    writer.written.append((wf, b""))
    return {"written_to": wf.path, "chars": len(text), "outline": _outline(body)}


def _describe_file(f: dict) -> str:
    extra = []
    if "duration_seconds" in f:
        extra.append(f"{f['duration_seconds']} s")
    if "width" in f:
        extra.append(f"{f['width']}x{f['height']}")
    return f"file: {f['path']} ({f['mime']}, {f['bytes']} bytes" + "".join(
        f", {e}" for e in extra) + ")"


def _blocks(structured: dict, writer: _Writer, *, preview: bool) -> list[dict]:
    lines = [f"model: {structured['model'] or 'unknown'}",
             f"revision: {structured['revision'] or 'unknown'}"]
    if structured["artifacts"]:
        lines.append(f"artifacts: {structured['artifacts']}")
    lines += [_describe_file(f) for f in structured["files"]]
    if structured["response"] is not None:
        lines.append("response: " + json.dumps(structured["response"], ensure_ascii=False))
    blocks: list[dict] = [{"type": "text", "text": "\n".join(lines)}]
    for f in structured["files"]:
        blocks.append({"type": "resource_link", "uri": pathlib.Path(f["path"]).as_uri(),
                       "name": os.path.basename(f["path"]), "mimeType": f["mime"],
                       "size": f["bytes"]})
    if preview:
        image = next((d for wf, d in writer.written if wf.mime.startswith("image/")), None)
        png = files.preview_png(image) if image else None
        if png:
            blocks.append({"type": "image", "mimeType": "image/png",
                           "data": base64.b64encode(png).decode("ascii")})
        elif image:
            blocks[0]["text"] += "\npreview unavailable: " + (
                "not a raster image" if _has_pillow() else "Pillow not installed")
    blocks.append({"type": "structured", "data": structured})
    return blocks


def _has_pillow() -> bool:
    import importlib.util

    return importlib.util.find_spec("PIL") is not None


def _render(resp: Any, gen: GeneratedTool, *, out_path: str | None,
            preview: bool, model: str | None) -> list[dict]:
    writer = _Writer(out_path)
    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype == "application/json" or ctype.endswith("+json"):
        body = resp.json()
        for path in gen.route.get("media_fields") or []:
            _walk(body, path.split("."), writer.media)
        for path, mime in (gen.route.get("text_media_fields") or {}).items():
            _walk(body, path.split("."),
                  lambda text, mime=mime: writer.write(text.encode("utf-8"), mime).path,
                  sibling_url=False)
        response = _maybe_spill(body, writer)
    else:
        writer.write(resp.content, ctype or _sniff(resp.content))
        response = None
    for header, key in DATA_HEADERS.items():
        if header in resp.headers:
            try:
                value = json.loads(resp.headers[header])
            except ValueError:
                value = resp.headers[header]
            response = {**(response if isinstance(response, dict) else {}), key: value}
    structured = {
        "model": resp.headers.get("x-muse-model") or model,
        "revision": resp.headers.get("x-muse-model-revision"),
        "artifacts": resp.headers.get("x-muse-model-artifacts"),
        "files": [_file_entry(wf, d) for wf, d in writer.written],
        "response": response,
    }
    return _blocks(structured, writer, preview=preview)


def call_describe(client: Any, args: dict) -> list[dict]:
    """``muse_describe``: the capabilities document (spilled when large)."""
    doc = client.capabilities(modality=args.get("modality"), model=args.get("model"))
    text = json.dumps(doc, ensure_ascii=False)
    if len(text) > LARGE_JSON_CHARS:
        wf = files.write_output(text.encode("utf-8"), "application/json", out_path=None)
        doc = {
            "written_to": wf.path,
            "note": "Full document written to a file; call again with `modality` "
                    "or `model` for full entries.",
            "server": doc.get("server"),
            "modalities": [{"modality": m.get("modality"), "summary": m.get("summary"),
                            "tools": [r.get("mcp_tool") for r in m.get("routes") or []],
                            "models": m.get("models")} for m in doc.get("modalities") or []],
            "models": [{k: m.get(k) for k in ("id", "modality", "license", "quality_tier",
                                              "summary", "loaded")}
                       for m in doc.get("models") or []],
        }
        text = json.dumps(doc, ensure_ascii=False)
    return [{"type": "text", "text": text}, {"type": "structured", "data": doc}]
