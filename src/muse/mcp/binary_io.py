"""Binary I/O helpers for MCP tool args and responses.

Inputs accept three alternative fields per binary slot:
  - <name>_b64:  base64-encoded bytes
  - <name>_url:  data: or http(s) URL
  - <name>_path: local filesystem path

Exactly one must be present. ``resolve_binary_input`` returns raw bytes
or raises ``ValueError`` with a structured message the MCP server
forwards to the LLM client (so it can correct itself on the next turn).

Outputs use MCP content blocks (TextContent + ImageContent +
AudioContent). The pack helpers build plain dicts; the MCPServer
converts those to SDK type instances at the call boundary so this
module stays import-safe without the SDK installed.
"""
from __future__ import annotations

import base64
import os
from typing import Any

import httpx


def resolve_binary_input(
    *,
    b64: str | None = None,
    url: str | None = None,
    path: str | None = None,
    field_name: str = "image",
) -> bytes:
    """Resolve a tri-modal binary input to raw bytes.

    Exactly one of ``b64``, ``url``, ``path`` must be provided. Anything
    else raises ``ValueError`` so the MCP server reports a structured
    error the LLM can correct from on retry.
    """
    provided = [k for k, v in (("b64", b64), ("url", url), ("path", path)) if v]
    if len(provided) == 0:
        raise ValueError(
            f"missing {field_name} input: provide exactly one of "
            f"{field_name}_b64 (base64), {field_name}_url (URL), or "
            f"{field_name}_path (local file path)"
        )
    if len(provided) > 1:
        raise ValueError(
            f"too many {field_name} inputs: provided {provided}; "
            f"provide exactly one"
        )
    if b64 is not None:
        # Strip a leading data: prefix if the LLM included it.
        if b64.startswith("data:"):
            comma = b64.find(",")
            if comma == -1:
                raise ValueError(f"malformed data URL in {field_name}_b64")
            b64 = b64[comma + 1:]
        try:
            return base64.b64decode(b64)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"malformed base64 in {field_name}_b64: {e}") from e
    if url is not None:
        if url.startswith("data:"):
            comma = url.find(",")
            if comma == -1:
                raise ValueError(f"malformed data URL in {field_name}_url")
            return base64.b64decode(url[comma + 1:])
        if url.startswith(("http://", "https://")):
            r = httpx.get(url, timeout=60.0, follow_redirects=True)
            r.raise_for_status()
            return r.content
        raise ValueError(
            f"unsupported {field_name}_url scheme: {url[:32]}; "
            f"expected http(s):// or data:"
        )
    if path is not None:
        if not os.path.exists(path):
            raise ValueError(f"{field_name}_path not found: {path}")
        with open(path, "rb") as f:
            return f.read()
    raise AssertionError("unreachable")


def binary_input_schema(field_name: str = "image") -> dict[str, Any]:
    """Build the JSON-Schema fragment for the three input fields.

    Caller merges this dict into its tool's ``properties``. The fields
    are mutually exclusive but JSON Schema's ``oneOf`` would force the
    LLM into a stricter shape than necessary; we describe the
    exclusivity in prose and validate at call time.
    """
    return {
        f"{field_name}_b64": {
            "type": "string",
            "description": (
                f"Base64-encoded {field_name} bytes. Provide exactly "
                f"one of {field_name}_b64, {field_name}_url, "
                f"{field_name}_path."
            ),
        },
        f"{field_name}_url": {
            "type": "string",
            "description": (
                f"URL to the {field_name} (data: or http(s)://). "
                f"Mutually exclusive with the other two."
            ),
        },
        f"{field_name}_path": {
            "type": "string",
            "description": (
                f"Local filesystem path to the {field_name}. Only "
                f"valid when the MCP server has access to that path."
            ),
        },
    }


def pack_image_output(
    image_bytes: bytes,
    *,
    mime: str = "image/png",
) -> dict[str, Any]:
    """Build an MCP ImageContent dict for raw image bytes."""
    return {
        "type": "image",
        "data": base64.b64encode(image_bytes).decode("ascii"),
        "mimeType": mime,
    }


def pack_audio_output(
    audio_bytes: bytes,
    *,
    mime: str = "audio/wav",
) -> dict[str, Any]:
    """Build an MCP AudioContent dict for raw audio bytes."""
    return {
        "type": "audio",
        "data": base64.b64encode(audio_bytes).decode("ascii"),
        "mimeType": mime,
    }


def pack_text_output(text: str) -> dict[str, Any]:
    """Build an MCP TextContent dict for a string payload."""
    return {"type": "text", "text": text}
