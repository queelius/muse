"""Helpers shared by the inference tool modules.

Content blocks, request-body shaping, binary-slot resolution, and the
handler most tools use: forward the arguments to one ``MuseClient``
method and return its JSON envelope as one TextContent block. Errors
are not handled here; ``MCPServer.call_handler`` maps them once.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from muse.mcp.binary_io import resolve_binary_input


def json_block(payload: Any, *, ensure_ascii: bool = True) -> dict:
    """Pack a JSON-serializable payload as a TextContent block."""
    return {
        "type": "text",
        "text": json.dumps(payload, indent=2, ensure_ascii=ensure_ascii),
    }


def request_body(args: dict, *binary_fields: str) -> dict:
    """``args`` minus None values and minus any key starting with
    ``<field>_`` for each binary field (its b64 / url / path slots)."""
    prefixes = tuple(f"{f}_" for f in binary_fields)
    return {
        k: v for k, v in args.items()
        if v is not None and not k.startswith(prefixes)
    }


def binary_arg(args: dict, field: str) -> bytes:
    """Resolve the ``<field>_b64`` / ``_url`` / ``_path`` trio to bytes."""
    return resolve_binary_input(
        b64=args.get(f"{field}_b64"),
        url=args.get(f"{field}_url"),
        path=args.get(f"{field}_path"),
        field_name=field,
    )


def json_handler(method: str, *, binary: str | None = None) -> Callable:
    """A handler calling ``MuseClient.<method>`` with the request body
    (plus the resolved ``binary`` input as a keyword, if given) and
    returning the server's JSON envelope verbatim."""
    def handler(client: Any, args: dict) -> list[dict]:
        if binary is None:
            return [json_block(getattr(client, method)(**request_body(args)))]
        data = binary_arg(args, binary)
        body = request_body(args, binary)
        return [json_block(getattr(client, method)(**{binary: data}, **body))]
    return handler
