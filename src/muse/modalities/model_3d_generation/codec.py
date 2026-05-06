"""Encoding for /v1/3d/generations and /v1/3d/from-image responses.

GLB is binary glTF. Two response modes:

  - `b64_json` (default): each data element is
    `{"b64_json": "<base64-encoded GLB>", "format": "glb"}`.
  - `url`: each data element is
    `{"url": "data:model/gltf-binary;base64,<base64-encoded GLB>",
       "format": "glb"}`.

Both modes embed the GLB bytes in the response. The url mode wraps the
same base64 payload as a data URL with the `model/gltf-binary` MIME
type (per glTF spec) so OpenAI-SDK-shape consumers can hand the URL
directly to a viewer / downloader without an intermediate fetch. We
intentionally do NOT serve a `file://` URL (unreachable from a remote
client) or a server-managed tempfile (would leak unboundedly because
no caller in normal HTTP usage can clean it up).

Future formats (USDZ, OBJ archive) flow through the same wire shape
by changing the `format` field, the encode body, and the data-URL
MIME; the `response_format` dimension stays orthogonal.
"""
from __future__ import annotations

import base64
import time
import uuid
from typing import Any

from muse.modalities.model_3d_generation.protocol import Generation3DResult


_VALID_RESPONSE_FORMATS = ("b64_json", "url")

# Per glTF 2.0 spec, registered IANA MIME type for binary glTF.
_GLB_MIME = "model/gltf-binary"


def encode_3d_response(
    results: list[Generation3DResult],
    *,
    model_id: str,
    response_format: str = "b64_json",
) -> dict[str, Any]:
    """Build the OpenAI-shape envelope for a 3D-generation response.

    Each entry in `data` carries the per-asset payload plus a `format`
    field. Mirrors `/v1/images/generations` shape so OpenAI SDK
    consumers can reuse helper code.

    Raises ValueError when `response_format` is not "b64_json" or "url".
    """
    if response_format not in _VALID_RESPONSE_FORMATS:
        raise ValueError(
            f"response_format must be one of {_VALID_RESPONSE_FORMATS!r}; "
            f"got {response_format!r}"
        )

    data: list[dict[str, Any]] = []
    for r in results:
        encoded = base64.b64encode(r.glb_bytes).decode("ascii")
        if response_format == "b64_json":
            data.append({
                "b64_json": encoded,
                "format": r.format,
            })
        else:  # "url"
            data.append({
                "url": f"data:{_GLB_MIME};base64,{encoded}",
                "format": r.format,
            })

    return {
        "id": f"3d-{uuid.uuid4().hex[:24]}",
        "created": int(time.time()),
        "model": model_id,
        "data": data,
    }
