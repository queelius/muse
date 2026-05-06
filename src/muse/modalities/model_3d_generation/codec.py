"""Encoding for /v1/3d/generations and /v1/3d/from-image responses.

GLB is binary glTF. Two response modes:

  - `b64_json` (default): each data element is
    `{"b64_json": "<base64-encoded GLB>", "format": "glb"}`.
  - `url`: writes the GLB blob to a tempfile and returns
    `{"url": "file://<absolute path>", "format": "glb"}`. The caller
    is responsible for cleaning up the temp file when done; muse does
    not auto-prune since in-process consumers may still be reading.

Future formats (USDZ, OBJ archive) flow through the same wire shape
by changing the `format` field and the encode body; the
`response_format` dimension stays orthogonal.
"""
from __future__ import annotations

import base64
import tempfile
import time
import uuid
from typing import Any

from muse.modalities.model_3d_generation.protocol import Generation3DResult


_VALID_RESPONSE_FORMATS = ("b64_json", "url")


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
        if response_format == "b64_json":
            data.append({
                "b64_json": base64.b64encode(r.glb_bytes).decode("ascii"),
                "format": r.format,
            })
        else:  # "url"
            with tempfile.NamedTemporaryFile(
                suffix=".glb", delete=False,
            ) as f:
                f.write(r.glb_bytes)
                tmp_path = f.name
            data.append({
                "url": f"file://{tmp_path}",
                "format": r.format,
            })

    return {
        "id": f"3d-{uuid.uuid4().hex[:24]}",
        "created": int(time.time()),
        "model": model_id,
        "data": data,
    }
