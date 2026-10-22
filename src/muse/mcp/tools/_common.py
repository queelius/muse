"""Helpers shared by the admin tool module."""
from __future__ import annotations

import json
from typing import Any


def json_block(payload: Any, *, ensure_ascii: bool = True) -> dict:
    """Pack a JSON-serializable payload as a TextContent block."""
    return {
        "type": "text",
        "text": json.dumps(payload, indent=2, ensure_ascii=ensure_ascii),
    }
