"""Encoding/decoding for /v1/translate and /languages (LibreTranslate-shape).

Pure functions: no FastAPI, no I/O. Tested standalone.
"""
from __future__ import annotations

from typing import Any

from muse.modalities.text_translation.lang_names import ISO_639_1_NAMES


def shape_response(texts: list[str], *, scalar: bool) -> dict:
    """Build the LibreTranslate-shape /translate response body.

    {"translatedText": texts[0]} when scalar (request `q` was a single
    string) else {"translatedText": texts} (request `q` was a list).
    """
    if scalar:
        return {"translatedText": texts[0]}
    return {"translatedText": list(texts)}


def normalize_q(q: Any) -> tuple[list[str], bool]:
    """Normalize the request's `q` field to a uniform (texts, scalar) pair.

    str -> ([q], True); list[str] -> (q, False). Raises ValueError on
    any other type, or a list containing a non-str item, so the route
    layer can map that to a 422/400.
    """
    if isinstance(q, str):
        return [q], True
    if isinstance(q, list):
        if not all(isinstance(item, str) for item in q):
            raise ValueError("q list items must all be strings")
        return q, False
    raise ValueError("q must be a string or a list of strings")


def languages_payload(supported: dict[str, list[str]]) -> list[dict]:
    """Build the LibreTranslate-shape /languages response body.

    [{code, name, targets}], sorted by code. name comes from
    ISO_639_1_NAMES, falling back to the code itself when unmapped.
    """
    return [
        {
            "code": code,
            "name": ISO_639_1_NAMES.get(code, code),
            "targets": targets,
        }
        for code, targets in sorted(supported.items())
    ]
