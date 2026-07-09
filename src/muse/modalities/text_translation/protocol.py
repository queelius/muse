"""Protocol + dataclass for text/translation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class TranslationResult:
    """One or more translated strings produced by a `text/translation`
    backend.

    texts: one translated string per input string, in the same order
           as the request's `texts` list. Scalar-vs-list wire shaping
           happens at the codec layer, not here.
    """

    texts: list[str]


@runtime_checkable
class TranslationBackend(Protocol):
    """Structural protocol any translation backend satisfies.

    `TranslationRuntime` (the generic HF seq2seq runtime, Task 2) and
    the bundled m2m100 Model satisfy this without inheritance.
    """

    def translate(
        self, texts: list[str], *, source: str, target: str
    ) -> TranslationResult:
        """Translate `texts` from `source` to `target` (ISO 639-1 codes)."""
        ...

    def supported_languages(self) -> dict[str, list[str]]:
        """{iso_code: [target iso codes]}"""
        ...
