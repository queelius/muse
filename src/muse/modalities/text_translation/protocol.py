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


class UnsupportedLanguageError(Exception):
    """Raised when a requested source/target language isn't supported.

    Raised by `TranslationBackend.translate()` implementations (e.g.
    `TranslationRuntime`, Task 2) when a requested ISO 639-1 code (or,
    for single-pair backends like opus-mt, a "source->target" pair
    string) has no mapping. The route layer (Task 3) catches this and
    maps it to a 400 `invalid_language` response.
    """

    code: str
    supported: list[str] | dict

    def __init__(self, code: str, supported: list[str] | dict) -> None:
        self.code = code
        self.supported = supported
        super().__init__(f"unsupported language: {code!r}")


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
