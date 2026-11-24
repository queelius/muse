"""Voice errors for audio/speech, stdlib only.

Kept apart from ``protocol`` (which imports numpy) so the voice catalog and
light routes can raise them in the gateway process. ``protocol`` re-exports
them, so existing import paths keep working.
"""
from __future__ import annotations


class VoiceError(ValueError):
    """A request-level voice problem; carries an OpenAI-envelope code + status."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


class UnknownVoiceError(VoiceError):
    """The named voice does not exist (404 voice_not_found)."""

    def __init__(self, name: str | None, known: list[str]):
        shown = ", ".join(known[:5]) if known else "none"
        more = ", ..." if len(known) > 5 else ""
        super().__init__(
            "voice_not_found",
            f"unknown voice {name!r}; known voices: {shown}{more}",
            404,
        )
        self.name = name
        self.known = list(known)


class VoiceUnreadableError(RuntimeError):
    """A stored voice exists but cannot be read (half-written or corrupt)."""
