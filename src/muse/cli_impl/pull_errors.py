"""Translate Hugging Face access errors into clean, actionable CLI messages.

`muse pull` downloads weights via huggingface_hub. When a repo is gated or
private and the caller is not authenticated, hub raises GatedRepoError /
RepositoryNotFoundError / HfHubHTTPError. Uncaught, those dump a ~200-line
traceback for what is really a one-line "log in and accept access" condition.

`friendly_pull_error` classifies those errors and returns a short
operator-facing message; anything it does not recognize returns None so
genuine bugs still surface with a full traceback.
"""
from __future__ import annotations


def friendly_pull_error(identifier: str, exc: BaseException) -> str | None:
    """Return a clean CLI message for an HF access error, else None.

    `identifier` is the alias/URI the user passed to `muse pull`. Returns
    None for anything that is not a recognized Hugging Face access failure,
    so the caller re-raises and the real traceback is preserved.
    """
    try:
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            RepositoryNotFoundError,
        )
    except Exception:  # noqa: BLE001  (hub missing/renamed -> can't classify)
        return None

    # Order matters: GatedRepoError <: RepositoryNotFoundError <: HfHubHTTPError.
    # Check most-specific first.
    if isinstance(exc, GatedRepoError):
        repo = getattr(exc, "repo_id", None) or identifier
        return (
            f"error: '{identifier}' is a gated Hugging Face repo ({repo}).\n"
            f"  1. Accept access:  https://huggingface.co/{repo}\n"
            f"  2. Authenticate:   hf auth login   (or export HF_TOKEN=...)\n"
            f"  3. Re-run:         muse pull {identifier}"
        )

    if isinstance(exc, RepositoryNotFoundError):
        repo = getattr(exc, "repo_id", None) or identifier
        return (
            f"error: Hugging Face repo for '{identifier}' ({repo}) was not found "
            f"or is private.\n"
            f"  If it is private, authenticate (hf auth login / export HF_TOKEN=...) "
            f"and retry; otherwise double-check the id."
        )

    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            return (
                f"error: access to the Hugging Face repo for '{identifier}' was "
                f"denied (HTTP {status}).\n"
                f"  Authenticate with `hf auth login` (or export HF_TOKEN=...); if "
                f"the repo is gated, accept access on its model page, then re-run "
                f"`muse pull {identifier}`."
            )

    return None
