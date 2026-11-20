"""Record whether functions run on an event-loop thread.

Image routes must decode inputs and encode outputs in worker threads; a
multi-second encode on the worker's loop stalls /health and gets the
worker restarted by the supervisor monitor.
"""
from __future__ import annotations

import asyncio


def on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def spy_on(monkeypatch, target, name: str, seen: list[bool]) -> None:
    """Wrap ``target.name`` to append ``on_event_loop()`` per call."""
    real = getattr(target, name)

    def _spy(*args, **kwargs):
        seen.append(on_event_loop())
        return real(*args, **kwargs)

    monkeypatch.setattr(target, name, _spy)


def spy_decode(monkeypatch) -> list[bool]:
    import muse.modalities._common.images as images

    seen: list[bool] = []
    spy_on(monkeypatch, images, "_bytes_to_pil", seen)
    return seen
