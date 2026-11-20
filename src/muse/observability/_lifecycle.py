"""Shared argument validation and bounded thread joins for observability.

Stdlib only: the recorder, sampler, log hub and ticket store all import
this from hot or import-light paths.
"""
from __future__ import annotations

import logging
import math
import threading


def finite_number(value: object, name: str, *, allow_zero: bool = False) -> float:
    """Return `value` as a float, or raise ValueError.

    Accepts a finite int or float (never a bool) that is positive, or
    non-negative when `allow_zero` is true.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        if allow_zero:
            raise ValueError(f"{name} must be a finite non-negative number")
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def positive_int(value: object, name: str) -> int:
    """Return `value` if it is a positive int (never a bool), else raise."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def join_bounded(
    thread: threading.Thread | None,
    timeout: float,
    *,
    label: str,
    logger: logging.Logger,
) -> bool:
    """Join `thread` within `timeout`; return whether it has stopped.

    False when called from `thread` itself, when the join raises while the
    thread is still alive, or when the thread outlives the deadline. The
    caller keeps its handle on False, so a later cleanup can retry and
    diagnostics never mistake a live thread for a completed shutdown.
    """
    if thread is None:
        return True
    if thread is threading.current_thread():
        logger.warning("%s cannot join its own thread", label)
        return False
    try:
        thread.join(timeout=timeout)
    except RuntimeError:
        if thread.is_alive() is True:
            logger.warning("could not join %s thread", label, exc_info=True)
            return False
    if thread.is_alive() is True:
        logger.warning("%s thread did not stop within %.1fs", label, timeout)
        return False
    return True
