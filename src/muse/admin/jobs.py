"""In-memory async-job tracker for admin operations.

Each enable / pull / probe call returns a Job; the caller polls
GET /v1/admin/jobs/{id} to observe progression. Jobs persist for ten
minutes after `finished_at`; older jobs are reaped on every list call
(lazy reap) to keep memory bounded without a dedicated reaper thread.

The job_id is a uuid4 hex string. Jobs go through:
  pending -> running -> (done | failed)
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_RETENTION_SECONDS = 600.0  # ten minutes
_MAX_JOBS = 100


@dataclass
class Job:
    """One async admin operation.

    `thread` is the daemon worker that runs the operation; tracked so
    the gateway can join it on shutdown. Not serialized into to_dict.
    `finished_at_monotonic` is for lazy expiry; not serialized either.
    """
    job_id: str
    op: str
    model_id: str
    state: str = "pending"
    started_at: str = ""
    finished_at: str | None = None
    result: dict | None = None
    error: str | None = None
    log_lines: list[str] = field(default_factory=list)
    thread: Any = field(default=None, repr=False)
    finished_at_monotonic: float | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "op": self.op,
            "model_id": self.model_id,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "log_lines": list(self.log_lines),
        }


class JobStore:
    """Thread-safe in-memory job map with lazy expiry.

    `retention_seconds` controls how long a finished job stays
    addressable via `get`/`list_recent`. The default is 10 minutes,
    matching the spec.

    `max_jobs` caps the live deque so we never grow unboundedly even
    when nothing finishes (e.g. all pending). The deque drops the
    oldest job_id when full; the dict entry stays addressable until
    expiry, but `list_recent` only returns entries that are also in
    the deque.
    """

    def __init__(self, retention_seconds: float = _RETENTION_SECONDS, max_jobs: int = _MAX_JOBS):
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque(maxlen=max_jobs)
        self._lock = threading.Lock()
        self._retention = retention_seconds

    def create(self, op: str, model_id: str) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex,
            op=op,
            model_id=model_id,
            state="pending",
            started_at=_now_iso(),
        )
        with self._lock:
            self._reap_expired()
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
        logger.info("job %s created (op=%s, model=%s)", job.job_id, op, model_id)
        return job

    def update(self, job_id: str, **fields: Any) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for k, v in fields.items():
                setattr(job, k, v)
            if job.state in ("done", "failed") and job.finished_at_monotonic is None:
                job.finished_at = _now_iso()
                job.finished_at_monotonic = time.monotonic()
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._reap_expired()
            return self._jobs.get(job_id)

    def list_recent(self) -> list[Job]:
        """Return jobs newest-first, capped at the deque's maxlen."""
        with self._lock:
            self._reap_expired()
            return [self._jobs[jid] for jid in reversed(self._order) if jid in self._jobs]

    def shutdown(self, timeout: float = 5.0) -> None:
        """Join live worker threads; called on gateway shutdown."""
        with self._lock:
            threads = [j.thread for j in self._jobs.values() if j.thread is not None]
        for t in threads:
            try:
                t.join(timeout=timeout)
            except Exception as e:  # noqa: BLE001
                logger.warning("error joining job thread: %s", e)

    def _reap_expired(self) -> None:
        """Drop jobs whose finished_at_monotonic is older than retention.

        Caller must hold `self._lock`.
        """
        if self._retention <= 0:
            return
        cutoff = time.monotonic() - self._retention
        expired = [
            jid for jid, j in self._jobs.items()
            if j.finished_at_monotonic is not None and j.finished_at_monotonic < cutoff
        ]
        for jid in expired:
            self._jobs.pop(jid, None)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Module-level default. Tests can build their own JobStore without
# touching this; production code reaches it through get_default_store.
_default_store: JobStore | None = None


def get_default_store() -> JobStore:
    global _default_store
    if _default_store is None:
        _default_store = JobStore()
    return _default_store


def reset_default_store() -> None:
    """Test hook: drop the singleton so next get_default_store rebuilds it."""
    global _default_store
    _default_store = None
