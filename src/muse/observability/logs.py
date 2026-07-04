"""Per-model log hub: byte-bounded ring buffer plus pub/sub for live tails.

Stdlib-only. Does not import torch, fastapi, or any other observability
module. One lock guards all mutation (buffers, byte counts, subscriber
sets) so snapshot/subscribe/unsubscribe/drop are all safe against a
concurrent append from another thread.
"""

from __future__ import annotations

import collections
import queue
import threading


class LogHub:
    """Buffers recent log lines per model_id and fans them out to subscribers.

    - append(model_id, line): buffer the line (evicting oldest lines once the
      running byte count exceeds buffer_bytes) then publish it to every
      subscriber for that model_id.
    - snapshot(model_id): a point-in-time copy of the current buffer.
    - subscribe(model_id)/unsubscribe(model_id, q): live-tail registration.
    - drop(model_id): remove a model's buffer and all its subscribers.
    """

    def __init__(self, *, buffer_bytes: int = 65536) -> None:
        self._buffer_bytes = buffer_bytes
        self._lock = threading.Lock()
        self._buffers: dict[str, collections.deque] = {}
        self._byte_counts: dict[str, int] = {}
        self._subscribers: dict[str, set[queue.Queue]] = {}

    def append(self, model_id: str, line: str) -> None:
        with self._lock:
            buf = self._buffers.setdefault(model_id, collections.deque())
            buf.append(line)
            self._byte_counts[model_id] = self._byte_counts.get(model_id, 0) + len(
                line.encode("utf-8")
            )

            # Evict oldest lines until under the byte bound, but always keep
            # at least one line so a single oversized line is retained
            # rather than evicted to empty.
            while len(buf) > 1 and self._byte_counts[model_id] > self._buffer_bytes:
                oldest = buf.popleft()
                self._byte_counts[model_id] -= len(oldest.encode("utf-8"))

            subscribers = self._subscribers.get(model_id, ())
            for q in subscribers:
                try:
                    q.put_nowait(line)
                except queue.Full:
                    # A slow/full subscriber must not block the reader
                    # thread or the append; drop the line for that
                    # subscriber only.
                    pass

    def snapshot(self, model_id: str) -> list[str]:
        with self._lock:
            buf = self._buffers.get(model_id)
            return list(buf) if buf is not None else []

    def subscribe(self, model_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.setdefault(model_id, set()).add(q)
        return q

    def unsubscribe(self, model_id: str, q: queue.Queue) -> None:
        with self._lock:
            subscribers = self._subscribers.get(model_id)
            if subscribers is not None:
                subscribers.discard(q)

    def drop(self, model_id: str) -> None:
        with self._lock:
            self._buffers.pop(model_id, None)
            self._byte_counts.pop(model_id, None)
            self._subscribers.pop(model_id, None)
