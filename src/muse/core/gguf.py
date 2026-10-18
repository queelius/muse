"""Split-GGUF naming (stdlib only; safe for single-file HF plugin imports).

llama.cpp's `gguf-split` names shards `<stem>-00001-of-00003.gguf`; the
loader is given the first shard and opens the rest from the same
directory, so every shard must be downloaded and counted.
"""
from __future__ import annotations

import re

_SHARD_RE = re.compile(r"^(?P<prefix>.*)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$")


def gguf_shard_files(filename: str) -> tuple[str, ...]:
    """Every shard in `filename`'s split set, in load order.

    A non-split file returns `(filename,)`. Raises ValueError for
    impossible shard numbers (index 0, index > total, total 0).
    """
    m = _SHARD_RE.match(filename)
    if m is None:
        return (filename,)
    index, total = int(m["index"]), int(m["total"])
    if total < 1 or not 1 <= index <= total:
        raise ValueError(f"invalid GGUF shard name {filename!r}")
    return tuple(
        f"{m['prefix']}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)
    )
