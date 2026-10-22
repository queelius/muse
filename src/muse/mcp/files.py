"""Local file I/O for the thin ``muse mcp`` process.

Inputs are read from local paths and media outputs are written to files named
by content hash, so large media never enters the agent's context.  Light by
design: stdlib plus ``muse.core.config`` only (no fastapi, no cli_impl); PIL is
optional.
"""
from __future__ import annotations

import base64
import hashlib
import io
import mimetypes
import os
import stat
import tempfile
import wave
from dataclasses import dataclass

from muse.core import config

PREVIEW_LONG_SIDE = 512

# mimetypes lacks (or varies on) these across platforms.
_EXT_FALLBACK = {
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
    "audio/flac": ".flac", "audio/ogg": ".ogg", "audio/opus": ".opus",
    "audio/mpeg": ".mp3", "image/webp": ".webp", "video/mp4": ".mp4",
    "video/webm": ".webm", "image/svg+xml": ".svg",
    "model/gltf-binary": ".glb",
}


def _real(path: str) -> str:
    return os.path.realpath(os.path.expanduser(path))


def allowed_prefixes() -> list[str]:
    """Realpath'd prefixes path inputs/outputs must lie within.

    An empty ``fetch.mcp_allowed_path_prefixes`` means the process working
    directory plus ``mcp.output_dir``.
    """
    raw = (config.get("fetch.mcp_allowed_path_prefixes") or "").strip()
    items = [p for p in raw.split(os.pathsep) if p.strip()]
    if not items:
        items = [os.getcwd(), _output_dir()]
    return [_real(p) for p in items]


def _output_dir() -> str:
    return str(config.get("mcp.output_dir") or "./muse-output")


def _check_within(path: str, *, label: str,
                  prefixes: list[str] | None = None) -> str:
    real = _real(path)
    for prefix in (prefixes if prefixes is not None else allowed_prefixes()):
        if real == prefix or real.startswith(prefix.rstrip(os.sep) + os.sep):
            return real
    raise ValueError(
        f"{label} {path!r} is not within an allowed prefix; "
        "set MUSE_MCP_ALLOWED_PATH_PREFIXES to include its directory"
    )


def read_input(path: str, *, field: str, max_bytes: int) -> tuple[bytes, str, str]:
    """Read a local file: ``(data, filename, mime)``."""
    real = _check_within(path, label=f"{field} path")
    fd = -1
    try:
        fd = _open_regular_nofollow(real)
        with os.fdopen(fd, "rb") as f:
            fd = -1
            data = f.read(max_bytes + 1)
    except FileNotFoundError as e:
        raise ValueError(f"{field} path not found: {path}") from e
    except OSError as e:
        raise ValueError(f"could not read {field} path {path!r}: {e}") from e
    finally:
        if fd >= 0:
            os.close(fd)
    if len(data) > max_bytes:
        raise ValueError(f"{field} file exceeds the max of {max_bytes} bytes")
    name = os.path.basename(real)
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return data, name, mime


def _open_regular_nofollow(path: str) -> int:
    """Open an absolute path with no symlink followed in any component."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    parts = [p for p in path.split(os.sep) if p]
    if not os.path.isabs(path) or not parts:
        raise OSError("path does not name a regular file")
    cur = os.open(os.sep, os.O_RDONLY | directory | cloexec)
    try:
        for comp in parts[:-1]:
            nxt = os.open(comp, os.O_RDONLY | directory | nofollow | cloexec,
                          dir_fd=cur)
            os.close(cur)
            cur = nxt
        # O_NONBLOCK so a FIFO cannot hang the open before the fstat check.
        fd = os.open(parts[-1],
                     os.O_RDONLY | nofollow | cloexec | os.O_NONBLOCK,
                     dir_fd=cur)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError("path does not name a regular file")
        except BaseException:
            os.close(fd)
            raise
        return fd
    finally:
        os.close(cur)


def as_data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


@dataclass(frozen=True)
class WrittenFile:
    path: str
    sha256: str
    mime: str
    bytes: int


def write_output(data: bytes, mime: str, *, out_path: str | None) -> WrittenFile:
    """Write ``data`` atomically; default name is ``<sha256[:16]><ext>``."""
    digest = hashlib.sha256(data).hexdigest()
    # Snapshot the prefixes once: the default prefix includes the (relative)
    # output dir, which must not be re-resolved after a symlink swap.
    prefixes = allowed_prefixes()
    if out_path:
        target = _check_within(out_path, label="out_path", prefixes=prefixes)
    else:
        base = mime.split(";")[0].strip().lower()
        ext = _EXT_FALLBACK.get(base) or mimetypes.guess_extension(base) or ""
        target = os.path.join(_real(_output_dir()), digest[:16] + ext)
        _check_within(target, label="output path", prefixes=prefixes)
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".muse-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            os.fchmod(f.fileno(), 0o644)
            f.flush()
            os.fsync(f.fileno())
        # Re-check immediately before the replace: an ancestor swapped for a
        # symlink since the first check would otherwise redirect the write.
        # A small race remains between this check and os.replace; that is
        # acceptable because this is a guardrail for an agent acting as the
        # same local user, not a privilege boundary.
        real_parent = os.path.realpath(parent)
        _check_within(real_parent, label="output directory", prefixes=prefixes)
        if os.path.dirname(os.path.realpath(tmp)) != real_parent:
            raise ValueError(
                f"output directory {parent!r} is not within an allowed prefix "
                "(changed during write)")
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return WrittenFile(path=target, sha256=digest, mime=mime, bytes=len(data))


def media_facts(data: bytes, mime: str) -> dict:
    """Cheap facts: WAV duration (stdlib), image size (PIL if importable)."""
    base = mime.split(";")[0].strip().lower()
    if base in ("audio/wav", "audio/x-wav", "audio/wave"):
        try:
            with wave.open(io.BytesIO(data), "rb") as w:
                rate = w.getframerate()
                if rate <= 0:
                    return {}
                return {"duration_seconds": round(w.getnframes() / rate, 3)}
        except (wave.Error, EOFError):
            return {}
    if base.startswith("image/"):
        img = _open_image(data)
        if img is not None:
            return {"width": img.size[0], "height": img.size[1]}
    return {}


def _open_image(data: bytes):
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception:  # noqa: BLE001 - not an image / corrupt
        return None


def preview_png(data: bytes) -> bytes | None:
    """PNG with long side <= 512, or None without PIL / not an image."""
    img = _open_image(data)
    if img is None:
        return None
    img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    w, h = img.size
    scale = PREVIEW_LONG_SIDE / max(w, h)
    if scale < 1:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()
