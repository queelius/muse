"""Local ``file://`` GGUF drop-in resolver.

Resolves local ``.gguf`` files (or directories containing a single GGUF model
or split-shard GGUF set) into ``chat/completion`` manifests backed by
``LlamaServerModel``, and scans ``paths.gguf_dir`` for ``muse search --backend file``.

Stdlib + core Muse helpers only: no heavy ML imports.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import uuid
from pathlib import Path
from typing import Any, Iterable

from muse.core import config
from muse.core.chat_formats import lookup_chat_format
from muse.core.gguf import gguf_shard_files
from muse.core.resolvers import (
    ResolvedModel,
    Resolver,
    ResolverError,
    SearchResult,
    register_resolver,
)
from muse.core.runtime_helpers import GGUF_DEFAULT_CONTEXT_LENGTH

_RUNTIME_PATH = "muse.modalities.chat_completion.runtimes.llama_server:LlamaServerModel"
_HASH_CHUNK_BYTES = 1024 * 1024  # 1 MiB
_SHARD_STEM_SUFFIX_RE = re.compile(r"-\d{5}-of-\d{5}$")
_SLUG_INVALID_RE = re.compile(r"[^a-z0-9.-]+")
_MULTI_DASH_RE = re.compile(r"-{2,}")


def _slugify_stem(stem: str) -> str:
    """Convert a GGUF filename stem into a safe lowercase catalog id."""
    cleaned = _SHARD_STEM_SUFFIX_RE.sub("", stem).lower()
    cleaned = _SLUG_INVALID_RE.sub("-", cleaned)
    cleaned = _MULTI_DASH_RE.sub("-", cleaned)
    cleaned = cleaned.strip(".-")
    if not cleaned or cleaned in (".", ".."):
        return "local-gguf"
    return cleaned


def _parse_file_uri(uri: str) -> Path:
    """Parse ``file:///abs/path`` or ``file://~/path`` into a local Path.

    Rejects non-file schemes, empty paths, NUL bytes, relative paths, and
    ``..`` traversal segments.
    """
    if not uri.startswith("file://"):
        raise ResolverError(f"not a file:// URI: {uri!r}")
    raw = uri[len("file://"):]
    if not raw:
        raise ResolverError(f"empty file:// path: {uri!r}")
    if "\x00" in raw:
        raise ResolverError(f"invalid NUL byte in file:// URI: {uri!r}")
    if not (raw.startswith("/") or raw == "~" or raw.startswith("~/")):
        raise ResolverError(
            f"file:// URI must use an absolute path or ~/..., got {uri!r}"
        )
    if ".." in Path(raw).parts:
        raise ResolverError(f"path traversal ('..') is not allowed in {uri!r}")
    return Path(raw).expanduser()


def _lstat_no_symlink(path: Path, *, label: str = "path") -> os.stat_result:
    """Stat ``path`` without following symlinks; raise ResolverError on failure."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise ResolverError(f"local {label} does not exist: {path}") from None
    except OSError as exc:
        raise ResolverError(f"cannot access local {label} {path}: {exc}") from None
    if stat.S_ISLNK(info.st_mode):
        raise ResolverError(f"refusing symlink {label}: {path}")
    return info


def _resolve_shard_paths(directory: Path, filename: str) -> tuple[Path, ...]:
    """Return every shard path for ``filename`` in ``directory``, verified on disk."""
    try:
        shard_names = gguf_shard_files(filename)
    except ValueError as exc:
        raise ResolverError(str(exc)) from None

    missing: list[str] = []
    shard_paths: list[Path] = []
    for name in shard_names:
        candidate = directory / name
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            missing.append(name)
            continue
        except OSError as exc:
            raise ResolverError(f"cannot access GGUF shard {candidate}: {exc}") from None
        if stat.S_ISLNK(info.st_mode):
            raise ResolverError(f"refusing symlink GGUF shard: {candidate}")
        if not stat.S_ISREG(info.st_mode):
            raise ResolverError(f"GGUF shard is not a regular file: {candidate}")
        shard_paths.append(candidate)

    if missing:
        raise ResolverError(
            f"missing GGUF shard(s) {missing} in {directory}"
        )
    return tuple(shard_paths)


def _locate_gguf_shards(target: Path) -> tuple[Path, ...]:
    """Locate and validate the GGUF file (or split-shard set) at ``target``."""
    info = _lstat_no_symlink(target)
    if stat.S_ISDIR(info.st_mode):
        try:
            children = sorted(target.iterdir(), key=lambda p: p.name)
        except OSError as exc:
            raise ResolverError(f"cannot list directory {target}: {exc}") from None
        gguf_children = [p for p in children if p.suffix.lower() == ".gguf"]
        if not gguf_children:
            raise ResolverError(f"no .gguf files found in directory: {target}")

        shard_groups: list[tuple[str, ...]] = []
        seen_groups: set[tuple[str, ...]] = set()
        for child in gguf_children:
            child_info = _lstat_no_symlink(child, label="GGUF file")
            if not stat.S_ISREG(child_info.st_mode):
                raise ResolverError(f"GGUF entry is not a regular file: {child}")
            try:
                group = gguf_shard_files(child.name)
            except ValueError as exc:
                raise ResolverError(str(exc)) from None
            if group not in seen_groups:
                seen_groups.add(group)
                shard_groups.append(group)

        if len(shard_groups) > 1:
            names = ", ".join(g[0] for g in shard_groups)
            raise ResolverError(
                f"directory {target} contains multiple GGUF models ({names}); "
                f"specify a .gguf file directly"
            )
        return _resolve_shard_paths(target, shard_groups[0][0])

    if not stat.S_ISREG(info.st_mode):
        raise ResolverError(f"not a regular file or directory: {target}")
    if target.suffix.lower() != ".gguf":
        raise ResolverError(f"not a .gguf file: {target}")
    return _resolve_shard_paths(target.parent, target.name)


def _hash_file(path: Path) -> tuple[str, int]:
    """Stream-hash ``path`` in 1 MiB chunks; return ``(sha256_hex, size_bytes)``."""
    h = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                h.update(chunk)
                total += len(chunk)
    except OSError as exc:
        raise ResolverError(f"cannot read GGUF file {path}: {exc}") from None
    return h.hexdigest(), total


def _atomic_stage_file(src: Path, dst: Path) -> None:
    """Link (or copy) ``src`` into ``dst`` atomically without mutating ``src``."""
    tmp = dst.parent / f".{dst.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class FileResolver(Resolver):
    """Resolver for local ``file://`` GGUF drop-ins."""

    scheme = "file"

    def resolve(self, uri: str, **options: Any) -> ResolvedModel:
        modality = options.pop("modality", None)
        if modality is not None and modality != "chat/completion":
            raise ResolverError(
                f"file:// resolver only supports 'chat/completion', not {modality!r}"
            )
        if options:
            unsupported = ", ".join(sorted(options))
            raise ResolverError(
                f"file:// resolver does not support option(s): {unsupported}"
            )

        target = _parse_file_uri(uri)
        shard_paths = _locate_gguf_shards(target)
        primary = shard_paths[0]
        resolved_path = primary.resolve()
        digest, size_bytes = _hash_file(primary)
        if size_bytes <= 0:
            raise ResolverError(f"GGUF file is empty: {primary}")
        slug = _slugify_stem(primary.stem)

        capabilities: dict[str, Any] = {
            "gguf_file": primary.name,
            "device": "auto",
            "parallel": 4,
            "cont_batching": True,
            "cache_prompt": True,
            "context_length": GGUF_DEFAULT_CONTEXT_LENGTH,
        }
        hints = lookup_chat_format(primary.stem) or {}
        if "chat_format" in hints:
            capabilities["chat_format"] = hints["chat_format"]
        if "supports_tools" in hints:
            capabilities["supports_tools"] = hints["supports_tools"]

        manifest: dict[str, Any] = {
            "model_id": slug,
            "modality": "chat/completion",
            "hf_repo": "",
            "description": f"Local GGUF model: {primary.name}",
            "license": None,
            "pip_extras": [],
            "system_packages": [],
            "capabilities": capabilities,
        }

        provenance = (
            {
                "kind": "file",
                "path": str(resolved_path),
                "filename": primary.name,
                "sha256": digest,
                "size_bytes": size_bytes,
            },
        )

        def _download(weights_cache: Path) -> Path:
            dest_dir = weights_cache / "local-gguf" / f"{slug}-{digest[:12]}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in shard_paths:
                _atomic_stage_file(src, dest_dir / src.name)
            return dest_dir

        return ResolvedModel(
            manifest=manifest,
            backend_path=_RUNTIME_PATH,
            download=_download,
            artifact_provenance=provenance,
        )

    def search(self, query: str, **filters: Any) -> Iterable[SearchResult]:
        modality = filters.get("modality")
        if modality is not None and modality != "chat/completion":
            return ()
        limit = filters.get("limit")
        gguf_dir = Path(config.get("paths.gguf_dir")).expanduser()
        if not gguf_dir.exists() or not gguf_dir.is_dir():
            return ()

        needle = (query or "").lower()
        seen_sets: set[tuple[Path, tuple[str, ...]]] = set()
        results: list[SearchResult] = []

        for root, dirs, files in os.walk(gguf_dir, followlinks=False):
            dirs.sort()
            root_path = Path(root)
            for name in sorted(files):
                if not name.lower().endswith(".gguf"):
                    continue
                candidate = root_path / name
                try:
                    info = candidate.lstat()
                except OSError:
                    continue
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                    continue
                try:
                    shards = gguf_shard_files(name)
                except ValueError:
                    continue
                key = (root_path, shards)
                if key in seen_sets:
                    continue
                seen_sets.add(key)

                primary = root_path / shards[0]
                rel = str(primary.relative_to(gguf_dir))
                if needle and needle not in rel.lower():
                    continue

                total_bytes = 0
                all_present = True
                for shard_name in shards:
                    sp = root_path / shard_name
                    try:
                        sinfo = sp.lstat()
                    except OSError:
                        all_present = False
                        break
                    if stat.S_ISLNK(sinfo.st_mode) or not stat.S_ISREG(sinfo.st_mode):
                        all_present = False
                        break
                    total_bytes += sinfo.st_size
                if not all_present:
                    continue

                results.append(
                    SearchResult(
                        uri=f"file://{primary}",
                        model_id=_slugify_stem(primary.stem),
                        modality="chat/completion",
                        size_gb=round(total_bytes / 1e9, 3),
                        description=f"Local GGUF: {primary.name}",
                    )
                )
                if isinstance(limit, int) and limit > 0 and len(results) >= limit:
                    return results

        return results


register_resolver(FileResolver())
