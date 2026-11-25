"""Supervised `llama-server` binary management and process lifecycle.

Stdlib + ``httpx`` only (no heavy ML imports). All child process lifecycle
operations route exclusively through :mod:`muse.core.owned_process` so
process-group ownership and leader pinning invariants are preserved.
"""
from __future__ import annotations

import atexit
import logging
import os
import platform
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx

from muse.core import config
from muse.core.owned_process import (
    BoundedTail,
    cleanup_deadlines,
    concrete_safe_id,
    drain_stream_into,
    kill_signal,
    new_process_group_kwargs,
    observe_leader,
    reap_exited_leader,
    signal_isolated_group,
    signal_live_fallback_leader,
    supports_pinned_leader,
    validated_isolated_group,
)


def _pick_ephemeral_port(host: str = "127.0.0.1") -> int:
    """Bind port 0 to obtain an OS-assigned ephemeral port outside worker range."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])

logger = logging.getLogger(__name__)

DEFAULT_RELEASE_TAG = "b12728"
DEFAULT_VARIANT = "auto"
DEFAULT_STARTUP_TIMEOUT_SECONDS = 120.0

_BINARY_NAME = "llama-server.exe" if os.name == "nt" else "llama-server"
_STDERR_TAIL_BYTES = 64 * 1024
_STDERR_READ_BYTES = 8 * 1024
_PROBE_TIMEOUT_SECONDS = 5.0
_DOWNLOAD_TIMEOUT_SECONDS = 300.0
_GITHUB_RELEASE_BASE = "https://github.com/ggml-org/llama.cpp/releases/download"

_CUDA_DEVICE_RE = re.compile(r"\bCUDA\d*\b", re.IGNORECASE)
_METAL_DEVICE_RE = re.compile(r"\bMetal\b", re.IGNORECASE)
_VULKAN_DEVICE_RE = re.compile(r"\bVulkan\d*\b", re.IGNORECASE)


def _config_get(key: str, default: Any, env: str | None = None) -> Any:
    """Read a config setting with safe fallback if the key is not registered yet."""
    try:
        return config.get(key)
    except KeyError:
        if env:
            raw = os.environ.get(env)
            if raw is not None and raw.strip() != "":
                return raw.strip()
        return default


def _managed_bin_dir() -> Path:
    raw = _config_get("paths.catalog_dir", "~/.muse", "MUSE_CATALOG_DIR")
    return Path(str(raw)).expanduser() / "bin"


def _managed_binary_path() -> Path:
    return _managed_bin_dir() / _BINARY_NAME


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def find_llama_server_binary(binary_path: str | Path | None = None) -> Path | None:
    """Locate an existing ``llama-server`` binary.

    Precedence:
      1. Explicit ``binary_path`` argument or ``llama_server.binary_path`` config
         (raises ``FileNotFoundError`` if set but missing/non-executable).
      2. Managed binary at ``<catalog_dir>/bin/llama-server`` (``.exe`` on Windows).
      3. ``shutil.which("llama-server")``.
    """
    explicit = binary_path
    if explicit is None:
        explicit = _config_get(
            "llama_server.binary_path", None, "MUSE_LLAMA_SERVER_BINARY",
        )
    if explicit is not None and str(explicit).strip() != "":
        candidate = Path(str(explicit)).expanduser()
        if not _is_executable_file(candidate):
            raise FileNotFoundError(
                f"configured llama-server binary is missing or not executable: {candidate}"
            )
        return candidate

    managed = _managed_binary_path()
    if _is_executable_file(managed):
        return managed

    on_path = shutil.which("llama-server")
    if on_path:
        return Path(on_path)

    return None


def _detect_auto_variant(*, system: str | None = None) -> str:
    sys_name = (system or platform.system()).lower()
    if sys_name == "darwin":
        return "metal"
    if sys_name in ("linux", "windows"):
        try:
            from muse.core import memory_probe
            if memory_probe.gpu_free_gb() is not None:
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        if shutil.which("nvidia-smi") is not None:
            return "cuda"
        return "cpu"
    return "cpu"


def _release_asset_names(
    release_tag: str,
    variant: str,
    *,
    system: str | None = None,
    machine: str | None = None,
) -> list[str]:
    """Return upstream ``ggml-org/llama.cpp`` release archive filenames.

    CUDA builds require both the main binary archive and the companion
    ``cudart`` shared-library archive so ``llama-server`` runs without a
    system-wide CUDA toolkit installed.
    """
    sys_name = (system or platform.system()).lower()
    arch = (machine or platform.machine()).lower()
    is_arm = arch in ("arm64", "aarch64")
    is_x64 = arch in ("x86_64", "amd64", "x64")

    if sys_name == "linux":
        if is_x64:
            if variant == "cuda":
                return [
                    f"llama-{release_tag}-bin-ubuntu-cuda-12.8-x64.tar.gz",
                    f"cudart-llama-{release_tag}-bin-ubuntu-cuda-12.8-x64.tar.gz",
                ]
            if variant == "vulkan":
                return [f"llama-{release_tag}-bin-ubuntu-vulkan-x64.tar.gz"]
            return [f"llama-{release_tag}-bin-ubuntu-x64.tar.gz"]
        if is_arm:
            if variant == "vulkan":
                return [f"llama-{release_tag}-bin-ubuntu-vulkan-arm64.tar.gz"]
            if variant == "cpu":
                return [f"llama-{release_tag}-bin-ubuntu-arm64.tar.gz"]
        raise RuntimeError(
            f"no prebuilt llama-server binary for Linux {arch!r} ({variant}); "
            "set llama_server.binary_path to a local build"
        )

    if sys_name == "darwin":
        if is_arm:
            return [f"llama-{release_tag}-bin-macos-arm64.tar.gz"]
        if is_x64:
            return [f"llama-{release_tag}-bin-macos-x64.tar.gz"]
        raise RuntimeError(
            f"no prebuilt llama-server binary for macOS architecture {arch!r}"
        )

    if sys_name == "windows":
        if is_x64:
            if variant == "cuda":
                return [
                    f"llama-{release_tag}-bin-win-cuda-12.4-x64.zip",
                    "cudart-llama-bin-win-cuda-12.4-x64.zip",
                ]
            if variant == "vulkan":
                return [f"llama-{release_tag}-bin-win-vulkan-x64.zip"]
            return [f"llama-{release_tag}-bin-win-cpu-x64.zip"]
        if is_arm:
            if variant == "vulkan":
                return [f"llama-{release_tag}-bin-win-vulkan-arm64.zip"]
            if variant == "cpu":
                return [f"llama-{release_tag}-bin-win-cpu-arm64.zip"]
        raise RuntimeError(
            f"no prebuilt llama-server binary for Windows {arch!r} ({variant}); "
            "set llama_server.binary_path to a local build"
        )

    raise RuntimeError(
        f"unsupported platform for llama-server auto-download: {sys_name}/{arch}"
    )


def _release_asset_name(
    release_tag: str,
    variant: str,
    *,
    system: str | None = None,
    machine: str | None = None,
) -> str:
    """Return the primary upstream ``ggml-org/llama.cpp`` release archive filename."""
    return _release_asset_names(
        release_tag, variant, system=system, machine=machine
    )[0]


def _is_needed_archive_member(name: str) -> bool:
    """Keep ``llama-server`` and bundled shared libraries only."""
    basename = Path(name).name
    if not basename:
        return False
    if basename in ("llama-server", "llama-server.exe"):
        return True
    if basename.endswith((".dylib", ".dll", ".so")):
        return True
    if ".so." in basename:
        return True
    return False


def _extract_archive(
    archive_path: Path,
    dest_dir: Path,
    *,
    require_binary: bool = True,
) -> Path | None:
    """Extract ``llama-server`` and bundled shared libraries into ``dest_dir``."""
    extracted_binary: Path | None = None

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if not _is_needed_archive_member(info.filename):
                    continue
                basename = Path(info.filename).name
                target = dest_dir / basename
                if target.exists() or target.is_symlink():
                    target.unlink()
                with zf.open(info, "r") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                if basename == _BINARY_NAME:
                    extracted_binary = target
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if not _is_needed_archive_member(member.name):
                    continue
                basename = Path(member.name).name
                target = dest_dir / basename
                if member.issym():
                    link_target = Path(member.linkname).name
                    if not link_target or link_target == basename:
                        continue
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    os.symlink(link_target, target)
                    continue
                if not member.isfile():
                    continue
                src = tf.extractfile(member)
                if src is None:
                    continue
                if target.exists() or target.is_symlink():
                    target.unlink()
                with src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                if basename == _BINARY_NAME:
                    extracted_binary = target
    else:
        raise RuntimeError(f"unsupported llama-server archive format: {archive_path.name}")

    if require_binary and (extracted_binary is None or not extracted_binary.is_file()):
        raise RuntimeError(
            f"downloaded archive {archive_path.name} did not contain {_BINARY_NAME}"
        )

    if extracted_binary is not None and os.name == "posix":
        extracted_binary.chmod(
            stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
        )
    return extracted_binary


def _atomic_symlink(target: Path, link_path: Path) -> None:
    """Atomically create or replace ``link_path`` pointing to ``target``."""
    tmp_link = link_path.parent / f".{link_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        os.symlink(target, tmp_link)
        os.replace(tmp_link, link_path)
    except OSError:
        if tmp_link.exists() or tmp_link.is_symlink():
            try:
                tmp_link.unlink()
            except OSError:
                pass
        # Fallback for environments without symlink privileges (e.g. Windows).
        shutil.copy2(target, link_path)


def ensure_llama_server_binary(
    *,
    variant: str | None = None,
    release_tag: str | None = None,
    binary_path: str | Path | None = None,
) -> Path:
    """Return an executable ``llama-server`` binary, downloading it if needed."""
    existing = find_llama_server_binary(binary_path=binary_path)
    if existing is not None:
        return existing

    tag = str(
        release_tag
        or _config_get("llama_server.release_tag", DEFAULT_RELEASE_TAG, "MUSE_LLAMA_SERVER_RELEASE")
    ).strip()
    raw_variant = str(
        variant
        or _config_get("llama_server.variant", DEFAULT_VARIANT, "MUSE_LLAMA_SERVER_VARIANT")
    ).strip().lower()
    resolved_variant = _detect_auto_variant() if raw_variant == "auto" else raw_variant

    asset_names = _release_asset_names(tag, resolved_variant)

    bin_dir = _managed_bin_dir()
    bin_dir.mkdir(parents=True, exist_ok=True)
    install_dir = bin_dir / f"llama-server-{tag}-{resolved_variant}"

    tmp_root = Path(tempfile.mkdtemp(prefix=".llama-server-dl-", dir=bin_dir))
    try:
        stage_dir = tmp_root / "extracted"
        stage_dir.mkdir()
        for idx, asset_name in enumerate(asset_names):
            url = f"{_GITHUB_RELEASE_BASE}/{tag}/{asset_name}"
            logger.info("downloading llama-server %s (%s) from %s", tag, resolved_variant, url)
            archive_path = tmp_root / asset_name
            with httpx.stream(
                "GET", url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS,
            ) as resp:
                resp.raise_for_status()
                with open(archive_path, "wb") as fh:
                    for chunk in resp.iter_bytes():
                        fh.write(chunk)
            _extract_archive(archive_path, stage_dir, require_binary=(idx == 0))

        if install_dir.exists():
            shutil.rmtree(install_dir)
        os.replace(stage_dir, install_dir)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    installed_binary = install_dir / _BINARY_NAME
    managed_link = _managed_binary_path()
    _atomic_symlink(installed_binary, managed_link)
    return managed_link


def probe_llama_server(binary: Path | None = None) -> tuple[bool, str | None]:
    """Probe ``llama-server --list-devices`` for GPU offload support.

    Returns ``(gpu_offload, gpu_backend)`` where ``gpu_backend`` is
    ``"cuda"``, ``"metal"``, or ``None``.
    """
    exe = binary if binary is not None else find_llama_server_binary()
    if exe is None or not _is_executable_file(exe):
        return (False, None)

    env = _child_env_for_binary(exe)
    try:
        result = subprocess.run(
            [str(exe), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("llama-server --list-devices failed for %s: %s", exe, exc)
        return (False, None)

    if result.returncode != 0:
        logger.warning(
            "llama-server --list-devices exited %d for %s", result.returncode, exe,
        )
        return (False, None)

    output = f"{result.stdout}\n{result.stderr}"
    if _CUDA_DEVICE_RE.search(output):
        return (True, "cuda")
    if _METAL_DEVICE_RE.search(output):
        return (True, "metal")
    if _VULKAN_DEVICE_RE.search(output):
        return (True, "vulkan")
    return (False, None)


def _child_env_for_binary(binary: Path) -> dict[str, str]:
    """Prepare child environment so bundled shared libraries beside ``binary`` load."""
    env = dict(os.environ)
    lib_dir = str(binary.resolve().parent)
    if sys.platform.startswith("linux"):
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{prev}" if prev else lib_dir
    elif sys.platform == "darwin":
        prev = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = f"{lib_dir}:{prev}" if prev else lib_dir
    return env


class LlamaServerProcess:
    """Supervised ``llama-server`` child process owning an isolated process group."""

    def __init__(
        self,
        *,
        binary: Path,
        model_path: Path,
        context_length: int = 8192,
        n_gpu_layers: int = -1,
        parallel: int = 4,
        cont_batching: bool = True,
        no_jinja: bool = False,
        chat_format: str | None = None,
        n_threads: int | None = None,
        n_threads_batch: int | None = None,
        n_batch: int | None = None,
        flash_attn: bool | None = None,
        use_mlock: bool | None = None,
        type_k: int | str | None = None,
        type_v: int | str | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        startup_timeout: float | None = None,
    ) -> None:
        self.binary = Path(binary)
        self.model_path = Path(model_path)
        self.context_length = int(context_length)
        self.n_gpu_layers = int(n_gpu_layers)
        self.parallel = max(1, int(parallel))
        self.cont_batching = bool(cont_batching)
        self.no_jinja = bool(no_jinja)
        self.chat_format = chat_format
        self.n_threads = n_threads
        self.n_threads_batch = n_threads_batch
        self.n_batch = n_batch
        self.flash_attn = flash_attn
        self.use_mlock = use_mlock
        self.type_k = type_k
        self.type_v = type_v
        self.host = host
        self.port = int(port) if port is not None else _pick_ephemeral_port(host)
        self.base_url = f"http://{self.host}:{self.port}"
        if startup_timeout is None:
            raw_timeout = _config_get(
                "llama_server.startup_timeout_seconds",
                DEFAULT_STARTUP_TIMEOUT_SECONDS,
                "MUSE_LLAMA_SERVER_STARTUP_TIMEOUT",
            )
            self.startup_timeout = float(raw_timeout)
        else:
            self.startup_timeout = float(startup_timeout)

        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._pid: int | None = None
        self._pgid: int | None = None
        self._pinned: bool = False
        self._stopped: bool = False
        self._stderr_tail = BoundedTail(_STDERR_TAIL_BYTES)
        self._stderr_thread: threading.Thread | None = None
        self._atexit_registered: bool = False

    def build_command(self) -> list[str]:
        cmd = [
            str(self.binary),
            "-m", str(self.model_path),
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(self.context_length),
            "-ngl", str(self.n_gpu_layers),
            "--parallel", str(self.parallel),
        ]
        if self.cont_batching:
            cmd.append("--cont-batching")
        if self.no_jinja:
            cmd.append("--no-jinja")
        elif self.chat_format:
            cmd.extend(["--chat-template", str(self.chat_format)])
        if self.flash_attn is True:
            cmd.append("--flash-attn")
        if self.n_threads is not None:
            cmd.extend(["-t", str(self.n_threads)])
        if self.n_threads_batch is not None:
            cmd.extend(["-tb", str(self.n_threads_batch)])
        if self.n_batch is not None:
            cmd.extend(["-b", str(self.n_batch)])
        if self.use_mlock is True:
            cmd.append("--mlock")
        if self.type_k is not None:
            cmd.extend(["-ctk", str(self.type_k)])
        if self.type_v is not None:
            cmd.extend(["-ctv", str(self.type_v)])
        return cmd

    def stderr_tail(self) -> str:
        data, discarded, _err = self._stderr_tail.snapshot()
        text = data.decode("utf-8", errors="replace").strip()
        if discarded:
            return f"[... {discarded} stderr bytes truncated ...]\n{text}"
        return text

    def _drain_stderr(self, stream: Any) -> None:
        try:
            drain_stream_into(stream, self._stderr_tail, read_bytes=_STDERR_READ_BYTES)
        except (OSError, ValueError):
            pass
        finally:
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    def is_alive(self) -> bool:
        with self._lock:
            if self._proc is None or self._stopped:
                return False
            alive = observe_leader(
                self._proc,
                self._pid,
                pinned=self._pinned,
                os_name=os.name,
                label="llama-server",
                process_group=self._pgid,
            )
            return bool(alive)

    def start(self) -> None:
        cmd = self.build_command()
        popen_kwargs: dict[str, Any] = new_process_group_kwargs(os.name)
        env = _child_env_for_binary(self.binary)

        logger.info("starting llama-server on %s: %s", self.base_url, " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
            **popen_kwargs,
        )
        pid = concrete_safe_id(getattr(proc, "pid", None))
        pgid = validated_isolated_group(
            pid,
            os_name=os.name,
            label="llama-server",
            new_session=popen_kwargs.get("start_new_session") is True,
        )
        pinned = supports_pinned_leader(proc, pid, pgid, os_name=os.name)

        with self._lock:
            self._proc = proc
            self._pid = pid
            self._pgid = pgid
            self._pinned = pinned
            self._stopped = False

        if proc.stderr is not None:
            thread = threading.Thread(
                target=self._drain_stderr,
                args=(proc.stderr,),
                name=f"muse-llama-server-stderr-{self.port}",
                daemon=True,
            )
            thread.start()
            self._stderr_thread = thread

        atexit.register(self.stop)
        self._atexit_registered = True

        try:
            self._wait_until_ready(self.startup_timeout)
        except BaseException:
            self.stop()
            raise

    def _wait_until_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + max(0.1, timeout)
        health_url = f"{self.base_url}/health"
        with httpx.Client(timeout=2.0) as client:
            while True:
                if not self.is_alive():
                    tail = self.stderr_tail()
                    raise RuntimeError(
                        f"llama-server exited before becoming healthy"
                        + (f": {tail}" if tail else "")
                    )
                try:
                    resp = client.get(health_url)
                    if resp.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    tail = self.stderr_tail()
                    raise TimeoutError(
                        f"llama-server on {self.base_url} did not become healthy "
                        f"within {timeout:.1f}s"
                        + (f": {tail}" if tail else "")
                    )
                time.sleep(0.1)

    def _signal_locked(
        self,
        proc: Any,
        pid: int | None,
        pgid: int | None,
        pinned: bool,
        sig: signal.Signals,
    ) -> bool:
        if pgid is not None:
            return signal_isolated_group(
                pid, pgid, sig, label="llama-server", pinned=pinned,
            )
        return signal_live_fallback_leader(
            proc, pid, pgid, sig, label="llama-server",
        )

    def stop(self, timeout: float = 5.0) -> None:
        if self._atexit_registered:
            try:
                atexit.unregister(self.stop)
            except Exception:  # noqa: BLE001
                pass
            self._atexit_registered = False

        stderr_stream = None
        stderr_thread = None

        with self._lock:
            proc = self._proc
            if proc is None:
                self._stopped = True
                return
            pid = self._pid
            pgid = self._pgid
            pinned = self._pinned
            stderr_stream = getattr(proc, "stderr", None)
            stderr_thread = self._stderr_thread

            term_deadline, kill_deadline = cleanup_deadlines(timeout)
            alive = observe_leader(
                proc,
                pid,
                pinned=pinned,
                os_name=os.name,
                label="llama-server",
                process_group=pgid,
            )
            if alive is True:
                self._signal_locked(proc, pid, pgid, pinned, signal.SIGTERM)
                while time.monotonic() < term_deadline:
                    alive = observe_leader(
                        proc,
                        pid,
                        pinned=pinned,
                        os_name=os.name,
                        label="llama-server",
                        process_group=pgid,
                    )
                    if alive is False:
                        break
                    time.sleep(0.05)

            if alive is not False:
                self._signal_locked(proc, pid, pgid, pinned, kill_signal())
                while time.monotonic() < kill_deadline:
                    alive = observe_leader(
                        proc,
                        pid,
                        pinned=pinned,
                        os_name=os.name,
                        label="llama-server",
                        process_group=pgid,
                    )
                    if alive is False:
                        break
                    time.sleep(0.05)

            reap_exited_leader(proc, pid, label="llama-server")
            self._proc = None
            self._pid = None
            self._pgid = None
            self._pinned = False
            self._stopped = True
            self._stderr_thread = None

        if stderr_stream is not None:
            try:
                stderr_stream.close()
            except (OSError, ValueError):
                pass
        if stderr_thread is not None and stderr_thread.is_alive():
            stderr_thread.join(timeout=0.5)
