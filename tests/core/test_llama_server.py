"""Tests for muse.core.llama_server, placement integration, and worker teardown."""
from __future__ import annotations

import io
import os
import stat
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from muse.core import llama_server, placement
from muse.core.ports import find_free_port
from muse.core.registry import ModalityRegistry


def _make_executable(path: Path, content: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

def test_find_binary_explicit_arg_wins(tmp_path):
    exe = _make_executable(tmp_path / "custom-llama-server")
    assert llama_server.find_llama_server_binary(exe) == exe


def test_find_binary_explicit_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing or not executable"):
        llama_server.find_llama_server_binary(tmp_path / "does-not-exist")


def test_find_binary_explicit_non_executable_raises(tmp_path):
    f = tmp_path / "not-exec"
    f.write_text("data", encoding="utf-8")
    f.chmod(0o644)
    with pytest.raises(FileNotFoundError, match="missing or not executable"):
        llama_server.find_llama_server_binary(f)


def test_find_binary_managed_beats_path(tmp_path, monkeypatch):
    managed = _make_executable(tmp_path / "bin" / llama_server._BINARY_NAME)
    path_bin = _make_executable(tmp_path / "other" / llama_server._BINARY_NAME)
    monkeypatch.setattr(llama_server, "_config_get", lambda key, default, env=None: default)
    monkeypatch.setattr(llama_server, "_managed_bin_dir", lambda: tmp_path / "bin")
    monkeypatch.setattr(llama_server.shutil, "which", lambda _name: str(path_bin))
    assert llama_server.find_llama_server_binary() == managed


def test_find_binary_falls_back_to_which(tmp_path, monkeypatch):
    path_bin = _make_executable(tmp_path / "other" / llama_server._BINARY_NAME)
    monkeypatch.setattr(llama_server, "_config_get", lambda key, default, env=None: default)
    monkeypatch.setattr(llama_server, "_managed_bin_dir", lambda: tmp_path / "bin")
    monkeypatch.setattr(llama_server.shutil, "which", lambda _name: str(path_bin))
    assert llama_server.find_llama_server_binary() == path_bin


def test_find_binary_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(llama_server, "_config_get", lambda key, default, env=None: default)
    monkeypatch.setattr(llama_server, "_managed_bin_dir", lambda: tmp_path / "bin")
    monkeypatch.setattr(llama_server.shutil, "which", lambda _name: None)
    assert llama_server.find_llama_server_binary() is None


# ---------------------------------------------------------------------------
# Auto-download & extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("system", "machine", "variant", "expected"),
    [
        ("Linux", "x86_64", "cpu", "llama-b5490-bin-ubuntu-x64.zip"),
        ("Linux", "x86_64", "cuda", "llama-b5490-bin-ubuntu-cuda-cu12.4-x64.zip"),
        ("Darwin", "arm64", "metal", "llama-b5490-bin-macos-arm64.zip"),
        ("Darwin", "x86_64", "metal", "llama-b5490-bin-macos-x64.zip"),
        ("Windows", "AMD64", "cpu", "llama-b5490-bin-win-cpu-x64.zip"),
        ("Windows", "AMD64", "cuda", "llama-b5490-bin-win-cuda-cu12.4-x64.zip"),
    ],
)
def test_release_asset_name(system, machine, variant, expected):
    assert (
        llama_server._release_asset_name("b5490", variant, system=system, machine=machine)
        == expected
    )


def test_ensure_binary_downloads_extracts_and_symlinks(tmp_path, monkeypatch):
    monkeypatch.setattr(llama_server, "_config_get", lambda key, default, env=None: default)
    monkeypatch.setattr(llama_server, "_managed_bin_dir", lambda: tmp_path / "bin")
    monkeypatch.setattr(llama_server.shutil, "which", lambda _name: None)
    monkeypatch.setattr(llama_server.platform, "system", lambda: "Linux")
    monkeypatch.setattr(llama_server.platform, "machine", lambda: "x86_64")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("build/bin/llama-server", b"#!/bin/sh\necho ok\n")
        zf.writestr("build/bin/libllama.so.0", b"ELF-shared-lib")
        zf.writestr("build/bin/README.md", b"ignored")
    archive_bytes = buf.getvalue()

    class _FakeStreamResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield archive_bytes[:32]
            yield archive_bytes[32:]

    seen_urls: list[str] = []

    def _fake_stream(method, url, **kwargs):
        seen_urls.append(url)
        return _FakeStreamResp()

    monkeypatch.setattr(llama_server.httpx, "stream", _fake_stream)

    result = llama_server.ensure_llama_server_binary(variant="cpu", release_tag="b5490")
    assert result == tmp_path / "bin" / llama_server._BINARY_NAME
    assert result.is_symlink() or result.is_file()
    assert os.access(result, os.X_OK)
    install_dir = tmp_path / "bin" / "llama-server-b5490-cpu"
    assert (install_dir / "llama-server").is_file()
    assert (install_dir / "libllama.so.0").is_file()
    assert not (install_dir / "README.md").exists()
    assert stat.S_IMODE((install_dir / "llama-server").stat().st_mode) & 0o111
    assert seen_urls == [
        "https://github.com/ggml-org/llama.cpp/releases/download/b5490/llama-b5490-bin-ubuntu-x64.zip"
    ]


# ---------------------------------------------------------------------------
# GPU probing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("Available devices:\n  CUDA0: NVIDIA RTX 4090 (24564 MiB)\n", (True, "cuda")),
        ("Available devices:\n  Metal: Apple M3 Max\n", (True, "metal")),
        ("Available devices:\n", (False, None)),
    ],
)
def test_probe_llama_server_parses_devices(tmp_path, monkeypatch, stdout, expected):
    exe = _make_executable(tmp_path / "llama-server")
    completed = subprocess.CompletedProcess([str(exe), "--list-devices"], 0, stdout, "")
    monkeypatch.setattr(llama_server.subprocess, "run", lambda *a, **kw: completed)
    assert llama_server.probe_llama_server(exe) == expected


def test_probe_llama_server_handles_timeout(tmp_path, monkeypatch):
    exe = _make_executable(tmp_path / "llama-server")

    def _timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd=a[0], timeout=5.0)

    monkeypatch.setattr(llama_server.subprocess, "run", _timeout)
    assert llama_server.probe_llama_server(exe) == (False, None)


# ---------------------------------------------------------------------------
# LlamaServerProcess command building & real subprocess lifecycle
# ---------------------------------------------------------------------------

def test_build_command_flags(tmp_path):
    proc = llama_server.LlamaServerProcess(
        binary=tmp_path / "llama-server",
        model_path=tmp_path / "model.gguf",
        context_length=4096,
        n_gpu_layers=32,
        parallel=8,
        cont_batching=True,
        no_jinja=False,
        chat_format="chatml",
        n_threads=6,
        n_threads_batch=12,
        n_batch=512,
        flash_attn=True,
        use_mlock=True,
        type_k="q8_0",
        type_v="q8_0",
        port=9123,
    )
    cmd = proc.build_command()
    assert cmd[:14] == [
        str(tmp_path / "llama-server"),
        "-m", str(tmp_path / "model.gguf"),
        "--host", "127.0.0.1",
        "--port", "9123",
        "--ctx-size", "4096",
        "-ngl", "32",
        "--parallel", "8",
        "--cont-batching",
    ]
    assert "--chat-template" in cmd
    assert cmd[cmd.index("--chat-template") + 1] == "chatml"
    assert "--no-jinja" not in cmd
    assert "--flash-attn" in cmd
    assert "--mlock" in cmd
    assert cmd[cmd.index("-t") + 1] == "6"
    assert cmd[cmd.index("-tb") + 1] == "12"
    assert cmd[cmd.index("-b") + 1] == "512"
    assert cmd[cmd.index("-ctk") + 1] == "q8_0"
    assert cmd[cmd.index("-ctv") + 1] == "q8_0"


def test_build_command_no_jinja_omits_chat_template(tmp_path):
    proc = llama_server.LlamaServerProcess(
        binary=tmp_path / "llama-server",
        model_path=tmp_path / "model.gguf",
        no_jinja=True,
        chat_format="chatml",
        cont_batching=False,
        port=9124,
    )
    cmd = proc.build_command()
    assert "--no-jinja" in cmd
    assert "--chat-template" not in cmd
    assert "--cont-batching" not in cmd


def test_llama_server_process_start_health_and_clean_stop(tmp_path):
    """Spawn a real fake llama-server script, verify 503->200 readiness and clean stop."""
    fake_bin = _make_executable(
        tmp_path / "fake-llama-server",
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import http.server
            import signal
            import sys

            port = int(sys.argv[sys.argv.index("--port") + 1])
            hits = [0]

            class Handler(http.server.BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == "/health":
                        hits[0] += 1
                        code = 503 if hits[0] == 1 else 200
                        self.send_response(code)
                        self.end_headers()
                        self.wfile.write(b'{{"status":"ok"}}')
                    else:
                        self.send_response(404)
                        self.end_headers()
                def log_message(self, *args):
                    pass

            server = http.server.HTTPServer(("127.0.0.1", port), Handler)
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
            server.serve_forever()
            """
        ),
    )
    model_file = tmp_path / "tiny.gguf"
    model_file.write_bytes(b"GGUF")

    proc = llama_server.LlamaServerProcess(
        binary=fake_bin,
        model_path=model_file,
        port=find_free_port(),
        startup_timeout=10.0,
    )
    proc.start()
    try:
        assert proc.is_alive() is True
    finally:
        proc.stop(timeout=5.0)
    assert proc.is_alive() is False
    # Idempotent second stop
    proc.stop(timeout=1.0)


def test_llama_server_process_fails_fast_on_early_exit(tmp_path):
    fake_bin = _make_executable(
        tmp_path / "failing-llama-server",
        f"#!{sys.executable}\nimport sys\nsys.stderr.write('bad gguf magic\\n')\nsys.exit(1)\n",
    )
    model_file = tmp_path / "bad.gguf"
    model_file.write_bytes(b"BAD")

    proc = llama_server.LlamaServerProcess(
        binary=fake_bin,
        model_path=model_file,
        port=find_free_port(),
        startup_timeout=5.0,
    )
    with pytest.raises(RuntimeError, match="bad gguf magic"):
        proc.start()
    assert proc.is_alive() is False


# ---------------------------------------------------------------------------
# Placement integration for LlamaServerModel
# ---------------------------------------------------------------------------

def test_placement_recognizes_llama_server_without_python_path(monkeypatch):
    manifest = {
        "model_id": "test-gguf",
        "modality": "chat/completion",
        "backend_path": placement.LLAMA_SERVER_BACKEND,
        "capabilities": {"gguf_file": "m.gguf"},
    }
    assert placement.is_native_gguf(manifest) is True

    monkeypatch.setattr(llama_server, "probe_llama_server", lambda _bin=None: (True, "cuda"))
    res = placement.resolve_model_placement("test-gguf", manifest, entry={})
    assert (res.device, res.pool, res.n_gpu_layers) == ("cuda", "cuda", -1)

    monkeypatch.setattr(llama_server, "probe_llama_server", lambda _bin=None: (True, "metal"))
    res_metal = placement.resolve_model_placement("test-gguf", manifest, entry={})
    assert (res_metal.device, res_metal.pool, res_metal.n_gpu_layers) == ("mps", "cpu", -1)

    monkeypatch.setattr(llama_server, "probe_llama_server", lambda _bin=None: (False, None))
    res_cpu = placement.resolve_model_placement("test-gguf", manifest, entry={})
    assert (res_cpu.device, res_cpu.pool, res_cpu.n_gpu_layers) == ("cpu", "cpu", 0)


def test_placement_cpu_skips_llama_server_probe(monkeypatch):
    manifest = {
        "model_id": "test-gguf",
        "modality": "chat/completion",
        "backend_path": placement.LLAMA_SERVER_BACKEND,
        "capabilities": {"gguf_file": "m.gguf"},
    }

    def _forbidden(_bin=None):
        pytest.fail("probe_llama_server must not run for cpu or n_gpu_layers=0")

    monkeypatch.setattr(llama_server, "probe_llama_server", _forbidden)
    res = placement.resolve_model_placement("test-gguf", manifest, requested="cpu", entry={})
    assert (res.device, res.pool, res.n_gpu_layers) == ("cpu", "cpu", 0)


# ---------------------------------------------------------------------------
# Worker teardown calls backend.close()
# ---------------------------------------------------------------------------

def test_worker_closes_registered_backends_on_shutdown():
    from muse.cli_impl.worker import _close_registry_backends

    reg = ModalityRegistry()
    closed: list[str] = []

    class _ClosableModel:
        model_id = "closable-gguf"

        def close(self):
            closed.append(self.model_id)

    class _PlainModel:
        model_id = "plain-model"

    reg.register("chat/completion", _ClosableModel())
    reg.register("embedding/text", _PlainModel())

    _close_registry_backends(reg)
    assert closed == ["closable-gguf"]
