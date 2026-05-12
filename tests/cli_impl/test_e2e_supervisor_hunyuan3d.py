"""Slow e2e: boot in-process supervisor with hunyuan3d-2 enabled,
issue BOTH image-to-3D AND text-to-3D requests, assert GLB bytes on both.

Skipped when:
  - hunyuan3d-2 weights not cached locally, or
  - no CUDA device available.

Hunyuan3D-2 is GPU-required; CPU inference is impractical at this size.
"""
import base64
import os
import pathlib
import subprocess
import sys
import time
from io import BytesIO

import pytest
import httpx


def _hunyuan_weights_cached() -> bool:
    hf_cache = pathlib.Path(
        os.environ.get("HF_HOME", pathlib.Path.home() / ".cache" / "huggingface")
    )
    candidate = hf_cache / "hub" / "models--tencent--Hunyuan3D-2"
    if not candidate.exists():
        return False
    for snapshot in (candidate / "snapshots").glob("*"):
        if (snapshot / "config.json").exists() or (snapshot / "model_index.json").exists():
            return True
    return False


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _red_png_bytes(side: int = 256) -> bytes:
    from PIL import Image
    img = Image.new("RGB", (side, side), "red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_HUNYUAN_INSTALLED = _hunyuan_weights_cached()
_GPU_AVAILABLE = _gpu_available()


pytestmark = pytest.mark.slow


@pytest.mark.timeout(900)
@pytest.mark.skipif(
    not _HUNYUAN_INSTALLED,
    reason="hunyuan3d-2 weights not cached at ~/.cache/huggingface/hub",
)
@pytest.mark.skipif(
    not _GPU_AVAILABLE,
    reason="no CUDA device available; hunyuan3d-2 requires cuda",
)
def test_supervisor_serves_hunyuan3d_both_directions(tmp_path, monkeypatch):
    """One supervisor boot, two requests covering both directions."""
    # Use the per-test catalog dir for isolation (matches the
    # sibling Trellis/Shape-E slow e2e tests).
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path / "muse-catalog"))

    port = 18793  # avoid collisions (vlm=18790, shape-e=18791, trellis=18792)
    proc = subprocess.Popen(
        [sys.executable, "-m", "muse.cli", "serve", "--port", str(port)],
        env={**os.environ},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        # Wait for /health.
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                r = httpx.get(f"{base_url}/health", timeout=2)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            pytest.fail("supervisor never became ready")

        # Verify hunyuan3d-2 is in /v1/models.
        r = httpx.get(f"{base_url}/v1/models", timeout=10)
        r.raise_for_status()
        ids = {m["id"] for m in r.json()["data"]}
        assert "hunyuan3d-2" in ids

        # Request 1: image-to-3D.
        with httpx.Client(timeout=600) as client:
            r = client.post(
                f"{base_url}/v1/3d/from-image",
                files={"image": ("input.png", _red_png_bytes(), "image/png")},
                data={"model": "hunyuan3d-2"},
            )
            r.raise_for_status()
            body = r.json()
            assert body["model"] == "hunyuan3d-2"
            assert len(body["data"]) >= 1
            glb_b64 = body["data"][0].get("b64_json")
            assert glb_b64
            glb_bytes = base64.b64decode(glb_b64)
            assert glb_bytes[:4] == b"glTF", "image-to-3D returned non-GLB bytes"

        # Request 2: text-to-3D (same supervisor, same loaded pipeline).
        with httpx.Client(timeout=600) as client:
            r = client.post(
                f"{base_url}/v1/3d/generations",
                json={"model": "hunyuan3d-2", "prompt": "a small cube"},
            )
            r.raise_for_status()
            body = r.json()
            assert body["model"] == "hunyuan3d-2"
            assert len(body["data"]) >= 1
            glb_b64 = body["data"][0].get("b64_json")
            assert glb_b64
            glb_bytes = base64.b64decode(glb_b64)
            assert glb_bytes[:4] == b"glTF", "text-to-3D returned non-GLB bytes"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
