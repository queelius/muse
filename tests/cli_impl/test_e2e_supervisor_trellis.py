"""Slow e2e: boot in-process supervisor with trellis-image enabled,
issue an image-to-3D request, assert GLB bytes returned.

No supervisor_process fixture exists; follows the inline subprocess
pattern from test_e2e_supervisor_shape_e.py. Test skips when
trellis-image weights are not cached locally (HF hub cache absent)
OR when no CUDA device is available. TRELLIS is GPU-required; CPU
inference is impractical (multiple minutes per request).

Marked @pytest.mark.slow so it is excluded from the fast lane
(`pytest -m "not slow"`).
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

import httpx
import pytest


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Skip guard: trellis-image weights must be in the HF hub cache.
# ---------------------------------------------------------------------------

_HF_REPO = "JeffreyXiang/TRELLIS-image-large"

# huggingface_hub stores snapshots under
#   {HF_HOME|~/.cache/huggingface}/hub/models--{org}--{name}/snapshots/
_HF_CACHE_ROOT = Path(
    os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")
) / "hub"
_MODEL_CACHE_DIR = _HF_CACHE_ROOT / "models--JeffreyXiang--TRELLIS-image-large"
_TRELLIS_INSTALLED = _MODEL_CACHE_DIR.exists() and (
    any(_MODEL_CACHE_DIR.glob("snapshots/*/config.json"))
    or any(_MODEL_CACHE_DIR.glob("snapshots/*/pipeline.py"))
)


# ---------------------------------------------------------------------------
# Skip guard: CUDA must be available (trellis-image requires cuda).
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except ImportError:
        return False


_GPU_AVAILABLE = _gpu_available()


# ---------------------------------------------------------------------------
# Helper: poll until the gateway is ready or the deadline passes.
# ---------------------------------------------------------------------------


def _wait_for_gateway(base_url: str, timeout_s: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# The slow e2e test.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    not _TRELLIS_INSTALLED,
    reason=(
        "trellis-image weights not cached locally "
        f"(expected at {_MODEL_CACHE_DIR}); "
        "run `muse pull trellis-image` to install."
    ),
)
@pytest.mark.skipif(
    not _GPU_AVAILABLE,
    reason="no CUDA device available; trellis-image requires cuda",
)
def test_supervisor_serves_trellis_image_to_3d(tmp_path, monkeypatch):
    """Boot a real supervisor with trellis-image and issue an image-to-3D request.

    Verifies:
    - /v1/models advertises ``supports_image_to_3d: true`` for trellis-image.
    - /v1/models advertises ``supports_text_to_3d: false`` for trellis-image.
    - POST /v1/3d/from-image (multipart) returns a non-empty b64_json GLB
      blob with the correct model id and GLB magic bytes (b'glTF').
    """
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    env = os.environ.copy()
    env["MUSE_CATALOG_DIR"] = str(tmp_path)

    # trellis-image is a resolver-pulled (curated) model; the supervisor
    # discovers it from catalog.json. Use an unusual port to avoid
    # collisions with other slow-lane tests (18790=shap-e, 18780=vlm).
    port = 18792
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "muse.cli",
            "serve", "--port", str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        # Wait up to 30s for the gateway to come up (lazy boot is instant;
        # we only need the gateway, not the worker).
        if not _wait_for_gateway(base_url, timeout_s=30.0):
            stdout, stderr = proc.communicate(timeout=5)
            pytest.fail(
                "gateway never became ready.\n"
                f"stdout: {stdout.decode()[:1000]}\n"
                f"stderr: {stderr.decode()[:1000]}"
            )

        # /v1/models must list trellis-image with the correct capability flags.
        r = httpx.get(f"{base_url}/v1/models", timeout=10.0)
        assert r.status_code == 200
        models = r.json()["data"]
        trellis = next(
            (m for m in models if m["id"] == "trellis-image"), None
        )
        assert trellis is not None, (
            f"trellis-image not in /v1/models; got: "
            f"{[m['id'] for m in models]}"
        )
        caps = trellis.get("capabilities") or {}
        assert caps.get("supports_image_to_3d") is True, (
            f"expected supports_image_to_3d=True, got capabilities={caps}"
        )
        assert caps.get("supports_text_to_3d") is False, (
            f"expected supports_text_to_3d=False, got capabilities={caps}"
        )

        # Build a minimal test image (red square).
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not available in test process")
        img = Image.new("RGB", (256, 256), "red")
        buf = BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        # Issue an image-to-3D request via the multipart route.
        # Cold-load latency for trellis-image on CUDA: potentially 30-120s;
        # inference: potentially 60-300s. Use timeout=480s.
        with httpx.Client(timeout=480.0) as client:
            r = client.post(
                f"{base_url}/v1/3d/from-image",
                files={"image": ("input.png", png_bytes, "image/png")},
                data={"model": "trellis-image"},
            )
        assert r.status_code == 200, (
            f"POST /v1/3d/from-image returned {r.status_code}: {r.text}"
        )
        body = r.json()
        assert body.get("model") == "trellis-image", (
            f"expected model='trellis-image', got {body.get('model')!r}"
        )
        data = body.get("data", [])
        assert len(data) >= 1, "expected at least one item in data"
        item = data[0]
        # Default response_format is b64_json.
        assert "b64_json" in item, (
            f"expected b64_json key in data item; got keys: {list(item.keys())}"
        )
        # Decode and verify GLB magic bytes.
        glb_bytes = base64.b64decode(item["b64_json"])
        assert len(glb_bytes) > 0, "GLB payload is empty"
        # GLB magic: first 4 bytes are ASCII 'glTF' (0x67 0x6C 0x54 0x46).
        assert glb_bytes[:4] == b"glTF", (
            f"GLB magic mismatch; got first 4 bytes: {glb_bytes[:4]!r}"
        )

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
