"""Slow e2e: boot in-process supervisor with smolvlm-256m-instruct,
issue a chat completion with a small data-URL image.

No supervisor_process fixture exists; follows the inline subprocess
pattern from test_e2e_supervisor.py.  Test skips gracefully when
smolvlm-256m-instruct weights are not cached locally (HF hub cache
absent), so it does not require a network download on CI fast-lane or
on developer machines that have not pulled the model.

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
# Skip guard: model weights must already be in the HF hub cache.
# ---------------------------------------------------------------------------

_HF_REPO = "HuggingFaceTB/SmolVLM-256M-Instruct"

# huggingface_hub stores snapshots under
#   {HF_HOME|~/.cache/huggingface}/hub/models--{org}--{name}/snapshots/
_HF_CACHE_ROOT = Path(
    os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")
) / "hub"
_MODEL_CACHE_DIR = _HF_CACHE_ROOT / "models--HuggingFaceTB--SmolVLM-256M-Instruct"
_SMOLVLM_INSTALLED = _MODEL_CACHE_DIR.exists() and any(
    _MODEL_CACHE_DIR.glob("snapshots/*/config.json")
)


def _data_url_for_red_square(side: int = 8) -> str:
    """Return a tiny PNG data-URL (solid red square) for VLM tests."""
    try:
        from PIL import Image
        img = Image.new("RGB", (side, side), "red")
        buf = BytesIO()
        img.save(buf, format="PNG")
    except ImportError:
        # Fallback: hard-coded 8×8 solid-red PNG bytes (hand-crafted minimal PNG).
        # Produced once offline; valid PNG that Pillow / transformers can decode.
        _RAW = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x08\x00\x00\x00"
            b"\x08\x08\x02\x00\x00\x00Km)\x1d\x00\x00\x00HIDAT"
            b"x\x9cc\xfc\xcf\xc0\xc0\xc0\xf0\x9f\x81\x81\x81\x81\x01"
            b"\x18\x18\x18\x00\x00\x00\x00\xff\xff\x03\x00<\x00\x01"
            b"\xb4V\xf8,\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return "data:image/png;base64," + base64.b64encode(_RAW).decode()
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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


@pytest.mark.timeout(180)
@pytest.mark.skipif(
    not _SMOLVLM_INSTALLED,
    reason=(
        "smolvlm-256m-instruct weights not cached locally "
        f"(expected at {_MODEL_CACHE_DIR}); "
        "run `muse pull smolvlm-256m-instruct` to install."
    ),
)
def test_supervisor_serves_vlm_chat_completion(tmp_path, monkeypatch):
    """Boot a real supervisor with smolvlm-256m-instruct and issue a
    VLM chat completion request with a small data-URL image.

    Verifies:
    - /v1/models advertises ``supports_vision: true`` for the model.
    - POST /v1/chat/completions with an image_url content part returns
      a non-empty assistant reply with the correct model id.
    """
    monkeypatch.setenv("MUSE_CATALOG_DIR", str(tmp_path))
    env = os.environ.copy()
    env["MUSE_CATALOG_DIR"] = str(tmp_path)

    # smolvlm-256m-instruct is a bundled script; the supervisor discovers
    # it via discover_models without needing a catalog.json entry.
    # Use an unusual port to avoid collisions with other slow-lane tests.
    port = 18780
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

        # /v1/models must list smolvlm-256m-instruct with supports_vision=True.
        r = httpx.get(f"{base_url}/v1/models", timeout=10.0)
        assert r.status_code == 200
        models = r.json()["data"]
        smolvlm = next(
            (m for m in models if m["id"] == "smolvlm-256m-instruct"), None
        )
        assert smolvlm is not None, (
            f"smolvlm-256m-instruct not in /v1/models; got: "
            f"{[m['id'] for m in models]}"
        )
        caps = smolvlm.get("capabilities") or {}
        assert caps.get("supports_vision") is True, (
            f"expected supports_vision=True, got capabilities={caps}"
        )

        # Issue a chat completion with one small data-URL image.
        # Cold-load latency for smolvlm on CPU: ~15-60s; timeout=150s.
        r = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "smolvlm-256m-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is this?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": _data_url_for_red_square()},
                            },
                        ],
                    }
                ],
                "max_tokens": 40,
            },
            timeout=150.0,
        )
        assert r.status_code == 200, (
            f"POST /v1/chat/completions returned {r.status_code}: {r.text}"
        )
        body = r.json()
        assert body.get("model") == "smolvlm-256m-instruct", (
            f"expected model='smolvlm-256m-instruct', got {body.get('model')!r}"
        )
        text = body["choices"][0]["message"]["content"]
        assert text, "expected non-empty assistant reply, got empty string"

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)
