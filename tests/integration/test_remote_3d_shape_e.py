# tests/integration/test_remote_3d_shape_e.py
"""Integration tests for Shap-E text-to-3D against a real muse server.

Opt-in via MUSE_REMOTE_SERVER. Skips when:
  - env unset, or
  - shap-e isn't enabled on the server.
"""
import os

import pytest
import requests


pytestmark = pytest.mark.skipif(
    not os.environ.get("MUSE_REMOTE_SERVER"),
    reason="MUSE_REMOTE_SERVER not set; integration tests skipped",
)


SHAPE_E_MODEL_ID = os.environ.get("MUSE_SHAPE_E_MODEL_ID", "shap-e")


@pytest.fixture(scope="module")
def base_url():
    return os.environ["MUSE_REMOTE_SERVER"].rstrip("/")


@pytest.fixture(scope="module")
def shape_e_loaded(base_url):
    """Skip if the configured Shap-E model isn't enabled on the server."""
    r = requests.get(f"{base_url}/v1/models", timeout=10)
    r.raise_for_status()
    ids = {m["id"] for m in r.json()["data"]}
    if SHAPE_E_MODEL_ID not in ids:
        pytest.skip(f"shap-e model {SHAPE_E_MODEL_ID} not on server")


def test_protocol_text_to_3d_returns_glb(base_url, shape_e_loaded):
    """Hard claim: a text prompt yields a non-empty GLB blob."""
    r = requests.post(
        f"{base_url}/v1/3d/generations",
        json={"model": SHAPE_E_MODEL_ID, "prompt": "a small cube", "n": 1},
        timeout=300,
    )
    r.raise_for_status()
    body = r.json()
    assert body["model"] == SHAPE_E_MODEL_ID
    assert len(body["data"]) >= 1
    # b64_json or url; both valid per codec contract.
    item = body["data"][0]
    assert "b64_json" in item or "url" in item


def test_protocol_image_to_3d_rejected(base_url, shape_e_loaded):
    """Shap-E declares supports_image_to_3d: false, so image-to-3D
    returns 400."""
    r = requests.post(
        f"{base_url}/v1/3d/from-image",
        files={"image": ("x.png", b"fake-image-bytes", "image/png")},
        data={"model": SHAPE_E_MODEL_ID},
        timeout=30,
    )
    assert r.status_code == 400


def test_protocol_capability_advertised_in_models(base_url, shape_e_loaded):
    """The /v1/models endpoint advertises supports_text_to_3d=True and
    supports_image_to_3d=False for shap-e."""
    r = requests.get(f"{base_url}/v1/models", timeout=10)
    r.raise_for_status()
    entries = [m for m in r.json()["data"] if m["id"] == SHAPE_E_MODEL_ID]
    assert len(entries) == 1
    caps = entries[0].get("capabilities") or {}
    assert caps.get("supports_text_to_3d") is True
    assert caps.get("supports_image_to_3d") is False


def test_protocol_legacy_string_prompt_unaffected(base_url, shape_e_loaded):
    """Existing prompt-string shape works (regression watchdog)."""
    r = requests.post(
        f"{base_url}/v1/3d/generations",
        json={"model": SHAPE_E_MODEL_ID, "prompt": "a sphere"},
        timeout=300,
    )
    assert r.status_code == 200


def test_observe_shape_e_describes_simple_prompt(base_url, shape_e_loaded):
    """Watchdog: not a hard claim. Logs the response size so a human
    can spot quality drift across runs."""
    r = requests.post(
        f"{base_url}/v1/3d/generations",
        json={
            "model": SHAPE_E_MODEL_ID,
            "prompt": "a chair shaped like an avocado",
            "n": 1,
        },
        timeout=300,
    )
    r.raise_for_status()
    body = r.json()
    item = body["data"][0]
    if "b64_json" in item:
        size = len(item["b64_json"])
    else:
        size = len(item.get("url", ""))
    print(f"\n[observed shape-e GLB blob size]: {size} chars")
