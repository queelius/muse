"""Shared fixtures for opt-in integration tests against a real muse server.

How to run:
    MUSE_REMOTE_SERVER=http://192.168.0.225:8000 pytest tests/integration/

Default (no env var): all integration tests are skipped. They never run
in CI unless CI explicitly sets MUSE_REMOTE_SERVER.

The fixtures probe the server before each test and skip if:
  - MUSE_REMOTE_SERVER is unset
  - the server doesn't respond on /health
  - the server doesn't host the model the test requires

This keeps the tests purely additive: they augment unit tests without
ever blocking the fast suite.
"""
from __future__ import annotations

import os
from typing import Iterable

import pytest


def _server_url() -> str | None:
    """Return the muse server URL or None if integration tests are off."""
    return os.environ.get("MUSE_REMOTE_SERVER")


@pytest.fixture(scope="session")
def remote_url() -> str:
    """Skip the test if MUSE_REMOTE_SERVER is unset."""
    url = _server_url()
    if not url:
        pytest.skip("MUSE_REMOTE_SERVER not set; integration tests are opt-in")
    return url.rstrip("/")


@pytest.fixture(scope="session")
def remote_health(remote_url) -> dict:
    """Probe /health; skip if the server isn't reachable or isn't muse-shaped."""
    import httpx
    try:
        r = httpx.get(f"{remote_url}/health", timeout=5.0)
    except httpx.HTTPError as e:
        pytest.skip(f"muse server at {remote_url} not reachable: {e}")
    if r.status_code != 200:
        pytest.skip(f"muse server returned status {r.status_code} for /health")
    body = r.json()
    if "modalities" not in body or "models" not in body:
        pytest.skip(
            f"server at {remote_url} returned non-muse /health body: {body}"
        )
    return body


@pytest.fixture(scope="session")
def openai_client(remote_url):
    """OpenAI SDK client pointed at the muse server."""
    try:
        from openai import OpenAI
    except ImportError:
        pytest.skip("openai package not installed; pip install openai")
    return OpenAI(base_url=f"{remote_url}/v1", api_key="not-used")


def _require_model(remote_health: dict, model_id: str) -> None:
    """Skip if the loaded muse server doesn't have a specific model."""
    loaded = remote_health.get("models") or []
    if model_id not in loaded:
        pytest.skip(
            f"muse server doesn't have {model_id!r} loaded "
            f"(loaded: {loaded}); pull and restart to enable this test"
        )


def require_model_fixture(model_id: str):
    """Build a session-scoped fixture that skips if model_id isn't loaded."""
    @pytest.fixture(scope="session")
    def _fixture(remote_health):
        _require_model(remote_health, model_id)
        return model_id
    return _fixture


# Embedding + audio models are static (one good choice each).
qwen3_embedding = require_model_fixture("qwen3-embedding-0.6b")
kokoro_82m = require_model_fixture("kokoro-82m")


@pytest.fixture(scope="session")
def chat_model(remote_health) -> str:
    """The chat/completion model id integration tests should target.

    Defaults to qwen3.5-4b-q4 (smallest/fastest verified). Override with
    MUSE_CHAT_MODEL_ID env var to test against a different model:

      MUSE_CHAT_MODEL_ID=qwen3.5-9b-q4 pytest tests/integration/

    The fixture skips the test if the chosen model isn't loaded on the
    server, so you can keep the env var set across runs without
    breakage.
    """
    model_id = os.environ.get("MUSE_CHAT_MODEL_ID", "qwen3.5-4b-q4")
    _require_model(remote_health, model_id)
    return model_id


# Backwards-compat alias for the few tests that still want to pin to 4b.
qwen3_5_4b = require_model_fixture("qwen3.5-4b-q4")


@pytest.fixture(scope="session")
def whisper_model(remote_health) -> str:
    """The audio/transcription model id integration tests should target.

    Defaults to whisper-tiny (fastest CPU-friendly). Override with
    MUSE_WHISPER_MODEL_ID for a different model:

      MUSE_WHISPER_MODEL_ID=whisper-base pytest tests/integration/

    Skips the test if the chosen model isn't loaded on the server.
    """
    model_id = os.environ.get("MUSE_WHISPER_MODEL_ID", "whisper-tiny")
    _require_model(remote_health, model_id)
    return model_id


@pytest.fixture(scope="session")
def text_moderation_model(remote_health) -> str:
    """The text/classification model id integration tests should target.

    Defaults to text-moderation. Override via MUSE_MODERATION_MODEL_ID.
    Skips the test if the chosen model isn't loaded on the server.
    """
    model_id = os.environ.get("MUSE_MODERATION_MODEL_ID", "text-moderation")
    _require_model(remote_health, model_id)
    return model_id


@pytest.fixture(scope="session")
def rerank_model(remote_health) -> str:
    """The text/rerank model id integration tests should target.

    Defaults to bge-reranker-v2-m3. Override via MUSE_RERANK_MODEL_ID.
    Skips the test if the chosen model isn't loaded on the server.
    """
    model_id = os.environ.get("MUSE_RERANK_MODEL_ID", "bge-reranker-v2-m3")
    _require_model(remote_health, model_id)
    return model_id


@pytest.fixture(scope="session")
def audio_generation_model(remote_health) -> str:
    """The audio/generation model id integration tests should target.

    Defaults to stable-audio-open-1.0. Override via
    MUSE_AUDIO_GENERATION_MODEL_ID. Skips the test if the chosen model
    isn't loaded on the server.
    """
    model_id = os.environ.get(
        "MUSE_AUDIO_GENERATION_MODEL_ID", "stable-audio-open-1.0",
    )
    _require_model(remote_health, model_id)
    return model_id


@pytest.fixture(scope="session")
def summarization_model(remote_health) -> str:
    """The text/summarization model id integration tests should target.

    Defaults to bart-large-cnn. Override via MUSE_SUMMARIZATION_MODEL_ID.
    Skips the test if the chosen model isn't loaded on the server.
    """
    model_id = os.environ.get(
        "MUSE_SUMMARIZATION_MODEL_ID", "bart-large-cnn",
    )
    _require_model(remote_health, model_id)
    return model_id


@pytest.fixture(scope="session")
def image_embedding_model(remote_health) -> str:
    """The image/embedding model id integration tests should target.

    Defaults to dinov2-small (smallest, CPU-friendly). Override via
    MUSE_IMAGE_EMBEDDING_MODEL_ID for a different bundled / curated id.
    Skips the test if the chosen model isn't loaded on the server.
    """
    model_id = os.environ.get(
        "MUSE_IMAGE_EMBEDDING_MODEL_ID", "dinov2-small",
    )
    _require_model(remote_health, model_id)
    return model_id


@pytest.fixture(scope="session")
def audio_embedding_model(remote_health) -> str:
    """The audio/embedding model id integration tests should target.

    Defaults to mert-v1-95m (smallest, CPU-friendly, music understanding).
    Override via MUSE_AUDIO_EMBEDDING_MODEL_ID for a different bundled /
    curated id. Skips the test if the chosen model isn't loaded on the
    server.
    """
    model_id = os.environ.get(
        "MUSE_AUDIO_EMBEDDING_MODEL_ID", "mert-v1-95m",
    )
    _require_model(remote_health, model_id)
    return model_id


@pytest.fixture(scope="session")
def upscale_model(remote_health) -> str:
    """The image/upscale model id integration tests should target.

    Defaults to stable-diffusion-x4-upscaler. Override via
    MUSE_UPSCALE_MODEL_ID. Skips the test if the chosen model isn't
    loaded on the server.
    """
    model_id = os.environ.get(
        "MUSE_UPSCALE_MODEL_ID", "stable-diffusion-x4-upscaler",
    )
    _require_model(remote_health, model_id)
    return model_id


@pytest.fixture(scope="session")
def segmentation_model(remote_health) -> str:
    """The image/segmentation model id integration tests should target.

    Defaults to sam2-hiera-tiny (smallest, CPU-friendly). Override via
    MUSE_SEGMENTATION_MODEL_ID. Skips the test if the chosen model
    isn't loaded on the server.
    """
    model_id = os.environ.get(
        "MUSE_SEGMENTATION_MODEL_ID", "sam2-hiera-tiny",
    )
    _require_model(remote_health, model_id)
    return model_id
