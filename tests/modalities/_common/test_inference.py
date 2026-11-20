"""Tests for the shared locked-call and text-batch helpers."""
import json
import threading

import pytest
from fastapi.responses import JSONResponse

from muse.modalities._common.inference import check_text_batch, run_locked


class _Backend:
    def __init__(self):
        self._inference_lock = threading.Lock()


async def test_run_locked_holds_the_lock_and_returns_the_result():
    backend = _Backend()
    assert await run_locked(
        backend, lambda: backend._inference_lock.locked(),
        label="x", message="m",
    ) is True


async def test_run_locked_hides_backend_text_behind_a_500(caplog):
    def _boom():
        raise RuntimeError("secret /srv/path")

    response = await run_locked(_Backend(), _boom, label="thing", message="thing failed")
    assert isinstance(response, JSONResponse)
    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error"] == {
        "code": "internal_error", "message": "thing failed", "type": "server_error",
    }
    assert "secret" not in response.body.decode()
    assert "thing failed" in caplog.text


async def test_run_locked_reraises_passthrough_types():
    class _Typed(Exception):
        pass

    def _raise():
        raise _Typed()

    with pytest.raises(_Typed):
        await run_locked(_Backend(), _raise, label="x", message="m", passthrough=(_Typed,))


def test_check_text_batch_names_the_env_knob(monkeypatch):
    monkeypatch.setenv("MUSE_MODERATIONS_MAX_BATCH", "2")
    response = check_text_batch(
        ["a", "b", "c"],
        max_batch_key="limits.moderations_max_batch",
        max_chars_key="limits.moderations_max_chars_per_item",
    )
    assert response.status_code == 400
    assert "MUSE_MODERATIONS_MAX_BATCH=2" in response.body.decode()


@pytest.mark.parametrize("env", [
    "MUSE_MODERATIONS_MAX_BATCH", "MUSE_MODERATIONS_MAX_CHARS_PER_ITEM",
])
def test_zero_cap_falls_back_to_default(monkeypatch, env):
    """R7: 0 is not "reject everything"; it falls back to the default."""
    monkeypatch.setenv(env, "0")
    assert check_text_batch(
        ["hello"],
        max_batch_key="limits.moderations_max_batch",
        max_chars_key="limits.moderations_max_chars_per_item",
    ) is None
