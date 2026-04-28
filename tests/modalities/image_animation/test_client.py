"""Tests for AnimationsClient HTTP client."""
import base64
from unittest.mock import patch, MagicMock

from muse.modalities.image_animation.client import AnimationsClient


def test_client_returns_webp_bytes_by_default():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [{"b64_json": base64.b64encode(b"RIFFfakeWEBP").decode()}],
        "metadata": {"format": "webp"},
    }
    with patch(
        "muse.modalities.image_animation.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = AnimationsClient(server_url="http://x")
        out = c.animate("a cat", model="anim")
    assert out == b"RIFFfakeWEBP"
    payload = mock_post.call_args.kwargs["json"]
    assert payload["prompt"] == "a cat"
    assert payload["model"] == "anim"


def test_client_response_format_frames_returns_list_of_pngs():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [
            {"b64_json": base64.b64encode(b"png1").decode()},
            {"b64_json": base64.b64encode(b"png2").decode()},
        ],
        "metadata": {"format": "frames_b64"},
    }
    with patch(
        "muse.modalities.image_animation.client.requests.post",
        return_value=fake_resp,
    ):
        c = AnimationsClient(server_url="http://x")
        out = c.animate("x", response_format="frames_b64")
    assert out == [b"png1", b"png2"]


def test_client_passes_optional_fields():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [{"b64_json": base64.b64encode(b"x").decode()}],
        "metadata": {"format": "webp"},
    }
    with patch(
        "muse.modalities.image_animation.client.requests.post",
        return_value=fake_resp,
    ) as mock_post:
        c = AnimationsClient(server_url="http://x")
        c.animate(
            "x", model="m", frames=24, fps=12, loop=False,
            negative_prompt="bad", steps=30, guidance=8.0, seed=7,
        )
    payload = mock_post.call_args.kwargs["json"]
    assert payload["frames"] == 24
    assert payload["fps"] == 12
    assert payload["loop"] is False
    assert payload["negative_prompt"] == "bad"
    assert payload["seed"] == 7


def test_client_raises_on_non_200():
    fake_resp = MagicMock()
    fake_resp.status_code = 400
    fake_resp.text = '{"error": {"message": "bad"}}'
    with patch(
        "muse.modalities.image_animation.client.requests.post",
        return_value=fake_resp,
    ):
        c = AnimationsClient(server_url="http://x")
        try:
            c.animate("x")
        except RuntimeError as e:
            assert "400" in str(e)
        else:
            assert False, "expected RuntimeError"
