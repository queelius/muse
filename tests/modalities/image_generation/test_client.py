"""Tests for GenerationsClient HTTP client."""
import base64
from unittest.mock import MagicMock, patch

import pytest

from muse.modalities.image_generation.client import GenerationsClient


def test_default_base_url():
    c = GenerationsClient()
    assert c.base_url == "http://localhost:8000"


def test_custom_base_url_strips_trailing_slash():
    c = GenerationsClient(base_url="http://lan:8000/")
    assert c.base_url == "http://lan:8000"


def test_muse_server_env_var_used_when_base_url_unset(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://custom-host:9999")
    c = GenerationsClient()
    assert c.base_url == "http://custom-host:9999"


def test_generate_sends_prompt_and_returns_decoded_bytes():
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    with patch("muse.modalities.image_generation.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"b64_json": base64.b64encode(fake_png).decode()}]},
        )
        c = GenerationsClient()
        images = c.generate("a cat", n=1)

        assert len(images) == 1
        assert images[0] == fake_png

        body = mock_post.call_args.kwargs["json"]
        assert body["prompt"] == "a cat"
        assert body["response_format"] == "b64_json"
        assert body["n"] == 1


def test_generate_sends_optional_kwargs_when_provided():
    with patch("muse.modalities.image_generation.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"b64_json": base64.b64encode(b"x").decode()}]},
        )
        c = GenerationsClient()
        c.generate(
            "a bird",
            model="sd-turbo",
            n=2,
            size="256x256",
            negative_prompt="blurry",
            steps=4,
            guidance=1.5,
            seed=7,
        )
        body = mock_post.call_args.kwargs["json"]
        assert body["model"] == "sd-turbo"
        assert body["n"] == 2
        assert body["size"] == "256x256"
        assert body["negative_prompt"] == "blurry"
        assert body["steps"] == 4
        assert body["guidance"] == 1.5
        assert body["seed"] == 7


def test_generate_omits_none_optional_fields():
    """Unsupplied optionals must not appear in the request body."""
    with patch("muse.modalities.image_generation.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"b64_json": base64.b64encode(b"x").decode()}]},
        )
        c = GenerationsClient()
        c.generate("hi")
        body = mock_post.call_args.kwargs["json"]
        # These were not passed as kwargs; they shouldn't leak as null keys
        for field in ("model", "negative_prompt", "steps", "guidance", "seed"):
            assert field not in body


def test_generate_raises_on_http_error():
    with patch("muse.modalities.image_generation.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=500, text="boom")
        c = GenerationsClient()
        with pytest.raises(RuntimeError, match="500"):
            c.generate("x")


def test_generate_returns_list_of_bytes_for_n_greater_than_1():
    fake_a = b"\x89PNG" + b"A" * 10
    fake_b = b"\x89PNG" + b"B" * 10
    with patch("muse.modalities.image_generation.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [
                {"b64_json": base64.b64encode(fake_a).decode()},
                {"b64_json": base64.b64encode(fake_b).decode()},
            ]},
        )
        c = GenerationsClient()
        images = c.generate("x", n=2)
        assert images == [fake_a, fake_b]


# ---------------- ImageEditsClient (#100, v0.21.0) ----------------


def test_edits_default_base_url():
    from muse.modalities.image_generation.client import ImageEditsClient
    c = ImageEditsClient()
    assert c.base_url == "http://localhost:8000"


def test_edits_custom_base_url_strips_trailing_slash():
    from muse.modalities.image_generation.client import ImageEditsClient
    c = ImageEditsClient(base_url="http://lan:8000/")
    assert c.base_url == "http://lan:8000"


def test_edits_muse_server_env_var_used_when_base_url_unset(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://custom-host:9999")
    from muse.modalities.image_generation.client import ImageEditsClient
    c = ImageEditsClient()
    assert c.base_url == "http://custom-host:9999"


def test_edits_posts_multipart_with_image_mask_prompt():
    """edit() POSTs multipart with image+mask files and prompt+model fields."""
    from muse.modalities.image_generation.client import ImageEditsClient

    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
    with patch("muse.modalities.image_generation.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [
                {"b64_json": base64.b64encode(fake_png).decode(),
                 "revised_prompt": "y"},
            ]},
        )
        c = ImageEditsClient()
        out = c.edit(
            "make it night",
            image=b"src-bytes", mask=b"mask-bytes",
            model="sd-turbo", n=1, size="64x64",
        )

    # Result decodes correctly
    assert out == [fake_png]

    call = mock_post.call_args
    assert call.args[0].endswith("/v1/images/edits")
    files = call.kwargs["files"]
    assert "image" in files and "mask" in files
    # files map to (filename, bytes, mime) tuples
    assert files["image"][1] == b"src-bytes"
    assert files["mask"][1] == b"mask-bytes"
    data = dict(call.kwargs["data"])
    assert data["prompt"] == "make it night"
    assert data["model"] == "sd-turbo"
    assert data["n"] == "1"
    assert data["size"] == "64x64"
    assert data["response_format"] == "b64_json"


def test_edits_omits_model_field_when_none():
    """If model arg is None, the field is not sent (server falls back to default)."""
    from muse.modalities.image_generation.client import ImageEditsClient

    with patch("muse.modalities.image_generation.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"b64_json": base64.b64encode(b"x").decode(),
                                    "revised_prompt": "y"}]},
        )
        c = ImageEditsClient()
        c.edit("y", image=b"a", mask=b"b")

    data = dict(mock_post.call_args.kwargs["data"])
    assert "model" not in data


def test_edits_n_greater_than_1_returns_list():
    from muse.modalities.image_generation.client import ImageEditsClient

    a = b"\x89PNGA" + b"\x00" * 4
    b = b"\x89PNGB" + b"\x00" * 4
    with patch("muse.modalities.image_generation.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [
                {"b64_json": base64.b64encode(a).decode(), "revised_prompt": "y"},
                {"b64_json": base64.b64encode(b).decode(), "revised_prompt": "y"},
            ]},
        )
        c = ImageEditsClient()
        out = c.edit("x", image=b"i", mask=b"m", n=2)
    assert out == [a, b]


def test_edits_raises_on_http_error():
    from muse.modalities.image_generation.client import ImageEditsClient

    with patch("muse.modalities.image_generation.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=400, text="bad")
        c = ImageEditsClient()
        with pytest.raises(RuntimeError, match="400"):
            c.edit("x", image=b"i", mask=b"m")


def test_edits_response_format_url_decodes_back_to_bytes():
    from muse.modalities.image_generation.client import ImageEditsClient

    fake_png = b"\x89PNG" + b"X" * 10
    data_url = f"data:image/png;base64,{base64.b64encode(fake_png).decode()}"
    with patch("muse.modalities.image_generation.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"url": data_url, "revised_prompt": "y"}]},
        )
        c = ImageEditsClient()
        out = c.edit("x", image=b"i", mask=b"m", response_format="url")
    assert out == [fake_png]


# ---------------- ImageVariationsClient (#100, v0.21.0) ----------------


def test_variations_default_base_url():
    from muse.modalities.image_generation.client import ImageVariationsClient
    c = ImageVariationsClient()
    assert c.base_url == "http://localhost:8000"


def test_variations_custom_base_url_strips_trailing_slash():
    from muse.modalities.image_generation.client import ImageVariationsClient
    c = ImageVariationsClient(base_url="http://lan:8000/")
    assert c.base_url == "http://lan:8000"


def test_variations_muse_server_env_var_used_when_base_url_unset(monkeypatch):
    monkeypatch.setenv("MUSE_SERVER", "http://custom-host:9999")
    from muse.modalities.image_generation.client import ImageVariationsClient
    c = ImageVariationsClient()
    assert c.base_url == "http://custom-host:9999"


def test_variations_posts_multipart_with_image_only():
    """vary() POSTs multipart with image only (no prompt, no mask)."""
    from muse.modalities.image_generation.client import ImageVariationsClient

    fake_png = b"\x89PNG" + b"V" * 8
    with patch("muse.modalities.image_generation.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [
                {"b64_json": base64.b64encode(fake_png).decode()},
            ]},
        )
        c = ImageVariationsClient()
        out = c.vary(image=b"src-bytes", model="sd-turbo", n=1, size="64x64")

    assert out == [fake_png]
    call = mock_post.call_args
    assert call.args[0].endswith("/v1/images/variations")
    files = call.kwargs["files"]
    assert list(files.keys()) == ["image"]
    assert files["image"][1] == b"src-bytes"
    data = dict(call.kwargs["data"])
    assert data["model"] == "sd-turbo"
    assert "prompt" not in data
    assert "mask" not in data


def test_variations_n_greater_than_1_returns_list():
    from muse.modalities.image_generation.client import ImageVariationsClient

    a = b"\x89PNGA" + b"\x00" * 4
    b = b"\x89PNGB" + b"\x00" * 4
    with patch("muse.modalities.image_generation.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [
                {"b64_json": base64.b64encode(a).decode()},
                {"b64_json": base64.b64encode(b).decode()},
            ]},
        )
        c = ImageVariationsClient()
        out = c.vary(image=b"i", n=2)
    assert out == [a, b]


def test_variations_raises_on_http_error():
    from muse.modalities.image_generation.client import ImageVariationsClient

    with patch("muse.modalities.image_generation.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=500, text="boom")
        c = ImageVariationsClient()
        with pytest.raises(RuntimeError, match="500"):
            c.vary(image=b"i")


def test_variations_response_format_url_decodes_back_to_bytes():
    from muse.modalities.image_generation.client import ImageVariationsClient

    fake_png = b"\x89PNG" + b"Y" * 6
    data_url = f"data:image/png;base64,{base64.b64encode(fake_png).decode()}"
    with patch("muse.modalities.image_generation.client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"url": data_url}]},
        )
        c = ImageVariationsClient()
        out = c.vary(image=b"i", response_format="url")
    assert out == [fake_png]
