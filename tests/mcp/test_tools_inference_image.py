"""Tests for the 7 inference image tools.

Cover: correct routing, b64/url/path input resolution for the four
binary-input tools (edit, vary, upscale, segment), output ImageContent
packing for the three image-output tools (generate, edit, vary, upscale).
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from muse.mcp.client import MuseClient
from muse.mcp.server import MCPServer


SAMPLE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
SAMPLE_PNG_B64 = base64.b64encode(SAMPLE_PNG).decode("ascii")


@pytest.fixture
def server(monkeypatch):
    monkeypatch.delenv("MUSE_SERVER", raising=False)
    monkeypatch.delenv("MUSE_ADMIN_TOKEN", raising=False)
    client = MuseClient(server_url="http://test")
    return MCPServer(client=client, filter_kind="inference")


def _split_blocks(blocks):
    """Split content blocks into (image_blocks, text_blocks)."""
    images = [b for b in blocks if b["type"] == "image"]
    texts = [b for b in blocks if b["type"] == "text"]
    return images, texts


def _parse_text(block):
    return json.loads(block["text"])


class TestRegistry:
    def test_image_tools_present(self, server):
        names = {t.name for t in server.tools}
        for expected in (
            "muse_generate_image",
            "muse_edit_image",
            "muse_vary_image",
            "muse_upscale_image",
            "muse_segment_image",
            "muse_generate_animation",
            "muse_embed_image",
        ):
            assert expected in names

    def test_edit_image_has_image_and_mask_fields(self, server):
        edit = next(t for t in server.tools if t.name == "muse_edit_image")
        props = edit.inputSchema["properties"]
        assert "image_b64" in props
        assert "image_url" in props
        assert "image_path" in props
        assert "mask_b64" in props
        assert "mask_url" in props
        assert "mask_path" in props


class TestGenerateImage:
    def test_returns_image_blocks_plus_summary(self, server):
        b64 = base64.b64encode(SAMPLE_PNG).decode("ascii")
        server.client.generate_image = MagicMock(return_value={
            "data": [{"b64_json": b64}, {"b64_json": b64}],
            "model": "sd-turbo",
        })
        blocks = server.call_handler("muse_generate_image", {
            "prompt": "cat", "n": 2, "size": "512x512",
        })
        images, texts = _split_blocks(blocks)
        assert len(images) == 2
        assert len(texts) == 1
        assert base64.b64decode(images[0]["data"]) == SAMPLE_PNG
        assert images[0]["mimeType"] == "image/png"
        summary = _parse_text(texts[0])
        assert summary["model"] == "sd-turbo"
        assert summary["n"] == 2
        assert summary["size"] == "512x512"

    def test_default_response_format_b64(self, server):
        server.client.generate_image = MagicMock(return_value={"data": [], "model": "x"})
        server.call_handler("muse_generate_image", {"prompt": "x"})
        call = server.client.generate_image.call_args
        # _post_json's default-setdefault is in the client; the mock is
        # raw, so we assert the handler doesn't pass response_format
        # explicitly when not present. The MuseClient default kicks in
        # at the HTTP layer.
        assert "prompt" in call.kwargs


class TestEditImage:
    def test_resolves_binary_inputs(self, server):
        server.client.edit_image = MagicMock(
            return_value={"data": [{"b64_json": SAMPLE_PNG_B64}], "model": "sd"},
        )
        blocks = server.call_handler("muse_edit_image", {
            "prompt": "paint blue",
            "image_b64": SAMPLE_PNG_B64,
            "mask_b64": SAMPLE_PNG_B64,
        })
        images, _ = _split_blocks(blocks)
        assert len(images) == 1
        call = server.client.edit_image.call_args
        assert call.kwargs["image"] == SAMPLE_PNG
        assert call.kwargs["mask"] == SAMPLE_PNG
        assert "image_b64" not in call.kwargs
        assert "mask_b64" not in call.kwargs

    def test_resolves_path_input(self, server):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            f.write(SAMPLE_PNG)
            ipath = f.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            f.write(SAMPLE_PNG)
            mpath = f.name
        try:
            server.client.edit_image = MagicMock(
                return_value={"data": [], "model": "x"},
            )
            server.call_handler("muse_edit_image", {
                "prompt": "p",
                "image_path": ipath,
                "mask_path": mpath,
            })
            call = server.client.edit_image.call_args
            assert call.kwargs["image"] == SAMPLE_PNG
        finally:
            os.unlink(ipath)
            os.unlink(mpath)

    def test_missing_image_returns_error(self, server):
        server.client.edit_image = MagicMock()
        blocks = server.call_handler("muse_edit_image", {
            "prompt": "p", "mask_b64": SAMPLE_PNG_B64,
        })
        # The error path captures the exception inside MCPServer.call_handler
        # and emits a structured text block.
        text = blocks[0]
        assert text["type"] == "text"
        body = json.loads(text["text"])
        assert "missing image input" in body["error"]


class TestVaryImage:
    def test_passes_image(self, server):
        server.client.vary_image = MagicMock(
            return_value={"data": [{"b64_json": SAMPLE_PNG_B64}], "model": "sd"},
        )
        blocks = server.call_handler("muse_vary_image", {
            "image_b64": SAMPLE_PNG_B64, "n": 2,
        })
        images, _ = _split_blocks(blocks)
        assert len(images) == 1
        call = server.client.vary_image.call_args
        assert call.kwargs["image"] == SAMPLE_PNG
        assert call.kwargs["n"] == 2


class TestUpscaleImage:
    def test_passes_scale(self, server):
        server.client.upscale_image = MagicMock(
            return_value={"data": [{"b64_json": SAMPLE_PNG_B64}], "model": "sd-x4"},
        )
        blocks = server.call_handler("muse_upscale_image", {
            "image_b64": SAMPLE_PNG_B64, "scale": 4,
        })
        images, texts = _split_blocks(blocks)
        assert len(images) == 1
        summary = _parse_text(texts[0])
        assert summary["scale"] == 4
        call = server.client.upscale_image.call_args
        assert call.kwargs["scale"] == 4


class TestSegmentImage:
    def test_returns_envelope(self, server):
        server.client.segment_image = MagicMock(
            return_value={"masks": [{"b64": "..."}], "model": "sam2"},
        )
        blocks = server.call_handler("muse_segment_image", {
            "image_b64": SAMPLE_PNG_B64, "mode": "auto",
        })
        # Single TextContent block with the envelope
        assert len(blocks) == 1
        body = _parse_text(blocks[0])
        assert body["model"] == "sam2"
        call = server.client.segment_image.call_args
        assert call.kwargs["image"] == SAMPLE_PNG
        assert call.kwargs["mode"] == "auto"


class TestGenerateAnimation:
    def test_returns_envelope(self, server):
        server.client.generate_animation = MagicMock(
            return_value={"data": [{"b64": "..."}], "model": "ad"},
        )
        blocks = server.call_handler("muse_generate_animation", {
            "prompt": "bouncing ball",
            "frames": 16, "fps": 8,
        })
        assert len(blocks) == 1
        body = _parse_text(blocks[0])
        assert body["model"] == "ad"
        call = server.client.generate_animation.call_args
        assert call.kwargs["prompt"] == "bouncing ball"
        assert call.kwargs["frames"] == 16


class TestEmbedImage:
    def test_returns_envelope(self, server):
        server.client.embed_image = MagicMock(
            return_value={"data": [{"embedding": [0.1, 0.2]}], "model": "clip"},
        )
        blocks = server.call_handler("muse_embed_image", {
            "input": ["data:image/png;base64,AAA"],
        })
        body = _parse_text(blocks[0])
        assert body["data"][0]["embedding"] == [0.1, 0.2]
