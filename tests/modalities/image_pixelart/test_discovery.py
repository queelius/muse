"""The image/pixelart modality's discovery, probe and import hygiene."""
from __future__ import annotations

import subprocess
import sys

from muse.core.discovery import modality_discovery


def _routes():
    return {r["path"]: r for r in modality_discovery()["image/pixelart"]["routes"]}


def test_discovery_declares_tools():
    routes = _routes()
    assert set(routes) == {"/v1/images/pixelize", "/v1/images/autotile"}
    pix, auto = routes["/v1/images/pixelize"], routes["/v1/images/autotile"]
    assert pix["mcp_tool"] == "muse_pixelize_image"
    assert pix["file_fields"] == ["image"]
    assert pix["media_fields"] == ["data[].b64_json"]
    assert pix["request_content_type"] == "multipart/form-data"
    assert auto["mcp_tool"] == "muse_autotile"
    assert auto["file_fields"] == ["a2", "fill", "base"]
    assert auto["media_fields"] == ["files.a2.b64_json", "files.blob47.b64_json"]
    assert auto["text_media_fields"] == {
        "files.godot_tres": "text/plain", "files.tiled_tsx": "application/xml",
    }
    assert auto["request_content_type"] == "multipart/form-data"


def test_autotile_mcp_tool_defaults_to_zip():
    """MCP saves each output under a content-hash name, which would break the
    .tres -> <name>_blob47.png reference, so the tool defaults to the zip."""
    disc = modality_discovery()["image/pixelart"]
    auto = _routes()["/v1/images/autotile"]
    assert auto["mcp_defaults"] == {"response_format": "zip"}
    assert "zip" in disc["summary"]


def test_modality_constant_and_exports():
    import muse.modalities.image_pixelart as pkg

    assert pkg.MODALITY == "image/pixelart"
    for name in ("build_router", "ImagePixelartModel", "AutotileRequest",
                 "AutotileResult", "PixelartClient", "PROBE_DEFAULTS", "DISCOVERY"):
        assert name in pkg.__all__, name
        assert hasattr(pkg, name), name


def test_package_import_is_light():
    code = (
        "import sys; import muse.modalities.image_pixelart as p; "
        "bad = [m for m in ('PIL', 'numpy') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "", f"importing the modality package loaded: {out}"


def test_probe_defaults_run():
    from muse.modalities.image_pixelart import PROBE_DEFAULTS
    from muse.modalities.image_pixelart.runtime import PixelTools

    assert "4x" in PROBE_DEFAULTS["shape"] and "T=16" in PROBE_DEFAULTS["shape"]
    out = PROBE_DEFAULTS["call"](PixelTools(model_id="pixel-tools", hf_repo="", local_dir=None))
    assert out is not None
