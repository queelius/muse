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


def test_discovery_steers_generated_sprites_to_target_size():
    """Live on the 3060, grid=auto found no grid on 8/8 klein sprites (their
    pixels drift between about 6 and 10 px), so agents are told to size them."""
    disc = modality_discovery()["image/pixelart"]
    assert "target_size" in disc["summary"]
    assert "upscaled" in disc["summary"]
    assert _routes()["/v1/images/pixelize"]["example"]["target_size"] == "64x64"


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
    assert out == {"grid": 4, "tiles": 47}


def test_model_optional_paths_declared_and_discovered():
    import muse.modalities.image_pixelart as pkg
    from muse.core.discovery import _reset_model_optional_paths_cache, model_optional_paths

    assert pkg.MODEL_OPTIONAL_PATHS == ("/v1/images/pixelize", "/v1/images/autotile")
    assert "MODEL_OPTIONAL_PATHS" in pkg.__all__
    _reset_model_optional_paths_cache()
    try:
        paths = model_optional_paths()
    finally:
        _reset_model_optional_paths_cache()
    assert paths["/v1/images/pixelize"] == "image/pixelart"
    assert paths["/v1/images/autotile"] == "image/pixelart"
