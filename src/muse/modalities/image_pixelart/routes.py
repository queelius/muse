"""FastAPI routes for /v1/images/pixelize, /v1/images/autotile and /v1/images/spritesheet (multipart).

Both mirror /v1/images/upscale: files arrive as multipart uploads read
through the shared bounded decoder (``muse.modalities._common.images``,
``limits.image_input_max_bytes``), the
other fields as Form parameters validated here (400 ``invalid_parameter``).
Inputs larger than ``limits.pixelart_max_input_side`` on either side, and
pixelize previews whose long side (art size x ``preview_scale``) would exceed
it, are 413 ``input_too_large``; oversize bytes are 413 ``payload_too_large``. Any
ValueError from the library is a client error (400), never a 500.

Image decoding and the CPU work run off the event loop through
``run_native_offload``; the work also runs under the backend's inference
lock, so concurrent requests cannot stack their working memory past the
model's declared ``memory_gb``.

The pixel library (numpy + Pillow) is imported inside the handlers so
importing the modality package stays light.
"""
from __future__ import annotations

import logging
import re

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from muse.core import config
from muse.core.errors import error_response
from muse.core.registry import ModalityRegistry
from muse.modalities._common.images import (
    read_and_decode_upload,
    read_and_decode_uploads,
)
from muse.modalities._common.inference import offload_call

logger = logging.getLogger(__name__)

MODALITY = "image/pixelart"
ALLOWED_TILE_SIZES = (16, 24, 32, 48)
MAX_PREVIEW_SCALE = 16
_PAIR_RE = re.compile(r"^\s*(\d{1,5})\s*[xX]\s*(\d{1,5})\s*$")
_SIDE_ENV = "MUSE_PIXELART_MAX_INPUT_SIDE"


class _BadParameter(ValueError):
    pass


def _max_input_side() -> int:
    """Per-request input-side cap (env MUSE_PIXELART_MAX_INPUT_SIDE)."""
    return int(config.get_positive("limits.pixelart_max_input_side"))


def _parse_grid(value: str) -> int | None:
    v = (value or "").strip().lower()
    if v == "auto":
        return None
    if not v.isdigit() or int(v) < 1:
        raise _BadParameter("grid must be 'auto' or an integer of at least 1")
    return int(v)


def _parse_pair(value: str | None, name: str, shape: str) -> tuple[int, int] | None:
    if value is None or not value.strip():
        return None
    m = _PAIR_RE.match(value)
    if not m or int(m.group(1)) < 1 or int(m.group(2)) < 1:
        raise _BadParameter(f"{name} must be {shape} with positive integers (e.g. 2x2)")
    return int(m.group(1)), int(m.group(2))


def _parse_palette(value: str | None):
    from muse.modalities.image_pixelart.palettes import parse_palette

    if value is None:
        return None
    try:
        return parse_palette(value)
    except ValueError as e:
        raise _BadParameter(str(e)) from e


def _parse_colour_options(background: str, outline: str) -> tuple[str, str]:
    """Validate ``background`` / ``outline`` before any upload is decoded."""
    from muse.modalities.image_pixelart.pixelize import check_background, check_outline

    bg, ol = (background or "").strip(), (outline or "").strip()
    try:
        check_background(bg)
        check_outline(ol)
    except ValueError as e:
        raise _BadParameter(str(e)) from e
    return bg, ol


async def _run(backend, call, inputs: list):
    """Run ``call`` off-loop under the inference lock; closes ``inputs``."""
    return await offload_call(call, lock=backend._inference_lock, inputs=inputs)


def build_router(registry: ModalityRegistry) -> APIRouter:
    router = APIRouter(prefix="/v1/images", tags=[MODALITY])

    @router.post("/pixelize")
    async def pixelize(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        grid: str = Form("auto"),
        target_size: str | None = Form(None),
        palette: str = Form("auto:32"),
        background: str = Form("auto"),
        remove_enclosed: bool = Form(False, description=(
            "Also key background-coloured regions enclosed by the sprite (with "
            "background=auto) of at least max(4 cells, 0.05% of the grid); can "
            "remove sprite pixels that match the background.")),
        alpha_threshold: int = Form(128),
        cleanup: bool | None = Form(None),
        outline: str = Form("none"),
        seamless: bool = Form(False),
        sheet: str | None = Form(None),
        preview_scale: int = Form(1),
        response_format: str = Form("json"),
    ):
        if response_format not in ("json", "png"):
            return error_response(400, "invalid_parameter",
                                  "response_format must be 'json' or 'png'")
        if not 1 <= preview_scale <= MAX_PREVIEW_SCALE:
            return error_response(400, "invalid_parameter",
                                  f"preview_scale must be in [1, {MAX_PREVIEW_SCALE}]")
        if not 0 <= alpha_threshold <= 255:
            return error_response(400, "invalid_parameter",
                                  "alpha_threshold must be in [0, 255]")
        try:
            target = _parse_pair(target_size, "target_size", "WxH")
            grid_value = _parse_grid(grid)
            if target is not None and grid_value is not None:
                raise _BadParameter("grid and target_size are mutually exclusive")
            sheet_value = _parse_pair(sheet, "sheet", "CxR")
            spec = _parse_palette(palette)
            background_value, outline_value = _parse_colour_options(background, outline)
        except _BadParameter as e:
            return error_response(400, "invalid_parameter", str(e))

        backend = registry.get_or_404(MODALITY, model)

        from muse.modalities.image_pixelart.codec import encode_pixelize, png_bytes
        from muse.modalities.image_pixelart.pixelize import (
            PixelizeOptions,
            PreviewTooLargeError,
        )

        opts = PixelizeOptions(
            grid=grid_value, target_size=target, palette=spec,
            background=background_value, remove_enclosed=remove_enclosed,
            alpha_threshold=alpha_threshold,
            cleanup=cleanup, outline=outline_value, seamless=seamless,
            sheet=sheet_value, preview_scale=preview_scale,
            max_preview_side=_max_input_side(),
        )
        source, error = await read_and_decode_upload(
            image, "image", max_side=_max_input_side(), side_setting=_SIDE_ENV,
        )
        if error is not None:
            return error

        def _call():
            result = backend.pixelize(source, opts)
            if response_format == "png":
                return png_bytes(result.image)
            return encode_pixelize(result, backend.model_id)

        try:
            out = await _run(backend, _call, [source])
        except PreviewTooLargeError as e:
            return error_response(
                413, "input_too_large",
                f"{e} (limits.pixelart_max_input_side; set {_SIDE_ENV} to raise)",
            )
        except ValueError as e:
            return error_response(400, "invalid_parameter", str(e))
        except Exception:  # noqa: BLE001
            logger.exception("pixelize failed")
            return error_response(500, "internal_error", "pixelize failed; see server logs")

        if response_format == "png":
            return Response(content=out, media_type="image/png")
        return JSONResponse(content=out)

    @router.post("/autotile")
    async def autotile(
        a2: UploadFile | None = File(None),
        fill: UploadFile | None = File(None),
        base: UploadFile | None = File(None),
        model: str | None = Form(None),
        tile_size: int = Form(48),
        edge_width: int | None = Form(None),
        outline: str | None = Form(None),
        highlight: bool | None = Form(None),
        roughness: float | None = Form(None),
        corner_radius: int | None = Form(None),
        seed: int | None = Form(None),
        palette: str | None = Form(None),
        name: str = Form("terrain"),
        response_format: str = Form("json"),
    ):
        if response_format not in ("json", "zip"):
            return error_response(400, "invalid_parameter",
                                  "response_format must be 'json' or 'zip'")
        if tile_size not in ALLOWED_TILE_SIZES:
            return error_response(
                400, "invalid_parameter",
                f"tile_size must be one of {list(ALLOWED_TILE_SIZES)}",
            )
        if (a2 is None) == (fill is None):
            return error_response(400, "invalid_parameter",
                                  "exactly one of a2 or fill is required")
        if a2 is not None and base is not None:
            return error_response(400, "invalid_parameter",
                                  "base is only used with fill, not with an a2 block")
        if roughness is not None and not 0.0 <= roughness <= 1.0:
            return error_response(400, "invalid_parameter", "roughness must be in [0, 1]")
        from muse.modalities.image_pixelart.exporters import safe_name

        try:
            safe_name(name)
            spec = _parse_palette(palette)
        except ValueError as e:
            return error_response(400, "invalid_parameter", str(e))

        style_fields = {
            "edge_width": edge_width, "outline": outline, "highlight": highlight,
            "roughness": roughness, "corner_radius": corner_radius, "seed": seed,
        }
        given = {k: v for k, v in style_fields.items() if v is not None}

        backend = registry.get_or_404(MODALITY, model)

        from muse.modalities.image_pixelart.autotile import EdgeStyle
        from muse.modalities.image_pixelart.codec import autotile_zip, encode_autotile
        from muse.modalities.image_pixelart.protocol import AutotileRequest
        from muse.modalities.image_pixelart.runtime import layout_entries

        uploads = [
            (field, upload)
            for field, upload in (("a2", a2), ("fill", fill), ("base", base))
            if upload is not None
        ]
        decoded, error = await read_and_decode_uploads(
            uploads, max_side=_max_input_side(), side_setting=_SIDE_ENV,
        )
        if error is not None:
            return error
        images = {field: img for (field, _), img in zip(uploads, decoded)}

        style = None
        if fill is not None:
            if isinstance(given.get("outline"), str):
                given["outline"] = given["outline"].strip()
            style = EdgeStyle(**given)
        request = AutotileRequest(
            tile_size=tile_size, a2=images.get("a2"), fill=images.get("fill"),
            base=images.get("base"), style=style, palette=spec, name=name,
        )

        def _call():
            result = backend.autotile(request)
            if a2 is not None and given:
                result.metadata["ignored"] = sorted(given)
            layout = layout_entries(result.tiles, tile_size)
            if response_format == "zip":
                return autotile_zip(result, name, layout)
            return encode_autotile(result, backend.model_id, tile_size, layout)

        try:
            out = await _run(backend, _call, list(images.values()))
        except ValueError as e:
            return error_response(400, "invalid_parameter", str(e))
        except Exception:  # noqa: BLE001
            logger.exception("autotile failed")
            return error_response(500, "internal_error", "autotile failed; see server logs")

        if response_format == "zip":
            return Response(
                content=out, media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="{name}.zip"'},
            )
        return JSONResponse(content=out)

    @router.post("/spritesheet")
    async def spritesheet(
        image: UploadFile = File(...),
        model: str | None = Form(None),
        grid: str = Form("3x2"),
        frame_size: str = Form("48x48"),
        fit_size: str | None = Form(None),
        mirror_mode: str = Form("none"),
        background: str = Form("auto"),
        bg_threshold: int = Form(236),
        fringe_threshold: int = Form(212),
        clear_enclosed_white: bool = Form(True),
        clean_stray: bool = Form(True),
        align: str = Form("bottom"),
        bottom_padding: int = Form(2),
        resample: str = Form("nearest"),
        hard_alpha: bool = Form(True),
        alpha_threshold: int = Form(128),
        palette: str = Form("none"),
        output_cols: int | None = Form(None),
        response_format: str = Form("json"),
    ):
        if response_format not in ("json", "png", "zip"):
            return error_response(400, "invalid_parameter",
                                  "response_format must be 'json', 'png' or 'zip'")
        if mirror_mode not in ("none", "5to8"):
            return error_response(400, "invalid_parameter",
                                  "mirror_mode must be 'none' or '5to8'")
        if background not in ("auto", "white", "none"):
            return error_response(400, "invalid_parameter",
                                  "background must be 'auto', 'white' or 'none'")
        if align not in ("bottom", "center"):
            return error_response(400, "invalid_parameter",
                                  "align must be 'bottom' or 'center'")
        if resample not in ("nearest", "lanczos"):
            return error_response(400, "invalid_parameter",
                                  "resample must be 'nearest' or 'lanczos'")
        if not 0 <= bg_threshold <= 255:
            return error_response(400, "invalid_parameter",
                                  "bg_threshold must be in [0, 255]")
        if not 0 <= fringe_threshold <= 255:
            return error_response(400, "invalid_parameter",
                                  "fringe_threshold must be in [0, 255]")
        if not 0 <= alpha_threshold <= 255:
            return error_response(400, "invalid_parameter",
                                  "alpha_threshold must be in [0, 255]")
        if bottom_padding < 0:
            return error_response(400, "invalid_parameter",
                                  "bottom_padding must be non-negative")
        if output_cols is not None and output_cols < 1:
            return error_response(400, "invalid_parameter",
                                  "output_cols must be a positive integer")

        try:
            grid_value = _parse_pair(grid, "grid", "CxR")
            if grid_value is None:
                raise _BadParameter("grid must be CxR with positive integers (e.g. 3x2)")
            frame_value = _parse_pair(frame_size, "frame_size", "WxH")
            if frame_value is None:
                raise _BadParameter("frame_size must be WxH with positive integers (e.g. 48x48)")
            fit_value = _parse_pair(fit_size, "fit_size", "WxH")
            if fit_value is not None and (
                fit_value[0] > frame_value[0] or fit_value[1] > frame_value[1]
            ):
                raise _BadParameter(
                    f"fit_size {fit_value[0]}x{fit_value[1]} exceeds "
                    f"frame_size {frame_value[0]}x{frame_value[1]}"
                )
            if align == "bottom" and bottom_padding >= frame_value[1]:
                raise _BadParameter(
                    f"bottom_padding ({bottom_padding}) must be smaller than "
                    f"frame height ({frame_value[1]})"
                )
            if mirror_mode == "5to8" and grid_value[0] * grid_value[1] < 5:
                raise _BadParameter(
                    f"mirror_mode='5to8' requires a grid with at least 5 cells, "
                    f"got {grid_value[0]}x{grid_value[1]}"
                )
            spec = _parse_palette(palette)
        except _BadParameter as e:
            return error_response(400, "invalid_parameter", str(e))

        cap = _max_input_side()
        n_frames = 8 if mirror_mode == "5to8" else grid_value[0] * grid_value[1]
        out_cols = (
            output_cols
            if output_cols is not None
            else (4 if mirror_mode == "5to8" else grid_value[0])
        )
        out_rows = (n_frames + out_cols - 1) // out_cols
        out_w, out_h = out_cols * frame_value[0], out_rows * frame_value[1]
        if max(out_w, out_h) > cap:
            return error_response(
                413, "input_too_large",
                f"spritesheet atlas {out_w}x{out_h} exceeds max side {cap} "
                f"(limits.pixelart_max_input_side; set {_SIDE_ENV} to raise)",
            )

        backend = registry.get_or_404(MODALITY, model)

        from muse.modalities.image_pixelart.codec import (
            encode_spritesheet,
            png_bytes,
            spritesheet_zip,
        )
        from muse.modalities.image_pixelart.spritesheet import (
            SpritesheetOptions,
            SpritesheetTooLargeError,
        )

        opts = SpritesheetOptions(
            grid=grid_value,
            frame_size=frame_value,
            fit_size=fit_value,
            mirror_mode=mirror_mode,
            background=background,
            bg_threshold=bg_threshold,
            fringe_threshold=fringe_threshold,
            clear_enclosed_white=clear_enclosed_white,
            clean_stray=clean_stray,
            align=align,
            bottom_padding=bottom_padding,
            resample=resample,
            hard_alpha=hard_alpha,
            alpha_threshold=alpha_threshold,
            palette=spec,
            output_cols=output_cols,
            max_output_side=cap,
        )
        source, error = await read_and_decode_upload(
            image, "image", max_side=cap, side_setting=_SIDE_ENV,
        )
        if error is not None:
            return error

        def _call():
            result = backend.spritesheet(source, opts)
            if response_format == "png":
                return png_bytes(result.atlas)
            if response_format == "zip":
                return spritesheet_zip(result)
            return encode_spritesheet(result, backend.model_id)

        try:
            out = await _run(backend, _call, [source])
        except SpritesheetTooLargeError as e:
            return error_response(
                413, "input_too_large",
                f"{e} (limits.pixelart_max_input_side; set {_SIDE_ENV} to raise)",
            )
        except ValueError as e:
            return error_response(400, "invalid_parameter", str(e))
        except Exception:  # noqa: BLE001
            logger.exception("spritesheet failed")
            return error_response(500, "internal_error",
                                  "spritesheet failed; see server logs")

        if response_format == "png":
            return Response(content=out, media_type="image/png")
        if response_format == "zip":
            return Response(
                content=out, media_type="application/zip",
                headers={"Content-Disposition": 'attachment; filename="spritesheet.zip"'},
            )
        return JSONResponse(content=out)

    return router
